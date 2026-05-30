import argparse
import json
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import tensorflow  # type: ignore  # noqa: F401
except Exception:
    import importlib.machinery
    tf_stub = types.ModuleType("tensorflow")
    keras_stub = types.ModuleType("keras")
    class _DummyKerasModel:
        pass
    keras_stub.Model = _DummyKerasModel
    tf_stub.keras = keras_stub
    tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    keras_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow.keras", loader=None)
    sys.modules["tensorflow"] = tf_stub
    sys.modules["tensorflow.keras"] = keras_stub

ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "tool"
SERVER_DIR = TOOL_DIR / "server"
VISUALIZE_DIR = TOOL_DIR / "visualize"
for path in (TOOL_DIR, SERVER_DIR, VISUALIZE_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from run_visualization import initialize_config  # noqa: E402
from data_provider import DataProvider  # noqa: E402
from strategy.timevis_strategy import TimeVis  # noqa: E402

RESULT_DIR = ROOT / "refine_validation_results" / "phase3_local_visualizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 validation: verify local visualizer refine leaves the global visualizer untouched."
    )
    parser.add_argument("--content-path", required=True)
    parser.add_argument("--source-vis-id", default="1")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--focus-indices", required=True, help="Comma-separated indices")
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError("--focus-indices must contain at least one integer")
    return deduped


def clone_session(content_path: Path, source_vis_id: str, target_vis_id: str) -> None:
    src = content_path / "visualize" / f"TimeVis_{source_vis_id}"
    dst = content_path / "visualize" / f"TimeVis_{target_vis_id}"
    if dst.exists():
        raise FileExistsError(f"Temporary target already exists: {dst}")
    shutil.copytree(src, dst)


def build_strategy(content_path: Path, vis_id: str) -> TimeVis:
    info_path = content_path / "visualize" / f"TimeVis_{vis_id}" / "info.json"
    source_info = json.loads(info_path.read_text(encoding="utf-8"))
    vis_config = dict(source_info.get("vis_config", {}))
    vis_config["gpu_id"] = -1
    config = initialize_config(
        str(content_path),
        "TimeVis",
        vis_id,
        source_info.get("data_type", "Image"),
        source_info.get("task_type", "Classification"),
        vis_config,
    )
    config["vis_config"]["gpu_id"] = -1
    device = torch.device("cpu")
    data_provider = DataProvider(config, device)
    return TimeVis(config, data_provider)


def load_projection(content_path: Path, folder_name: str, epoch: int) -> np.ndarray:
    path = content_path / "visualize" / folder_name / "epochs" / f"epoch_{epoch}" / "projection.npy"
    if not path.exists():
        raise FileNotFoundError(f"Projection not found: {path}")
    return np.load(path)


def main() -> int:
    args = parse_args()
    content_path = Path(args.content_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    focus_indices = parse_int_list(args.focus_indices)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_vis_id = f"phase3_{timestamp}"
    clone_session(content_path, args.source_vis_id, temp_vis_id)

    strategy = build_strategy(content_path, temp_vis_id)
    baseline = load_projection(content_path, f"TimeVis_{temp_vis_id}", args.epoch)
    features = strategy.data_provider.get_representation(args.epoch)
    sample_tensor = torch.from_numpy(features[focus_indices].copy()).float().to(strategy.device)

    model_path = content_path / "visualize" / f"TimeVis_{temp_vis_id}" / "vis_model.pth"
    ckpt = torch.load(model_path, map_location=strategy.device)
    strategy.visualize_model.load_state_dict(ckpt["state_dict"])
    strategy.visualize_model.to(strategy.device)
    strategy.visualize_model.eval()

    with torch.no_grad():
        before_embed = strategy.visualize_model.encoder(sample_tensor).cpu().numpy()

    mask = strategy.get_focus_mask(focus_indices)
    strategy.update_ttav_context(focus_indices, "balanced", mask)
    strategy.refine(
        focus_indices=focus_indices,
        neighbor_indices=[],
        current_epoch=args.epoch,
        epochs_to_update=10,
        _skip_avg_benchmark=True,
    )

    strategy.visualize_model.eval()
    with torch.no_grad():
        after_embed = strategy.visualize_model.encoder(sample_tensor).cpu().numpy()

    refined = load_projection(content_path, f"TimeVis_{temp_vis_id}_refined", args.epoch)
    local_model = getattr(strategy, "_last_local_visualizer", None)
    if local_model is None:
        raise RuntimeError("Local visualizer was not stored on strategy")

    with torch.no_grad():
        local_embed = local_model.encoder(sample_tensor).cpu().numpy()

    result = {
        "phase": "phase3_local_visualizer",
        "content_path": args.content_path,
        "source_vis_id": args.source_vis_id,
        "temp_vis_id": temp_vis_id,
        "epoch": args.epoch,
        "focus_indices": focus_indices,
        "global_model_unchanged_max_abs": float(np.max(np.abs(after_embed - before_embed))),
        "local_vs_global_embed_max_abs": float(np.max(np.abs(local_embed - before_embed))),
        "focus_projection_shift_mean": float(np.linalg.norm(refined[np.array(focus_indices)] - baseline[np.array(focus_indices)], axis=1).mean()),
        "refine_metrics": {
            "neighbor_preservation": float(getattr(strategy, "_last_refine_np", 0.0)),
            "mean_rank_hd": float(getattr(strategy, "_last_refine_mrh", 0.0)),
            "trustworthiness": float(getattr(strategy, "_last_refine_trust", 0.0)),
            "continuity": float(getattr(strategy, "_last_refine_cont", 0.0)),
        },
    }

    out_path = output_dir / f"local_visualizer_check_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote result: {out_path}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
