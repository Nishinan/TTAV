import argparse
import csv
import json
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import types

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
import torch  # noqa: E402

RESULT_DIR = ROOT / "refine_validation_results" / "phase1b_refine_compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1b validation: run real TimeVis refine for candidate focus_set definitions."
    )
    parser.add_argument("--content-path", required=True, help="Dataset content path")
    parser.add_argument("--source-vis-id", default="1", help="Existing TimeVis session id to clone")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch to evaluate")
    parser.add_argument("--seeds", required=True, help="Comma-separated seed indices")
    parser.add_argument("--bbox", required=True, help="bbox as x_min,x_max,y_min,y_max")
    parser.add_argument("--hd-k", type=int, default=15, help="High-D neighbors per seed")
    parser.add_argument("--focus-mode", default="balanced", choices=["coarse", "balanced", "fine"])
    parser.add_argument("--output-dir", default=str(RESULT_DIR), help="Directory to store outputs")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError("--seeds must contain at least one valid integer index")
    return deduped


def parse_bbox(raw: str) -> dict[str, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must be x_min,x_max,y_min,y_max")
    x_min, x_max, y_min, y_max = map(float, parts)
    return {
        "x_min": min(x_min, x_max),
        "x_max": max(x_min, x_max),
        "y_min": min(y_min, y_max),
        "y_max": max(y_min, y_max),
    }


def load_embeddings(content_path: Path, epoch: int) -> np.ndarray:
    path = content_path / "epochs" / f"epoch_{epoch}" / "embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found: {path}")
    return np.load(path)


def load_projection(content_path: Path, vis_id: str, epoch: int, refined: bool = False) -> np.ndarray:
    suffix = "_refined" if refined else ""
    path = content_path / "visualize" / f"TimeVis_{vis_id}{suffix}" / "epochs" / f"epoch_{epoch}" / "projection.npy"
    if not path.exists():
        raise FileNotFoundError(f"Projection not found: {path}")
    return np.load(path)


def compute_hd_neighbors(features: np.ndarray, seeds: list[int], hd_k: int) -> set[int]:
    n = len(features)
    valid_seeds = [idx for idx in seeds if 0 <= idx < n]
    if not valid_seeds or n <= 1:
        return set()
    k = min(max(hd_k + 1, 2), n)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nbrs.fit(features)
    _, nn_idx = nbrs.kneighbors(features[valid_seeds])
    result = set(nn_idx.flatten().tolist())
    result.difference_update(valid_seeds)
    return result


def compute_bbox_points(projection: np.ndarray, bbox: dict[str, float]) -> set[int]:
    mask = (
        (projection[:, 0] >= bbox["x_min"]) & (projection[:, 0] <= bbox["x_max"]) &
        (projection[:, 1] >= bbox["y_min"]) & (projection[:, 1] <= bbox["y_max"])
    )
    return set(np.where(mask)[0].tolist())


def mean_movement(before: np.ndarray, after: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    pts_before = before[np.array(indices)]
    pts_after = after[np.array(indices)]
    dist = np.linalg.norm(pts_after - pts_before, axis=1)
    return float(dist.mean())


def clone_timevis_session(content_path: Path, source_vis_id: str, target_vis_id: str) -> Path:
    src = content_path / "visualize" / f"TimeVis_{source_vis_id}"
    dst = content_path / "visualize" / f"TimeVis_{target_vis_id}"
    if dst.exists():
        raise FileExistsError(f"Temporary target already exists: {dst}")
    shutil.copytree(src, dst)
    return dst


def build_config(content_path: Path, vis_id: str) -> dict:
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
    return config


def run_candidate(
    content_path: Path,
    candidate_name: str,
    focus_indices: list[int],
    focus_mode: str,
    epoch: int,
    source_vis_id: str,
    timestamp: str,
) -> tuple[dict[str, object], np.ndarray]:
    temp_vis_id = f"phase1b_{timestamp}_{candidate_name}"
    clone_timevis_session(content_path, source_vis_id, temp_vis_id)

    config = build_config(content_path, temp_vis_id)
    device = torch.device("cpu")
    data_provider = DataProvider(config, device)
    strategy = TimeVis(config, data_provider)

    mask = strategy.get_focus_mask(focus_indices)
    strategy.update_ttav_context(focus_indices, focus_mode, mask)

    baseline = load_projection(content_path, temp_vis_id, epoch, refined=False)

    start = time.time()
    strategy.refine(
        focus_indices=focus_indices,
        neighbor_indices=[],
        current_epoch=epoch,
        epochs_to_update=10,
        _skip_avg_benchmark=True,
    )
    latency = time.time() - start

    refined = load_projection(content_path, temp_vis_id, epoch, refined=True)

    focus_set = sorted(set(int(i) for i in focus_indices))
    focus_lookup = set(focus_set)
    non_focus = [idx for idx in range(len(baseline)) if idx not in focus_lookup]

    row = {
        "candidate": candidate_name,
        "temp_vis_id": temp_vis_id,
        "focus_set_size": len(focus_set),
        "neighbor_preservation": float(getattr(strategy, "_last_refine_np", 0.0)),
        "mean_rank_hd": float(getattr(strategy, "_last_refine_mrh", 0.0)),
        "trustworthiness": float(getattr(strategy, "_last_refine_trust", 0.0)),
        "continuity": float(getattr(strategy, "_last_refine_cont", 0.0)),
        "focus_shift": mean_movement(baseline, refined, focus_set),
        "global_drift": mean_movement(baseline, refined, non_focus),
        "latency_seconds": latency,
    }
    return row, refined


def save_outputs(
    output_dir: Path,
    timestamp: str,
    args: argparse.Namespace,
    bbox: dict[str, float],
    summaries: list[dict[str, object]],
    candidate_indices: dict[str, list[int]],
    candidate_projections: dict[str, np.ndarray],
    baseline: np.ndarray,
    seeds: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"metrics_{timestamp}.json"
    csv_path = output_dir / f"summary_{timestamp}.csv"
    fig_path = output_dir / f"metrics_{timestamp}.png"
    scatter_path = output_dir / f"refined_scatter_{timestamp}.png"

    payload = {
        "phase": "phase1b_refine_compare",
        "content_path": args.content_path,
        "source_vis_id": args.source_vis_id,
        "epoch": args.epoch,
        "focus_mode": args.focus_mode,
        "seeds": seeds,
        "bbox": bbox,
        "hd_k": args.hd_k,
        "summaries": summaries,
        "candidate_indices": candidate_indices,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    candidates = [row["candidate"] for row in summaries]
    np_vals = [row["neighbor_preservation"] for row in summaries]
    trust_vals = [row["trustworthiness"] for row in summaries]
    cont_vals = [row["continuity"] for row in summaries]
    drift_vals = [row["global_drift"] for row in summaries]
    latency_vals = [row["latency_seconds"] for row in summaries]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes[0, 0].bar(candidates, np_vals, color="#3182bd")
    axes[0, 0].set_title("Neighbor Preservation")
    axes[0, 1].bar(candidates, trust_vals, color="#31a354")
    axes[0, 1].set_title("Trustworthiness")
    axes[1, 0].bar(candidates, cont_vals, color="#756bb1")
    axes[1, 0].set_title("Continuity")
    axes[1, 1].bar(candidates, drift_vals, color="#e6550d", label="global_drift")
    ax2 = axes[1, 1].twinx()
    ax2.plot(candidates, latency_vals, color="#636363", marker="o", label="latency")
    axes[1, 1].set_title("Global Drift / Latency")
    for ax in axes.flatten():
        ax.tick_params(axis="x", rotation=15)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    scatter_fig, scatter_axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (name, proj) in zip(scatter_axes.flatten(), candidate_projections.items()):
        ax.scatter(baseline[:, 0], baseline[:, 1], s=3, c="#d9d9d9", alpha=0.35, linewidths=0)
        indices = candidate_indices[name]
        if indices:
            local = proj[np.array(indices)]
            ax.scatter(local[:, 0], local[:, 1], s=7, c="#cb181d", alpha=0.9, linewidths=0)
        if seeds:
            seed_proj = proj[np.array(seeds)]
            ax.scatter(seed_proj[:, 0], seed_proj[:, 1], s=18, c="#08519c", marker="x")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    scatter_fig.savefig(scatter_path, dpi=200)
    plt.close(scatter_fig)

    print(f"[INFO] Wrote metrics: {json_path}")
    print(f"[INFO] Wrote summary: {csv_path}")
    print(f"[INFO] Wrote figure:  {fig_path}")
    print(f"[INFO] Wrote scatter: {scatter_path}")


def main() -> int:
    args = parse_args()
    content_path = Path(args.content_path)
    output_dir = Path(args.output_dir)

    seeds = parse_int_list(args.seeds)
    bbox = parse_bbox(args.bbox)

    source_projection = load_projection(content_path, args.source_vis_id, args.epoch, refined=False)
    features = load_embeddings(content_path, args.epoch)
    bbox_set = compute_bbox_points(source_projection, bbox)
    hd_set = compute_hd_neighbors(features, seeds, args.hd_k)
    seed_set = set(seeds)

    candidate_sets = OrderedDict([
        ("seeds_only", seed_set),
        ("seeds_plus_hd", seed_set | hd_set),
        ("seeds_plus_bbox", seed_set | bbox_set),
        ("seeds_plus_bbox_plus_hd", seed_set | bbox_set | hd_set),
    ])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = []
    candidate_indices = {}
    candidate_projections = {}

    for name, candidate in candidate_sets.items():
        focus_indices = sorted(candidate)
        candidate_indices[name] = focus_indices
        print(f"[INFO] Running candidate {name} with {len(focus_indices)} focus points...")
        row, refined = run_candidate(
            content_path=content_path,
            candidate_name=name,
            focus_indices=focus_indices,
            focus_mode=args.focus_mode,
            epoch=args.epoch,
            source_vis_id=args.source_vis_id,
            timestamp=timestamp,
        )
        summaries.append(row)
        candidate_projections[name] = refined

    save_outputs(
        output_dir=output_dir,
        timestamp=timestamp,
        args=args,
        bbox=bbox,
        summaries=summaries,
        candidate_indices=candidate_indices,
        candidate_projections=candidate_projections,
        baseline=source_projection,
        seeds=seeds,
    )

    print("[INFO] Phase 1b refine comparison completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
