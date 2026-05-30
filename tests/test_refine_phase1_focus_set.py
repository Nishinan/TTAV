import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "tool"
SERVER_DIR = TOOL_DIR / "server"
VISUALIZE_DIR = TOOL_DIR / "visualize"
for path in (TOOL_DIR, SERVER_DIR, VISUALIZE_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from server_utils import load_raw_projection_array  # noqa: E402

RESULT_DIR = ROOT / "refine_validation_results" / "phase1_focus_set"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 validation: compare candidate focus_set definitions."
    )
    parser.add_argument("--content-path", required=True, help="Dataset content path")
    parser.add_argument("--vis-method", default="TimeVis", help="Visualization method")
    parser.add_argument("--vis-id", default="0", help="Visualization ID")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch to evaluate")
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated raw seed indices, e.g. 1,5,9",
    )
    parser.add_argument(
        "--bbox",
        required=True,
        help="bbox as x_min,x_max,y_min,y_max in projection coordinates",
    )
    parser.add_argument(
        "--hd-k",
        type=int,
        default=15,
        help="Number of high-D neighbors per seed to include",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULT_DIR),
        help="Directory to store phase-1 outputs",
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
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
    emb_path = content_path / "epochs" / f"epoch_{epoch}" / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    return np.load(emb_path)


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
    if projection.ndim != 2 or projection.shape[1] < 2:
        raise ValueError("Projection must be shaped [N, 2]")

    mask = (
        (projection[:, 0] >= bbox["x_min"])
        & (projection[:, 0] <= bbox["x_max"])
        & (projection[:, 1] >= bbox["y_min"])
        & (projection[:, 1] <= bbox["y_max"])
    )
    return set(np.where(mask)[0].tolist())


def summarize_candidate(name: str, seeds: set[int], bbox_set: set[int], hd_set: set[int], candidate: set[int]) -> dict[str, object]:
    candidate_size = len(candidate)
    seed_overlap = len(candidate & seeds)
    bbox_overlap = len(candidate & bbox_set)
    hd_overlap = len(candidate & hd_set)

    return {
        "candidate": name,
        "focus_set_size": candidate_size,
        "seed_overlap": seed_overlap,
        "bbox_overlap": bbox_overlap,
        "hd_overlap": hd_overlap,
        "seed_coverage": seed_overlap / max(len(seeds), 1),
        "bbox_coverage": bbox_overlap / max(len(bbox_set), 1),
        "hd_coverage": hd_overlap / max(len(hd_set), 1),
        "bbox_ratio": bbox_overlap / max(candidate_size, 1),
        "hd_ratio": hd_overlap / max(candidate_size, 1),
    }


def save_outputs(
    output_dir: Path,
    timestamp: str,
    args: argparse.Namespace,
    bbox: dict[str, float],
    summaries: list[dict[str, object]],
    candidate_indices: dict[str, list[int]],
    projection: np.ndarray,
    seeds: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"metrics_{timestamp}.json"
    summary_path = output_dir / f"summary_{timestamp}.csv"
    figure_path = output_dir / f"focus_set_comparison_{timestamp}.png"

    payload = {
        "phase": "phase1_focus_set",
        "content_path": args.content_path,
        "vis_method": args.vis_method,
        "vis_id": args.vis_id,
        "epoch": args.epoch,
        "seeds": seeds,
        "bbox": bbox,
        "hd_k": args.hd_k,
        "summaries": summaries,
        "candidate_indices": candidate_indices,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (name, indices) in zip(axes.flatten(), candidate_indices.items()):
        ax.scatter(projection[:, 0], projection[:, 1], s=4, c="#d6d6d6", alpha=0.45, linewidths=0)
        if indices:
            local_proj = projection[np.array(indices)]
            ax.scatter(local_proj[:, 0], local_proj[:, 1], s=7, c="#e6550d", alpha=0.85, linewidths=0)
        if seeds:
            seed_proj = projection[np.array(seeds)]
            ax.scatter(seed_proj[:, 0], seed_proj[:, 1], s=18, c="#08519c", alpha=1.0, marker="x")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Phase 1 Focus-Set Comparison | epoch={args.epoch} | vis={args.vis_method}_{args.vis_id}",
        fontsize=13,
    )
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    print(f"[INFO] Wrote metrics: {metrics_path}")
    print(f"[INFO] Wrote summary: {summary_path}")
    print(f"[INFO] Wrote figure:  {figure_path}")


def main() -> int:
    args = parse_args()
    content_path = Path(args.content_path)
    output_dir = Path(args.output_dir)

    seeds = parse_int_list(args.seeds)
    bbox = parse_bbox(args.bbox)

    projection = load_raw_projection_array(
        str(content_path),
        args.vis_method,
        args.vis_id,
        args.epoch,
        refine_flag=False,
    )
    features = load_embeddings(content_path, args.epoch)

    bbox_set = compute_bbox_points(projection, bbox)
    hd_set = compute_hd_neighbors(features, seeds, args.hd_k)
    seed_set = set(seeds)

    candidate_sets = OrderedDict(
        [
            ("seeds_only", seed_set),
            ("seeds_plus_hd", seed_set | hd_set),
            ("seeds_plus_bbox", seed_set | bbox_set),
            ("seeds_plus_bbox_plus_hd", seed_set | bbox_set | hd_set),
        ]
    )

    summaries = [
        summarize_candidate(name, seed_set, bbox_set, hd_set, candidate)
        for name, candidate in candidate_sets.items()
    ]
    candidate_indices = {
        name: sorted(candidate)
        for name, candidate in candidate_sets.items()
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_outputs(
        output_dir=output_dir,
        timestamp=timestamp,
        args=args,
        bbox=bbox,
        summaries=summaries,
        candidate_indices=candidate_indices,
        projection=projection,
        seeds=seeds,
    )

    print("[INFO] Phase 1 focus-set validation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
