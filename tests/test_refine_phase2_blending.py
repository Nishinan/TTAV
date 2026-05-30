import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "refine_validation_results" / "phase2_blending"
DEFAULT_DECAY_RATIO = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 validation: compare direct replace vs blended projection."
    )
    parser.add_argument("--content-path", required=True, help="Dataset content path")
    parser.add_argument("--baseline-vis-id", default="1", help="Baseline TimeVis session id")
    parser.add_argument("--phase1b-metrics", required=True, help="metrics_*.json from Phase 1b")
    parser.add_argument("--decay-ratio", type=float, default=DEFAULT_DECAY_RATIO, help="Blend decay ratio")
    parser.add_argument("--output-dir", default=str(RESULT_DIR), help="Directory to store outputs")
    return parser.parse_args()


def distance_to_bbox(x: float, y: float, bbox: dict[str, float]) -> float:
    dx = bbox["x_min"] - x if x < bbox["x_min"] else (x - bbox["x_max"] if x > bbox["x_max"] else 0.0)
    dy = bbox["y_min"] - y if y < bbox["y_min"] else (y - bbox["y_max"] if y > bbox["y_max"] else 0.0)
    return float(np.sqrt(dx * dx + dy * dy))


def build_blended_projection(baseline: np.ndarray, refined: np.ndarray, bbox: dict[str, float], decay_ratio: float) -> np.ndarray:
    bbox_w = max(abs(bbox["x_max"] - bbox["x_min"]), 1e-6)
    bbox_h = max(abs(bbox["y_max"] - bbox["y_min"]), 1e-6)
    decay = max(float(np.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)) * decay_ratio, 1e-6)

    blended = np.empty_like(baseline)
    for idx, base in enumerate(baseline):
        d = distance_to_bbox(float(base[0]), float(base[1]), bbox)
        weight = 1.0 if d <= 1e-12 else float(np.exp(-d / decay))
        blended[idx] = base * (1.0 - weight) + refined[idx] * weight
    return blended


def mean_movement(before: np.ndarray, after: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    pts_before = before[np.array(indices)]
    pts_after = after[np.array(indices)]
    return float(np.linalg.norm(pts_after - pts_before, axis=1).mean())


def load_projection(content_path: Path, folder_name: str, epoch: int) -> np.ndarray:
    path = content_path / "visualize" / folder_name / "epochs" / f"epoch_{epoch}" / "projection.npy"
    if not path.exists():
        raise FileNotFoundError(f"Projection not found: {path}")
    return np.load(path)


def save_outputs(output_dir: Path, timestamp: str, payload: dict, rows: list[dict[str, object]], scatter_payload: dict[str, np.ndarray], focus_map: dict[str, list[int]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"metrics_{timestamp}.json"
    summary_path = output_dir / f"summary_{timestamp}.csv"
    figure_path = output_dir / f"metrics_{timestamp}.png"
    scatter_path = output_dir / f"scatter_{timestamp}.png"

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    candidates = [row["candidate"] for row in rows]
    direct_focus = [row["direct_focus_shift"] for row in rows]
    blended_focus = [row["blended_focus_shift"] for row in rows]
    direct_drift = [row["direct_global_drift"] for row in rows]
    blended_drift = [row["blended_global_drift"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(candidates))
    width = 0.36
    axes[0].bar(x - width / 2, direct_focus, width, label="direct")
    axes[0].bar(x + width / 2, blended_focus, width, label="blended")
    axes[0].set_title("Focus Shift")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(candidates, rotation=15)
    axes[0].legend()

    axes[1].bar(x - width / 2, direct_drift, width, label="direct")
    axes[1].bar(x + width / 2, blended_drift, width, label="blended")
    axes[1].set_title("Global Drift")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(candidates, rotation=15)
    axes[1].legend()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    scatter_fig, scatter_axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, candidate in zip(scatter_axes.flatten(), candidates):
        baseline = scatter_payload[f"{candidate}_baseline"]
        blended = scatter_payload[f"{candidate}_blended"]
        focus = focus_map[candidate]
        ax.scatter(baseline[:, 0], baseline[:, 1], s=3, c="#d9d9d9", alpha=0.35, linewidths=0)
        if focus:
            ax.scatter(blended[np.array(focus), 0], blended[np.array(focus), 1], s=7, c="#cb181d", alpha=0.9, linewidths=0)
        ax.set_title(candidate)
        ax.set_xticks([])
        ax.set_yticks([])
    scatter_fig.savefig(scatter_path, dpi=200)
    plt.close(scatter_fig)

    print(f"[INFO] Wrote metrics: {metrics_path}")
    print(f"[INFO] Wrote summary: {summary_path}")
    print(f"[INFO] Wrote figure:  {figure_path}")
    print(f"[INFO] Wrote scatter: {scatter_path}")


def main() -> int:
    args = parse_args()
    content_path = Path(args.content_path)
    output_dir = Path(args.output_dir)
    phase1b_path = Path(args.phase1b_metrics)
    phase1b = json.loads(phase1b_path.read_text(encoding="utf-8"))

    epoch = int(phase1b["epoch"])
    bbox = phase1b["bbox"]
    candidate_indices = {k: [int(x) for x in v] for k, v in phase1b["candidate_indices"].items()}
    temp_vis_ids = {row["candidate"]: row["temp_vis_id"] for row in phase1b["summaries"]}

    baseline = load_projection(content_path, f"TimeVis_{args.baseline_vis_id}", epoch)

    rows = []
    scatter_payload: dict[str, np.ndarray] = {}
    for candidate, focus in candidate_indices.items():
        temp_vis_id = temp_vis_ids[candidate]
        refined = load_projection(content_path, f"TimeVis_{temp_vis_id}_refined", epoch)
        blended = build_blended_projection(baseline, refined, bbox, args.decay_ratio)
        non_focus = [idx for idx in range(len(baseline)) if idx not in set(focus)]

        rows.append({
            "candidate": candidate,
            "focus_set_size": len(focus),
            "direct_focus_shift": mean_movement(baseline, refined, focus),
            "blended_focus_shift": mean_movement(baseline, blended, focus),
            "direct_global_drift": mean_movement(baseline, refined, non_focus),
            "blended_global_drift": mean_movement(baseline, blended, non_focus),
            "decay_ratio": args.decay_ratio,
        })
        scatter_payload[f"{candidate}_baseline"] = baseline
        scatter_payload[f"{candidate}_blended"] = blended

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "phase": "phase2_blending",
        "content_path": args.content_path,
        "baseline_vis_id": args.baseline_vis_id,
        "epoch": epoch,
        "bbox": bbox,
        "decay_ratio": args.decay_ratio,
        "rows": rows,
        "source_phase1b_metrics": str(phase1b_path),
    }
    save_outputs(output_dir, timestamp, payload, rows, scatter_payload, candidate_indices)
    print("[INFO] Phase 2 blending validation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
