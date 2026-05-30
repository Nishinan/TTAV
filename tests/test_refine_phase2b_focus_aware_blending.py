import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "refine_validation_results" / "phase2b_focus_aware_blending"
DEFAULT_DECAY_RATIO = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2b: compare bbox-only vs bbox+focus blending.")
    parser.add_argument("--content-path", required=True)
    parser.add_argument("--baseline-vis-id", default="1")
    parser.add_argument("--phase1b-metrics", required=True)
    parser.add_argument("--decay-ratio", type=float, default=DEFAULT_DECAY_RATIO)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    return parser.parse_args()


def distance_to_bbox(x: float, y: float, bbox: dict[str, float]) -> float:
    dx = bbox["x_min"] - x if x < bbox["x_min"] else (x - bbox["x_max"] if x > bbox["x_max"] else 0.0)
    dy = bbox["y_min"] - y if y < bbox["y_min"] else (y - bbox["y_max"] if y > bbox["y_max"] else 0.0)
    return float(np.sqrt(dx * dx + dy * dy))


def build_bbox_blend(baseline: np.ndarray, refined: np.ndarray, bbox: dict[str, float], decay_ratio: float) -> np.ndarray:
    bbox_w = max(abs(bbox["x_max"] - bbox["x_min"]), 1e-6)
    bbox_h = max(abs(bbox["y_max"] - bbox["y_min"]), 1e-6)
    decay = max(float(np.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)) * decay_ratio, 1e-6)
    blended = np.empty_like(baseline)
    for idx, base in enumerate(baseline):
        d = distance_to_bbox(float(base[0]), float(base[1]), bbox)
        w = 1.0 if d <= 1e-12 else float(np.exp(-d / decay))
        blended[idx] = base * (1.0 - w) + refined[idx] * w
    return blended


def build_focus_aware_blend(baseline: np.ndarray, refined: np.ndarray, bbox: dict[str, float], focus_indices: list[int], decay_ratio: float) -> np.ndarray:
    bbox_blend = build_bbox_blend(baseline, refined, bbox, decay_ratio)
    valid_focus = [int(i) for i in focus_indices if 0 <= int(i) < len(baseline)]
    if not valid_focus:
        return bbox_blend
    focus_coords = baseline[np.array(valid_focus)]
    bbox_w = max(abs(bbox["x_max"] - bbox["x_min"]), 1e-6)
    bbox_h = max(abs(bbox["y_max"] - bbox["y_min"]), 1e-6)
    bbox_decay = max(float(np.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)) * decay_ratio, 1e-6)
    if len(focus_coords) == 1:
        focus_decay = bbox_decay
    else:
        center = focus_coords.mean(axis=0)
        radii = np.linalg.norm(focus_coords - center, axis=1)
        focus_decay = max(float(np.percentile(radii, 75)) * 1.5, bbox_decay * 0.5, 1e-6)

    out = np.empty_like(baseline)
    focus_set = set(valid_focus)
    for idx, base in enumerate(baseline):
        bbox_d = distance_to_bbox(float(base[0]), float(base[1]), bbox)
        bbox_wt = 1.0 if bbox_d <= 1e-12 else float(np.exp(-bbox_d / bbox_decay))
        if idx in focus_set:
            focus_wt = 1.0
        else:
            d_focus = float(np.min(np.linalg.norm(focus_coords - base, axis=1)))
            focus_wt = float(np.exp(-d_focus / focus_decay))
        w = max(bbox_wt, focus_wt)
        out[idx] = base * (1.0 - w) + refined[idx] * w
    return out


def mean_movement(before: np.ndarray, after: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    b = before[np.array(indices)]
    a = after[np.array(indices)]
    return float(np.linalg.norm(a - b, axis=1).mean())


def load_projection(content_path: Path, folder_name: str, epoch: int) -> np.ndarray:
    path = content_path / "visualize" / folder_name / "epochs" / f"epoch_{epoch}" / "projection.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def save_outputs(output_dir: Path, timestamp: str, payload: dict, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"metrics_{timestamp}.json"
    csv_path = output_dir / f"summary_{timestamp}.csv"
    fig_path = output_dir / f"metrics_{timestamp}.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    candidates = [row['candidate'] for row in rows]
    bbox_focus = [row['bbox_focus_shift'] for row in rows]
    aware_focus = [row['focus_aware_focus_shift'] for row in rows]
    bbox_drift = [row['bbox_global_drift'] for row in rows]
    aware_drift = [row['focus_aware_global_drift'] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(candidates))
    width = 0.36
    axes[0].bar(x - width / 2, bbox_focus, width, label='bbox-only')
    axes[0].bar(x + width / 2, aware_focus, width, label='bbox+focus')
    axes[0].set_title('Focus Shift')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(candidates, rotation=15)
    axes[0].legend()
    axes[1].bar(x - width / 2, bbox_drift, width, label='bbox-only')
    axes[1].bar(x + width / 2, aware_drift, width, label='bbox+focus')
    axes[1].set_title('Global Drift')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(candidates, rotation=15)
    axes[1].legend()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Wrote metrics: {json_path}")
    print(f"[INFO] Wrote summary: {csv_path}")
    print(f"[INFO] Wrote figure:  {fig_path}")


def main() -> int:
    args = parse_args()
    content_path = Path(args.content_path)
    output_dir = Path(args.output_dir)
    phase1b = json.loads(Path(args.phase1b_metrics).read_text(encoding='utf-8'))
    epoch = int(phase1b['epoch'])
    bbox = phase1b['bbox']
    candidate_indices = {k: [int(x) for x in v] for k, v in phase1b['candidate_indices'].items()}
    temp_vis_ids = {row['candidate']: row['temp_vis_id'] for row in phase1b['summaries']}
    baseline = load_projection(content_path, f"TimeVis_{args.baseline_vis_id}", epoch)

    rows = []
    for candidate, focus in candidate_indices.items():
        refined = load_projection(content_path, f"TimeVis_{temp_vis_ids[candidate]}_refined", epoch)
        bbox_blend = build_bbox_blend(baseline, refined, bbox, args.decay_ratio)
        aware_blend = build_focus_aware_blend(baseline, refined, bbox, focus, args.decay_ratio)
        non_focus = [idx for idx in range(len(baseline)) if idx not in set(focus)]
        rows.append({
            'candidate': candidate,
            'focus_set_size': len(focus),
            'bbox_focus_shift': mean_movement(baseline, bbox_blend, focus),
            'focus_aware_focus_shift': mean_movement(baseline, aware_blend, focus),
            'bbox_global_drift': mean_movement(baseline, bbox_blend, non_focus),
            'focus_aware_global_drift': mean_movement(baseline, aware_blend, non_focus),
            'decay_ratio': args.decay_ratio,
        })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    payload = {
        'phase': 'phase2b_focus_aware_blending',
        'content_path': args.content_path,
        'baseline_vis_id': args.baseline_vis_id,
        'epoch': epoch,
        'bbox': bbox,
        'decay_ratio': args.decay_ratio,
        'rows': rows,
        'source_phase1b_metrics': args.phase1b_metrics,
    }
    save_outputs(output_dir, timestamp, payload, rows)
    print('[INFO] Phase 2b focus-aware blending validation completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
