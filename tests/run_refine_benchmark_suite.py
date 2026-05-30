import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full TTAV refine benchmark workflow from one config.")
    parser.add_argument("--config", required=True, help="Path to benchmark config JSON")
    parser.add_argument("--skip-phase1b", action="store_true", help="Skip the expensive real refine comparison")
    parser.add_argument("--skip-phase3", action="store_true", help="Skip the local visualizer validation")
    return parser.parse_args()


def latest_file(directory: Path, prefix: str, suffix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No file matching {prefix}*{suffix} in {directory}")
    return candidates[-1]


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def bbox_arg(bbox: dict[str, float]) -> str:
    return f"{bbox['x_min']},{bbox['x_max']},{bbox['y_min']},{bbox['y_max']}"


def seeds_arg(seeds: list[int]) -> str:
    return ",".join(str(x) for x in seeds)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    content_path = cfg["content_path"]
    baseline_vis_id = str(cfg["baseline_vis_id"])
    epoch = int(cfg["epoch"])
    seeds = list(map(int, cfg["seeds"]))
    bbox = cfg["bbox"]
    hd_k = int(cfg.get("hd_k", 15))
    focus_mode = cfg.get("focus_mode", "balanced")
    blend_decay_ratio = float(cfg.get("blend_decay_ratio", 0.35))

    run_cmd([
        sys.executable,
        str(TESTS_DIR / "test_refine_phase1_focus_set.py"),
        "--content-path", content_path,
        "--vis-method", "TimeVis",
        "--vis-id", baseline_vis_id,
        "--epoch", str(epoch),
        "--seeds", seeds_arg(seeds),
        "--bbox", bbox_arg(bbox),
        "--hd-k", str(hd_k),
    ])

    if not args.skip_phase1b:
        run_cmd([
            sys.executable,
            str(TESTS_DIR / "test_refine_phase1b_refine_compare.py"),
            "--content-path", content_path,
            "--source-vis-id", baseline_vis_id,
            "--epoch", str(epoch),
            "--seeds", seeds_arg(seeds),
            "--bbox", bbox_arg(bbox),
            "--hd-k", str(hd_k),
            "--focus-mode", focus_mode,
        ])

    phase1b_metrics = latest_file(ROOT / "refine_validation_results" / "phase1b_refine_compare", "metrics_", ".json")

    run_cmd([
        sys.executable,
        str(TESTS_DIR / "test_refine_phase2_blending.py"),
        "--content-path", content_path,
        "--baseline-vis-id", baseline_vis_id,
        "--phase1b-metrics", str(phase1b_metrics),
        "--decay-ratio", str(blend_decay_ratio),
    ])

    run_cmd([
        sys.executable,
        str(TESTS_DIR / "test_refine_phase2b_focus_aware_blending.py"),
        "--content-path", content_path,
        "--baseline-vis-id", baseline_vis_id,
        "--phase1b-metrics", str(phase1b_metrics),
        "--decay-ratio", str(blend_decay_ratio),
    ])

    if not args.skip_phase3:
        run_cmd([
            sys.executable,
            str(TESTS_DIR / "test_refine_phase3_local_visualizer.py"),
            "--content-path", content_path,
            "--source-vis-id", baseline_vis_id,
            "--epoch", str(epoch),
            "--focus-indices", seeds_arg(seeds),
        ])

    run_cmd([
        sys.executable,
        str(TESTS_DIR / "build_refine_benchmark_summary.py"),
    ])

    print("[INFO] Refine benchmark suite completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
