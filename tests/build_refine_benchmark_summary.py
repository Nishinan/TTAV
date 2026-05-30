import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "refine_validation_results"
OUTPUT_DIR = RESULT_ROOT / "benchmark_summary"


def latest_file(directory: Path, prefix: str, suffix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No file matching {prefix}*{suffix} in {directory}")
    return candidates[-1]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value is None or value == "":
        return None
    return float(value)


def main() -> int:
    phase1_dir = RESULT_ROOT / "phase1_focus_set"
    phase1b_dir = RESULT_ROOT / "phase1b_refine_compare"
    phase2_dir = RESULT_ROOT / "phase2_blending"
    phase2b_dir = RESULT_ROOT / "phase2b_focus_aware_blending"
    phase3_dir = RESULT_ROOT / "phase3_local_visualizer"

    phase1_json = latest_file(phase1_dir, "metrics_", ".json")
    phase1b_csv = latest_file(phase1b_dir, "summary_", ".csv")
    phase2_csv = latest_file(phase2_dir, "summary_", ".csv")
    phase2b_csv = latest_file(phase2b_dir, "summary_", ".csv")
    phase3_json = latest_file(phase3_dir, "local_visualizer_check_", ".json")

    phase1 = json.loads(phase1_json.read_text(encoding="utf-8"))
    phase3 = json.loads(phase3_json.read_text(encoding="utf-8"))
    phase1b_rows = {row["candidate"]: row for row in load_csv_rows(phase1b_csv)}
    phase2_rows = {row["candidate"]: row for row in load_csv_rows(phase2_csv)}
    phase2b_rows = {row["candidate"]: row for row in load_csv_rows(phase2b_csv)}

    candidates = sorted(set(phase1b_rows) & set(phase2_rows) & set(phase2b_rows))
    combined_rows = []
    for candidate in candidates:
        p1b = phase1b_rows[candidate]
        p2 = phase2_rows[candidate]
        p2b = phase2b_rows[candidate]
        row = {
            "candidate": candidate,
            "focus_set_size": int(p1b["focus_set_size"]),
            "np_direct": to_float(p1b["neighbor_preservation"]),
            "mrh_direct": to_float(p1b["mean_rank_hd"]),
            "trust_direct": to_float(p1b["trustworthiness"]),
            "continuity_direct": to_float(p1b["continuity"]),
            "focus_shift_direct": to_float(p1b["focus_shift"]),
            "global_drift_direct": to_float(p1b["global_drift"]),
            "latency_seconds": to_float(p1b["latency_seconds"]),
            "focus_shift_bbox_blend": to_float(p2["blended_focus_shift"]),
            "global_drift_bbox_blend": to_float(p2["blended_global_drift"]),
            "focus_shift_focus_blend": to_float(p2b["focus_aware_focus_shift"]),
            "global_drift_focus_blend": to_float(p2b["focus_aware_global_drift"]),
        }
        row["focus_blend_gain_vs_bbox"] = row["focus_shift_focus_blend"] / max(row["focus_shift_bbox_blend"], 1e-12)
        row["focus_blend_drift_vs_direct"] = row["global_drift_focus_blend"] / max(row["global_drift_direct"], 1e-12)
        combined_rows.append(row)

    # Heuristic recommendation: among non-trivial focus sets, maximize local quality
    # while keeping global drift far below direct replace.
    non_trivial = [r for r in combined_rows if r["focus_set_size"] > 3]
    recommended = min(
        non_trivial,
        key=lambda r: (
            -r["np_direct"],
            r["mrh_direct"],
            r["global_drift_focus_blend"],
            r["latency_seconds"],
        ),
    ) if non_trivial else combined_rows[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"refine_benchmark_summary_{timestamp}.csv"
    md_path = OUTPUT_DIR / f"refine_benchmark_summary_{timestamp}.md"

    fieldnames = list(combined_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    md_lines = [
        "# Refine Benchmark Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        f"- Phase 1 focus-set source: `{phase1_json.name}`",
        f"- Phase 1b refine compare: `{phase1b_csv.name}`",
        f"- Phase 2 bbox-only blending: `{phase2_csv.name}`",
        f"- Phase 2b focus-aware blending: `{phase2b_csv.name}`",
        f"- Phase 3 local visualizer check: `{phase3_json.name}`",
        "",
        "## Current Recommendation",
        f"- Default focus-set candidate on the current `backdoor` experiment: `{recommended['candidate']}`",
        f"- Reason: it gives the strongest non-trivial direct local quality (`NP={recommended['np_direct']:.2f}`, `MRH={recommended['mrh_direct']:.1f}`) while focus-aware blending keeps global drift to `{recommended['global_drift_focus_blend']:.4f}`.",
        f"- Local visualizer validation: global model unchanged max abs diff = `{phase3['global_model_unchanged_max_abs']}`; local-vs-global embed max abs diff = `{phase3['local_vs_global_embed_max_abs']:.4f}`.",
        "",
        "## How To Evaluate Improvement",
        "- Local quality: `NP`, `MRH`, `Trustworthiness`, `Continuity`.",
        "- Global stability: `Global Drift` after blending, plus non-focus displacement.",
        "- Interaction cost: `Latency`.",
        "- Use all three together. A method is better only if local quality improves without global drift or latency becoming unacceptable.",
        "",
        "## Combined Table",
        "",
        "| candidate | focus_set_size | NP direct | MRH direct | focus_shift direct | drift direct | focus_shift bbox+blend | drift bbox+blend | focus_shift focus+blend | drift focus+blend | latency(s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in combined_rows:
        md_lines.append(
            f"| {row['candidate']} | {row['focus_set_size']} | {row['np_direct']:.2f} | {row['mrh_direct']:.1f} | {row['focus_shift_direct']:.4f} | {row['global_drift_direct']:.4f} | {row['focus_shift_bbox_blend']:.4f} | {row['global_drift_bbox_blend']:.4f} | {row['focus_shift_focus_blend']:.4f} | {row['global_drift_focus_blend']:.4f} | {row['latency_seconds']:.2f} |"
        )
    md_lines.extend([
        "",
        "## Notes",
        f"- Phase 1 selected seeds: `{phase1.get('seeds')}`",
        f"- Phase 1 bbox: `{phase1.get('bbox')}`",
        "- `bbox-only blend` is the most conservative option.",
        "- `bbox+focus blend` better preserves local changes for larger focus sets while still reducing drift far below direct replacement.",
    ])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[INFO] Wrote CSV: {csv_path}")
    print(f"[INFO] Wrote MD:  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
