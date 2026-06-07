"""
Refine quality benchmark using the real strategy.refine() call path.

Mirrors exactly what server.py /updateFocusContext does: calls
strategy.refine(focus_indices=..., current_epoch=...) and reads
_last_refine_np / _last_refine_trust / _last_refine_cont from the
strategy object afterwards.

CONFIG switches
---------------
refine_enabled   : False → skip refine, report baseline metrics only
focus_indices    : list  → use these specific point indices
                   None  → draw n_random_points at random (seed=random_seed)

Usage:
    conda run -n visualizer python tests/test_refine_avg.py
"""

import sys
import os
import json
import time
import copy

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_SRV = os.path.join(ROOT, "tool", "server")
TOOL_VIS = os.path.join(ROOT, "tool", "visualize")

for p in [TOOL_SRV, TOOL_VIS]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(TOOL_SRV)  # run_visualization.py imports relative to server/

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit here, no server restart needed
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Dataset / session ──────────────────────────────────────────────────────
    "content_path": "/home/shinan/Dataset/backdoor",
    "vis_method"  : "TimeVis",
    "vis_id"      : "1",
    "data_type"   : "Image",
    "task_type"   : "Classification",
    "vis_config"  : {"gpu_id": -1},   # -1 = CPU; change to 0 for GPU
    "epoch"       : 10,

    # ── Toggle ─────────────────────────────────────────────────────────────────
    # Set False to skip refine and only report baseline NP/T/C.
    "refine_enabled": True,

    # ── Focus points ───────────────────────────────────────────────────────────
    # Provide a fixed list, or set None to draw randomly.
    "focus_indices"  : None,
    "n_random_points": 5,
    "random_seed"    : 42,

    # ── Output ─────────────────────────────────────────────────────────────────
    "save_json": True,
    "json_path": os.path.join(ROOT, "tests", "refine_avg_results.json"),
}

# ══════════════════════════════════════════════════════════════════════════════
# Baseline metric helper (same formula used inside strategy.refine)
# ══════════════════════════════════════════════════════════════════════════════

def compute_baseline_metrics(full_feat, proj_np, focus_idx, hd_neighbors_all, k=10, K_ext=200):
    """NP / T / C for one focus point against the original (unrefined) projection."""
    N     = len(full_feat)
    K_ext = min(K_ext, N - 1)

    hd_d   = np.linalg.norm(full_feat - full_feat[focus_idx], axis=1)
    hd_d[focus_idx] = np.inf
    hd_rank = np.argsort(hd_d)

    ld_d   = np.linalg.norm(proj_np - proj_np[focus_idx], axis=1)
    ld_d[focus_idx] = np.inf
    ld_rank = np.argsort(ld_d)

    hd_rank_of = {int(hd_rank[r]): r + 1 for r in range(K_ext)}
    ld_rank_of = {int(ld_rank[r]): r + 1 for r in range(K_ext)}

    hd_topk = set(hd_rank[:k].tolist())
    ld_topk = set(ld_rank[:k].tolist())

    np_v = len(hd_topk & ld_topk) / k

    t_pen = sum(max(0, hd_rank_of.get(j, K_ext + 1) - k) for j in (ld_topk - hd_topk))
    c_pen = sum(max(0, ld_rank_of.get(j, K_ext + 1) - k) for j in (hd_topk - ld_topk))

    worst = k * (K_ext - k)
    t_v = 1.0 - t_pen / worst if worst > 0 else 1.0
    c_v = 1.0 - c_pen / worst if worst > 0 else 1.0

    return np_v * 100, t_v * 100, c_v * 100


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg         = CONFIG
    epoch       = cfg["epoch"]
    content_path = cfg["content_path"]

    print("=" * 64)
    print(f"  Refine Avg Benchmark — {cfg['vis_method']}_{cfg['vis_id']}  epoch={epoch}")
    print(f"  refine_enabled = {cfg['refine_enabled']}")
    print("=" * 64)

    # ── Build strategy via the same path as server.py ─────────────────────────
    from run_visualization import initialize_config, init_visualize_component

    config   = initialize_config(
        content_path, cfg["vis_method"], cfg["vis_id"],
        cfg["data_type"], cfg["task_type"], cfg["vis_config"]
    )
    _, strategy = init_visualize_component(config)
    print(f"Strategy initialised: {type(strategy).__name__}")

    # ── Load data for baseline metrics ────────────────────────────────────────
    feat_path = os.path.join(content_path, "epochs", f"epoch_{epoch}", "embeddings.npy")
    proj_path = os.path.join(
        content_path, "visualize",
        f"{cfg['vis_method']}_{cfg['vis_id']}",
        "epochs", f"epoch_{epoch}", "projection.npy"
    )
    hd_nb_path = os.path.join(content_path, "epochs", f"epoch_{epoch}", "hd_neighbors_10.json")

    full_feat = np.load(feat_path)
    full_proj = np.load(proj_path)
    with open(hd_nb_path) as f:
        hd_neighbors_all = json.load(f)

    N = len(full_feat)
    print(f"N={N}  dim={full_feat.shape[1]}")

    # ── Select focus points ───────────────────────────────────────────────────
    if cfg["focus_indices"] is not None:
        focus_list = list(cfg["focus_indices"])
    else:
        rng        = np.random.default_rng(seed=cfg["random_seed"])
        focus_list = rng.choice(N, size=min(cfg["n_random_points"], N),
                                replace=False).tolist()

    print(f"Focus points ({len(focus_list)}): {focus_list}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Baseline
    # ══════════════════════════════════════════════════════════════════════════
    print("─── Baseline (original projection, no refine) ───")
    baseline_rows = []
    for fi in focus_list:
        np_v, t_v, c_v = compute_baseline_metrics(
            full_feat, full_proj, fi, hd_neighbors_all
        )
        baseline_rows.append({"idx": fi, "NP": np_v, "T": t_v, "C": c_v})
        print(f"  point {fi:6d}  NP={np_v:5.1f}%  T={t_v:5.1f}%  C={c_v:5.1f}%")

    avg_np_base = np.mean([r["NP"] for r in baseline_rows])
    avg_t_base  = np.mean([r["T"]  for r in baseline_rows])
    avg_c_base  = np.mean([r["C"]  for r in baseline_rows])
    print(f"\n  AVG  NP={avg_np_base:.1f}%  T={avg_t_base:.1f}%  C={avg_c_base:.1f}%")

    if not cfg["refine_enabled"]:
        print("\n[refine_enabled=False] Done — baseline only.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Refine — one call per focus point, exactly as server.py does
    # ══════════════════════════════════════════════════════════════════════════
    print("\n─── Refine (strategy.refine per point) ───")
    refine_rows = []
    for fi in focus_list:
        t0 = time.time()
        strategy.refine(
            focus_indices=[fi],
            neighbor_indices=[],
            current_epoch=epoch,
        )
        elapsed = time.time() - t0

        np_v = getattr(strategy, "_last_refine_np",    float("nan"))
        t_v  = getattr(strategy, "_last_refine_trust", float("nan"))
        c_v  = getattr(strategy, "_last_refine_cont",  float("nan"))
        refine_rows.append({"idx": fi, "NP": np_v, "T": t_v, "C": c_v, "t": elapsed})
        print(f"  point {fi:6d}  NP={np_v:5.1f}%  T={t_v:5.1f}%  C={c_v:5.1f}%  t={elapsed:.1f}s")

    avg_np_ref = np.mean([r["NP"] for r in refine_rows])
    avg_t_ref  = np.mean([r["T"]  for r in refine_rows])
    avg_c_ref  = np.mean([r["C"]  for r in refine_rows])
    avg_time   = np.mean([r["t"]  for r in refine_rows])
    print(f"\n  AVG  NP={avg_np_ref:.1f}%  T={avg_t_ref:.1f}%  C={avg_c_ref:.1f}%  t={avg_time:.1f}s")

    # ── Delta ─────────────────────────────────────────────────────────────────
    print("\n─── Delta (refine − baseline) ───")
    print(f"  ΔNP = {avg_np_ref - avg_np_base:+.1f}%")
    print(f"  ΔT  = {avg_t_ref  - avg_t_base:+.1f}%")
    print(f"  ΔC  = {avg_c_ref  - avg_c_base:+.1f}%")

    print("\n─── Per-point delta ───")
    print(f"  {'idx':>8}  {'ΔNP':>7}  {'ΔT':>7}  {'ΔC':>7}")
    for base, ref in zip(baseline_rows, refine_rows):
        print(f"  {base['idx']:>8}  "
              f"{ref['NP'] - base['NP']:>+6.1f}%  "
              f"{ref['T']  - base['T']:>+6.1f}%  "
              f"{ref['C']  - base['C']:>+6.1f}%")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if cfg["save_json"]:
        record = {
            "timestamp"    : time.strftime("%Y-%m-%d %H:%M:%S"),
            "vis_method"   : cfg["vis_method"],
            "vis_id"       : cfg["vis_id"],
            "epoch"        : epoch,
            "focus_points" : focus_list,
            "baseline": {
                "avg_NP": round(avg_np_base, 2),
                "avg_T" : round(avg_t_base,  2),
                "avg_C" : round(avg_c_base,  2),
                "rows"  : baseline_rows,
            },
            "refine": {
                "avg_NP"  : round(avg_np_ref, 2),
                "avg_T"   : round(avg_t_ref,  2),
                "avg_C"   : round(avg_c_ref,  2),
                "avg_time": round(avg_time,   2),
                "rows"    : refine_rows,
            },
            "delta": {
                "dNP": round(avg_np_ref - avg_np_base, 2),
                "dT" : round(avg_t_ref  - avg_t_base,  2),
                "dC" : round(avg_c_ref  - avg_c_base,  2),
            },
        }
        existing = []
        if os.path.exists(cfg["json_path"]):
            try:
                with open(cfg["json_path"]) as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(record)
        with open(cfg["json_path"], "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\nSaved → {cfg['json_path']}")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
