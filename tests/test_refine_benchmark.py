"""
Refine quality benchmark for TimeVis on a real dataset.

This script loads epoch 10 data directly (no Flask server) and runs the
anchor-constrained refine on a configurable set of focus points, then reports
per-point and aggregate NP / Trustworthiness / Continuity.

Usage:
    conda run -n visualizer python tests/test_refine_benchmark.py

Toggle switches (top of CONFIG dict):
    refine_enabled      : if False, report baseline metrics without any refine
    save_per_point_csv  : if True, write per-point results to tests/refine_bench.csv
"""

import sys
import os
import time
import json
import copy

import numpy as np
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_VIS = os.path.join(ROOT, "tool", "visualize")
TOOL_STR = os.path.join(TOOL_VIS, "strategy")
for p in [TOOL_VIS, TOOL_STR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from visualize_model import VisModel

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit here, no server restart needed
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Dataset paths ──────────────────────────────────────────────────────────
    "content_path"   : "/home/shinan/Dataset/backdoor",
    "vis_method"     : "TimeVis",
    "vis_id"         : "1",
    "epoch"          : 10,

    # ── Model architecture (must match the checkpoint) ─────────────────────────
    "encoder_dims"   : [512, 256, 128, 64, 32, 2],
    "decoder_dims"   : [2,  32,   64, 128, 256, 512],

    # ── Focus points to benchmark ──────────────────────────────────────────────
    # Fixed list of point indices in the dataset.  Use None to pick randomly.
    # Example: [0, 100, 500, 1000, 5000]
    "focus_indices"  : None,   # None → random selection
    "n_random_points": 10,     # how many to pick when focus_indices is None
    "random_seed"    : 42,

    # ── Refine toggle ──────────────────────────────────────────────────────────
    # Set False to skip refine and only report baseline metrics.
    "refine_enabled" : True,

    # ── Refine hyper-parameters ────────────────────────────────────────────────
    "max_steps"      : 500,    # max optimisation steps per focus point
    "lr"             : 0.005,
    "mu_anchor"      : 10.0,   # anchor constraint weight
    "gamma_repel"    : 1.0,    # repulsion weight
    "n_anchors"      : 300,
    "k_hd"           : 10,     # how many HD neighbours to attract
    "k_neg_mult"     : 5,      # n_neg = k_neg_mult * k_hd per focus point
    "np_stall_limit" : 5,      # early-stop if NP unchanged for this many 10-step checks
    "n_trainable_linear_layers": 2,  # how many last Linear layers to fine-tune

    # ── Metrics ────────────────────────────────────────────────────────────────
    "k_metric"       : 10,     # NP / T / C computed at top-k
    "K_ext"          : 200,    # extended neighbourhood for T/C normaliser

    # ── Output ─────────────────────────────────────────────────────────────────
    "save_per_point_csv": True,
    "csv_path"       : os.path.join(ROOT, "tests", "refine_bench.csv"),
}

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(full_feat, proj_np, focus_idx, hd_neighbors_all, k=10, K_ext=200):
    """Return (NP, T, C) for a single focus point given current 2-D projection."""
    N = len(full_feat)
    K_ext = min(K_ext, N - 1)

    hd_dists = np.linalg.norm(full_feat - full_feat[focus_idx], axis=1)
    hd_dists[focus_idx] = np.inf
    hd_rank = np.argsort(hd_dists)

    ld_dists = np.linalg.norm(proj_np - proj_np[focus_idx], axis=1)
    ld_dists[focus_idx] = np.inf
    ld_rank = np.argsort(ld_dists)

    hd_rank_of = {int(hd_rank[r]): r + 1 for r in range(K_ext)}
    ld_rank_of = {int(ld_rank[r]): r + 1 for r in range(K_ext)}

    hd_topk = set(hd_rank[:k].tolist())
    ld_topk = set(ld_rank[:k].tolist())

    np_val = len(hd_topk & ld_topk) / k

    t_penalty = sum(max(0, hd_rank_of.get(j, K_ext + 1) - k) for j in (ld_topk - hd_topk))
    c_penalty = sum(max(0, ld_rank_of.get(j, K_ext + 1) - k) for j in (hd_topk - ld_topk))

    worst = k * (K_ext - k)
    t_val = 1.0 - t_penalty / worst if worst > 0 else 1.0
    c_val = 1.0 - c_penalty / worst if worst > 0 else 1.0

    return np_val, t_val, c_val


def run_refine_for_point(model, full_feat, full_proj_baseline, focus_idx,
                         hd_neighbors_all, device, cfg):
    """
    Run anchor-constrained refine for a single focus point.
    Returns (np_val, t_val, c_val, steps_taken, elapsed_s) using refined projection,
    and also the updated full projection array (numpy, copy of baseline with focus region patched).
    """
    k_hd        = cfg["k_hd"]
    k_neg_mult  = cfg["k_neg_mult"]
    mu_anchor   = cfg["mu_anchor"]
    gamma_repel = cfg["gamma_repel"]
    n_anchors   = cfg["n_anchors"]
    max_steps   = cfg["max_steps"]
    lr          = cfg["lr"]
    np_stall_limit = cfg["np_stall_limit"]
    n_train_lin = cfg["n_trainable_linear_layers"]
    k           = cfg["k_metric"]
    K_ext       = cfg["K_ext"]

    N = len(full_feat)
    rng = np.random.default_rng(seed=cfg["random_seed"])

    # HD neighbours
    hd_nbr_idx = hd_neighbors_all[focus_idx][:k_hd]
    hd_nbr_set = set(hd_nbr_idx)

    # Anchor sampling (exclude focus + HD neighbours)
    exclude_set = {focus_idx} | hd_nbr_set
    candidates  = [i for i in range(N) if i not in exclude_set]
    n_a = min(n_anchors, len(candidates))
    anchor_idx = rng.choice(candidates, size=n_a, replace=False).tolist()
    anchor_feat_np = full_feat[anchor_idx]
    anchor_z0      = full_proj_baseline[anchor_idx]

    # Negative sampling (random, excluding focus + HD neighbours)
    neg_exclude  = exclude_set
    neg_cands    = [i for i in range(N) if i not in neg_exclude]
    n_neg        = min(k_neg_mult * k_hd, len(neg_cands))
    neg_idx      = rng.choice(neg_cands, size=n_neg, replace=False).tolist()
    neg_feat_np  = full_feat[neg_idx]

    # Adaptive margin
    all_dists = np.linalg.norm(full_proj_baseline - full_proj_baseline[focus_idx], axis=1)
    all_dists[focus_idx] = np.inf
    margin_m = float(np.median(np.sort(all_dists)[:20])) * 2.0
    margin_m = max(margin_m, 0.1)

    # Tensors
    focus_feat_t  = torch.from_numpy(full_feat[[focus_idx]].copy()).float().to(device)
    nbr_feat_t    = torch.from_numpy(full_feat[hd_nbr_idx].copy()).float().to(device)
    anchor_feat_t = torch.from_numpy(anchor_feat_np).float().to(device)
    anchor_z0_t   = torch.from_numpy(anchor_z0).float().to(device)
    neg_feat_t    = torch.from_numpy(neg_feat_np).float().to(device)
    full_feat_t   = torch.from_numpy(full_feat).float().to(device)

    # Freeze all but last n_train_lin Linear layers
    enc_layers = list(model.encoder.children())
    trainable  = []
    lin_count  = 0
    for layer in reversed(enc_layers):
        if isinstance(layer, torch.nn.Linear):
            for p in layer.parameters():
                p.requires_grad = True
            trainable += list(layer.parameters())
            lin_count += 1
            if lin_count >= n_train_lin:
                break
    for name, param in model.named_parameters():
        if param.requires_grad and not any(param is tp for tp in trainable):
            param.requires_grad = False

    weight_backup = {id(p): p.data.clone() for p in trainable}
    optimizer = torch.optim.Adam(trainable, lr=lr)

    best_np   = -1.0
    best_step = 0
    np_stall  = 0
    t0 = time.time()

    model.train()
    for step in range(max_steps):
        optimizer.zero_grad()

        z_focus   = model.encoder(focus_feat_t)    # [1, 2]
        z_nbrs    = model.encoder(nbr_feat_t)      # [k_hd, 2]
        z_neg     = model.encoder(neg_feat_t)      # [n_neg, 2]
        z_anchors = model.encoder(anchor_feat_t)   # [A, 2]

        l_attract = (z_focus - z_nbrs).pow(2).sum(dim=1).mean()
        dist_rep  = (z_focus - z_neg).pow(2).sum(dim=1).sqrt()
        l_repel   = torch.clamp(margin_m - dist_rep, min=0.0).pow(2).mean()
        l_anchor  = (z_anchors - anchor_z0_t).pow(2).sum(dim=1).mean()

        loss = l_attract + gamma_repel * l_repel + mu_anchor * l_anchor
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                z_eval = model.encoder(full_feat_t).cpu().numpy()
            ld_d = np.linalg.norm(z_eval - z_eval[focus_idx], axis=1)
            ld_d[focus_idx] = np.inf
            ld_topk = set(np.argsort(ld_d)[:k].tolist())
            cur_np = len(ld_topk & hd_nbr_set) / k * 100

            if cur_np > best_np:
                best_np   = cur_np
                best_step = step
                np_stall  = 0
            else:
                np_stall += 1

            model.train()

            if np_stall >= np_stall_limit and step >= 50:
                break

    # Final projection
    model.eval()
    with torch.no_grad():
        z_final = model.encoder(full_feat_t).cpu().numpy()

    np_val, t_val, c_val = compute_metrics(
        full_feat, z_final, focus_idx, hd_neighbors_all, k=k, K_ext=K_ext
    )

    # Restore weights
    with torch.no_grad():
        for p in trainable:
            p.data.copy_(weight_backup[id(p)])
    for param in model.parameters():
        param.requires_grad = True

    elapsed = time.time() - t0
    return np_val, t_val, c_val, step + 1, elapsed, z_final


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG
    epoch        = cfg["epoch"]
    content_path = cfg["content_path"]
    vis_method   = cfg["vis_method"]
    vis_id       = cfg["vis_id"]

    print("=" * 60)
    print(f"  Refine Benchmark — epoch {epoch}")
    print(f"  refine_enabled = {cfg['refine_enabled']}")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    feat_path    = os.path.join(content_path, "epochs", f"epoch_{epoch}", "embeddings.npy")
    proj_path    = os.path.join(content_path, "visualize",
                                f"{vis_method}_{vis_id}", "epochs", f"epoch_{epoch}", "projection.npy")
    hd_nb_path   = os.path.join(content_path, "epochs", f"epoch_{epoch}", "hd_neighbors_10.json")
    model_path   = os.path.join(content_path, "visualize",
                                f"{vis_method}_{vis_id}", "vis_model.pth")

    print(f"Loading features  : {feat_path}")
    full_feat = np.load(feat_path)          # [N, D]
    print(f"Loading projection: {proj_path}")
    full_proj = np.load(proj_path).copy()   # [N, 2]

    print(f"Loading HD neighbours: {hd_nb_path}")
    with open(hd_nb_path) as f:
        hd_neighbors_all = json.load(f)     # list[list[int]]

    N = len(full_feat)
    print(f"Dataset size N = {N}, feat dim = {full_feat.shape[1]}")

    # ── Load model ─────────────────────────────────────────────────────────────
    device = torch.device("cpu")  # change to "cuda:0" if GPU available
    model  = VisModel(cfg["encoder_dims"], cfg["decoder_dims"]).to(device)
    ckpt   = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model loaded from : {model_path}")

    # ── Select focus points ────────────────────────────────────────────────────
    if cfg["focus_indices"] is not None:
        focus_list = list(cfg["focus_indices"])
    else:
        rng        = np.random.default_rng(seed=cfg["random_seed"])
        focus_list = rng.choice(N, size=cfg["n_random_points"], replace=False).tolist()

    print(f"\nFocus points ({len(focus_list)}): {focus_list}\n")

    # ── Baseline metrics (without refine) ──────────────────────────────────────
    print("─── Baseline metrics (no refine) ───")
    baseline_results = []
    for fi in focus_list:
        np_v, t_v, c_v = compute_metrics(
            full_feat, full_proj, fi, hd_neighbors_all,
            k=cfg["k_metric"], K_ext=cfg["K_ext"]
        )
        baseline_results.append((fi, np_v, t_v, c_v))
        print(f"  point {fi:6d}  NP={np_v*100:5.1f}%  T={t_v*100:5.1f}%  C={c_v*100:5.1f}%")

    avg_np_base = np.mean([r[1] for r in baseline_results]) * 100
    avg_t_base  = np.mean([r[2] for r in baseline_results]) * 100
    avg_c_base  = np.mean([r[3] for r in baseline_results]) * 100
    print(f"\n  AVG baseline  NP={avg_np_base:.1f}%  T={avg_t_base:.1f}%  C={avg_c_base:.1f}%")

    if not cfg["refine_enabled"]:
        print("\n[refine_enabled=False] Skipping refine. Done.")
        return

    # ── Refine metrics ─────────────────────────────────────────────────────────
    print("\n─── Refine metrics (anchor-constrained) ───")
    refine_results = []
    for fi in focus_list:
        # Deep-copy model so each focus point starts from the same clean weights
        m_copy = copy.deepcopy(model)

        np_v, t_v, c_v, steps, elapsed, _ = run_refine_for_point(
            m_copy, full_feat, full_proj, fi,
            hd_neighbors_all, device, cfg
        )
        refine_results.append((fi, np_v, t_v, c_v, steps, elapsed))
        print(f"  point {fi:6d}  NP={np_v*100:5.1f}%  T={t_v*100:5.1f}%  C={c_v*100:5.1f}%"
              f"  steps={steps:4d}  t={elapsed:.1f}s")

    avg_np_ref = np.mean([r[1] for r in refine_results]) * 100
    avg_t_ref  = np.mean([r[2] for r in refine_results]) * 100
    avg_c_ref  = np.mean([r[3] for r in refine_results]) * 100
    avg_steps  = np.mean([r[4] for r in refine_results])
    avg_time   = np.mean([r[5] for r in refine_results])

    print(f"\n  AVG refine    NP={avg_np_ref:.1f}%  T={avg_t_ref:.1f}%  C={avg_c_ref:.1f}%"
          f"  steps={avg_steps:.0f}  t={avg_time:.1f}s")

    # ── Delta ──────────────────────────────────────────────────────────────────
    print("\n─── Delta (refine − baseline) ───")
    print(f"  ΔNP = {avg_np_ref - avg_np_base:+.1f}%")
    print(f"  ΔT  = {avg_t_ref  - avg_t_base:+.1f}%")
    print(f"  ΔC  = {avg_c_ref  - avg_c_base:+.1f}%")

    # ── Per-point delta ────────────────────────────────────────────────────────
    print("\n─── Per-point delta ───")
    print(f"  {'idx':>8}  {'ΔNP':>7}  {'ΔT':>7}  {'ΔC':>7}")
    for base, ref in zip(baseline_results, refine_results):
        fi = base[0]
        dnp = (ref[1] - base[1]) * 100
        dt  = (ref[2] - base[2]) * 100
        dc  = (ref[3] - base[3]) * 100
        print(f"  {fi:>8}  {dnp:>+6.1f}%  {dt:>+6.1f}%  {dc:>+6.1f}%")

    # ── CSV export ─────────────────────────────────────────────────────────────
    if cfg["save_per_point_csv"]:
        import csv
        with open(cfg["csv_path"], "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["focus_idx",
                             "base_NP", "base_T", "base_C",
                             "ref_NP",  "ref_T",  "ref_C",
                             "delta_NP","delta_T","delta_C",
                             "steps", "time_s"])
            for base, ref in zip(baseline_results, refine_results):
                fi = base[0]
                writer.writerow([
                    fi,
                    f"{base[1]*100:.2f}", f"{base[2]*100:.2f}", f"{base[3]*100:.2f}",
                    f"{ref[1]*100:.2f}",  f"{ref[2]*100:.2f}",  f"{ref[3]*100:.2f}",
                    f"{(ref[1]-base[1])*100:.2f}",
                    f"{(ref[2]-base[2])*100:.2f}",
                    f"{(ref[3]-base[3])*100:.2f}",
                    ref[4], f"{ref[5]:.1f}",
                ])
        print(f"\nPer-point results saved to: {cfg['csv_path']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
