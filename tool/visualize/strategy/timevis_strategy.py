import os
import shutil
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from strategy.trainer import SingleVisTrainer
from strategy.custom_weighted_random_sampler import CustomWeightedRandomSampler
from strategy.edge_dataset import DataHandler
from strategy.spatial_edge_constructor import kcSpatialEdgeConstructor
from strategy.temporal_edge_constructor import GlobalTemporalEdgeConstructor
from strategy.losses import SingleVisLoss, UmapLoss, ReconstructionLoss
from visualize_model import VisModel
from strategy.strategy_abstract import StrategyAbstractClass
from data_provider import DataProvider
from umap.umap_ import find_ab_params

import sys as _sys
_server_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server'))
if _server_dir not in _sys.path:
    _sys.path.insert(0, _server_dir)
from refine_behavior_config import resolve_refine_behavior_config

class TimeVis(StrategyAbstractClass):
    def __init__(self, config, data_provider):
        super().__init__(config)
        self.initialize_model()
        self.data_provider = data_provider
        
    def initialize_model(self):
        gpu_id = self.config['vis_config']['gpu_id']
        self.device = torch.device("cuda:{}".format(self.config['vis_config']['gpu_id']) if torch.cuda.is_available() and gpu_id != -1 else "cpu")
        self.visualize_model = VisModel(self.config['vis_config']['encoder_dims'], self.config['vis_config']['decoder_dims']).to(self.device)
        
        # define losses
        negative_sample_rate = 5
        min_dist = 0.1
        _a, _b = find_ab_params(1.0, min_dist)
        self.umap_fn = UmapLoss(negative_sample_rate, self.device, _a, _b, repulsion_strength=1.0)
        self.recon_fn = ReconstructionLoss(beta=1.0)
        self.criterion = SingleVisLoss(self.umap_fn, self.recon_fn, lambd=self.config['vis_config']['lambda'],negative_sample_rate=negative_sample_rate)
    
    def train(self):
        self.train_vis_model()

    def train_vis_model(self):
        # parameters
        N_NEIGHBORS = self.config['vis_config']["n_neighbors"]
        S_N_EPOCHS = self.config['vis_config']["s_n_epochs"]
        B_N_EPOCHS = self.config['vis_config']['b_n_epochs']
        T_N_EPOCHS = self.config['vis_config']['t_n_epochs'] # 5
        PATIENT = self.config['vis_config']['patient']
        MAX_EPOCH = self.config['vis_config']['max_epochs']
        
        optimizer = torch.optim.Adam(self.visualize_model.parameters(), lr=.01, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

        INIT_NUM = 10
        ALPHA, BETA = 1, 1
        spatial_cons = kcSpatialEdgeConstructor(data_provider=self.data_provider, init_num=INIT_NUM, s_n_epochs=S_N_EPOCHS, b_n_epochs=B_N_EPOCHS, n_neighbors=N_NEIGHBORS, MAX_HAUSDORFF=None, ALPHA=ALPHA, BETA=BETA)
        s_edge_to, s_edge_from, s_probs, feature_vectors, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention = spatial_cons.construct()
        temporal_cons = GlobalTemporalEdgeConstructor(X=feature_vectors, time_step_nums=time_step_nums, sigmas=sigmas, rhos=rhos, n_neighbors=N_NEIGHBORS, n_epochs=T_N_EPOCHS)
        t_edge_to, t_edge_from, t_probs = temporal_cons.construct()

        edge_to = np.concatenate((s_edge_to, t_edge_to),axis=0)
        edge_from = np.concatenate((s_edge_from, t_edge_from), axis=0)
        probs = np.concatenate((s_probs, t_probs), axis=0)
        probs = probs / (probs.max()+1e-3)
        eliminate_zeros = probs>1e-3
        edge_to = edge_to[eliminate_zeros]
        edge_from = edge_from[eliminate_zeros]
        probs = probs[eliminate_zeros]
        
        dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)
        # Save edge graph so refine() can look up neighbors without rebuilding the graph.
        self.data_handler = dataset
        self.feature_vectors = feature_vectors

        n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
        # 2^24 written as a Python XOR (^) is a bug; use the correct bit-shift form.
        if len(edge_to) > 2 ** 24:
            sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
        else:
            sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
        edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)

        trainer = SingleVisTrainer(self.visualize_model, self.criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=self.device)
        trainer.train(MAX_EPOCH)

        self.save_vis_model(self.visualize_model, trainer.loss, trainer.optimizer)
        
        selected_idxs_path = os.path.join(self.config["content_path"],  "selected_idxs")
        if os.path.exists(selected_idxs_path):
            shutil.rmtree(selected_idxs_path)

    def save_vis_model(self, model, loss = None, optimizer = None):
        save_model = {
            "loss": loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        # path：Dataset/backdoor/visualize/DynaVis-0/
        target_path = os.path.join(self.config["content_path"], "visualize", 
            f"{self.config.get('vis_method')}_{self.config.get('vis_id')}")
        os.makedirs(target_path, exist_ok=True)
        # os.makedirs(os.path.join(self.config["content_path"],"visualize", self.config["vis_id"]), exist_ok=True)
        full_model_path = os.path.join(target_path, "vis_model.pth")
        torch.save(save_model, full_model_path)
    
    def get_focus_mask(self, selected_indices):
        import numpy as np
        import torch
        
        # 1. 尝试从最可靠的地方获取总点数
        total_count = 0
        try:
            # 方法 A: 检查 data_provider 是否有已加载的数据
            if hasattr(self.data_provider, 'train_data'):
                total_count = len(self.data_provider.train_data)
            # 方法 B: 这里的 TimeVis 实例应该能通过 data_provider 获取数据
            else:
                # 这里的 0 代表获取第 0 个 epoch 的数据来计算总数
                test_data = self.data_provider.get_train_data(0)
                total_count = len(test_data)
        except Exception as e:
            # 方法 C: 如果上述都失败，读取硬盘 index.npy（这是最稳妥的）
            try:
                index_path = os.path.join(self.config["content_path"], "epochs", "index.npy")
                total_count = len(np.load(index_path))
            except:
                # 最后的保底
                total_count = max(selected_indices) + 1 if selected_indices else 10000

        # 2. 确保使用 self.device (initialize_model 中已定义)
        mask = torch.zeros(total_count, dtype=torch.bool).to(self.device)
        
        if selected_indices:
            mask[selected_indices] = True
            
        return mask
        
    def update_ttav_context(self, indices, mode, mask):
        """
        Update trainer state. Called by the server before incremental training.
        """
        self.ttav_indices = indices
        self.ttav_mode = mode
        self.ttav_mask = mask # Boolean mask on GPU
                
    def refine(self, focus_indices=None, focus_index=None, neighbor_indices=None,
               current_epoch=None, epochs_to_update=10, _skip_avg_benchmark=False,
               progress_callback=None, should_stop_callback=None,
               progress_refresh_indices=None,
               secondary_indices=None):
        """
        Locally refine projections for a set of focus points.

        Key design decisions for speed and global stability:
        1. Freeze encoder — only decoder weights are updated, so the global
           embedding topology cannot drift.
        2. Subset inference — after fine-tuning, only the focus + neighbor
           subset is re-projected; all other points keep their original
           coordinates (patch strategy).
        3. Single-epoch write — only `current_epoch` is updated immediately;
           the caller may trigger background updates for other epochs separately.
        """
        import torch
        import numpy as np
        import time
        import os

        # Normalise focus list
        if focus_indices is None:
            focus_indices = [focus_index] if focus_index is not None else []

        start_time = time.time()
        vis_method = self.config['vis_method']
        vis_id    = self.config['vis_id']
        content_path = self.config['content_path']
        available_epochs = self.config['available_epochs']
        vis_cfg = self.config['vis_config']

        # Default current_epoch to the last available epoch
        if current_epoch is None:
            current_epoch = available_epochs[-1]

        # --- 0. Hot-load ablation config from project root (no server restart) --
        import json
        _ablation_cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))), "tests", "ablation_config.json"
        )
        _ab_sigma_mode = "adaptive"
        _ab_lambda_reg = None   # None → use _lambda_reg_map defaults below
        if os.path.exists(_ablation_cfg_path):
            try:
                with open(_ablation_cfg_path) as _f:
                    _ab = json.load(_f)
                _ab_sigma_mode = _ab.get("sigma_mode", "adaptive")
                _ab_lambda_reg = _ab.get("lambda_reg", None)
                print(f"[TimeVis] refine cfg: sigma_mode={_ab_sigma_mode}, lambda_reg={_ab_lambda_reg}")
            except Exception:
                pass  # malformed JSON → fall through to defaults

        # --- 1. Ensure model weights are loaded (load path) -------------------
        if not hasattr(self, '_model_loaded'):
            model_path = os.path.join(
                content_path, 'visualize', f"{vis_method}_{vis_id}", 'vis_model.pth'
            )
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location=self.device)
                self.visualize_model.load_state_dict(ckpt['state_dict'])
                self.visualize_model.to(self.device)
            self._model_loaded = True

        # --- 2. Build neighbour list ------------------------------------------
        if not neighbor_indices:
            if hasattr(self, 'data_handler'):
                all_edges_to   = self.data_handler.edge_to
                all_edges_from = self.data_handler.edge_from
                collected = set()
                for fi in focus_indices:
                    idx = np.where(all_edges_to == fi)[0]
                    collected.update(all_edges_from[idx].tolist())
                neighbor_indices = list(collected - set(focus_indices))[:15 * max(len(focus_indices), 1)]
            else:
                from sklearn.neighbors import NearestNeighbors
                feats = self.data_provider.get_representation(current_epoch)
                k = min(16, len(feats) - 1)
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(feats)
                _, nn_idx = nbrs.kneighbors(feats[focus_indices])
                collected = set(nn_idx.flatten().tolist()) - set(focus_indices)
                neighbor_indices = list(collected)[:15 * max(len(focus_indices), 1)]

        all_indices = list(focus_indices) + neighbor_indices

        # --- 3. Gather full-dataset features and baseline projection ------------
        full_feat = self.data_provider.get_representation(current_epoch)  # [N, D]
        N = len(full_feat)

        baseline_path = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}",
            'epochs', f'epoch_{current_epoch}', 'projection.npy'
        )
        if os.path.exists(baseline_path):
            full_proj_baseline = np.load(baseline_path).copy()  # [N, 2]
        else:
            local_visualizer.eval()
            with torch.no_grad():
                full_proj_baseline = self.visualize_model.encoder(
                    torch.from_numpy(full_feat).float().to(self.device)
                ).cpu().numpy()

        # --- 3a. Load cached high-D neighbors for focus points ----------------
        import json as _json
        hd_cache_path = os.path.join(
            content_path, 'epochs', f'epoch_{current_epoch}', 'hd_neighbors_10.json'
        )
        if os.path.exists(hd_cache_path):
            with open(hd_cache_path) as _f:
                hd_neighbors_all = _json.load(_f)  # list[list[int]], length N
        else:
            # Fallback: compute on-the-fly for focus points only
            from sklearn.neighbors import NearestNeighbors as _NNS
            _nbrs = _NNS(n_neighbors=11, algorithm='auto').fit(full_feat)
            _, _nn_idx = _nbrs.kneighbors(full_feat)
            hd_neighbors_all = [_nn_idx[i, 1:].tolist() for i in range(N)]

        # Collect all high-D neighbor indices referenced by focus points
        hd_nbr_set = set()
        for fi in focus_indices:
            hd_nbr_set.update(hd_neighbors_all[fi][:10])
        hd_nbr_set -= set(focus_indices)

        # Secondary indices expand the training context but are NOT attract targets.
        sec_set = set()
        if secondary_indices:
            sec_set = {int(i) for i in secondary_indices} - set(focus_indices)
            sec_hd_set = set()
            for si in sec_set:
                if si < len(hd_neighbors_all):
                    sec_hd_set.update(hd_neighbors_all[si][:10])
            sec_hd_set -= set(focus_indices) | sec_set
            sec_set |= sec_hd_set
            if sec_set:
                print(f"[TimeVis] Tiered refine: {len(focus_indices)} primary + {len(sec_set)} secondary context points")

        training_context_indices = sorted(set(focus_indices) | hd_nbr_set | sec_set)
        # patch_support_indices = extra points to update beyond the training context
        patch_support_indices = neighbor_indices
        patch_indices = sorted(set(training_context_indices) | set(patch_support_indices))
        self._last_focus_indices = focus_indices
        self._last_training_context_indices = training_context_indices
        self._last_patch_indices = patch_indices

        # --- 3b. Anchor points: ALL non-focus points --------------------------
        # For small datasets (e.g. EIF ~50 tokens), hd_nbr_set covers nearly
        # all N points, leaving candidate_anchors empty and mu_anchor * 0 = 0,
        # so the encoder collapses everything.  The fix: anchor EVERY non-focus
        # point regardless of whether it is a HD neighbor.  Combined with
        # detaching z_edge_from in the training loop, only focus points'
        # encoder path receives gradient — all others stay at baseline.
        anchor_indices = [i for i in range(N) if i not in set(focus_indices)]
        if anchor_indices:
            anchor_feat_np = full_feat[anchor_indices]
            anchor_z0      = full_proj_baseline[anchor_indices]
        else:
            anchor_feat_np = np.zeros((0, full_feat.shape[1]), dtype=full_feat.dtype)
            anchor_z0      = np.zeros((0, full_proj_baseline.shape[1]), dtype=full_proj_baseline.dtype)

        # --- 4. Direct 2D coordinate optimisation (no encoder fine-tuning) --------
        # Fine-tuning the encoder changes ALL points' projections through the shared
        # f(x) function, causing global drift that anchor loss cannot fully prevent
        # (especially on small EIF datasets where every point is a HD neighbour).
        #
        # Solution: keep the encoder frozen.  Treat the 2D coordinates of the focus
        # points as the sole trainable parameters.  All other points stay exactly at
        # their baseline positions — global layout is preserved by construction.
        #
        # UmapLoss operates on 2D space directly:
        #   positive pairs  = (focus_z, baseline_nbr_z)  — attract neighbours
        #   negative pairs  = UmapLoss internal shuffle of baseline_nbr_z — repel non-nbrs
        # Because baseline_nbr_z are constants, gradients flow only through focus_z.

        focus_z = torch.nn.Parameter(
            torch.from_numpy(full_proj_baseline[focus_indices].copy()).float().to(self.device)
        )
        optimizer = torch.optim.Adam([focus_z], lr=0.01)

        # Build fixed neighbour target positions (constant — never optimised)
        _edge_from_parts = []
        k_hd_per_focus   = []
        for fi in focus_indices:
            nbr_idx = hd_neighbors_all[fi][:10]
            k_hd_per_focus.append(len(nbr_idx))
            _edge_from_parts.append(
                torch.from_numpy(full_proj_baseline[nbr_idx].copy()).float().to(self.device)
            )
        edge_from_fixed_t = torch.cat(_edge_from_parts, dim=0)  # [n_edges, 2] — constant

        behavior_cfg = resolve_refine_behavior_config(vis_cfg)
        _progress_cfg = behavior_cfg["progressive_updates"]
        _snapshot_every = max(1, int(_progress_cfg.get("snapshot_every_steps", 30)))

        _loss_window     = []
        _WINDOW          = 20
        _MIN_STEPS       = int(self.config['vis_config'].get('refine_min_steps', 80))
        _REL_TOL         = 5e-4
        _MAX_STEPS       = int(self.config['vis_config'].get('refine_max_steps', 1000))
        _PATIENCE        = int(self.config['vis_config'].get('refine_patience', 120))
        _MIN_DELTA       = float(self.config['vis_config'].get('refine_loss_min_delta', 1e-4))
        _TIME_LIMIT_S    = float(self.config['vis_config'].get('refine_time_limit_s', 45.0))
        _best_loss       = float('inf')
        _stale_steps     = 0

        print(f"[TimeVis] coord-opt refine: n_focus={len(focus_indices)}, "
              f"n_edges={edge_from_fixed_t.shape[0]}, max_steps={_MAX_STEPS}, "
              f"time_limit_s={_TIME_LIMIT_S}")

        for step in range(_MAX_STEPS):
            optimizer.zero_grad()

            # Rebuild edge_to each step from the current focus_z values
            edge_to_t = torch.cat([
                focus_z[i].unsqueeze(0).expand(k_hd_per_focus[i], -1)
                for i in range(len(focus_indices))
            ], dim=0)  # [n_edges, 2]

            l_umap = self.umap_fn(edge_to_t, edge_from_fixed_t)
            l_umap.backward()
            optimizer.step()

            cur_loss = float(l_umap.detach().item())
            if cur_loss + _MIN_DELTA < _best_loss:
                _best_loss   = cur_loss
                _stale_steps = 0
            else:
                _stale_steps += 1

            _loss_window.append(cur_loss)
            if len(_loss_window) > _WINDOW:
                _loss_window.pop(0)

            if step % 50 == 0:
                print(f"[TimeVis] step={step:4d}  L_umap={cur_loss:.4f}  "
                      f"stale={_stale_steps}  t={time.time()-start_time:.1f}s")

            if _PATIENCE > 0 and step >= _MIN_STEPS and _stale_steps >= _PATIENCE:
                print(f"[TimeVis] Early stop at step {step}: stale {_stale_steps} steps")
                break

            if step >= _MIN_STEPS and len(_loss_window) == _WINDOW:
                _w_max, _w_min = max(_loss_window), min(_loss_window)
                if _w_max > 0 and (_w_max - _w_min) / _w_max < _REL_TOL:
                    print(f"[TimeVis] Loss converged at step {step}")
                    break

            if _TIME_LIMIT_S > 0 and time.time() - start_time > _TIME_LIMIT_S:
                print(f"[TimeVis] Time limit at step {step} ({_TIME_LIMIT_S}s)")
                break

            # Progress snapshot: global layout = baseline; only focus coords updated
            if progress_callback and (step + 1) % _snapshot_every == 0:
                _snap_z = full_proj_baseline.copy()
                _snap_z[focus_indices] = focus_z.detach().cpu().numpy()
                progress_callback({
                    "steps_completed": step + 1,
                    "projection": _snap_z.tolist(),
                    "focus_indices": list(focus_indices),
                    "training_context_indices": list(training_context_indices),
                    "patch_indices": list(patch_indices),
                })

            if should_stop_callback and should_stop_callback():
                print(f"[TimeVis] Stop requested at step {step}")
                break

        # Build final projection: baseline everywhere, optimised coords for focus
        z_np = full_proj_baseline.copy()
        z_np[focus_indices] = focus_z.detach().cpu().numpy()

        # Guard k so it never exceeds the number of other points (e.g. very small N).
        k     = min(10, N - 1)
        K_ext = min(200, N - 1)

        trust_sum = 0.0
        cont_sum  = 0.0
        np_sum    = 0.0
        mrh_sum   = 0.0

        try:
            for fi_glob in focus_indices:
                # High-dim ranks (argsort over all N, excluding self)
                hd_feat_fi = full_feat[fi_glob]
                hd_dists = np.linalg.norm(full_feat - hd_feat_fi, axis=1)
                hd_dists[fi_glob] = np.inf
                hd_rank = np.argsort(hd_dists)

                # Low-dim ranks
                ld_dists = np.linalg.norm(z_np - z_np[fi_glob], axis=1)
                ld_dists[fi_glob] = np.inf
                ld_rank = np.argsort(ld_dists)

                # Rank lookup: point_idx → 1-based rank within K_ext
                hd_rank_of = {int(hd_rank[r]): r + 1 for r in range(K_ext)}
                ld_rank_of = {int(ld_rank[r]): r + 1 for r in range(K_ext)}

                # Full LD rank lookup for MRH (covers all N-1 non-self points)
                ld_rank_full = {int(ld_rank[r]): r + 1 for r in range(N - 1)}

                hd_topk = set(int(hd_rank[r]) for r in range(k) if int(hd_rank[r]) != fi_glob)
                ld_topk = set(int(ld_rank[r]) for r in range(k) if int(ld_rank[r]) != fi_glob)

                # NP: strict top-k set intersection
                denom_np = max(k, 1)
                np_sum += len(hd_topk & ld_topk) / denom_np

                # MRH: mean LD rank of HD top-k neighbors
                mrh_vals = [ld_rank_full.get(int(j), N) for j in hd_topk]
                mrh_sum += float(np.mean(mrh_vals)) if mrh_vals else float(N)

                # Trustworthiness penalty: fake LD neighbors (in LD but not HD top-k)
                t_penalty = 0.0
                for j in (ld_topk - hd_topk):
                    r_hd = hd_rank_of.get(int(j), K_ext + 1)
                    t_penalty += max(0, r_hd - k)

                # Continuity penalty: missing HD neighbors (in HD but not LD top-k)
                c_penalty = 0.0
                for j in (hd_topk - ld_topk):
                    r_ld = ld_rank_of.get(int(j), K_ext + 1)
                    c_penalty += max(0, r_ld - k)

                # Per-point worst-case: k fake neighbors each at max rank K_ext
                worst = k * (K_ext - k)
                trust_sum += 1.0 - t_penalty / worst if worst > 0 else 1.0
                cont_sum  += 1.0 - c_penalty / worst if worst > 0 else 1.0

            n_f = max(len(focus_indices), 1)
            final_np    = np_sum  / n_f * 100.0
            final_mrh   = mrh_sum / n_f
            final_trust = max(0.0, trust_sum / n_f * 100.0)
            final_cont  = max(0.0, cont_sum  / n_f * 100.0)

            self._last_refine_np    = final_np
            self._last_refine_mrh   = final_mrh
            self._last_refine_trust = final_trust
            self._last_refine_cont  = final_cont
        except Exception as _metrics_err:
            print(f"[TimeVis] WARNING: metrics computation failed: {_metrics_err}")
            import traceback as _tb; _tb.print_exc()
            self._last_refine_np    = None
            self._last_refine_mrh   = None
            self._last_refine_trust = None
            self._last_refine_cont  = None
            final_np = final_mrh = final_trust = final_cont = float('nan')

        print(f"[TimeVis] coord-opt refine done: "
              f"steps={step+1}  NP={final_np:.1f}%  MRH={final_mrh:.1f}  "
              f"T={final_trust:.1f}%  C={final_cont:.1f}%  "
              f"t={time.time()-start_time:.1f}s")

        # --- 5. Write patched projection directly (no encoder re-run) -----------
        # z_np = baseline with only focus coords updated.  Non-focus points are
        # untouched by construction — no encoder fine-tuning happened.
        refined_dir = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}_refined",
            'epochs', f'epoch_{current_epoch}'
        )
        os.makedirs(refined_dir, exist_ok=True)
        np.save(os.path.join(refined_dir, 'projection.npy'), z_np)

        # Store the coordinate delta so patch_other_epochs can apply the same
        # spatial shift to other epochs without re-running the optimiser.
        self._last_focus_delta = focus_z.detach().cpu().numpy() - full_proj_baseline[focus_indices]
        self._last_local_visualizer = None  # no encoder model in coord-opt mode

        print(f"[TimeVis] coord-opt refine finished in {time.time() - start_time:.2f}s "
              f"(epoch={current_epoch}, {len(focus_indices)} focus points moved)")

        # Store all_indices so patch_other_epochs() can reuse them.
        self._last_refine_indices = all_indices

        # --- 6. Background ablation (non-blocking) ----------------------------
        if os.path.exists(_ablation_cfg_path):
            try:
                with open(_ablation_cfg_path) as _f:
                    _ab_full = json.load(_f)
                if _ab_full.get("ablation_enabled", False):
                    import threading
                    _ab_model_copy = copy.deepcopy(self.visualize_model)
                    # Pass full-dataset features and the on-disk baseline projection
                    _full_feat = self.data_provider.get_representation(current_epoch)
                    _baseline_proj_path = os.path.join(
                        content_path, 'visualize', f"{vis_method}_{vis_id}",
                        'epochs', f'epoch_{current_epoch}', 'projection.npy'
                    )
                    _ab_thread = threading.Thread(
                        target=self._run_ablation_background,
                        args=(_ab_full, _ab_model_copy, current_epoch,
                              content_path, _full_feat, _baseline_proj_path,
                              _ablation_cfg_path),
                        daemon=True,
                    )
                    _ab_thread.start()
                    print("[TimeVis] Background ablation started.")
            except Exception as _e:
                print(f"[TimeVis] Ablation launch failed: {_e}")

        # --- 7. Refine-avg benchmark (background, non-blocking) ----------------
        # Reads tests/refine_avg_config.json. When "enabled" is true, runs
        # self.refine() independently on each test point (same code path as a
        # real interactive refine), then writes the per-point NP/T/C and the
        # running average to tests/refine_avg_results.json.
        _avg_cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))), "tests", "refine_avg_config.json"
        )
        if not _skip_avg_benchmark and os.path.exists(_avg_cfg_path):
            try:
                with open(_avg_cfg_path) as _f:
                    _avg_cfg = json.load(_f)
                if _avg_cfg.get("enabled", False):
                    import threading
                    _avg_thread = threading.Thread(
                        target=self._run_refine_avg_benchmark,
                        args=(_avg_cfg, _avg_cfg_path, current_epoch),
                        daemon=True,
                    )
                    _avg_thread.start()
                    print("[TimeVis] refine-avg benchmark started in background.")
            except Exception as _e:
                print(f"[TimeVis] refine-avg launch failed: {_e}")

    def _run_refine_avg_benchmark(self, avg_cfg, cfg_path, epoch):
        """
        Background benchmark: call self.refine() on each test point (identical
        code path to an interactive refine), then write per-point NP/T/C and
        the running average to the results file.
        """
        import json, time, os, traceback
        import numpy as np

        try:
            content_path = self.config['content_path']
            N = len(self.data_provider.get_representation(epoch))

            # Determine test points: config-specified list or random sample
            test_indices = avg_cfg.get("focus_indices", None)
            if not test_indices:
                n_pts = avg_cfg.get("n_points", 10)
                rng   = np.random.default_rng(seed=avg_cfg.get("seed", 0))
                test_indices = rng.choice(N, size=min(n_pts, N),
                                          replace=False).tolist()

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            results_rel  = avg_cfg.get("results_path", "tests/refine_avg_results.json")
            results_path = os.path.join(project_root, results_rel)

            print(f"[TimeVis] refine-avg: running on {len(test_indices)} points "
                  f"epoch={epoch} → {results_path}")

            point_rows = []
            for fi in test_indices:
                # Call the real refine() for this single focus point.
                # Weight backup/restore is already inside refine(), so every
                # point starts from the same clean model state.
                self.refine(
                    focus_indices=[fi],
                    neighbor_indices=[],
                    current_epoch=epoch,
                    _skip_avg_benchmark=True,
                )
                point_rows.append({
                    "focus_idx": fi,
                    "NP":  round(self._last_refine_np,    2),
                    "MRH": round(self._last_refine_mrh,   2),
                    "T":   round(self._last_refine_trust,  2),
                    "C":   round(self._last_refine_cont,   2),
                })
                print(f"[TimeVis] refine-avg  point {fi:6d}  "
                      f"NP={self._last_refine_np:.1f}%  "
                      f"MRH={self._last_refine_mrh:.1f}  "
                      f"T={self._last_refine_trust:.1f}%  "
                      f"C={self._last_refine_cont:.1f}%")

            avg_np  = float(np.mean([r["NP"]  for r in point_rows]))
            avg_mrh = float(np.mean([r["MRH"] for r in point_rows]))
            avg_t   = float(np.mean([r["T"]   for r in point_rows]))
            avg_c   = float(np.mean([r["C"]   for r in point_rows]))

            record = {
                "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch":         epoch,
                "focus_indices": test_indices,
                "per_point":     point_rows,
                "avg_NP":        round(avg_np,  2),
                "avg_MRH":       round(avg_mrh, 2),
                "avg_T":         round(avg_t,   2),
                "avg_C":         round(avg_c,   2),
            }

            existing = []
            if os.path.exists(results_path):
                try:
                    with open(results_path) as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            existing.append(record)
            with open(results_path, "w") as f:
                json.dump(existing, f, indent=2)

            print(f"[TimeVis] refine-avg DONE: "
                  f"avg NP={avg_np:.1f}%  MRH={avg_mrh:.1f}  "
                  f"T={avg_t:.1f}%  C={avg_c:.1f}%  "
                  f"({len(point_rows)} points)  → {results_path}")

        except Exception:
            print(f"[TimeVis] refine-avg benchmark ERROR:\n{traceback.format_exc()}")

    def _run_ablation_background(self, ab_cfg, ab_model, epoch,
                                 content_path, full_feat, baseline_proj_path, cfg_path):
        """
        Run multiple ablation configs in a background thread using an independent
        model copy. Results are appended to ablation_results.json next to cfg_path.

        full_feat           : full-dataset high-D features  [N, D]
        baseline_proj_path  : path to the on-disk baseline projection.npy [N, 2]
                              used as proj_before so global_drift is meaningful.
        """
        import json, time, os, traceback
        import numpy as np
        import torch
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import pairwise_distances_argmin_min
        try:
            self.__run_ablation_inner(ab_cfg, ab_model, epoch, content_path,
                                      full_feat, baseline_proj_path, cfg_path)
        except Exception:
            print(f"[Ablation] ERROR:\n{traceback.format_exc()}")

    def __run_ablation_inner(self, ab_cfg, ab_model, epoch,
                             content_path, full_feat, baseline_proj_path, cfg_path):
        import json, time, os
        import numpy as np
        import torch
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import pairwise_distances_argmin_min

        n_pts   = ab_cfg.get("n_random_points", 20)
        configs = ab_cfg.get("configs", [])
        if not configs:
            return

        n_total = len(full_feat)

        # Fix 1: proj_before = on-disk baseline, not model inference
        # This matches what the real refine() uses as its starting point.
        if not os.path.exists(baseline_proj_path):
            print(f"[Ablation] baseline projection not found: {baseline_proj_path}, aborting.")
            return
        proj_before = np.load(baseline_proj_path)   # [N, 2]

        # Fix 2: random focus points are global indices into full_feat
        rng           = np.random.default_rng(seed=42)
        focus_indices = rng.choice(n_total, size=min(n_pts, n_total),
                                   replace=False).tolist()

        vis_method = self.config['vis_method']
        vis_id     = self.config['vis_id']
        model_path = os.path.join(content_path, 'visualize',
                                  f"{vis_method}_{vis_id}", 'vis_model.pth')

        # Pre-compute high-D neighbors once (shared across configs)
        k_m     = 10
        hd_nbrs = NearestNeighbors(n_neighbors=k_m + 1).fit(full_feat)
        _, hd_idx = hd_nbrs.kneighbors(full_feat)   # [N, k+1]

        run_results = []
        for cfg in configs:
            t0 = time.time()

            # Reset to checkpoint so every config starts from identical weights
            ckpt = torch.load(model_path, map_location=self.device)
            ab_model.load_state_dict(ckpt['state_dict'])

            # Build local subset (focus + kNN neighbors) using full_feat
            k_nb  = min(16, n_total - 1)
            nbrs  = NearestNeighbors(n_neighbors=k_nb, algorithm='auto').fit(full_feat)
            _, nn_idx = nbrs.kneighbors(full_feat[focus_indices])
            collected = set(nn_idx.flatten().tolist()) - set(focus_indices)
            neighbor_indices = list(collected)[:15 * len(focus_indices)]
            all_idx  = list(focus_indices) + neighbor_indices

            sub_feat = full_feat[all_idx]             # [M, D]
            sub_t    = torch.from_numpy(sub_feat).float()
            k_local  = min(5, len(all_idx) - 1)
            _nbrs2   = NearestNeighbors(n_neighbors=k_local + 1).fit(sub_feat)
            _nn_idx2 = _nbrs2.kneighbors(sub_feat, return_distance=False)
            src_rows = np.repeat(np.arange(len(all_idx)), k_local)
            tgt_rows = _nn_idx2[:, 1:k_local + 1].flatten()

            edge_to   = sub_t[src_rows].to(self.device)
            edge_from = sub_t[tgt_rows].to(self.device)
            a_dummy   = torch.ones(edge_to.shape[0], edge_to.shape[1], device=self.device)

            # Sigma mode (operates on sub_feat local indices — correct)
            sigma_mode = cfg.get("sigma_mode", "adaptive")
            n_focus    = len(focus_indices)
            if sigma_mode == "adaptive" and n_focus > 0 and n_focus < len(sub_feat):
                _, dist_to_focus = pairwise_distances_argmin_min(sub_feat, sub_feat[:n_focus])
                sigma        = float(np.median(dist_to_focus[n_focus:])) + 1e-8
                node_weights = np.exp(-dist_to_focus / sigma)
            else:
                node_weights = np.ones(len(sub_feat), dtype=np.float32)
            edge_w   = np.sqrt(node_weights[src_rows] * node_weights[tgt_rows])
            edge_w_t = torch.from_numpy(edge_w.astype(np.float32)).to(self.device)

            # Lambda
            lambda_reg = float(cfg.get("lambda_reg", 0.1))
            theta_0    = {n: p.data.clone() for n, p in ab_model.named_parameters()}
            num_params = sum(p.numel() for p in ab_model.parameters())

            def _l2(model=ab_model, t0=theta_0, n=num_params):
                reg = torch.tensor(0., device=self.device)
                for name, param in model.named_parameters():
                    reg = reg + torch.sum(torch.square(param - t0[name]))
                return reg / n

            n_steps = ab_cfg.get("n_steps", 5)
            optimizer = torch.optim.Adam(ab_model.parameters(), lr=0.001)
            ab_model.train()
            for _ in range(n_steps):
                optimizer.zero_grad()
                outputs = ab_model(edge_to, edge_from)
                _, _, loss_local = self.criterion(
                    edge_to, edge_from, a_dummy, a_dummy, outputs, weights=edge_w_t
                )
                (loss_local + lambda_reg * _l2()).backward()
                optimizer.step()

            # Full inference on full_feat → proj_after [N, 2]
            ab_model.eval()
            with torch.no_grad():
                proj_after = ab_model.encoder(
                    torch.from_numpy(full_feat).float().to(self.device)
                ).cpu().numpy()

            # Metrics (all indices are global into full_feat / proj_*)
            focus_set    = set(focus_indices)
            non_focus    = [i for i in range(n_total) if i not in focus_set]
            focus_disp   = float(np.mean([
                np.linalg.norm(proj_after[i] - proj_before[i]) for i in focus_indices
            ]))
            global_drift = float(np.mean([
                np.linalg.norm(proj_after[i] - proj_before[i]) for i in non_focus
            ])) if non_focus else 0.0

            # NP: low-D neighbors from proj_after (global), high-D from full_feat (global)
            ld_nbrs   = NearestNeighbors(n_neighbors=k_m + 1).fit(proj_after)
            _, ld_idx = ld_nbrs.kneighbors(proj_after)
            np_scores = []
            for i in focus_indices:
                hd_set  = set(int(hd_idx[i, j]) for j in range(1, k_m + 1))
                ld_list = [int(ld_idx[i, j]) for j in range(1, k_m + 1)]
                np_scores.append(len([x for x in ld_list if x in hd_set]) / k_m)

            avg_np  = float(np.mean(np_scores)) * 100
            elapsed = time.time() - t0
            run_results.append({
                "config":               cfg["name"],
                "sigma_mode":           sigma_mode,
                "lambda_reg":           lambda_reg,
                "focus_displacement":   round(focus_disp,   5),
                "global_drift":         round(global_drift,  5),
                "neighbor_preservation":round(avg_np,        2),
                "elapsed_s":            round(elapsed,       2),
            })
            print(f"[Ablation] {cfg['name']:20s}  FocusDisp={focus_disp:.4f}  "
                  f"GlobalDrift={global_drift:.5f}  NP={avg_np:.1f}%  t={elapsed:.1f}s")

        # Write results
        results_path = os.path.join(os.path.dirname(cfg_path), "ablation_results.json")
        record = {
            "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch":        epoch,
            "n_points":     n_pts,
            "focus_sample": focus_indices[:10],   # first 10 for reference
            "runs":         run_results,
        }
        existing = []
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(record)
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"[Ablation] Results saved → {results_path}")

    def _patch_epoch(self, epoch, all_indices, model=None):
        """Write a full re-projection for one epoch using the provided local visualizer."""
        import numpy as np, os, torch

        vis_method = self.config['vis_method']
        vis_id     = self.config['vis_id']
        content_path = self.config['content_path']
        patch_model = model if model is not None else self.visualize_model

        refined_dir = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}_refined",
            'epochs', f'epoch_{epoch}'
        )

        patch_model.eval()
        with torch.no_grad():
            full_feat = self.data_provider.get_representation(epoch)
            full_proj = patch_model.encoder(
                torch.from_numpy(full_feat).float().to(self.device)
            ).cpu().numpy()

        os.makedirs(refined_dir, exist_ok=True)
        np.save(os.path.join(refined_dir, 'projection.npy'), full_proj)

    def patch_other_epochs(self, skip_epoch):
        """Patch all available epochs except skip_epoch."""
        import threading
        import numpy as np, os
        available_epochs = self.config['available_epochs']
        other_epochs = [e for e in available_epochs if e != skip_epoch]
        if not other_epochs:
            return

        focus_delta    = getattr(self, '_last_focus_delta', None)
        focus_indices  = getattr(self, '_last_focus_indices', None)
        local_visualizer = getattr(self, '_last_local_visualizer', None)

        vis_method   = self.config['vis_method']
        vis_id       = self.config['vis_id']
        content_path = self.config['content_path']

        if focus_delta is not None and focus_indices is not None:
            # Coord-opt mode: apply the same 2D delta to the baseline projection
            # of each other epoch.  No encoder re-run needed.
            def _run_delta():
                for e in other_epochs:
                    try:
                        baseline_path = os.path.join(
                            content_path, 'visualize', f"{vis_method}_{vis_id}",
                            'epochs', f'epoch_{e}', 'projection.npy'
                        )
                        if not os.path.exists(baseline_path):
                            continue
                        proj = np.load(baseline_path).copy()
                        proj[focus_indices] += focus_delta
                        refined_dir = os.path.join(
                            content_path, 'visualize', f"{vis_method}_{vis_id}_refined",
                            'epochs', f'epoch_{e}'
                        )
                        os.makedirs(refined_dir, exist_ok=True)
                        np.save(os.path.join(refined_dir, 'projection.npy'), proj)
                    except Exception as ex:
                        print(f"[TimeVis] Background delta-patch epoch {e} failed: {ex}")
                print(f"[TimeVis] Background delta-patch complete for {len(other_epochs)} epochs.")
            threading.Thread(target=_run_delta, daemon=True).start()

        elif local_visualizer is not None:
            # Encoder fine-tune mode (legacy): re-run encoder for each epoch
            all_indices = getattr(self, '_last_refine_indices', None)
            if not all_indices:
                return
            def _run():
                for e in other_epochs:
                    try:
                        self._patch_epoch(e, all_indices, model=local_visualizer)
                    except Exception as ex:
                        print(f"[TimeVis] Background patch epoch {e} failed: {ex}")
                print(f"[TimeVis] Background patch complete for {len(other_epochs)} epochs.")
            threading.Thread(target=_run, daemon=True).start()

