import os
import shutil
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
               current_epoch=None, epochs_to_update=10):
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

        # Default current_epoch to the last available epoch
        if current_epoch is None:
            current_epoch = available_epochs[-1]

        # --- 0. Hot-load ablation config from project root (no server restart) --
        import json
        _ablation_cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))), "ablation_config.json"
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

        # --- 3. Gather features and build local high-D neighbor pairs ----------
        # `feat[i]` is the high-dim representation of all_indices[i].
        feat = self.data_provider.get_representation(current_epoch)[all_indices]
        feat_t = torch.from_numpy(feat).float()

        # Build (edge_to, edge_from) pairs from high-dim kNN within the subset.
        # This gives UmapLoss a meaningful attract signal: pairs that are close
        # in high-D should also be close in low-D.
        from sklearn.neighbors import NearestNeighbors as _NNS
        k_local = min(5, len(all_indices) - 1)
        _nbrs = _NNS(n_neighbors=k_local + 1, algorithm='auto').fit(feat)
        _dist_mat, _nn_idx = _nbrs.kneighbors(feat, return_distance=True)   # shape [M, k_local+1]

        # _nn_idx[:,0] is self → skip; columns 1.. are true neighbors
        src_rows = np.repeat(np.arange(len(all_indices)), k_local)   # [M*k]
        tgt_rows = _nn_idx[:, 1:k_local + 1].flatten()               # [M*k]

        edge_to   = feat_t[src_rows].to(self.device)   # [M*k, D]
        edge_from = feat_t[tgt_rows].to(self.device)   # [M*k, D]
        a_dummy   = torch.ones(edge_to.shape[0], edge_to.shape[1], device=self.device)

        # --- Plan A: Hierarchical distance-decay weights ----------------------
        n_focus = len(focus_indices)
        if _ab_sigma_mode == "adaptive" and n_focus > 0 and n_focus < len(feat):
            from sklearn.metrics import pairwise_distances_argmin_min
            _, dist_to_focus = pairwise_distances_argmin_min(feat, feat[:n_focus])
            sigma = float(np.median(dist_to_focus[n_focus:])) + 1e-8
            node_weights = np.exp(-dist_to_focus / sigma)
        else:
            node_weights = np.ones(len(feat), dtype=np.float32)   # uniform (ablation baseline)
        edge_weights   = np.sqrt(node_weights[src_rows] * node_weights[tgt_rows])
        edge_weights_t = torch.from_numpy(edge_weights.astype(np.float32)).to(self.device)

        # --- Plan B: Normalised L2 parameter drift constraint -----------------
        theta_0 = {name: param.data.clone()
                   for name, param in self.visualize_model.named_parameters()}
        num_params = sum(p.numel() for p in self.visualize_model.parameters())
        _lambda_reg_map = {"fine": 1.0, "balanced": 0.5, "coarse": 0.1}
        focus_mode_now = getattr(self, 'ttav_mode', 'coarse')
        lambda_reg = _ab_lambda_reg if _ab_lambda_reg is not None \
                     else _lambda_reg_map.get(focus_mode_now, 0.1)

        def _l2_reg():
            reg = torch.tensor(0., device=self.device)
            for name, param in self.visualize_model.named_parameters():
                reg = reg + torch.sum(torch.square(param - theta_0[name]))
            return reg / num_params 

        # --- 4. Fine-tune with hierarchical weights + L2 constraint ------------
        optimizer = torch.optim.Adam(self.visualize_model.parameters(), lr=0.001)
        self.visualize_model.train()
        for _ in range(5):
            optimizer.zero_grad()
            outputs = self.visualize_model(edge_to, edge_from)
            _, _, loss_local = self.criterion(
                edge_to, edge_from, a_dummy, a_dummy, outputs, weights=edge_weights_t
            )
            l2_reg = _l2_reg()
            loss_total = loss_local + lambda_reg * l2_reg
            loss_total.backward()
            optimizer.step()
            if time.time() - start_time > 0.6:
                break

        # --- 5. Subset-patch projection for current_epoch --------------------
        # Load the baseline full projection (from the standard non-refined dir).
        # Then overwrite ONLY the focus+neighbour rows with freshly computed
        # encoder outputs.  Every other point is untouched → zero global drift.
        baseline_path = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}",
            'epochs', f'epoch_{current_epoch}', 'projection.npy'
        )
        refined_dir = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}_refined",
            'epochs', f'epoch_{current_epoch}'
        )

        # Always seed from the original baseline — never from _refined — to ensure
        # each refine() call is idempotent and cannot accumulate drift over iterations.
        if os.path.exists(baseline_path):
            full_proj = np.load(baseline_path).copy()
        else:
            # No baseline on disk (first-time / DVI per-epoch model not yet saved):
            # fall back to full inference and warn.
            print(f"[TimeVis] WARNING: baseline projection not found at {baseline_path}. "
                  f"Falling back to full encoder inference — this will be slow.")
            self.visualize_model.eval()
            with torch.no_grad():
                full_feat = self.data_provider.get_representation(current_epoch)
                full_proj = self.visualize_model.encoder(
                    torch.from_numpy(full_feat).float().to(self.device)
                ).cpu().numpy()

        # Re-project only the local subset for current_epoch
        self._patch_epoch(current_epoch, all_indices)

        print(f"[TimeVis] Subset-patch refine finished in {time.time() - start_time:.2f}s "
              f"({len(all_indices)} points patched, epoch={current_epoch})")

        # Store all_indices so patch_other_epochs() can reuse them without re-running kNN.
        self._last_refine_indices = all_indices

        # --- 6. Background ablation (non-blocking) ----------------------------
        if os.path.exists(_ablation_cfg_path):
            try:
                with open(_ablation_cfg_path) as _f:
                    _ab_full = json.load(_f)
                if _ab_full.get("ablation_enabled", False):
                    import threading, copy
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

    def _patch_epoch(self, epoch, all_indices):
        """Write a full re-projection for one epoch using the current (refined) model weights."""
        import numpy as np, os, torch

        vis_method = self.config['vis_method']
        vis_id     = self.config['vis_id']
        content_path = self.config['content_path']

        refined_dir = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}_refined",
            'epochs', f'epoch_{epoch}'
        )

        self.visualize_model.eval()
        with torch.no_grad():
            full_feat = self.data_provider.get_representation(epoch)
            full_proj = self.visualize_model.encoder(
                torch.from_numpy(full_feat).float().to(self.device)
            ).cpu().numpy()

        os.makedirs(refined_dir, exist_ok=True)
        np.save(os.path.join(refined_dir, 'projection.npy'), full_proj)

    def patch_other_epochs(self, skip_epoch):
        """Patch all available epochs except skip_epoch using the last refine indices."""
        import threading
        all_indices = getattr(self, '_last_refine_indices', None)
        if not all_indices:
            return
        available_epochs = self.config['available_epochs']
        other_epochs = [e for e in available_epochs if e != skip_epoch]

        def _run():
            for e in other_epochs:
                try:
                    self._patch_epoch(e, all_indices)
                except Exception as ex:
                    print(f"[TimeVis] Background patch epoch {e} failed: {ex}")
            print(f"[TimeVis] Background patch complete for {len(other_epochs)} other epochs.")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

