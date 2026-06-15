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

        # --- 3b. Sample anchor points (global, excluding patch region) ---------
        exclude_set = set(patch_indices)
        candidate_anchors = [i for i in range(N) if i not in exclude_set]
        rng = np.random.default_rng(seed=42)
        n_anchors = min(300, len(candidate_anchors))
        if n_anchors > 0:
            anchor_indices = rng.choice(candidate_anchors, size=n_anchors, replace=False).tolist()
            anchor_feat_np = full_feat[anchor_indices]                # [A, D]
            anchor_z0      = full_proj_baseline[anchor_indices]       # [A, 2]
        else:
            anchor_indices = []
            anchor_feat_np = np.zeros((0, full_feat.shape[1]), dtype=full_feat.dtype)
            anchor_z0 = np.zeros((0, full_proj_baseline.shape[1]), dtype=full_proj_baseline.dtype)

        # --- 3c. Sample random negative examples (global, excluding HD neighbors) --
        k_neg = 10
        n_neg_per_focus = 5 * k_neg
        neg_exclude = set(all_indices) | hd_nbr_set
        neg_candidates = [i for i in range(N) if i not in neg_exclude]
        n_neg_total = min(n_neg_per_focus * len(focus_indices), len(neg_candidates))
        if n_neg_total > 0:
            neg_indices = rng.choice(neg_candidates, size=n_neg_total, replace=False).tolist()
            neg_feat_np = full_feat[neg_indices]                      # [n_neg, D]
        else:
            neg_indices = []
            neg_feat_np = np.zeros((0, full_feat.shape[1]), dtype=full_feat.dtype)

        # --- 3d. Compute adaptive margin m -----------------------------------
        # Use the 90th-percentile of each focus point's top-20 LD distances ×1.5.
        # The 90th-percentile (rather than median) gives a larger clearance so that
        # non-neighbors must be pushed well beyond the local neighborhood boundary.
        # A hard lower bound of 0.3 prevents the margin from collapsing in
        # densely crowded regions, which would cause L_repel and L_attract to
        # fight each other and prevent convergence.
        focus_z0 = full_proj_baseline[focus_indices]                  # [|F|, 2]
        _nbr_dists = []
        for _fi in focus_indices:
            _all_dists = np.linalg.norm(full_proj_baseline - full_proj_baseline[_fi], axis=1)
            _all_dists[_fi] = np.inf
            _nbr_dists.extend(np.sort(_all_dists)[:20].tolist())
        margin_m = float(np.percentile(_nbr_dists, 90)) * 1.5
        margin_m = max(margin_m, 0.3)

        # Convert all feature arrays to tensors once
        focus_feat_t   = torch.from_numpy(full_feat[focus_indices].copy()).float().to(self.device)
        anchor_feat_t  = torch.from_numpy(anchor_feat_np).float().to(self.device)
        anchor_z0_t    = torch.from_numpy(anchor_z0).float().to(self.device)
        neg_feat_t     = torch.from_numpy(neg_feat_np).float().to(self.device)
        # Pre-compute intra-focus pairwise structure targets (computed once, reused every step).
        #
        # Why this works better than a temporal anchor or a simple hinge repulsion:
        #   • Temporal anchor (mu * ||z - z0||²) prevents collapse but also prevents any
        #     movement → NP stays low.  It treats all motion as bad, not just collapse.
        #   • Hinge repulsion on non-HD pairs misses pairs that ARE mutual HD neighbours yet
        #     still collapse (they have no repulsion mask entry).
        #   • Pairwise structure loss ((d_LD(i,j) - α·d_HD(i,j))²) directly encodes the
        #     desired geometry: focus points may compress toward the centre, but their
        #     RELATIVE distances must stay proportional to HD space.  Since every pair has
        #     d_HD > 0, the target d_LD > 0, so co-location is always penalised regardless
        #     of whether the pair is in each other's HD-neighbour set.
        n_focus = len(focus_indices)
        hd_pairwise_scaled_t = None
        if n_focus > 1:
            focus_feats_np = full_feat[focus_indices]                      # [n_focus, D]
            hd_pw = np.linalg.norm(
                focus_feats_np[:, None, :] - focus_feats_np[None, :, :],
                axis=-1,
            )                                                               # [n_focus, n_focus]
            # Scale HD distances so the mean pairwise distance maps to
            # target_radius = 25 % of the local neighbourhood margin.
            # This compresses the focus cluster enough that all HD neighbours
            # rank in the LD top-10, while keeping them visibly separated.
            _mean_hd = float(hd_pw[hd_pw > 0].mean()) if (hd_pw > 0).any() else 1.0
            target_radius = margin_m * 0.25
            alpha_structure = target_radius / (_mean_hd + 1e-8)
            hd_pairwise_scaled_t = torch.from_numpy(
                (hd_pw * alpha_structure).astype(np.float32)
            ).to(self.device)                                               # [n_focus, n_focus]

        # For each focus point, pre-build tensor of its HD neighbor features
        # Shape: list of tensors, each [k_hd, D]
        focus_hd_nbr_feats = []
        for fi in focus_indices:
            nbr_idx = hd_neighbors_all[fi][:10]
            focus_hd_nbr_feats.append(
                torch.from_numpy(full_feat[nbr_idx].copy()).float().to(self.device)
            )

        # --- 4. Fine-tune a dedicated local visualizer copy ---------------------
        # Keep the global visualizer untouched; the copied local visualizer is
        # trained only for the current refine request and used to patch outputs.
        local_visualizer = copy.deepcopy(self.visualize_model).to(self.device)
        encoder_layers = list(local_visualizer.encoder.children())
        trainable_params = []
        linear_count = 0
        for layer in reversed(encoder_layers):
            if isinstance(layer, torch.nn.Linear):
                for p in layer.parameters():
                    p.requires_grad = True
                trainable_params += list(layer.parameters())
                linear_count += 1
                if linear_count >= 2:
                    break
        for _, param in local_visualizer.named_parameters():
            if not param.requires_grad:
                continue
            if not any(param is tp for tp in trainable_params):
                param.requires_grad = False

        # Pre-batch all focus points and their HD neighbors for efficient forward pass.
        # focus_hd_nbr_feats[i] has shape [k_hd, D]; stack into [n_focus*k_hd, D].
        all_nbr_feat_t = torch.cat(focus_hd_nbr_feats, dim=0)  # [n_focus*k_hd, D]
        k_hd_per_focus = [f.shape[0] for f in focus_hd_nbr_feats]

        optimizer = torch.optim.Adam(trainable_params, lr=0.005)
        local_visualizer.train()

        # In crowded regions (small margin) repulsion fights attraction — reduce it.
        # margin >= 0.5: normal repulsion; margin < 0.3 (clamped floor): disable repulsion.
        gamma_repel = float(np.clip((margin_m - 0.3) / 0.2, 0.0, 1.0))
        mu_anchor   = 10.0  # anchor constraint weight
        # Weight for the intra-focus pairwise structure loss.
        # Higher values → stronger structure preservation, harder to collapse,
        # but also harder for the attract loss to compress the cluster.
        # Default 1.5 gives a good balance: cluster compresses ~60 % toward HD
        # neighbours while keeping all pairwise separations visible.
        gamma_structure = float(vis_cfg.get("refine_structure_weight", 1.5))

        full_feat_t = torch.from_numpy(full_feat).float().to(self.device)

        behavior_cfg = resolve_refine_behavior_config(vis_cfg)
        progress_cfg = behavior_cfg["progressive_updates"]
        stopping_cfg = behavior_cfg["stopping"]

        log_every_steps = int(vis_cfg.get("refine_log_every_steps", 50))
        progress_enabled = bool(progress_callback and progress_cfg.get("enabled", False))
        snapshot_every_steps = max(1, int(progress_cfg.get("snapshot_every_steps", 3)))
        sampled_metrics_enabled = bool(progress_callback and progress_cfg.get("enable_sampled_metrics", True))
        sample_metrics_every_steps = max(1, int(progress_cfg.get("sample_metrics_every_steps", 100)))

        enable_max_steps = bool(stopping_cfg.get("enable_max_steps", True))
        max_steps = max(1, int(stopping_cfg.get("max_steps", 20000)))
        enable_time_budget = bool(stopping_cfg.get("enable_time_budget", False))
        time_budget_seconds = stopping_cfg.get("time_budget_seconds", None)
        time_budget_seconds = None if time_budget_seconds in (None, "", False) else float(time_budget_seconds)
        enable_loss_converged = bool(stopping_cfg.get("enable_loss_converged", True))
        stop_priority = list(stopping_cfg.get("priority", ["loss_converged", "time_budget", "max_steps"]))
        safety_loop_cap = max(max_steps, int(stopping_cfg.get("safety_loop_cap", 200000)))

        # Loss-convergence early stop: track L_attract over a sliding window.
        # NP is meaningless as a stopping signal when 2-D cannot faithfully
        # represent high-D neighborhoods (NP stays 0% even while the layout
        # is genuinely improving). L_attract measures whether HD neighbors are
        # still being pulled closer, which is the actual optimisation goal.
        _loss_window  = []   # recent L_attract values
        _WINDOW       = 20   # check over last 20 steps
        _MIN_STEPS    = int(self.config['vis_config'].get('refine_min_steps', 80))
        _REL_TOL      = 5e-4 # stop if (max-min)/max < tol over the window
        _MAX_STEPS    = int(self.config['vis_config'].get('refine_max_steps', 1000))
        _PATIENCE     = int(self.config['vis_config'].get('refine_patience', 120))
        _MIN_DELTA    = float(self.config['vis_config'].get('refine_loss_min_delta', 1e-4))
        _TIME_LIMIT_S = float(self.config['vis_config'].get('refine_time_limit_s', 45.0))
        _best_total_loss = float('inf')
        _stale_steps = 0

        print(f"[TimeVis] refine early-stop config: max_steps={_MAX_STEPS}, min_steps={_MIN_STEPS}, patience={_PATIENCE}, min_delta={_MIN_DELTA}, time_limit_s={_TIME_LIMIT_S}")

        for step in range(_MAX_STEPS):
            optimizer.zero_grad()

            z_focus   = local_visualizer.encoder(focus_feat_t)    # [n_focus, 2]
            z_all_nbr = local_visualizer.encoder(all_nbr_feat_t)  # [n_focus*k_hd, 2]
            z_neg     = local_visualizer.encoder(neg_feat_t) if neg_feat_t.shape[0] > 0 else None
            z_anchors = local_visualizer.encoder(anchor_feat_t) if anchor_feat_t.shape[0] > 0 else None

            l_attract = torch.tensor(0., device=self.device)
            l_repel   = torch.tensor(0., device=self.device)
            nbr_offset = 0

            for idx in range(len(focus_indices)):
                k_i = k_hd_per_focus[idx]
                z_i    = z_focus[idx].unsqueeze(0)               # [1, 2]
                z_nbrs = z_all_nbr[nbr_offset:nbr_offset + k_i] # [k_i, 2]
                nbr_offset += k_i

                # Attract HD neighbors
                l_attract = l_attract + (z_i - z_nbrs).pow(2).sum(dim=1).mean()

                # Repel random negatives (fixed set, no full inference in loop)
                if z_neg is not None and z_neg.shape[0] > 0:
                    dist_repel = (z_i - z_neg).pow(2).sum(dim=1).sqrt()
                    hinge = torch.clamp(margin_m - dist_repel, min=0.0)
                    l_repel = l_repel + hinge.pow(2).mean()

            n_f = max(len(focus_indices), 1)
            l_attract = l_attract / n_f
            l_repel   = l_repel   / n_f

            if z_anchors is not None and z_anchors.shape[0] > 0:
                l_anchor = (z_anchors - anchor_z0_t).pow(2).sum(dim=1).mean()
            else:
                l_anchor = torch.tensor(0., device=self.device)

            # Intra-focus pairwise structure loss.
            # Forces the LD pairwise distances within the focus set to be proportional
            # to their HD pairwise distances (scaled by alpha_structure so the cluster
            # fits inside the local neighbourhood).  This simultaneously:
            #   (a) prevents collapse — every pair has a positive HD distance → positive
            #       LD target, so co-location is penalised for ALL pairs, including
            #       those that are mutual HD neighbours (the gap left by hinge repulsion).
            #   (b) preserves local geometry — the internal topology of the focus cluster
            #       mirrors the HD structure rather than being arbitrarily deformed.
            #   (c) allows alignment — the whole cluster can compress toward centre
            #       because α rescales the targets to fit inside target_radius.
            l_structure = torch.tensor(0., device=self.device)
            if n_focus > 1 and gamma_structure > 0 and hd_pairwise_scaled_t is not None:
                diff_ff = z_focus.unsqueeze(0) - z_focus.unsqueeze(1)        # [n, n, 2]
                dist_ff = diff_ff.pow(2).sum(dim=-1).clamp(min=1e-8).sqrt()  # [n, n]
                l_structure = (dist_ff - hd_pairwise_scaled_t).pow(2).mean()

            loss_total = (l_attract
                          + gamma_repel * l_repel
                          + mu_anchor * l_anchor
                          + gamma_structure * l_structure)
            loss_total.backward()
            optimizer.step()

            current_total_loss = float(loss_total.detach().item())
            if current_total_loss + _MIN_DELTA < _best_total_loss:
                _best_total_loss = current_total_loss
                _stale_steps = 0
            else:
                _stale_steps += 1

            _loss_window.append(l_attract.item())
            if len(_loss_window) > _WINDOW:
                _loss_window.pop(0)

            if step % 50 == 0:
                print(f"[TimeVis] step={step:4d}  L_attract={l_attract.item():.4f}  "
                      f"L_anchor={l_anchor.item():.5f}  "
                      f"L_structure={l_structure.item():.4f}  "
                      f"L_total={current_total_loss:.5f}  stale={_stale_steps}  "
                      f"t={time.time()-start_time:.1f}s")

            if _PATIENCE > 0 and step >= _MIN_STEPS and _stale_steps >= _PATIENCE:
                print(f"[TimeVis] Early stop at step {step}: no total-loss improvement for {_stale_steps} steps (best={_best_total_loss:.6f})")
                break

            # Loss-convergence early stop (only after _MIN_STEPS)
            if step >= _MIN_STEPS and len(_loss_window) == _WINDOW:
                _w_max = max(_loss_window)
                _w_min = min(_loss_window)
                if _w_max > 0 and (_w_max - _w_min) / _w_max < _REL_TOL:
                    print(f"[TimeVis] Loss converged at step {step}: "
                          f"L_attract range={_w_max-_w_min:.6f} < tol")
                    break

            if _TIME_LIMIT_S > 0 and time.time() - start_time > _TIME_LIMIT_S:
                print(f"[TimeVis] Time limit at step {step} ({_TIME_LIMIT_S}s)")
                break

        # --- Compute T, C, NP for focus points using full-dataset distances ------
        # z_eval was computed in the last NP-check iteration inside the loop.
        # Re-compute here to ensure it uses the final model state.
        self.visualize_model.eval()
        with torch.no_grad():
            z_final = local_visualizer.encoder(full_feat_t)  # [N, 2]
        z_np = z_final.cpu().numpy()  # [N, 2]

        trust_sum = 0.0
        cont_sum  = 0.0
        np_sum    = 0.0
        mrh_sum   = 0.0   # Mean Rank of HD neighbors in LD space (lower = better)

        k = 10
        # Extended neighborhood to compute meaningful ranks: top K_ext points
        # gives penalty a real range instead of collapsing near zero.
        K_ext = min(200, N - 1)

        for fi_glob in focus_indices:
            # High-dim ranks (argsort over all N, excluding self)
            hd_feat_fi = full_feat[fi_glob]
            hd_dists = np.linalg.norm(full_feat - hd_feat_fi, axis=1)
            hd_dists[fi_glob] = np.inf
            hd_rank = np.argsort(hd_dists)   # index array: hd_rank[r] = point at rank r

            # Low-dim ranks
            ld_dists = np.linalg.norm(z_np - z_np[fi_glob], axis=1)
            ld_dists[fi_glob] = np.inf
            ld_rank = np.argsort(ld_dists)

            # Build rank lookup: point_idx → 1-based rank (only within K_ext)
            # Used for T/C penalty computation (capped at K_ext for normalisation).
            hd_rank_of = {int(hd_rank[r]): r + 1 for r in range(K_ext)}
            ld_rank_of = {int(ld_rank[r]): r + 1 for r in range(K_ext)}

            # Full LD rank lookup for MRH — must cover all N points so HD
            # neighbors that landed far away in 2D get their true rank, not 201.
            ld_rank_full = {int(ld_rank[r]): r + 1 for r in range(N - 1)}

            hd_topk = set(hd_rank[:k].tolist())
            ld_topk = set(ld_rank[:k].tolist())

            # NP: strict top-k set intersection
            np_sum += len(hd_topk & ld_topk) / k

            # MRH: average LD rank of the HD top-k neighbors (full-range lookup).
            # Ideal ≈ 5.5 (perfectly centred in top-10).
            # Crowded-but-good: MRH ≈ 10–30 (just outside top-10 due to density).
            # Truly bad projection: MRH >> 100.
            mrh_sum += float(np.mean([
                ld_rank_full[j] for j in hd_topk
            ]))

            # Trustworthiness: penalise fake low-D neighbors (in LD but not HD top-k)
            # Penalty = HD rank - k; if HD rank > K_ext, use K_ext as conservative bound.
            t_penalty = 0.0
            for j in (ld_topk - hd_topk):
                r_hd = hd_rank_of.get(j, K_ext + 1)
                t_penalty += max(0, r_hd - k)

            # Continuity: penalise missing HD neighbors (in HD but not LD top-k)
            c_penalty = 0.0
            for j in (hd_topk - ld_topk):
                r_ld = ld_rank_of.get(j, K_ext + 1)
                c_penalty += max(0, r_ld - k)

            # Normalizer: worst-case penalty for one point with k fake neighbors,
            # each at rank K_ext.  This gives T/C a meaningful [0,1] range
            # for a single focus point rather than the global N-point formula.
            worst = k * (K_ext - k)
            trust_sum += 1.0 - t_penalty / worst if worst > 0 else 1.0
            cont_sum  += 1.0 - c_penalty / worst if worst > 0 else 1.0

        n_f = max(len(focus_indices), 1)
        final_np    = np_sum    / n_f * 100.0
        final_mrh   = mrh_sum   / n_f          # raw rank number, not a percentage
        final_trust = max(0.0, trust_sum / n_f * 100.0)
        final_cont  = max(0.0, cont_sum  / n_f * 100.0)

        self._last_refine_np    = final_np
        self._last_refine_mrh   = final_mrh
        self._last_refine_trust = final_trust
        self._last_refine_cont  = final_cont

        print(f"[TimeVis] Anchor-constrained refine done: "
              f"steps={step+1}  NP={final_np:.1f}%  MRH={final_mrh:.1f}  "
              f"T={final_trust:.1f}%  C={final_cont:.1f}%  "
              f"margin={margin_m:.3f}  t={time.time()-start_time:.1f}s")

        # --- 5. Full-inference projection for current_epoch --------------------
        # Patch using the trained local visualizer while keeping the global
        # visualizer untouched for future sessions.
        self._patch_epoch(current_epoch, all_indices, model=local_visualizer)
        self._last_local_visualizer = local_visualizer

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
        """Patch all available epochs except skip_epoch using the last local visualizer."""
        import threading
        all_indices = getattr(self, '_last_refine_indices', None)
        local_visualizer = getattr(self, '_last_local_visualizer', None)
        if not all_indices or local_visualizer is None:
            return
        available_epochs = self.config['available_epochs']
        other_epochs = [e for e in available_epochs if e != skip_epoch]

        def _run():
            for e in other_epochs:
                try:
                    self._patch_epoch(e, all_indices, model=local_visualizer)
                except Exception as ex:
                    print(f"[TimeVis] Background patch epoch {e} failed: {ex}")
            print(f"[TimeVis] Background patch complete for {len(other_epochs)} other epochs.")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

