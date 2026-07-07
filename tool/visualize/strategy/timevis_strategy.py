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

def _compute_focus_metrics(snap_z, focus_indices, hd_cache, N, k, K_ext):
    """Compute NP, MRH, Trustworthiness, Continuity for the given 2D snapshot.

    hd_cache: dict mapping focus_index → {'topk': set[int], 'rank_of': dict[int,int]}
    Reuses precomputed HD rankings so only LD distances are computed here (fast).
    """
    np_sum = mrh_sum = trust_sum = cont_sum = 0.0
    for fi in focus_indices:
        ld_dists = np.linalg.norm(snap_z - snap_z[fi], axis=1)
        ld_dists[fi] = np.inf
        ld_rank = np.argsort(ld_dists)

        ld_rank_of   = {int(ld_rank[r]): r + 1 for r in range(K_ext)}
        ld_rank_full = {int(ld_rank[r]): r + 1 for r in range(N - 1)}

        hd_topk    = hd_cache[fi]['topk']
        hd_rank_of = hd_cache[fi]['rank_of']
        ld_topk    = set(int(ld_rank[r]) for r in range(k))

        np_sum += len(hd_topk & ld_topk) / max(k, 1)

        mrh_vals = [ld_rank_full.get(j, N) for j in hd_topk]
        mrh_sum += float(np.mean(mrh_vals)) if mrh_vals else float(N)

        t_penalty = sum(max(0, hd_rank_of.get(j, K_ext + 1) - k) for j in (ld_topk - hd_topk))
        c_penalty = sum(max(0, ld_rank_of.get(j,  K_ext + 1) - k) for j in (hd_topk - ld_topk))
        worst = k * (K_ext - k)
        trust_sum += 1.0 - t_penalty / worst if worst > 0 else 1.0
        cont_sum  += 1.0 - c_penalty / worst if worst > 0 else 1.0

    n_f = max(len(focus_indices), 1)
    return {
        "neighbor_preservation": np_sum  / n_f * 100.0,
        "mean_rank_hd":          mrh_sum / n_f,
        "trustworthiness":  max(0.0, trust_sum / n_f * 100.0),
        "continuity":       max(0.0, cont_sum  / n_f * 100.0),
    }


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

    def _get_refine_epoch_data(self, epoch, k=10):
        """A1: cache the per-epoch immutable refine inputs — features, baseline
        projection, HD-neighbor lists — keyed by source-file mtimes (and k). Only
        the most recent epoch is retained (bounded memory). The cache rebuilds
        automatically when any underlying file changes (e.g. a retrain overwrites
        embeddings.npy / projection.npy) or when k changes.

        k (C3): number of HD neighbors to load per point — reads the parametric
        hd_neighbors_{k}.json cache, falling back to an on-the-fly NN computation.

        Returns (full_feat, full_proj_baseline_or_None, hd_neighbors_all, cache_hit).
        full_proj_baseline is None when projection.npy is absent — the caller
        then falls back to model inference (model-dependent, not cacheable here).
        """
        import os
        import json as _json
        import numpy as np

        vis_method   = self.config['vis_method']
        vis_id       = self.config['vis_id']
        content_path = self.config['content_path']

        emb_path  = os.path.join(content_path, 'epochs', f'epoch_{epoch}', 'embeddings.npy')
        proj_path = os.path.join(content_path, 'visualize', f"{vis_method}_{vis_id}",
                                 'epochs', f'epoch_{epoch}', 'projection.npy')
        hd_path   = os.path.join(content_path, 'epochs', f'epoch_{epoch}', f'hd_neighbors_{k}.json')

        def _mtime(p):
            try:
                return os.path.getmtime(p)
            except OSError:
                return None

        key = (epoch, k, _mtime(emb_path), _mtime(proj_path), _mtime(hd_path))
        cache = getattr(self, '_refine_epoch_cache', None)
        if cache is not None and cache.get('key') == key:
            return (cache['full_feat'], cache['full_proj_baseline'],
                    cache['hd_neighbors_all'], True)

        # (Re)build — one disk read per immutable input.
        full_feat = self.data_provider.get_representation(epoch)  # reads embeddings.npy
        N = len(full_feat)

        if os.path.exists(proj_path):
            full_proj_baseline = np.load(proj_path).copy()  # [N, 2]
        else:
            full_proj_baseline = None  # caller does model-inference fallback

        if os.path.exists(hd_path):
            with open(hd_path) as _f:
                hd_neighbors_all = _json.load(_f)  # list[list[int]], length N, k each
        else:
            from sklearn.neighbors import NearestNeighbors as _NNS
            _nbrs = _NNS(n_neighbors=min(k + 1, N), algorithm='auto').fit(full_feat)
            _, _nn_idx = _nbrs.kneighbors(full_feat)
            hd_neighbors_all = [_nn_idx[i, 1:].tolist() for i in range(N)]

        # Retain only this epoch (bounded memory).
        self._refine_epoch_cache = {
            'key':                key,
            'full_feat':          full_feat,
            'full_proj_baseline': full_proj_baseline,
            'hd_neighbors_all':   hd_neighbors_all,
        }
        return full_feat, full_proj_baseline, hd_neighbors_all, False

    def refine(self, focus_indices=None, focus_index=None, neighbor_indices=None,
               current_epoch=None, epochs_to_update=10, _skip_avg_benchmark=False,
               progress_callback=None, should_stop_callback=None,
               progress_refresh_indices=None,
               secondary_indices=None, top_k=None, priority=None):
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

        print(f"[TimeVis refine ENTRY] focus_indices count={len(focus_indices)}, sample={focus_indices[:10]}")
        start_time = time.time()
        vis_method = self.config['vis_method']
        vis_id    = self.config['vis_id']
        content_path = self.config['content_path']
        available_epochs = self.config['available_epochs']
        vis_cfg = self.config['vis_config']

        # C3: neighbourhood size to preserve (HD top-k == LD top-k). Per-refine
        # override wins; otherwise session config; clamped to [3, 20].
        _k = int(top_k if top_k is not None else vis_cfg.get('refine_top_k', 10))
        _k = max(3, min(20, _k))

        # B1: accuracy↔layout tradeoff. t=0.5 reproduces the current defaults
        # exactly; t→1 prioritises neighbor accuracy (strong ranking pull, weak
        # anchor, full escalation), t→0 prioritises preserving the layout (weak
        # ranking, strong anchor, escalation relaxed — may accept <100%).
        _priority = float(priority if priority is not None else vis_cfg.get('refine_priority', 0.5))
        _priority = max(0.0, min(1.0, _priority))
        print(f"[TimeVis refine] top_k = {_k}, priority = {_priority:.2f}")

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

        # --- 1. Always reload model from checkpoint at start of each refine -----
        # Refine modifies encoder weights in-place. If a previous refine caused
        # collapse, the next refine would start from that corrupted state while
        # full_proj_baseline is loaded from the original projection.npy — the
        # mismatch makes anchor loss huge at step 0 and prevents learning.
        model_path = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}", 'vis_model.pth'
        )
        _ckpt_hit = False
        if os.path.exists(model_path):
            # A1: cache the checkpoint state_dict in memory keyed by (path, mtime)
            # so repeated refines skip the disk read. load_state_dict still runs
            # every time — the "reset to checkpoint" behaviour is unchanged.
            _ck_mtime = os.path.getmtime(model_path)
            _ck_cache = getattr(self, '_refine_ckpt_cache', None)
            if _ck_cache is not None and _ck_cache['path'] == model_path \
                    and _ck_cache['mtime'] == _ck_mtime:
                _state = _ck_cache['state_dict']
                _ckpt_hit = True
            else:
                _state = torch.load(model_path, map_location=self.device)['state_dict']
                self._refine_ckpt_cache = {
                    'path': model_path, 'mtime': _ck_mtime, 'state_dict': _state}
            self.visualize_model.load_state_dict(_state)
            self.visualize_model.to(self.device)
            print(f"[TimeVis refine] Reloaded model from checkpoint (ckpt_cache_hit={_ckpt_hit})")
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

        # --- 3. Gather full-dataset features, baseline projection, HD neighbors --
        # A1: per-epoch immutable inputs come from an mtime-keyed cache so
        # repeated refines on the same epoch skip all three disk reads.
        full_feat, full_proj_baseline, hd_neighbors_all, _epoch_hit = \
            self._get_refine_epoch_data(current_epoch, k=_k)
        N = len(full_feat)

        if full_proj_baseline is None:
            # projection.npy absent → model-inference fallback (not cacheable:
            # depends on current encoder weights).
            self.visualize_model.eval()
            with torch.no_grad():
                full_proj_baseline = self.visualize_model.encoder(
                    torch.from_numpy(full_feat).float().to(self.device)
                ).cpu().numpy()

        print(f"[TimeVis] refine setup: {time.time()-start_time:.2f}s "
              f"(epoch_cache_hit={_epoch_hit}, ckpt_cache_hit={_ckpt_hit})")

        # Collect all high-D neighbor indices referenced by focus points
        hd_nbr_set = set()
        for fi in focus_indices:
            hd_nbr_set.update(hd_neighbors_all[fi][:_k])
        hd_nbr_set -= set(focus_indices)

        # Secondary indices expand the training context but are NOT attract targets.
        sec_set = set()
        if secondary_indices:
            sec_set = {int(i) for i in secondary_indices} - set(focus_indices)
            sec_hd_set = set()
            for si in sec_set:
                if si < len(hd_neighbors_all):
                    sec_hd_set.update(hd_neighbors_all[si][:_k])
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

        # --- 4. Encoder fine-tuning with three-component loss ---------------------
        #
        # All points can move through the shared encoder, enabling a locally
        # accurate neighbourhood while the global layout adapts smoothly.
        #
        # Three loss terms:
        #   A. Focus attraction  (high weight): correct UMAP CE loss on
        #      (focus, HD_neighbor) positive pairs; negatives sampled from
        #      background non-HD-neighbor points — fixes the old coord-opt bug
        #      where negatives were drawn from HD-neighbor positions.
        #   B. Anchor regularisation (moderate): L2 penalty on how far each
        #      point drifts from its baseline projection, weighted by distance
        #      from the focus centre — focus area is free to move; distant
        #      points are held near baseline preventing global collapse.
        #   C. Global edge maintenance (light): standard UMAP loss on a random
        #      batch from the full pre-built edge graph — keeps topology outside
        #      the focus region stable.

        # Save model weights; restore on error so the server stays usable.
        _model_backup = copy.deepcopy(self.visualize_model.state_dict())

        behavior_cfg = resolve_refine_behavior_config(vis_cfg)
        _progress_cfg = behavior_cfg["progressive_updates"]
        _snapshot_every = max(1, int(_progress_cfg.get("snapshot_every_steps", 30)))

        # Precompute HD rank lookup (shared by accuracy checks, snapshots, final)
        _k_metrics = min(_k, N - 1)
        _K_ext     = min(200, N - 1)
        _hd_cache: dict = {}
        for _fi in focus_indices:
            _hd_dists = np.linalg.norm(full_feat - full_feat[_fi], axis=1)
            _hd_dists[_fi] = np.inf
            _hd_rank = np.argsort(_hd_dists)
            _hd_cache[_fi] = {
                'topk':    set(int(_hd_rank[r]) for r in range(_k_metrics)),
                'rank_of': {int(_hd_rank[r]): r + 1 for r in range(_K_ext)},
            }

        # --- 4a. Pre-build tensors used every step ----------------------------

        full_feat_t      = torch.from_numpy(full_feat).float().to(self.device)       # [N, D]
        baseline_proj_t  = torch.from_numpy(full_proj_baseline).float().to(self.device)  # [N, 2]

        # HD neighbor sets per focus point — the top-k targets we want to realise
        _hd_nbr_dict = {fi: set(hd_neighbors_all[fi][:_k]) for fi in focus_indices}

        # Anchor weights: distance-based for background; explicit values for
        # focus/HD to balance against the attraction loss.
        # Focus points need some freedom to move, but HD neighbors must resist
        # being pulled all the way into the focus cluster.
        _focus_center = full_proj_baseline[np.array(focus_indices)].mean(axis=0)
        _d_from_focus = np.linalg.norm(full_proj_baseline - _focus_center, axis=1)
        _d_ref        = max(float(np.percentile(_d_from_focus, 80)), 1e-6)
        anchor_w_np   = np.clip(_d_from_focus / _d_ref, 0.0, 1.0).astype(np.float32)
        for _fi in focus_indices:
            anchor_w_np[_fi] = 0.20   # can move toward HD neighbors
        for _hi in hd_nbr_set:
            anchor_w_np[_hi] = min(anchor_w_np[_hi], 0.35)  # pulled toward focus but not collapsed
        anchor_w_t = torch.from_numpy(anchor_w_np).to(self.device)  # [N]

        # Anti-flight pin targets: baseline positions of the focus points. NP is
        # translation-invariant, so without this the optimizer reaches 100% by
        # moving the focus into empty space (see _W_PIN below).
        _focus_idx_t      = torch.from_numpy(np.array(list(focus_indices), dtype=np.int64)).to(self.device)
        _baseline_focus_t = baseline_proj_t[_focus_idx_t]   # [F, 2]

        # --- Local-MDS precompute (Loss L_shape) ------------------------------
        # Cluster = focus ∪ HD top-k neighbors ∪ local LD context. Including the
        # focus's baseline LD-nearest points (which contain the current
        # "impostors") lets metric MDS place them at their TRUE, larger HD
        # distance — so they settle just outside the true neighbors NATURALLY,
        # instead of L_rank flinging them out and tearing an artificial void.
        # We preserve the RELATIVE high-D geometry of this cluster in 2D, scaled
        # to the baseline LD size so the patch neither collapses nor explodes.
        _K_CTX = max(3 * _k, 30)
        _ld_ctx = set()
        for _fi in focus_indices:
            _dld = np.linalg.norm(full_proj_baseline - full_proj_baseline[_fi], axis=1)
            _dld[_fi] = np.inf
            _ld_ctx.update(int(x) for x in np.argsort(_dld)[:min(_K_CTX, N - 1)])
        _cluster_arr = np.array(sorted(set(focus_indices) | hd_nbr_set | _ld_ctx), dtype=np.int64)
        _M = len(_cluster_arr)
        # Positions of focus points within the cluster (focus-centric L_shape sampling)
        _cl_pos = {int(g): i for i, g in enumerate(_cluster_arr)}
        _focus_pos_in_cl = np.array([_cl_pos[int(f)] for f in focus_indices], dtype=np.int64)
        _cl_feat_t = full_feat_t[_cluster_arr]                       # [M, D]
        # Pairwise HD distances within the cluster (defines target ratios)
        _cl_feat_np = full_feat[_cluster_arr]                        # [M, D]
        _hd_pdist_np = np.linalg.norm(
            _cl_feat_np[:, None, :] - _cl_feat_np[None, :, :], axis=2)  # [M, M]
        # Pairwise baseline LD distances within the cluster (defines target scale)
        _cl_base_np = full_proj_baseline[_cluster_arr]              # [M, 2]
        _ld_base_pdist_np = np.linalg.norm(
            _cl_base_np[:, None, :] - _cl_base_np[None, :, :], axis=2)  # [M, M]
        # Scale s: map HD distances → LD units so overall spread ≈ baseline.
        _iu = np.triu_indices(_M, k=1) if _M > 1 else (np.array([], int), np.array([], int))
        _med_hd = float(np.median(_hd_pdist_np[_iu])) if len(_iu[0]) else 1.0
        _med_ld = float(np.median(_ld_base_pdist_np[_iu])) if len(_iu[0]) else 1.0
        _s_shape = (_med_ld / _med_hd) if _med_hd > 1e-9 else 1.0
        # Local LD scale — used to set the ranking margin in coordinate units.
        _local_scale = max(_med_ld, 1e-6)
        _hd_pdist_t = torch.from_numpy(
            (_s_shape * _hd_pdist_np).astype(np.float32)).to(self.device)  # target LD dists [M,M]
        print(f"[TimeVis refine] local-MDS: M={_M} s_shape={_s_shape:.4f} "
              f"local_scale={_local_scale:.4f}")

        # Soft bearing preservation (L_dir): baseline UNIT direction of each
        # cluster member from the focus center. L_dir penalises change of bearing
        # only (radius is free), breaking the MDS rotation/reflection ambiguity
        # toward the baseline so neighbors slide IN and impostors slide OUT
        # without flipping to the opposite side — no rigid radial pinning.
        _focus_center_base = full_proj_baseline[np.array(focus_indices)].mean(axis=0)  # [2]
        _cl_vec_base = full_proj_baseline[_cluster_arr] - _focus_center_base           # [M, 2]
        _cl_r_base   = np.linalg.norm(_cl_vec_base, axis=1)                            # [M]
        _dir_valid   = _cl_r_base > (0.05 * _local_scale)   # skip members ~at center
        _cl_udir_base = np.zeros_like(_cl_vec_base)
        _cl_udir_base[_dir_valid] = _cl_vec_base[_dir_valid] / _cl_r_base[_dir_valid, None]
        _cl_udir_base_t = torch.from_numpy(_cl_udir_base.astype(np.float32)).to(self.device)  # [M,2]
        _dir_valid_t    = torch.from_numpy(_dir_valid.astype(np.float32)).to(self.device)      # [M]

        # Ranking triplets (focus, hd_neighbor, impostor) — rebuilt every ACC_CHECK.
        # Empty initially; populated at the first accuracy check.
        _tri_f = np.array([], dtype=np.int64)
        _tri_h = np.array([], dtype=np.int64)
        _tri_m = np.array([], dtype=np.int64)

        # UMAP a/b params (reuse from self.umap_fn)
        _a = float(self.umap_fn.a)
        _b = float(self.umap_fn.b)

        # Global edge arrays (if available from training-time edge graph)
        _has_global = hasattr(self, 'data_handler')
        if _has_global:
            _g_edge_to   = np.asarray(self.data_handler.edge_to,   dtype=np.int64)
            _g_edge_from = np.asarray(self.data_handler.edge_from, dtype=np.int64)
            _g_n_edges   = len(_g_edge_to)
        else:
            _g_n_edges = 0

        # Hyperparameters. _W_RANK / _LAMBDA_ANCHOR are scaled by the B1 priority
        # (t=0.5 → 5.0 / 1.5, i.e. the original balanced defaults).
        _prio_rank_mult   = _priority / 0.5           # t=0→0, 0.5→1, 1→2
        _prio_anchor_mult = (1.0 - _priority) / 0.5   # t=0→2, 0.5→1, 1→0
        _W_SHAPE       = 1.0                    # local-MDS shape-preservation weight (main local driver)
        _W_RANK        = 5.0 * _prio_rank_mult  # margin-ranking weight (boundary top-k correction)
        _W_PIN         = 20.0                   # anti-flight focus-position pin (scaled by rank_scale)
        _W_DIR         = 3.0                    # soft bearing preservation (anti-flip, scaled by rank_scale)
        _RANK_MARGIN   = 0.15 * _local_scale    # ordering buffer, in LD coordinate units
        _LAMBDA_ANCHOR = 1.5 * _prio_anchor_mult  # anchor regularisation weight (global tether)
        _GAMMA_GLOBAL  = 0.3    # global edge maintenance weight
        _SHAPE_BATCH   = 256    # cluster pairs sampled for L_shape each step
        _ANCHOR_SAMPLE = 400    # points sampled for anchor loss each step
        _GLOBAL_BATCH  = 256    # global edges per step
        _ACC_CHECK_EVERY = 50   # steps between accuracy evaluations
        _MIN_STEPS     = int(vis_cfg.get('refine_min_steps', 100))
        _MAX_STEPS     = int(vis_cfg.get('refine_max_steps', 10000))
        _NO_IMPROVE_PAT = 1000  # multi-focus: steps without accuracy gain → stop

        # --- Single-focus guarantee: penalty escalation -----------------------
        # For a single focus point, HD top-10 == LD top-10 is (almost) always
        # geometrically achievable in 2D. To GUARANTEE it, we never quit on a
        # plateau — instead we escalate the ranking weight until every impostor
        # is pushed out, and only stop when triplets==0 (100%) or hard caps hit.
        _single_focus    = len(focus_indices) <= 1

        # Wall-clock budget: single-focus gets a generous cap so escalation can
        # actually reach 100%; multi-focus stays tight (may never converge).
        _stop_cfg      = behavior_cfg["stopping"]
        _TIME_LIMIT_S  = float(_stop_cfg["time_limit_single_seconds"]) if _single_focus \
                         else float(_stop_cfg["time_limit_seconds"])

        _rank_scale      = 1.0     # current L_rank multiplier (grows on stall)
        _ESCALATE_PAT    = 200     # single-focus: stall steps before escalating
        _RANK_ESCALATE   = 1.5     # multiply factor per escalation
        # B1: escalation cap follows priority — full (200×) at balanced/accuracy,
        # relaxed below balanced so "preserve layout" won't force 100% at all costs.
        _RANK_SCALE_MAX  = max(1.0, 200.0 * min(1.0, _priority / 0.5))

        optimizer = torch.optim.Adam(self.visualize_model.parameters(), lr=0.001)

        _best_acc    = 0.0
        _stale_since = 0
        _stop_reason = "max_steps"   # D2: overwritten at whichever break fires

        print(f"[TimeVis] encoder-finetune refine: n_focus={len(focus_indices)}, "
              f"n_hd_nbrs={len(hd_nbr_set)}, N={N}, cluster_M={_M}, "
              f"max_steps={_MAX_STEPS}")

        try:
            for step in range(_MAX_STEPS):
                self.visualize_model.train()
                optimizer.zero_grad()

                # === L_shape. Local-MDS shape preservation =====================
                # Preserve the RELATIVE high-D geometry of the cluster in 2D:
                # target LD distance for pair (i,j) is s * d_hd(i,j).  Because the
                # targets are nonzero and vary with true HD distance, the cluster
                # keeps its intrinsic spread (dimensionality) and CANNOT collapse
                # to a point — near HD pairs stay near, far HD pairs stay far.
                # One encoder pass over the whole cluster, reused for every pair.
                l_shape = torch.tensor(0.0, device=self.device)
                if _M > 1:
                    emb_cl = self.visualize_model.encoder(_cl_feat_t)      # [M, 2]
                    # Focus-centric pairs (dominant for NP): every focus → every
                    # cluster member. These distances decide which points end up
                    # as the focus's nearest neighbors, so they must not be diluted
                    # by the now-larger cluster. Always include them in full.
                    _fa = np.repeat(_focus_pos_in_cl, _M)
                    _fb = np.tile(np.arange(_M), len(_focus_pos_in_cl))
                    # Plus random intra-cluster pairs for overall shape context.
                    _ra = np.random.randint(0, _M, _SHAPE_BATCH)
                    _rb = np.random.randint(0, _M, _SHAPE_BATCH)
                    _si_a = np.concatenate([_fa, _ra])
                    _si_b = np.concatenate([_fb, _rb])
                    _sm   = _si_a == _si_b
                    _si_b[_sm] = (_si_b[_sm] + 1) % _M
                    _ta = torch.from_numpy(_si_a).to(self.device)
                    _tb = torch.from_numpy(_si_b).to(self.device)
                    d_ld_pair  = torch.norm(emb_cl[_ta] - emb_cl[_tb], dim=1)
                    d_tgt_pair = _hd_pdist_t[_ta, _tb]
                    l_shape = (d_ld_pair - d_tgt_pair).pow(2).mean()

                    # L_dir: soft bearing preservation (radius free, angle kept).
                    _emb_center = emb_cl[_focus_pos_in_cl].mean(dim=0)          # [2]
                    _v_now = emb_cl - _emb_center                              # [M, 2]
                    _u_now = _v_now / (_v_now.norm(dim=1, keepdim=True) + 1e-6)
                    _cos   = (_u_now * _cl_udir_base_t).sum(dim=1)             # [M]
                    l_dir  = ((1.0 - _cos) * _dir_valid_t).sum() / (_dir_valid_t.sum() + 1e-6)
                else:
                    l_dir = torch.tensor(0.0, device=self.device)

                # === L_rank. Boundary margin ranking ===========================
                # For triplets (focus f, HD neighbor h, impostor m): require h to
                # be closer to f than m by a margin.  This ONE hinge does both
                # jobs — pull true neighbors in AND push impostors out — but only
                # relatively, so it goes silent once h ranks ahead of m (no
                # collapse).  Multiple focus points whose neighborhoods conflict
                # simply reach a hinge equilibrium → naturally lenient.
                n_tri = len(_tri_f)
                if n_tri > 0:
                    emb_tf = self.visualize_model.encoder(full_feat_t[_tri_f])
                    emb_th = self.visualize_model.encoder(full_feat_t[_tri_h])
                    emb_tm = self.visualize_model.encoder(full_feat_t[_tri_m])
                    d_fh = torch.norm(emb_tf - emb_th, dim=1)
                    d_fm = torch.norm(emb_tf - emb_tm, dim=1)
                    l_rank = torch.clamp(d_fh - d_fm + _RANK_MARGIN, min=0.0).mean()
                else:
                    l_rank = torch.tensor(0.0, device=self.device)

                # === Focus-position pin (anti-flight) ==========================
                # NP is translation-invariant, so the optimizer can reach 100% by
                # relocating the focus into empty space and dragging its HD
                # neighbors along — abandoning the surrounding structure. Pin the
                # focus to its baseline location so the neighborhood reorganizes
                # AROUND it, in context. Scaled by rank_scale so escalation can
                # never buy accuracy through translation (equilibrium focus
                # displacement stays bounded regardless of escalation level).
                emb_focus_pin = self.visualize_model.encoder(full_feat_t[_focus_idx_t])
                l_pin = (emb_focus_pin - _baseline_focus_t).pow(2).sum(dim=1).mean()

                # === B. Anchor regularisation ===================================
                _anc_idx  = np.random.choice(N, min(_ANCHOR_SAMPLE, N), replace=False)
                _anc_feat = full_feat_t[_anc_idx]
                _anc_emb  = self.visualize_model.encoder(_anc_feat)                   # [S, 2]
                _anc_tgt  = baseline_proj_t[_anc_idx]                                 # [S, 2]
                _anc_w    = anchor_w_t[_anc_idx]                                      # [S]
                l_anchor  = (_anc_w * (_anc_emb - _anc_tgt).pow(2).sum(dim=1)).mean()

                # === C. Global edge maintenance =================================
                l_global = torch.tensor(0.0, device=self.device)
                if _has_global and _g_n_edges > 0:
                    _gb = min(_GLOBAL_BATCH, _g_n_edges)
                    _gi = np.random.choice(_g_n_edges, _gb, replace=False)
                    _g_emb_to   = self.visualize_model.encoder(full_feat_t[_g_edge_to[_gi]])
                    _g_emb_from = self.visualize_model.encoder(full_feat_t[_g_edge_from[_gi]])
                    l_global = self.umap_fn(_g_emb_to, _g_emb_from)

                # === Total loss =================================================
                # L_shape governs local geometry/dimensionality; L_rank fixes the
                # residual top-10 ordering at the boundary; anchor tethers the far
                # field; global keeps outside topology stable.
                loss = (_W_SHAPE * l_shape + _W_RANK * _rank_scale * l_rank
                        + _W_PIN * _rank_scale * l_pin
                        + _W_DIR * _rank_scale * l_dir
                        + _LAMBDA_ANCHOR * l_anchor + _GAMMA_GLOBAL * l_global)
                loss.backward()
                optimizer.step()

                if step % 50 == 0:
                    print(f"[TimeVis] step={step:4d}  "
                          f"L={float(loss):.4f}  "
                          f"shape={float(l_shape):.4f}({_M}pts)  "
                          f"rank={float(l_rank):.4f}({n_tri}tri)  "
                          f"pin={float(l_pin):.4f}  "
                          f"dir={float(l_dir):.4f}  "
                          f"anchor={float(l_anchor):.4f}  "
                          f"t={time.time()-start_time:.1f}s")

                # === Neighbour accuracy check + dynamic pair update ===========
                if (step + 1) % _ACC_CHECK_EVERY == 0:
                    self.visualize_model.eval()
                    with torch.no_grad():
                        _snap_z_np = self.visualize_model.encoder(full_feat_t).cpu().numpy()

                    _all_100   = True
                    _total_acc = 0.0
                    _new_tri_f, _new_tri_h, _new_tri_m = [], [], []

                    for _fi in focus_indices:
                        _ld_d = np.linalg.norm(_snap_z_np - _snap_z_np[_fi], axis=1)
                        _ld_d[_fi] = np.inf
                        _ld_top10 = set(int(x) for x in np.argsort(_ld_d)[:_k])
                        _hd_top10 = _hd_nbr_dict[_fi]
                        _acc = len(_hd_top10 & _ld_top10) / float(_k)
                        _total_acc += _acc
                        if _acc < 1.0:
                            _all_100 = False

                        # Build ranking triplets: each unsatisfied HD neighbor h
                        # (HD top-10 but not yet LD top-10) must beat each impostor
                        # m (LD top-10 but not HD).  h should end up closer than m.
                        _unsat = [j for j in _hd_top10 if j not in _ld_top10]
                        _imps  = [j for j in _ld_top10 if j not in _hd_top10]
                        for h in _unsat:
                            for m in _imps:
                                _new_tri_f.append(_fi)
                                _new_tri_h.append(h)
                                _new_tri_m.append(m)

                    # Update dynamic triplet arrays (cap to bound per-step cost)
                    _TRI_CAP = 4096
                    if len(_new_tri_f) > _TRI_CAP:
                        _sel = np.random.choice(len(_new_tri_f), _TRI_CAP, replace=False)
                        _tri_f = np.array(_new_tri_f, dtype=np.int64)[_sel]
                        _tri_h = np.array(_new_tri_h, dtype=np.int64)[_sel]
                        _tri_m = np.array(_new_tri_m, dtype=np.int64)[_sel]
                    else:
                        _tri_f = np.array(_new_tri_f, dtype=np.int64)
                        _tri_h = np.array(_new_tri_h, dtype=np.int64)
                        _tri_m = np.array(_new_tri_m, dtype=np.int64)
                    n_tri = len(_tri_f)

                    _avg_acc = _total_acc / len(focus_indices)
                    if _avg_acc > _best_acc:
                        _best_acc    = _avg_acc
                        _stale_since = step

                    print(f"[TimeVis] step={step+1:4d}  NP_acc={_avg_acc*100:.1f}%  "
                          f"best={_best_acc*100:.1f}%  triplets={n_tri}  "
                          f"rank_scale={_rank_scale:.1f}")

                    if n_tri == 0 and step >= _MIN_STEPS:
                        # No unsatisfied HD neighbor / impostor pairs remain — every
                        # top-10 ranking constraint is satisfied.
                        print(f"[TimeVis] All ranking constraints satisfied at step {step+1} — stopping")
                        _stop_reason = "converged"
                        break

                    if _all_100 and step >= _MIN_STEPS:
                        print(f"[TimeVis] 100% neighbour accuracy at step {step+1} — stopping")
                        _stop_reason = "converged"
                        break

                    # --- Stall handling -------------------------------------------
                    _stall = step - _stale_since
                    if _single_focus:
                        # Never give up on a plateau: escalate the ranking penalty
                        # until the last impostors are pushed out. Only bail if the
                        # weight cap is reached and it STILL can't (rare geometric
                        # obstruction) — then keep the best result we have.
                        if step >= _MIN_STEPS and _stall >= _ESCALATE_PAT:
                            if _rank_scale < _RANK_SCALE_MAX:
                                _rank_scale = min(_rank_scale * _RANK_ESCALATE, _RANK_SCALE_MAX)
                                _stale_since = step   # reset stall window after escalation
                                print(f"[TimeVis] single-focus stalled at NP={_avg_acc*100:.1f}% "
                                      f"({n_tri} impostors) — escalating rank weight → "
                                      f"×{_rank_scale:.1f}")
                            else:
                                print(f"[TimeVis] rank weight capped at ×{_RANK_SCALE_MAX:.0f} "
                                      f"and still {n_tri} impostors — stopping "
                                      f"(best NP={_best_acc*100:.1f}%)")
                                _stop_reason = "rank_cap"
                                break
                    else:
                        # Multi-focus: neighborhoods conflict, 100% may be impossible.
                        # Best-effort — stop once accuracy plateaus.
                        if step >= _MIN_STEPS and _stall >= _NO_IMPROVE_PAT:
                            print(f"[TimeVis] No accuracy gain for {_NO_IMPROVE_PAT} steps — stopping")
                            _stop_reason = "no_improve"
                            break

                # Progress snapshot for the frontend
                if progress_callback and (step + 1) % _snapshot_every == 0:
                    self.visualize_model.eval()
                    with torch.no_grad():
                        _snap_z_np = self.visualize_model.encoder(full_feat_t).cpu().numpy()
                    try:
                        _snap_metrics = _compute_focus_metrics(
                            _snap_z_np, focus_indices, _hd_cache, N, _k_metrics, _K_ext)
                    except Exception:
                        _snap_metrics = None
                    _snap_payload: dict = {
                        "steps_completed": step + 1,
                        "projection":      _snap_z_np.tolist(),
                        "focus_indices":   list(focus_indices),
                        "training_context_indices": list(training_context_indices),
                        "patch_indices":   list(patch_indices),
                    }
                    if _snap_metrics is not None:
                        _snap_payload["sampled_metrics"] = _snap_metrics

                    # A2: live neighbor status for progress messaging + highlighting.
                    # Partition each focus's HD top-10 into satisfied (already in LD
                    # top-10) vs unsatisfied, and collect impostors (LD top-10 that
                    # are not HD neighbors). Reuses _snap_z_np — no extra forward.
                    _sat_hd, _unsat_hd, _imp = set(), set(), set()
                    for _fi in focus_indices:
                        _d = np.linalg.norm(_snap_z_np - _snap_z_np[_fi], axis=1)
                        _d[_fi] = np.inf
                        _ld10 = set(int(x) for x in np.argsort(_d)[:_k])
                        _hd10 = _hd_nbr_dict[_fi]
                        _sat_hd   |= (_hd10 & _ld10)
                        _unsat_hd |= (_hd10 - _ld10)
                        _imp      |= (_ld10 - _hd10)
                    _snap_payload["refine_live"] = {
                        "np": round(float(_snap_metrics["neighbor_preservation"]), 1)
                              if _snap_metrics is not None else None,
                        "unsatisfied_count": len(_unsat_hd),
                        "rank_scale":        round(float(_rank_scale), 2),
                        "escalating":        bool(_single_focus and _rank_scale > 1.0),
                        "single_focus":      _single_focus,
                        "satisfied_hd":      sorted(int(x) for x in _sat_hd),
                        "unsatisfied_hd":    sorted(int(x) for x in _unsat_hd),
                        "impostor_indices":  sorted(int(x) for x in _imp),
                    }
                    progress_callback(_snap_payload)

                if should_stop_callback and should_stop_callback():
                    print(f"[TimeVis] Stop requested at step {step+1}")
                    _stop_reason = "user_stopped"
                    break

                if _TIME_LIMIT_S > 0 and time.time() - start_time > _TIME_LIMIT_S:
                    _stop_reason = "time_limited" if n_tri > 0 else "converged"
                    if n_tri > 0:
                        # Cut off before all ranking constraints were satisfied —
                        # make this explicit rather than silently returning <100%.
                        print(f"[TimeVis] TIME-LIMITED at step {step+1} "
                              f"({_TIME_LIMIT_S:.0f}s, single_focus={_single_focus}): "
                              f"best NP={_best_acc*100:.1f}%, {n_tri} impostors remain "
                              f"— consider raising refine_time_limit"
                              f"{'_single' if _single_focus else ''}_s")
                    else:
                        print(f"[TimeVis] Time limit at step {step+1} ({_TIME_LIMIT_S:.0f}s)")
                    break

        except Exception as _train_err:
            print(f"[TimeVis] Training error — rolling back model: {_train_err}")
            self.visualize_model.load_state_dict(_model_backup)
            raise

        # --- 5. Final projection: run encoder over ALL points ------------------
        self.visualize_model.eval()
        with torch.no_grad():
            z_np = self.visualize_model.encoder(full_feat_t).cpu().numpy()  # [N, 2]

        try:
            _final = _compute_focus_metrics(z_np, focus_indices, _hd_cache, N, _k_metrics, _K_ext)
            final_np    = _final["neighbor_preservation"]
            final_mrh   = _final["mean_rank_hd"]
            final_trust = _final["trustworthiness"]
            final_cont  = _final["continuity"]
            self._last_refine_np    = final_np
            self._last_refine_mrh   = final_mrh
            self._last_refine_trust = final_trust
            self._last_refine_cont  = final_cont
        except Exception as _metrics_err:
            print(f"[TimeVis] WARNING: metrics computation failed: {_metrics_err}")
            self._last_refine_np = self._last_refine_mrh = None
            self._last_refine_trust = self._last_refine_cont = None
            final_np = final_mrh = final_trust = final_cont = float('nan')

        print(f"[TimeVis] encoder-finetune refine done: "
              f"steps={step+1}  NP={final_np:.1f}%  MRH={final_mrh:.1f}  "
              f"T={final_trust:.1f}%  C={final_cont:.1f}%  "
              f"t={time.time()-start_time:.1f}s")

        # --- D2. Structured outcome for the frontend --------------------------
        # Explain WHY refine stopped and, when it fell short of 100%, expose the
        # stubborn HD neighbors / impostors so the UI can surface the insight
        # (a single point that can't reach 100% has a highly non-planar HD
        # neighborhood — that is diagnostic signal, not a silent failure).
        _remaining_hd  = sorted({int(x) for x in _tri_h}) if n_tri > 0 else []
        _remaining_imp = sorted({int(x) for x in _tri_m}) if n_tri > 0 else []
        _reason_msg = {
            "converged":    "Reached 100% top-10 neighbor accuracy.",
            "rank_cap":     "Single point could not reach 100%: some HD neighbors "
                            "are geometrically inseparable in 2D (non-planar neighborhood).",
            "time_limited": f"Stopped at the {_TIME_LIMIT_S:.0f}s time budget before converging.",
            "no_improve":   "Multi-focus best-effort: accuracy plateaued (neighborhoods conflict).",
            "max_steps":    "Reached the step budget before converging.",
            "user_stopped": "Stopped on user request.",
        }.get(_stop_reason, _stop_reason)
        self._last_refine_status = {
            "reason":               _stop_reason,
            "message":              _reason_msg,
            "converged":            _stop_reason == "converged",
            "single_focus":         _single_focus,
            "final_np":             None if final_np != final_np else round(float(final_np), 2),  # NaN-safe
            "steps":                int(step + 1),
            "rank_scale":           round(float(_rank_scale), 2),
            "remaining_impostors":  int(n_tri),
            "unsatisfied_hd_neighbors": _remaining_hd,
            "impostor_indices":     _remaining_imp,
        }

        # --- 6. Write refined projection to disk -------------------------------
        refined_dir = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}_refined",
            'epochs', f'epoch_{current_epoch}'
        )
        os.makedirs(refined_dir, exist_ok=True)
        np.save(os.path.join(refined_dir, 'projection.npy'), z_np)

        # Store encoder snapshot for patch_other_epochs (encoder mode path)
        self._last_local_visualizer = copy.deepcopy(self.visualize_model)
        self._last_focus_delta      = None   # not applicable in encoder mode
        self._last_focus_indices    = focus_indices
        self._last_refine_indices   = all_indices

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

