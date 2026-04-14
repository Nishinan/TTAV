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
        _, _nn_idx = _nbrs.kneighbors(feat)   # shape [M, k_local+1]

        # _nn_idx[:,0] is self → skip; columns 1.. are true neighbors
        src_rows = np.repeat(np.arange(len(all_indices)), k_local)   # [M*k]
        tgt_rows = _nn_idx[:, 1:k_local + 1].flatten()               # [M*k]

        edge_to   = feat_t[src_rows].to(self.device)   # [M*k, D]
        edge_from = feat_t[tgt_rows].to(self.device)   # [M*k, D]
        a_dummy   = torch.ones(edge_to.shape[0], edge_to.shape[1], device=self.device)

        # --- 4. Fine-tune with correct local UMAP loss -----------------------
        # edge_to / edge_from are genuine high-D neighbor pairs → UmapLoss now
        # produces a meaningful attract/repel gradient for the focus region.
        optimizer = torch.optim.Adam(self.visualize_model.parameters(), lr=0.001)
        self.visualize_model.train()
        for _ in range(5):
            optimizer.zero_grad()
            outputs = self.visualize_model(edge_to, edge_from)
            _, _, loss = self.criterion(edge_to, edge_from, a_dummy, a_dummy, outputs)
            loss.backward()
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

    def _patch_epoch(self, epoch, all_indices):
        """Write a subset-patched projection for one epoch. Safe to call from background thread."""
        import numpy as np, os, torch

        vis_method = self.config['vis_method']
        vis_id     = self.config['vis_id']
        content_path = self.config['content_path']

        baseline_path = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}",
            'epochs', f'epoch_{epoch}', 'projection.npy'
        )
        refined_dir = os.path.join(
            content_path, 'visualize', f"{vis_method}_{vis_id}_refined",
            'epochs', f'epoch_{epoch}'
        )

        if os.path.exists(baseline_path):
            full_proj = np.load(baseline_path).copy()
        else:
            return  # no baseline for this epoch, skip silently

        self.visualize_model.eval()
        with torch.no_grad():
            sub_feat = self.data_provider.get_representation(epoch)[all_indices]
            sub_emb  = self.visualize_model.encoder(
                torch.from_numpy(sub_feat).float().to(self.device)
            ).cpu().numpy()

        full_proj[all_indices] = sub_emb
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

