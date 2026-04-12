import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from strategy.trainer import DVITrainer
from strategy.custom_weighted_random_sampler import CustomWeightedRandomSampler
from strategy.edge_dataset import DVIDataHandler
from strategy.spatial_edge_constructor import SingleEpochSpatialEdgeConstructor
from strategy.losses import DVILoss, DummyTemporalLoss, TemporalLoss, UmapLoss, ReconstructionLoss, SingleVisLoss
from visualize_model import VisModel
from strategy.strategy_abstract import StrategyAbstractClass
from data_provider import DataProvider
from umap.umap_ import find_ab_params
from utils import find_neighbor_preserving_rate

class DeepVisualInsight(StrategyAbstractClass):
    def __init__(self, config, data_provider):
        super().__init__(config)
        self.initialize_model()
        self.data_provider = data_provider

    def initialize_model(self):
        gpu_id = self.config['vis_config']['gpu_id']
        self.device = torch.device("cuda:{}".format(self.config['vis_config']['gpu_id']) if torch.cuda.is_available() and gpu_id != -1 else "cpu")
        # VisModel returns 4-tuple (emb_to, emb_from, recon_to, recon_from), now consistent with DVILoss
        self.visualize_model = VisModel(
            self.config['vis_config']['encoder_dims'],
            self.config['vis_config']['decoder_dims']
        ).to(self.device)
        
        # define losses
        negative_sample_rate = 5
        min_dist = 0.1
        _a, _b = find_ab_params(1.0, min_dist)
        self.umap_fn = UmapLoss(negative_sample_rate, self.device, _a, _b, repulsion_strength=1.0)
        self.recon_fn = ReconstructionLoss(beta=1.0)
    
    def train(self):
        self.train_vis_model()

    def train_vis_model(self):
        # parameters    
        LAMBDA1 = self.config['vis_config']['lambda1']
        LAMBDA2 = self.config['vis_config']['lambda2']
        N_NEIGHBORS = self.config['vis_config']['n_neighbors']
        S_N_EPOCHS = self.config['vis_config']['s_n_epochs']
        B_N_EPOCHS = self.config['vis_config']['b_n_epochs']
        PATIENT = self.config['vis_config']['patient']
        MAX_EPOCH = self.config['vis_config']['max_epochs']
        
        INIT_NUM = 100
        
        prev_model = VisModel(
            self.config['vis_config']['encoder_dims'],
            self.config['vis_config']['decoder_dims']
        ).to(self.device)
        prev_model.load_state_dict(self.visualize_model.state_dict())
        for param in prev_model.parameters():
            param.requires_grad = False
        w_prev = dict(self.visualize_model.named_parameters())

        # for each epch
        available_epochs = self.config['available_epochs']
        for i in range(len(available_epochs)):
            epoch = available_epochs[i]
            # Define DVI Loss
            if i == 0:
                temporal_loss_fn = DummyTemporalLoss(self.device)
                criterion = DVILoss(self.umap_fn, self.recon_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=0.0, device = self.device)
            else:
                self.temporal_fn = TemporalLoss(w_prev,self.device)
                prev_data = self.data_provider.get_representation(available_epochs[i-1])
                curr_data = self.data_provider.get_representation(epoch)
                prev_data = prev_data.reshape(-1,prev_data.shape[-1])
                curr_data = curr_data.reshape(-1,curr_data.shape[-1])
                
                npr = find_neighbor_preserving_rate(prev_data, curr_data, N_NEIGHBORS)
                # criterion = DVILoss(self.umap_fn, self.recon_fn, self.temporal_fn, lambd1=LAMBDA1, lambd2=LAMBDA2*npr, device = self.device)
                criterion = DVILoss(self.umap_fn, self.recon_fn, self.temporal_fn, lambd1=LAMBDA1, lambd2=LAMBDA2*npr.mean(), device = self.device)
                
            optimizer = torch.optim.Adam(self.visualize_model.parameters(), lr=.01, weight_decay=1e-5)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
            # Define Edge dataset
            spatial_cons = SingleEpochSpatialEdgeConstructor(self.data_provider, epoch, INIT_NUM, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS)
            edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct()

            probs = probs / (probs.max()+1e-3)
            eliminate_zeros = probs>1e-3
            edge_to = edge_to[eliminate_zeros]
            edge_from = edge_from[eliminate_zeros]
            probs = probs[eliminate_zeros]
            
            dataset = DVIDataHandler(edge_to, edge_from, feature_vectors, attention)
            # Store last epoch's dataset so refine() can look up neighbors.
            self._last_data_handler = dataset
            self._last_feature_vectors = feature_vectors

            n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
            # chose sampler based on the number of dataset
            if len(edge_to) > 2 ** 24:
                sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
            else:
                sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
            edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)

            trainer = DVITrainer(self.visualize_model, criterion, optimizer, lr_scheduler,edge_loader=edge_loader, DEVICE=self.device)
            trainer.train(PATIENT, MAX_EPOCH)
            
            self.save_vis_model(self.visualize_model, epoch, trainer.loss, trainer.optimizer)

            prev_model.load_state_dict(self.visualize_model.state_dict())
            for param in prev_model.parameters():
                param.requires_grad = False
            w_prev = dict(prev_model.named_parameters())

    def get_focus_mask(self, selected_indices):
        """Build a boolean mask on device indicating which points are in focus."""
        total_count = 0
        try:
            if hasattr(self.data_provider, 'train_data'):
                total_count = len(self.data_provider.train_data)
            else:
                available_epochs = self.config['available_epochs']
                test_data = self.data_provider.get_representation(available_epochs[0])
                total_count = len(test_data)
        except Exception:
            try:
                index_path = os.path.join(self.config["content_path"], "epochs", "index.npy")
                total_count = len(np.load(index_path))
            except Exception:
                total_count = max(selected_indices) + 1 if selected_indices else 10000

        mask = torch.zeros(total_count, dtype=torch.bool).to(self.device)
        if selected_indices:
            mask[selected_indices] = True
        return mask

    def update_ttav_context(self, indices, mode, mask):
        """Update trainer state. Called by the server before incremental training."""
        self.ttav_indices = indices
        self.ttav_mode = mode
        self.ttav_mask = mask

    def refine(self, focus_indices=None, focus_index=None, neighbor_indices=None, epochs_to_update=10):
        """
        Incrementally refine projections for one or more focus points.
        `focus_indices` is preferred (list); `focus_index` kept for backwards compat.

        For DVI, we load the latest saved per-epoch model, fine-tune with
        SingleVisLoss (no temporal term) on the local neighbourhood, then
        write every epoch's projection to the _refined path so the frontend
        can load it with refine_flag=True.
        """
        import time

        # Normalise to a list of focus points
        if focus_indices is None:
            focus_indices = [focus_index] if focus_index is not None else []

        start_time = time.time()
        available_epochs = self.config['available_epochs']
        vis_method = self.config['vis_method']
        vis_id = self.config['vis_id']
        content_path = self.config['content_path']

        # --- 1. Build neighbour list from the last epoch's edge graph -----------
        if not neighbor_indices:
            if hasattr(self, '_last_data_handler'):
                all_edges_to = self._last_data_handler.edge_to
                all_edges_from = self._last_data_handler.edge_from
                collected = set()
                for fi in focus_indices:
                    idx = np.where(all_edges_to == fi)[0]
                    collected.update(all_edges_from[idx].tolist())
                neighbor_indices = list(collected - set(focus_indices))[:15 * max(len(focus_indices), 1)]
            else:
                neighbor_indices = []

        all_indices = list(focus_indices) + neighbor_indices
        target_epochs = available_epochs[-epochs_to_update:]

        # --- 2. Load the latest per-epoch model as starting point ---------------
        last_epoch = available_epochs[-1]
        model_path = os.path.join(
            content_path, 'visualize',
            f"{vis_method}_{vis_id}",
            'epochs', f'epoch_{last_epoch}', 'vis_model.pth'
        )
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            self.visualize_model.load_state_dict(ckpt['state_dict'])

        # --- 3. Build a lightweight SingleVisLoss (no temporal) for refine ------
        negative_sample_rate = 5
        min_dist = 0.1
        from umap.umap_ import find_ab_params
        _a, _b = find_ab_params(1.0, min_dist)
        umap_fn = UmapLoss(negative_sample_rate, self.device, _a, _b, repulsion_strength=1.0)
        recon_fn = ReconstructionLoss(beta=1.0)
        refine_criterion = SingleVisLoss(umap_fn, recon_fn, lambd=1.0, negative_sample_rate=negative_sample_rate)

        # --- 4. Gather training data from target epochs -------------------------
        train_data = []
        for e in target_epochs:
            feat = self.data_provider.get_representation(e)[all_indices]
            train_data.append(torch.from_numpy(feat).float())
        train_batch = torch.cat(train_data, dim=0).to(self.device)

        # --- 5. Fine-tune with a 1.5-second budget ------------------------------
        optimizer = torch.optim.Adam(self.visualize_model.parameters(), lr=0.01)
        self.visualize_model.train()
        dummy_a = torch.ones_like(train_batch)
        for _ in range(20):
            optimizer.zero_grad()
            outputs = self.visualize_model(train_batch, train_batch)
            _, _, loss = refine_criterion(train_batch, train_batch, dummy_a, dummy_a, outputs)
            loss.backward()
            optimizer.step()
            if time.time() - start_time > 1.5:
                break

        # --- 6. Write refined projections for every epoch -----------------------
        self.visualize_model.eval()
        with torch.no_grad():
            for e in available_epochs:
                full_feat = self.data_provider.get_representation(e)
                embedding = self.visualize_model.encoder(
                    torch.from_numpy(full_feat).float().to(self.device)
                ).cpu().numpy()

                save_dir = os.path.join(
                    content_path, 'visualize',
                    f"{vis_method}_{vis_id}_refined",
                    'epochs', f'epoch_{e}'
                )
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, 'projection.npy'), embedding)

        print(f"[DVI] Refine finished in {time.time() - start_time:.2f}s")
