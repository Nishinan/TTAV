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
                
    def refine(self, focus_indices=None, focus_index=None, neighbor_indices=None, epochs_to_update=10):
        """
        Refine projections for one or more focus points.
        `focus_indices` is preferred (list); `focus_index` kept for backwards compat.
        """
        import torch
        import numpy as np
        import time
        import os

        # Normalise to a list of focus points
        if focus_indices is None:
            focus_indices = [focus_index] if focus_index is not None else []

        start_time = time.time()

        if not neighbor_indices:
            all_edges_to = self.data_handler.edge_to
            all_edges_from = self.data_handler.edge_from
            collected = set()
            for fi in focus_indices:
                idx = np.where(all_edges_to == fi)[0]
                collected.update(all_edges_from[idx].tolist())
            # Cap at 15 neighbours per focus point to avoid memory blow-up
            neighbor_indices = list(collected - set(focus_indices))[:15 * max(len(focus_indices), 1)]

        all_indices = list(focus_indices) + neighbor_indices
        available_epochs = self.config['available_epochs']
        target_epochs = available_epochs[-epochs_to_update:]

        train_data = []
        for e in target_epochs:
            feat = self.data_provider.get_representation(e)[all_indices]
            train_data.append(torch.from_numpy(feat).float())
        train_batch = torch.cat(train_data, dim=0).to(self.device)

        optimizer = torch.optim.Adam(self.visualize_model.parameters(), lr=0.01)
        self.visualize_model.train()
        # Dummy attention (ones) so ReconstructionLoss treats all features equally.
        dummy_a = torch.ones_like(train_batch)
        for _ in range(10):
            optimizer.zero_grad()
            outputs = self.visualize_model(train_batch, train_batch)
            _, _, loss = self.criterion(train_batch, train_batch, dummy_a, dummy_a, outputs)
            loss.backward()
            optimizer.step()
            if time.time() - start_time > 1.2:
                break

        # Save refined projections to visualize/{vis_method}_{vis_id}_refined/epochs/epoch_N/projection.npy
        # This matches the path that load_projection(refine_flag=True) expects.
        vis_method = self.config['vis_method']
        vis_id = self.config['vis_id']
        content_path = self.config['content_path']

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

        print(f"Refine & Save finished in {time.time() - start_time:.2f}s")

