"""The Projector class for visualization, serve as a helper module for evaluator and visualizer"""
from abc import ABC, abstractmethod
import os
import json
import numpy as np
import torch
import umap

from visualize_model import VisModel, SingleVisualizationModel

# ------------------
# Projector:
# using visualize model to project N-d feature to 2-d embedding
# ------------------
class ProjectorAbstractClass(ABC):

    @abstractmethod
    def __init__(self, vis_model, content_path, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_project(self, *args, **kwargs):
        pass

    @abstractmethod
    def individual_project(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_inverse(self, *args, **kwargs):
        pass

    @abstractmethod
    def individual_inverse(self, *args, **kwargs):
        pass

class Projector(ProjectorAbstractClass):
    def __init__(self, config):
        self.config = config
        self.content_path = config['content_path']
        self.vis_method = config['vis_method']
        self.vis_id = config['vis_id']
        
    def init_model(self):
        gpu_id = self.config['vis_config']['gpu_id']
        self.device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() and gpu_id != -1 else "cpu")
        self.vis_model = VisModel(self.config['vis_config']['encoder_dims'], self.config['vis_config']['decoder_dims']).to(self.device)

    def load(self, iteration):
        file_path = os.path.join(self.content_path, 'visualize', f"{self.vis_method}_{self.vis_id}", 'epochs', f'epoch_{iteration}', 'vis_model.pth')
        save_model = torch.load(file_path, map_location="cpu")
        self.vis_model.load_state_dict(save_model["state_dict"])
        self.vis_model.to(self.device)
        self.vis_model.eval()
        print("Successfully load the visualization model for iteration {}".format(iteration))

    
    def batch_project(self, iteration, data):
        self.load(iteration)
        if len(data.shape) == 2: # only have one feature
            embedding = self.vis_model.encoder(torch.from_numpy(data).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        else: # have more than one feature: [ [[],[]], [[],[]],  ......     [[],[]] ]
            data_flatten = data.reshape(-1, data.shape[-1])
            embedding = self.vis_model.encoder(torch.from_numpy(data_flatten).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        return embedding # [sample_num * feature_num, 2]
    
    def individual_project(self, iteration, data):
        self.load(iteration)
        embedding = self.vis_model.encoder(torch.from_numpy(np.expand_dims(data, axis=0)).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        return embedding.squeeze(axis=0)
    
    def batch_inverse(self, iteration, embedding):
        self.load(iteration)
        data = self.vis_model.decoder(torch.from_numpy(embedding).to(dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        return data
    
    def individual_inverse(self, iteration, embedding):
        self.load(iteration)
        data = self.vis_model.decoder(torch.from_numpy(np.expand_dims(embedding, axis=0)).to(dtype=torch.float32, device="cpu")).cpu().detach().numpy()
        return data.squeeze(axis=0)

class DVIProjector(Projector):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.init_model()

    def load(self, iteration):
        file_path = os.path.join(self.content_path, 'visualize', f"{self.vis_method}_{self.vis_id}", 'epochs', f'epoch_{iteration}', 'vis_model.pth')
        save_model = torch.load(file_path, map_location="cpu")
        self.vis_model.load_state_dict(save_model["state_dict"])
        self.vis_model.to(self.device)
        self.vis_model.eval()


class TimeVisProjector(Projector):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.init_model()

    def load(self, iteration):
        file_path = os.path.join(self.content_path, 'visualize', f"{self.vis_method}_{self.vis_id}", 'vis_model.pth')
        save_model = torch.load(file_path, map_location="cpu")
        self.vis_model.load_state_dict(save_model["state_dict"])
        self.vis_model.to(self.device)
        self.vis_model.eval()
        
class DynaVisProjector(Projector):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.init_model()
    
    def init_model(self):
        gpu_id = self.config['vis_config']['gpu_id']
        self.device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() and gpu_id != -1 else "cpu")
        self.vis_model = SingleVisualizationModel(
            input_dims = self.config['vis_config']['dimension'],
            output_dims = 2,
            units = 256,
            hidden_layer = 3,
            device = self.device
        )
    
    
    def load(self, iteration):
        """
        专门适配 DynaVis 格式的加载逻辑：包含分体式权重、归一化统计量及超参数
        """
        # 1. 路径定位
        # 路径：content_path/visualize/DynaVis_ID/vis_model.pth
        file_path = os.path.join(
            self.content_path, 
            'visualize', 
            f"{self.config['vis_method']}_{self.config['vis_id']}", 
            'vis_model.pth'
        )

        if not os.path.exists(file_path):
            print(f"[Error] DynaVis model not found at: {file_path}")
            return

        # 2. 加载字典
        try:
            checkpoint = torch.load(file_path, map_location="cpu")
            
            # 3. 核心加载逻辑：识别 DynaVis 专属 Key
            if "encoder_state_dict" in checkpoint and "decoder_state_dict" in checkpoint:
                # 恢复 Encoder 和 Decoder 权重
                # 使用 strict=False 是为了兼容后续可能注入的 LoRA 层
                self.vis_model.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
                self.vis_model.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=False)
                
                # 4. 恢复环境上下文 (Stats & HParams)
                # 这是为了确保 batch_project 时的归一化空间与训练时完全一致
                self.stats = checkpoint.get("stats", None)
                self.hparams_checkpoint = checkpoint.get("hparams", None)
                
                print(f"[TTAV] DynaVis Model & Stats successfully restored for iteration {iteration}")
            else:
                print("[Warning] File found but it's not in DynaVis complex format. Check your save_vis_model logic.")

        except Exception as e:
            print(f"[Fatal] Failed to load DynaVis model: {e}")

        # 5. 设备同步与推理模式
        self.vis_model.to(self.device)
        self.vis_model.eval()
        
class UmapProjector():
    def __init__(self, config):
        """
        Initialize UMAP projection parameters.

        Parameters:
        ----------
        n_neighbors : int
            The size of the local neighborhood used for manifold approximation.
        min_dist : float
            The minimum distance between points in the low-dimensional space.
        metric : str
            The metric to use for computing distances in high-dimensional space.
        """
        n_neighbors = config['vis_config']['n_neighbors']
        min_dist = config['vis_config']['min_dist']
        metric = config['vis_config']['metric']
        
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)

    def batch_project(self, embeddings):
        """
        Perform UMAP dimensionality reduction on high-dimensional embeddings.

        Parameters:
        ----------
        embeddings : numpy.ndarray
            High-dimensional input data of shape [n_samples, dimension].

        Returns:
        -------
        numpy.ndarray
            Low-dimensional projection of shape [n_samples, 2].
        """
        return self.reducer.fit_transform(embeddings)


