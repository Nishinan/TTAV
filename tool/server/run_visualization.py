import os
import json
import numpy as np

from server_utils import generate_dimension_array

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
#                     filename='app.log', filemode='w')

def initialize_config(content_path, vis_method, vis_id, data_type, task_type, vis_config):
    saved_vis_config = {}
    vis_info_path = os.path.join(content_path, "visualize", f"{vis_method}_{vis_id}", "info.json")
    if os.path.exists(vis_info_path):
        try:
            with open(vis_info_path, "r", encoding="utf-8") as f:
                vis_info = json.load(f)
            if isinstance(vis_info.get("vis_config"), dict):
                saved_vis_config = vis_info["vis_config"]
        except Exception as exc:
            print(f"Failed to load saved vis_config from {vis_info_path}: {exc}")

    config = {}
    config["content_path"] = content_path
    config["vis_method"] = vis_method
    config["vis_id"] = vis_id
    config["data_type"] = data_type
    config["task_type"] = task_type
    config["vis_config"] = {
        **saved_vis_config,
        **(vis_config or {}),
    }
    
    with open(os.path.join(content_path, 'dataset', 'info.json')) as f:
        dataset_info = json.load(f)
    config["classes"] = dataset_info['classes']
    config["model"] = dataset_info['model']
    
    # available epochs
    epochs_dir = os.path.join(content_path, 'epochs')
    available_epochs = []
    if os.path.exists(epochs_dir) and os.path.isdir(epochs_dir):
        for folder_name in os.listdir(epochs_dir):
            if folder_name.startswith("epoch_"):
                try:
                    k = int(folder_name.split("_")[1])
                    available_epochs.append(k)
                except ValueError:
                    print(f"Invalid epoch folder name: {folder_name}")
    
    available_epochs.sort()
    config["available_epochs"] = available_epochs
    # 2. 为不同方法提供默认的 resolution 字符串
    if 'resolution' not in config['vis_config']:
        if vis_method in ["DVI", "DynaVis"]:
            config['vis_config']['resolution'] = [200,200]
        elif vis_method == "TimeVis":
            config['vis_config']['resolution'] = [300,300]
        else:
            config['vis_config']['resolution'] = [200,200] 
            
    # vis_model dims
    if vis_method == "DVI" or vis_method == "TimeVis" or vis_method == "DynaVis":
        vc = config['vis_config']
        reuse_existing_dims = (
            isinstance(vc.get('encoder_dims'), list)
            and isinstance(vc.get('decoder_dims'), list)
            and isinstance(vc.get('dimension'), int)
        )
        if not reuse_existing_dims:
            epoch_0 = available_epochs[0]
            embedding_path = os.path.join(content_path, 'epochs', f'epoch_{epoch_0}', 'embeddings.npy')
            embedding = np.load(embedding_path)
            encoder_dims, decoder_dims = generate_dimension_array(embedding.shape[1])
            config['vis_config']['dimension'] = embedding.shape[1]
            config['vis_config']['encoder_dims'] = encoder_dims
            config['vis_config']['decoder_dims'] = decoder_dims
        
        # resolution_str = config['vis_config']['resolution']
        # r = resolution_str.split(",")
        # config['vis_config']['resolution'] = [int(i) for i in r]
    
    # 为不同方法补充缺失的默认超参数
    vc = config['vis_config']
    
    # 通用默认值
    if 'n_neighbors' not in vc:
        vc['n_neighbors'] = 10
    if 'max_epochs' not in vc:
        vc['max_epochs'] = 10
    if 'patient' not in vc:
        vc['patient'] = 3
    if 's_n_epochs' not in vc:
        vc['s_n_epochs'] = 500
    if 'b_n_epochs' not in vc:
        vc['b_n_epochs'] = 0
    if 'refine_max_steps' not in vc:
        vc['refine_max_steps'] = 800
    if 'refine_min_steps' not in vc:
        vc['refine_min_steps'] = 800
    if 'refine_patience' not in vc:
        vc['refine_patience'] = 0
    if 'refine_loss_min_delta' not in vc:
        vc['refine_loss_min_delta'] = 1e-4
    if 'refine_time_limit_s' not in vc:
        vc['refine_time_limit_s'] = 0.0

    # 特定方法的默认值
    if vis_method == "TimeVis":
        if 't_n_epochs' not in vc:
            vc['t_n_epochs'] = 5 # TimeVis 特有的时间边训练轮次
        if 'lambda' not in vc:
            vc['lambda'] = 1.0   # TimeVis 的损失权重
            
    elif vis_method == "DVI":
        if 'lambda1' not in vc:
            vc['lambda1'] = 1.0
        if 'lambda2' not in vc:
            vc['lambda2'] = 1.0 # DVI 的时间连续性权重

    return config

def init_visualize_component(config):
    import torch
    from visualize.strategy.projector import DVIProjector, TimeVisProjector, UmapProjector, DynaVisProjector
    from visualize.strategy.dvi_strategy import DeepVisualInsight
    from visualize.strategy.timevis_strategy import TimeVis
    # [修改点 1]：删除不存在的 dynavis_strategy 导入，改为导入 Runner
    from visualize.dynavis.runner import DynaVisRunner
    from visualize.data_provider import DataProvider
    from visualize.result_generator import ResultGenerator, UmapResultGenerator
    # 做一些操作
    if 'gpu_id' not in config['vis_config']:
        config['vis_config']['gpu_id'] = -1
    if  config['vis_config']['gpu_id'] == -1:    
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(config['vis_config']['gpu_id']) if torch.cuda.is_available() else "cpu")
    
    if config.get('vis_method') == "TimeVis":
        # 确保 vis_config 字典存在
        if 'vis_config' not in config:
            config['vis_config'] = {}
        
        # 补全 TimeVis 强依赖的参数
        if 'lambda' not in config['vis_config']:
            config['vis_config']['lambda'] = 1.0
            
    if config['vis_method'] == "DVI":
        data_provider = DataProvider(config, device)
        projector = DVIProjector(config)
        visualizer = ResultGenerator(config, data_provider, projector)
        strategy = DeepVisualInsight(config, data_provider)
    elif config['vis_method'] == "TimeVis":
        data_provider = DataProvider(config, device)
        projector = TimeVisProjector(config)
        visualizer = ResultGenerator(config, data_provider, projector)
        strategy = TimeVis(config, data_provider)
    elif config['vis_method'] == "DynaVis":
        data_provider = DataProvider(config, device)
        projector = DynaVisProjector(config)
        visualizer = ResultGenerator(config, data_provider, projector)
        runner = DynaVisRunner(
            config["content_path"], config["vis_id"],
            config["data_type"], config["task_type"], config["vis_config"]
        )
        strategy = runner
        
    elif config['vis_method'] == "UMAP":
        data_provider = DataProvider(config, device)  
        projector = UmapProjector(config)
        visualizer = UmapResultGenerator(config, data_provider, projector)
        strategy = None
    else:
        raise NotImplementedError
    
    return visualizer, strategy

def visualize_run(content_path, vis_method, vis_id, data_type, task_type, vis_config):
    # step 1: initialize config
    config = initialize_config(content_path, vis_method, vis_id, data_type, task_type, vis_config)

    visualizer, strategy = init_visualize_component(config)

    if vis_method == "DynaVis":
        # DynaVis manages its own training and output; do not call visualize_all_epochs().
        strategy.run()
    else:
        if vis_method in ("DVI", "TimeVis"):
            print("Start training visualization model...")
            strategy.train_vis_model()
            print("Train visualization model finished.")

        # Project all epochs with the trained model.
        print("Start generating visualization results...")
        visualizer.visualize_all_epochs()
        print("Generate visualization results finished, visualization process completed successfully!")
    
    # step 4: save config
    os.makedirs(os.path.join(content_path, 'visualize', f"{vis_method}_{vis_id}"), exist_ok=True)


    json.dump(config, open(os.path.join(content_path, 'visualize', f"{vis_method}_{vis_id}", 'info.json'), 'w'), indent=2)
    return visualizer, strategy 
        