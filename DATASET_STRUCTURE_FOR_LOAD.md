# TTAV 数据目录规范（用于 Load Visualization）

本文用于统一 `content_path` 的目录结构与文件格式，保证可直接 `Load Visualization`，并兼容后续常见分析功能。

---

## 1. 结论：`train.py` 是否必需

- `train.py` **不是**当前在线可视化服务（`start/load/updateProjection`）的必需文件。  
- 可将 `train.py` 视为离线训练脚本，默认归类为**可选**。

---

## 2. 目录结构（推荐）

```text
<content_path>/
├── dataset/
│   ├── info.json
│   ├── labels.npy
│   └── index.json
├── epochs/
│   ├── epoch_1/
│   │   ├── embeddings.npy
│   │   ├── predictions.npy
│   │   └── model.pth
│   ├── epoch_2/
│   │   ├── embeddings.npy
│   │   ├── predictions.npy
│   │   └── model.pth
│   └── ...
├── scripts/
│   └── model.py
└── visualize/
    └── <vis_method>_<vis_id>/
        ├── info.json
        └── epochs/
            ├── epoch_1/projection.npy
            ├── epoch_2/projection.npy
            └── ...
```

---

## 3. 分级要求

## 3.1 Load Visualization 最低必需

1. `dataset/info.json`
2. `epochs/epoch_*/embeddings.npy`（至少一个 `epoch_*`）
3. `visualize/<vis_method>_<vis_id>/info.json`
4. `visualize/<vis_method>_<vis_id>/epochs/epoch_*/projection.npy`

如果你当前加载的是 `TimeVis + vis_id=1`，则关键路径是：

`visualize/TimeVis_1/epochs/epoch_*/projection.npy`

## 3.2 后续常用任务（建议保留）

1. `dataset/labels.npy`（标签、筛选、属性相关）
2. `dataset/index.json`（样本索引划分；缺失时可自动生成）
3. `epochs/epoch_*/predictions.npy`（预测相关分析）
4. `epochs/epoch_*/model.pth` + `scripts/model.py`（需加载模型的功能会用到）

## 3.3 可选/可重建

- `epochs/epoch_*/hd_neighbors_*.json`（缓存）
- `visualize/.../proj_neighbors_*.json`（缓存）
- `scripts/__pycache__/`（缓存）
- `selected_idxs/`（仅特定流程使用）
- `train.py`（离线训练脚本）

---

## 4. 文件内部格式要求

## 4.1 `dataset/info.json`（必需）

最少字段：

```json
{
  "model": "ResNet18",
  "classes": ["class0", "class1", "class2"]
}
```

说明：
- `model`：字符串，需与 `scripts/model.py` 里可实例化的模型类名一致（如 `ResNet18`）。
- `classes`：类别名数组，长度 = 类别数。

## 4.2 `dataset/labels.npy`（建议保留）

- 一维数组，长度为 `N`（样本总数）。
- 每个元素是类别索引（通常为整数 `0..C-1`）。

## 4.3 `dataset/index.json`（建议保留）

```json
{
  "train": [0, 1, 2],
  "test": [3, 4]
}
```

说明：
- `train`、`test` 为样本下标数组（指向全量样本索引）。
- 若不存在，系统通常会自动生成：
  - `train = [0..N-1]`
  - `test = []`

## 4.4 `epochs/epoch_*/embeddings.npy`（Load 必需）

- 二维数组，形状 `N x D`。
- `N` 需与 `labels.npy` 对齐（或至少覆盖同一索引空间）。
- `D` 为特征维度（如 128、256、512 等）。

## 4.5 `epochs/epoch_*/predictions.npy`（建议保留）

- 二维数组，形状通常为 `N x C`（每类得分/logits/probability）。
- 系统会据此计算预测类别（`argmax`）。

## 4.6 `epochs/epoch_*/model.pth`（建议保留）

- PyTorch 模型权重文件，与 `scripts/model.py` 中模型定义匹配。

## 4.7 `scripts/model.py`（建议保留）

- 需包含与 `info.json` 的 `model` 同名类，如 `ResNet18`。
- 该类应可被后端动态加载并 `load_state_dict`。

## 4.8 `visualize/<vis_method>_<vis_id>/info.json`（Load 必需）

- 记录该可视化会话配置（`content_path`, `vis_method`, `vis_id`, `vis_config` 等）。
- `load/sync session` 时会读取并用于还原会话。

## 4.9 `visualize/<vis_method>_<vis_id>/epochs/epoch_*/projection.npy`（Load 必需）

- 二维数组，形状 `N x 2`。
- 表示每个样本在 2D 可视化空间中的坐标。
- `N` 需与对应 epoch 的样本索引体系一致。

---

## 5. 一致性检查（强烈建议）

在交付数据前，至少确认：

1. `labels.npy`、`embeddings.npy`、`predictions.npy` 的样本数 `N` 一致。  
2. `visualize/.../projection.npy` 的行数与样本索引体系一致。  
3. `info.json:model` 与 `scripts/model.py` 中类名一致。  
4. `visualize` 下的 `<vis_method>_<vis_id>` 与前端加载参数一致（如 `TimeVis_1`）。

---

## 6. 最小可加载包（仅 Load）

如果你只想“能 Load，不做复杂分析”，可只打包：

```text
dataset/info.json
epochs/epoch_*/embeddings.npy
visualize/<vis_method>_<vis_id>/info.json
visualize/<vis_method>_<vis_id>/epochs/epoch_*/projection.npy
```

若要保证后续分析能力，按第 3.2 节再补齐建议文件。
