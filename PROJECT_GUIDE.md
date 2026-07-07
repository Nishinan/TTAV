# Time-Travelling Visualizer · 项目代码指南

> 用途：新开对话时先读本文件，快速掌握项目结构、Refine 全链路、关键文件与"坑位"。
> 配套文档：`REFINE_PROGRESS_REPORT.md`（本轮改进的问题→根因→修复叙事）。

---

## 1. 项目概述

VS Code 扩展，可视化 ML 模型训练中**高维表示空间的演变**：逐 epoch 步进、观察点如何移动、分析邻居关系。核心交互功能是 **Refine**——用户点选一个/多个点，局部微调投影，使该点的**高维 top-k 邻居**在低维里也成为 top-k 邻居。

**三层架构（HTTP 通信）**：
```
VS Code 扩展 (TypeScript)  ←IPC→  Web 前端 (React/TS)  ←HTTP:5050→  Python 后端 (Flask/gunicorn)
extension/                        web/                              tool/
```

---

## 2. 如何运行 / 重启（重要）

后端以 **gunicorn 常驻服务**运行（systemd）：
```
ExecStart: gunicorn -w 1 -b 0.0.0.0:5050 --timeout 3600 --preload server:app
WorkingDirectory: tool/server
```
- **`--preload` 且无 `--reload`** → Python 代码改动**必须重启**才生效：
  ```bash
  sudo systemctl restart ttav-backend
  tail -f /var/log/ttav-backend.log     # 看日志
  ```
- **例外（无需重启）**：`tests/ablation_config.json`、`tests/refine_avg_config.json` 是每次 refine 从磁盘热读的。
- **前端**：Vite dev（端口 5173）热更；改动一般刷新浏览器即可。

端口：后端 **5050**（硬编码），Vite **5173**。

---

## 3. 目录 / 关键文件地图

### 后端 `tool/`
| 文件 | 职责 |
|------|------|
| `server/server.py` | Flask 入口，所有 HTTP 路由；refine 会话管理 |
| `server/server_utils.py` | `build_focus_set`、`load_projection`（含优雅回落）、`calculate_high_dimensional_neighbors`（参数化 k）、邻居缓存 |
| `server/run_visualization.py` | 初始化配置、启动可视化流水线；refine 步数等默认值 |
| `server/refine_behavior_config.py` | refine 行为/停止条件配置（含时间预算）|
| `visualize/strategy/timevis_strategy.py` | **核心**：`TimeVis` 策略，`refine()` 即本轮重构的主战场 |
| `visualize/strategy/{dvi,timevis}_strategy.py` 等 | 各可视化算法策略（DVI / TimeVis / UMAP）|
| `visualize/dynavis/runner.py` | DynaVis 独立 runner |
| `visualize/data_provider.py` | `get_representation(epoch)` 读 embeddings.npy（**无缓存**，见坑位）|

### 前端 `web/src/`
| 文件 | 职责 |
|------|------|
| `views/plotView.tsx` | **主视图**：refine 触发、流式进度轮询、投影混合/写回、完成提示 |
| `component/chart.tsx` | 散点图渲染；`NeighborOverlay`（红/蓝/灰 邻居环）；框选；before/after 注入点（`epochData = allEpochData[epoch]`）|
| `component/function-panel.tsx` | 右侧控制面板：选点、refine 参数（top-k / Refine goal）、指标、before/after 开关、Reset refine |
| `component/main-block.tsx` | 时间轴（epoch 节点、播放、C2 绿环）|
| `state/state.unified.ts` | **Zustand 全局 store**；`useDefaultStore([...])`；setter 按字段自动生成 |
| `communication/backend.ts` | axios 调用后端（`basicPostWithJsonResponse`）|

### 扩展 `extension/src/`
`extension.ts`（激活）/ `control.ts`（配置/流程）/ `views.ts`（面板注册）。方法超参在 `extension/package.json` 的 `contributes.configuration`。

---

## 4. Refine 全链路（端到端）

**前端主路径（流式，带进度）**：
```
plotView.tsx: BackendAPI.startRefineSession(...)          # 发 selected_indices, focus_mode, epoch, zoom_bbox, secondary_indices, refine_top_k, refine_priority
  → server.py /startRefineSession
      → _prepare_refine_request(req)                       # build_focus_set 扩展 + _split_focus_targets(Option A)
      → 后台线程 _run_refine_session_worker
          → _run_refine_request → strategy.refine(...)     # 核心优化
  → 前端轮询 /getRefineSessionProgress                     # 拿 projection 快照 + sampled_metrics + refine_live
  → 完成后 fetchEpochProjection(refine_flag=true) 取最终结果，混合显示
  → strategy.patch_other_epochs() 后台 patch 其余 epoch
```
另有同步旧路径 `/updateFocusContext`（逻辑相同，可能少用）。

**Option A（关键语义）**：`_split_focus_targets`（server.py）
- 有显式选点 → `focus_indices = seeds`（唯一 attract 目标），扩展部分并入 `secondary_indices`（上下文）→ 单击=单 focus。
- 无选点框选 → 整簇作为多 focus（best-effort）。
- 响应里 `focus_indices` 仍返回扩展集 → 前端高亮不变。

---

## 5. Refine 当前损失设计（`refine()` in `timevis_strategy.py`）

微调共享 encoder；每次入口从 `vis_model.pth` checkpoint 重载。

```
L_total =  1.0        · L_shape     # 局部 MDS 保形（主力）：目标 ‖enc(i)-enc(j)‖ ≈ s·d_hd(i,j)
        +  5.0·prio   · rank_scale · L_rank   # 边界排序 hinge（残余）：max(0, ‖f-h‖-‖f-m‖+margin)
        + 20.0        · rank_scale · L_pin    # 抗逃逸：‖enc(focus)-baseline_focus‖²
        +  3.0        · rank_scale · L_dir    # 软方向：1-cos(u_now,u_base)，只惩罚方位角
        +  1.5·prio   ·             L_anchor  # 远场锚定（距离加权）
        +  0.3        ·             L_global  # 全局 edge 拓扑
```
- **簇** `_cluster_arr` = focus ∪ HD-top-k ∪ focus 的局部 LD 邻域(K_CTX=max(3k,30)，含 impostor)
- **s** = median(baseline LD 距离)/median(HD 距离)（尺度对齐）
- **prio** = B1 精度↔保形（0.5 复现默认；0.15 保布局 / 0.9 保精度）
- **rank_scale** = escalation 倍率（单点停滞 200 步 ×1.5，封顶 200）
- **L_shape 采样**：每步全量 focus→各成员 + 随机簇内对
- **停止**：triplets==0 / escalation 封顶 / 300s(单)·90s(多) / max_steps=10000

> 各损失的来龙去脉见 `REFINE_PROGRESS_REPORT.md`。一句话：**L_shape 忠实重建局部几何为主，L_rank 只做残余补正；L_pin 防平移逃逸，L_dir 防镜像翻面。**

**结构化结果**：`self._last_refine_status`（reason / converged / final_np / 顽固邻居索引），经响应返回前端（D2）。

---

## 6. 关键参数与位置

| 参数 | 默认 | 位置 |
|------|------|------|
| 损失权重 `_W_SHAPE/_W_RANK/_W_PIN/_W_DIR/_LAMBDA_ANCHOR/_GAMMA_GLOBAL` | 1/5·prio/20/3/1.5·prio/0.3 | `timevis_strategy.py` refine() 超参块（硬编码）|
| escalation `_ESCALATE_PAT/_RANK_ESCALATE/_RANK_SCALE_MAX` | 200/1.5/200 | 同上 |
| `refine_top_k`（C3） | 10（3–20）| 前端 store `refineTopK` → 请求 `refine_top_k` → refine `top_k` 参数 |
| `refine_priority`（B1） | 0.5 | 前端 store `refinePriority` → `refine_priority` → `priority` 参数 |
| 时间预算 `time_limit_seconds / time_limit_single_seconds` | 90 / 300 | `refine_behavior_config.py`（flat 覆盖：`refine_time_limit_s` / `_single_s`）|
| `refine_max_steps / refine_min_steps` | 10000 / 100 | `run_visualization.py` / vis_config |

---

## 7. 后端 HTTP 端点（refine 相关）

| 端点 | 作用 |
|------|------|
| `POST /startRefineSession` | 启动 refine 会话（流式主路径），返回 session_id |
| `POST /getRefineSessionProgress` | 轮询进度：projection 快照、sampled_metrics、`refine_live`（A2）、result |
| `POST /stopRefineSession` | 请求停止 |
| `POST /updateFocusContext` | 同步 refine（旧路径）|
| `POST /discardRefine` | **Undo**：删 `_refined` 投影 → `load_projection` 优雅回落到 baseline（B3）|
| `POST /refinedEpochs` | 列出已有 refined 投影的 epoch（时间轴绿环 C2）|
| `POST /updateProjection` | 取某 epoch 投影（`refine_flag` 选 refined/baseline）|
| `POST /getProjectionNeighbors` | 取投影近邻（支持传入 projection_data 直接算）|

---

## 8. 磁盘数据布局（`content_path` 下）

```
epochs/epoch_{E}/
    embeddings.npy            # [N, D] 高维特征（get_representation 读）
    hd_neighbors_{k}.json     # 每点 HD top-k 邻居（惰性生成，参数化 k）
    predictions.npy, index.npy ...
visualize/{method}_{id}/epochs/epoch_{E}/
    projection.npy            # [N, 2] baseline 投影（refine 从不覆盖它）
    vis_model.pth             # encoder/decoder checkpoint（refine 每次重载）
visualize/{method}_{id}_refined/epochs/epoch_{E}/
    projection.npy            # refine 结果；缺失时 load_projection 回落 baseline
```

---

## 9. 前端 store 关键字段（`state.unified.ts`）

`selectedIndices` / `secondaryIndices`（选点）、`allEpochData[epoch].{projection, originalProjection, indexList, originalNeighbors, projectionNeighbors}`、`refineMetrics`、`refineStatus`、`refineProgress`、`neighborDisplayIndices`（画谁的邻居环）、`refineTopK`、`refinePriority`、`showPreRefine`（before/after）、`refinedEpochs`（C2）。

setter 自动生成：字段 `foo` → `setFoo`；或通用 `setValue('foo', v)`。

---

## 10. 坑位 / 必知事实

1. **改后端 .py 必须重启**（gunicorn preload，无 reload）；JSON 配置热读。
2. **`build_focus_set` 会把单个 seed 扩展成 ~16 点** → 用户"单击"到 refine 里 focus_indices 不止 1 个；靠 `_split_focus_targets`(Option A) 才还原成单 focus。
3. **NP 有平移/镜像规范自由度** → 优化器会"逃逸到空白"或"翻面"作弊 → 已用 `L_pin`/`L_dir` 打破。理解新问题时先想"是不是又一个规范自由度"。
4. **`load_projection(refine_flag=True)` 优雅回落**：refined 缺失自动读 baseline → Undo 靠删文件即可。
5. **`get_representation` 无缓存**（每次 np.load）；A1 在 refine 入口加了 per-epoch mtime 缓存（`_get_refine_epoch_data`）。
6. **baseline `projection.npy` 从不被 refine 覆盖**（refined 写到 `_refined/`）→ 同 session 反复 refine 缓存/before-after 一直有效。
7. **overlay 邻居环的 k 目前固定 10**（chart.tsx），与 refine 的可调 k 尚未完全对齐（待办）。
8. **单点 100% 是否可达 = 数据在 2D 的可嵌入性**；达不到时 D2 报 `rank_cap` / "non-planar neighborhood"，这是诚实结果不是 bug。

---

## 11. 当前状态

- **算法**：损失重构完成（L_shape/L_rank/L_pin/L_dir + escalation + Option A），已验证：单点在原位、保方向、忠实距离地拉近邻居，不坍塌/不逃逸/不翻面/少空洞。
- **交互**：10 项 UX 路线图全部落地（D1/A1/D2/A2/B2/C3/B1/B3/C1/C2）。
- **编译**：Python `py_compile` + 前端 `tsc --noEmit` 均零错误。
- **验证**：用户已多轮真机验证通过。

## 12. 常用调参入口（若效果需微调）

- 邻居拉不够近/够远 → `_s_shape` 标定、`_K_CTX`
- 仍逃逸 → `_W_PIN`↑
- 仍翻面 → `_W_DIR`↑（3→5~8）；太僵 → ↓
- 空洞明显 → 降 `_W_RANK` 或 escalation 激进度
- 精度↔保形整体偏好 → 前端 Refine goal（B1 priority）
