# TTAV Refine: Algorithm, Implementation, Testing, and Results

更新时间：2026-05-30

本文档完整记录 TTAV（Time Travelling Visualizer）中围绕 `refine` 功能所做的算法设计、代码实现、测试方案、实验结果与当前默认结论。目标是让后续继续做代码修改、性能比较、论文整理时，都可以直接以本文档为入口。

---

## 1. 背景与目标

TTAV 原有的 refine 机制更接近“局部 patch”：

- 用户选中少量点后触发 refine
- 后端基于当前 `TimeVis` / `DVI` 策略做一次局部微调
- 结果写入 `_refined` 目录
- 前端重新读取 refine 后的投影与邻居

这种方式已经能做局部修正，但还不够系统。主要问题是：

1. 用户实际关注的是“局部区域”，不是只有几个 seed 点。
2. 如果直接用 refine 结果替换整张图，容易破坏全局稳定性。
3. 原有实现没有把“全局模型”和“局部模型”分离，局部训练逻辑不够清晰。
4. 缺少一套系统化 benchmark，难以回答“新方法是否真的提升了”。

因此，本轮工作的目标是把 refine 升级成一套更清晰的局部优化系统：

- 在 `baseline global visualizer` 基础上，以用户选中的 `seeds` 和 `zoom-in focus area` 为条件，快速训练一个 `local visualizer`
- 随后按与 `focus area` 的距离对 `old/new` 结果做连续加权融合
- 在保证全局稳定的前提下，提高局部结构表达精度
- 建立完整的测试脚本、结果目录与 benchmark workflow

---

## 2. 最终算法思路

### 2.1 核心设计

最终的 refine 方案包含三层：

1. `global visualizer`
- 即原始训练好的 `TimeVis` 主模型
- 负责整张图的稳定骨架
- 提供 `baseline projection`
- 作为局部模型初始化权重来源

2. `local visualizer`
- 在 refine 时从 `global visualizer` 复制一份副本
- 只针对当前局部区域快速训练
- 负责给出更精细的局部结果 `local projection`

3. `blended projection`
- 不直接整张图替换为局部结果
- 而是对 `baseline projection` 和 `local/refined projection` 做连续融合
- 在 `focus area` 附近更多使用新结果，远离时更多使用旧结果

### 2.2 关键词定义

- `seeds`
  - 用户手动选中的点

- `bbox`
  - `bounding box`
  - 当前 zoom-in 局部视野的二维矩形范围
  - 用于描述用户当前正在看的局部区域

- `focus area`
  - 用户真正关注的区域
  - 不是只由 `seeds` 决定，也不是只由 `bbox` 决定

- `focus_set`
  - 用户当前关注区域的点集合

- `training_context`
  - 局部训练真正使用的上下文集合
  - 当前实现中由 `focus_set` 与其 high-D 正邻居共同构成

- `patch_set`
  - 当前 refine 会写回 `_refined/projection.npy` 的点集合
  - 当前实现中比 `training_context` 更大，会额外包含一圈 patch-support neighbors

- `baseline projection`
  - 原始全局模型输出的二维投影

- `local projection`
  - 局部模型输出的二维投影

- `blended projection`
  - `baseline projection` 与 `local/refined projection` 融合后的最终显示结果

### 2.3 第一版采用的具体策略

当前落地的第一版采用：

- 只支持 `TimeVis`
- `focus_set` 支持多种候选定义，但当前运行时默认值已收敛为 `seeds_plus_hd`
- `bbox` 仍作为 runtime blended projection 的重要条件，而不是默认被硬并入训练集合
- `blended projection` 只在运行时内存中生成，不单独落盘
- 用户暂时不能调 `blend_decay`
- 第一版不做 `instability audit`

---

## 3. 分阶段算法设计与实现路线

整个实现不是一步到位，而是分阶段验证。

### 3.1 Phase 1: `bbox + seeds + high-D neighbors` 定义 `focus_set`

目的：先验证“怎么定义局部区域”这个问题。

第一版候选定义：

- `seeds_only`
- `seeds_plus_hd`
- `seeds_plus_bbox`
- `seeds_plus_bbox_plus_hd`

其中：

`focus_set = seeds ∪ bbox内所有点 ∪ seeds的高维近邻`

设计原因：

- 只用 `seeds` 太少
- 只用 `bbox` 会把二维上靠近但高维不相关的点也纳入
- `high-D neighbors` 能补高维语义相似性
- 两者联合更接近用户真实意图

### 3.2 Phase 1b: 用真实 refine 对比不同 `focus_set`

目的：不是只看集合大小，而是真正跑 refine 比较效果。

对每个 candidate：

- 克隆一份临时 `TimeVis` 会话
- 真实执行 refine
- 比较局部指标、全局漂移和耗时

这一步帮助回答：

- 哪种 `focus_set` 定义更适合当前数据集
- 引入 `bbox` 到底是增益还是噪声

### 3.3 Phase 2: `bbox-only blending`

目的：先验证 blended projection 是否比 direct replace 更好。

设计：

- 对每个点计算它到 `bbox` 的距离
- 距离越近，越多使用 refined 结果
- 距离越远，越多回退到 baseline

这是最保守的融合方式。

### 3.4 Phase 2b: `bbox + focus_set` focus-aware blending

目的：解决 `bbox-only` 太保守的问题。

观察发现：

- 当 `focus_set` 明显超出 `bbox` 时
- 单纯按 `bbox` 融合会把很多局部收益压掉

因此升级为：

- 同时考虑 `bbox` 距离和 `focus_set` 距离
- 取两者中更大的局部权重

这样可以：

- 保住更多局部变化
- 同时把全局漂移控制在远低于 direct replace 的范围内

### 3.5 Phase 2.5: blended view 的邻居与前端指标对齐

目的：让前端显示的结果和评价指标尽量一致。

问题：

- 如果前端显示的是 blended projection
- 但邻居、NP、Trustworthiness 仍来自 refined projection
- 那么显示和指标不一致

因此补做：

- 后端支持对 runtime blended projection 计算 projection neighbors
- 前端 refine 后优先使用 blended 对应的 neighbors
- 后端新增对当前 blended projection 统一计算 `NP / MRH / Trustworthiness / Continuity` 的路径
- 前端只保留位移类指标（`focus shift / global drift`）的本地计算，结构质量指标统一以后端 blended-view 结果为准

### 3.6 Phase 3: 显式 `local visualizer`

目的：让 refine 真正变成“全局模型 + 局部模型”的双模型流程。

原始 TimeVis refine 的问题：

- 更像是临时修改全局模型，再恢复权重
- 模型职责边界不清晰

现在改成：

- `global visualizer = self.visualize_model`
- refine 开始时 `deepcopy` 一份，得到 `local visualizer`
- 只在 `local visualizer` 上训练
- patch 当前 epoch 和其他 epoch 时都用 `local visualizer`
- 全局模型不再被 refine 过程污染

这一步是整个方案结构上最关键的一次升级。

---

## 4. 每一步借鉴了什么思想

本方案不是直接照抄某一篇论文，而是组合多条近年思路：

### 4.1 `focus_set` 联合构建
借鉴思想：

- 用户交互区域不应只由点级约束决定，也要吸收局部视野信息
- 与层级探索和 details-on-demand 类工作思路一致

实际落地：

- `seeds ∪ bbox内点 ∪ high-D neighbors(seeds)`

### 4.2 `global visualizer + local visualizer`
借鉴思想：

- 来自流式/在线 DR 工作中的 global/updating dual-model 思路
- 本地化快速更新，不立即污染全局主模型

实际落地：

- refine 时复制 `TimeVis` 模型副本作为 local visualizer

### 4.3 `baseline + local/refined -> blended projection`
借鉴思想：

- 局部增强与全局稳定需要同时优化
- 可以先在训练目标层做 local/global 平衡，也可以先在显示层做连续融合

第一版实际落地：

- 先用显示层的距离加权融合验证价值
- 暂时不把 blend 写成训练正则项

### 4.4 `bbox + focus_set` 联合加权
借鉴思想：

- 仅靠用户视窗区域不够，局部语义相关点也应该被照顾
- 所以融合权重不仅看 `bbox`，还看 `focus_set`

### 4.5 benchmark 三层评价
借鉴思想：

- 不能只看局部结构指标
- 还要同时看全局漂移和交互耗时

因此统一采用三类指标：

- 局部质量
- 全局稳定
- 交互成本

---

## 5. 具体代码修改了什么

下面按模块整理当前主要改动。

## 5.1 前端状态与视图

### [web/src/state/state.unified.ts](/home/yilu/workspace/time-travelling-visualizer/web/src/state/state.unified.ts:1)
新增：

- `ViewportBBox` 类型
- `currentViewportBBox`
- `focusIndices`
- `refineMetrics`

作用：

- 保存当前用户视野 `bbox`
- 保存 refine 后真正使用的 `focus_set`
- 保存 refine 质量指标

### [web/src/component/chart.tsx](/home/yilu/workspace/time-travelling-visualizer/web/src/component/chart.tsx:1)
新增/修改：

- 根据当前 viewport 计算 `bbox`
- 将 `bbox` 写入全局状态
- focus mode 时高亮 `focusIndices`

作用：

- 让后端知道用户当前 zoom-in 的局部区域

### [web/src/views/plotView.tsx](/home/yilu/workspace/time-travelling-visualizer/web/src/views/plotView.tsx:1)
这是 refine 前端主逻辑的核心文件。

新增/修改包括：

1. `buildBlendedProjection(...)`
- 先做 `bbox-only` blending
- 后升级成 `bbox + focus_set` focus-aware blending

2. refine 请求链路
- refine 时把 `currentViewportBBox` 发给后端
- 后端返回 `focus_indices`
- 前端再基于 `focus_indices` 计算 blended projection

3. blended neighbors
- refine 后请求 runtime blended projection 对应的 neighbors

4. refine metrics
- 前端只保留位移类指标（`focus shift / global drift`）的本地计算
- `NP / MRH / Trustworthiness / Continuity` 由后端按当前 blended projection 统一计算

5. bug 修复
- 之前存在“先请求 blended neighbors，后解析 focusIndices”的顺序问题
- 现已修正为先拿 `focusIndices`，再请求 blended neighbors

### [web/src/communication/backend.ts](/home/yilu/workspace/time-travelling-visualizer/web/src/communication/backend.ts:1)
新增/修改：

- `updateFocusContext(...)` 支持 `zoom_bbox`
- `getProjectionNeighbors(...)` 支持：
  - `blend_bbox`
  - `blend_focus_indices`
  - `blend_decay_ratio`

作用：

- 前后端 refine 参数打通
- 支持运行时 blended neighbors

### [web/src/config/refine.ts](/home/yilu/workspace/time-travelling-visualizer/web/src/config/refine.ts:1)
新增：

- 前端 refine 默认配置
- 当前包含 `blendDecayRatio = 0.35`

作用：

- 避免默认值散落在前端多个文件里

---

## 5.2 后端服务与运行时工具

### [tool/server/server.py](/home/yilu/workspace/time-travelling-visualizer/tool/server/server.py:1)
这是 refine 后端入口的核心文件。

新增/修改：

1. `/updateFocusContext`
- 接收 `zoom_bbox`
- 对 `TimeVis` 使用 `build_focus_set(...)`
- 构造 `focus_indices`
- 返回：
  - `focus_set_size`
  - `focus_seed_count`
  - `focus_bbox_count`
  - `focus_hd_neighbor_count`
  - `focus_indices`

2. refine 默认值收敛
- 从 `refine_runtime_config.py` 读取 `focus_hd_k` 和 `blend_decay_ratio`
- 新增 `focus_set_strategy`
- `TimeVis` 的 `hd_k` 可从 `vis_config.refine_hd_k` 覆盖

3. `/getProjectionNeighbors`
- 支持 runtime blended projection 路径
- 若请求里带 `blend_bbox` / `blend_focus_indices`
- 则先构建 blended projection，再计算 neighbors

4. `/getRefineMetrics`
- 支持对 runtime blended projection 统一计算：
  - `neighbor_preservation`
  - `mean_rank_hd`
  - `trustworthiness`
  - `continuity`

### [tool/server/server_utils.py](/home/yilu/workspace/time-travelling-visualizer/tool/server/server_utils.py:1)
这是 refine 支撑逻辑最集中的工具文件。

新增/修改：

1. `load_raw_projection_array(...)`
- 读取原始 `projection.npy`
- 不做 train/test 重排
- 用于 bbox 点筛选

2. `build_focus_set(...)`
- 支持：
  - `seeds_only`
  - `seeds_plus_hd`
  - `seeds_plus_bbox`
  - `seeds_plus_bbox_plus_hd`
- 当前运行时默认值为 `seeds_plus_hd`
- 返回 `focus_indices` 和统计信息

3. `build_runtime_blended_projection(...)`
- 运行时生成 blended projection
- 当前支持 `bbox + focus_indices` 联合加权

4. 任意 projection 的 neighbors
- 新增基于任意二维投影直接计算 neighbors 的 helper

5. refine metrics helper
- 新增基于任意 runtime projection 统一计算 `NP / MRH / Trustworthiness / Continuity` 的 helper
- 用于保证 blended view 与结构质量指标的定义一致

6. `faiss -> sklearn` 回退
- 若环境无 `faiss`
- 自动回退到 `sklearn.NearestNeighbors`

作用：

- 支撑 focus-aware blending
- 降低环境依赖风险

### [tool/server/refine_runtime_config.py](/home/yilu/workspace/time-travelling-visualizer/tool/server/refine_runtime_config.py:1)
新增：

- 后端 refine 默认配置
- 当前包含：
  - `focus_hd_k = 15`
  - `blend_decay_ratio = 0.35`
  - `focus_mode = balanced`
  - `focus_set_strategy = seeds_plus_hd`

作用：

- 集中管理 refine 运行时默认值

---

## 5.3 TimeVis 策略层

### [tool/visualize/strategy/timevis_strategy.py](/home/yilu/workspace/time-travelling-visualizer/tool/visualize/strategy/timevis_strategy.py:1)
这是算法结构上最重要的一处修改。

改动重点：

1. `refine(...)` 现在显式创建 `local_visualizer`
- `local_visualizer = copy.deepcopy(self.visualize_model)`

2. 只训练 `local_visualizer`
- refine 的局部训练不再直接修改全局模型

3. 集合语义显式拆分
- `focus_indices`
  - 用户关注区域 / 指标评估 / blending 中心
- `training_context_indices`
  - `focus_indices ∪ high-D positive neighbors`
  - 局部训练真正使用的核心上下文
- `patch_indices`
  - `training_context_indices ∪ patch-support neighbors`
  - `_refined/projection.npy` 真正更新的区域

4. `_patch_epoch(...)`
- 支持传入 `model`
- 只对 `patch_indices` 做 subset encoder 推理并写回 baseline projection
- 不再在 refine 路径中对整张图做 full encoder inference fallback

5. `patch_other_epochs(...)`
- 使用 `self._last_local_visualizer`
- 并输出逐 epoch 的后台 patch 耗时日志

6. refine 分段耗时日志
- 当前会输出：
  - `prepare_baseline`
  - `prepare_context`
  - `train`
  - `metrics`
  - `patch`
  - `total`

当前效果：

- 全局模型 refine 前后保持不变
- 局部模型能学到不同的局部嵌入
- 当前 epoch 的 refined projection 只更新 `patch_set`
- refine 结束后可直接看到各阶段耗时

这正是“全局稳、局部动”的双模型机制。

---

## 6. 测试是怎么设计的

整个测试体系放在：

- [tests](/home/yilu/workspace/time-travelling-visualizer/tests)

设计原则是：

- 每一阶段验证一个关键问题
- 所有结果都落盘
- 输出 `json/csv/png/md`
- 格式尽量接近论文里常见的数值表和图

### 6.1 Phase 1: focus_set 构造对比
文件：

- [tests/test_refine_phase1_focus_set.py](/home/yilu/workspace/time-travelling-visualizer/tests/test_refine_phase1_focus_set.py)

回答的问题：

- `bbox + seeds + hd-neighbors` 是否能更合理地定义局部区域

比较对象：

- `seeds_only`
- `seeds_plus_hd`
- `seeds_plus_bbox`
- `seeds_plus_bbox_plus_hd`

输出：

- focus_set 组成统计
- `json`
- `csv`
- 可视化图

### 6.2 Phase 1b: 不同 focus_set 的真实 refine 对比
文件：

- [tests/test_refine_phase1b_refine_compare.py](/home/yilu/workspace/time-travelling-visualizer/tests/test_refine_phase1b_refine_compare.py)

回答的问题：

- 哪种 `focus_set` 定义真正带来更好的 refine 效果

比较指标：

- `Neighbor Preservation (NP)`
- `Mean Rank of HD neighbors (MRH)`
- `Trustworthiness`
- `Continuity`
- `Focus Shift`
- `Global Drift`
- `Latency`

### 6.3 Phase 2: bbox-only blending
文件：

- [tests/test_refine_phase2_blending.py](/home/yilu/workspace/time-travelling-visualizer/tests/test_refine_phase2_blending.py)

回答的问题：

- blended projection 是否比 direct replace 更稳

比较对象：

- `direct replace`
- `bbox-only blended`

看什么：

- `focus_shift`
- `global_drift`

### 6.4 Phase 2b: focus-aware blending
文件：

- [tests/test_refine_phase2b_focus_aware_blending.py](/home/yilu/workspace/time-travelling-visualizer/tests/test_refine_phase2b_focus_aware_blending.py)

回答的问题：

- `bbox + focus_set` 是否优于 `bbox-only`

比较对象：

- `bbox-only`
- `bbox + focus_set`

### 6.5 Phase 3: local visualizer 检查
文件：

- [tests/test_refine_phase3_local_visualizer.py](/home/yilu/workspace/time-travelling-visualizer/tests/test_refine_phase3_local_visualizer.py)

回答的问题：

- 全局模型是否被局部 refine 污染
- 局部模型是否真的学到了不同的局部表示

检查指标：

- `global_model_unchanged_max_abs`
- `local_vs_global_embed_max_abs`

### 6.6 汇总与一键 workflow
文件：

- [tests/build_refine_benchmark_summary.py](/home/yilu/workspace/time-travelling-visualizer/tests/build_refine_benchmark_summary.py)
- [tests/run_refine_benchmark_suite.py](/home/yilu/workspace/time-travelling-visualizer/tests/run_refine_benchmark_suite.py)
- [tests/refine_benchmark_config.example.json](/home/yilu/workspace/time-travelling-visualizer/tests/refine_benchmark_config.example.json)

作用：

- 自动读取各阶段最新结果
- 生成统一 benchmark summary
- 一条命令跑完整套实验

---

## 7. 测试结果保存在哪里

统一保存在：

- [refine_validation_results](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results)

当前目录结构包括：

- [phase1_focus_set](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase1_focus_set)
- [phase1b_refine_compare](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase1b_refine_compare)
- [phase2_blending](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase2_blending)
- [phase2b_focus_aware_blending](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase2b_focus_aware_blending)
- [phase3_local_visualizer](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase3_local_visualizer)
- [benchmark_summary](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/benchmark_summary)

常见文件类型：

- `metrics_*.json`
- `summary_*.csv`
- `*.png`
- `refine_benchmark_summary_*.md`

---

## 8. 当前真实实验用了什么数据

当前已经跑过的主实验使用：

- 数据集：`/home/yilu/workspace/Dataset/backdoor`
- baseline session：`TimeVis_1`
- epoch：`10`
- seeds：`[10975, 11490, 17685]`
- bbox：
  - `x_min = 3.577545`
  - `x_max = 3.653369`
  - `y_min = 1.164337`
  - `y_max = 1.236651`

benchmark 配置模板见：

- [tests/refine_benchmark_config.example.json](/home/yilu/workspace/time-travelling-visualizer/tests/refine_benchmark_config.example.json)

---

## 9. 当前已经产出的关键结果文件

### 9.1 Phase 1
- [summary_20260530_093731.csv](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase1_focus_set/summary_20260530_093731.csv)
- [metrics_20260530_093731.json](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase1_focus_set/metrics_20260530_093731.json)
- [focus_set_comparison_20260530_093731.png](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase1_focus_set/focus_set_comparison_20260530_093731.png)

### 9.2 Phase 1b
- [summary_20260530_095238.csv](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase1b_refine_compare/summary_20260530_095238.csv)
- [metrics_20260530_095238.json](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase1b_refine_compare/metrics_20260530_095238.json)
- [metrics_20260530_095238.png](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase1b_refine_compare/metrics_20260530_095238.png)
- [refined_scatter_20260530_095238.png](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase1b_refine_compare/refined_scatter_20260530_095238.png)

### 9.3 Phase 2
- [summary_20260530_100118.csv](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase2_blending/summary_20260530_100118.csv)
- [metrics_20260530_100118.json](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase2_blending/metrics_20260530_100118.json)
- [metrics_20260530_100118.png](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase2_blending/metrics_20260530_100118.png)
- [scatter_20260530_100118.png](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase2_blending/scatter_20260530_100118.png)

### 9.4 Phase 2b
- [summary_20260530_103218.csv](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase2b_focus_aware_blending/summary_20260530_103218.csv)
- [metrics_20260530_103218.json](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase2b_focus_aware_blending/metrics_20260530_103218.json)
- [metrics_20260530_103218.png](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase2b_focus_aware_blending/metrics_20260530_103218.png)

### 9.5 Phase 3
- [local_visualizer_check_20260530_101636.json](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/phase3_local_visualizer/local_visualizer_check_20260530_101636.json)

### 9.6 汇总结果
- [refine_benchmark_summary_20260530_105119.md](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/benchmark_summary/refine_benchmark_summary_20260530_105119.md)
- [refine_benchmark_summary_20260530_105119.csv](/home/yilu/workspace/time-travelling-visualizer/refine_validation_results/benchmark_summary/refine_benchmark_summary_20260530_105119.csv)

---

## 10. 当前实验结果说明了什么

### 10.1 Phase 1: `bbox` 与 `high-D neighbors` 确实补充了不同信息

当前结果显示：

- `seeds_only`: `focus_set_size = 3`
- `seeds_plus_hd`: `focus_set_size = 48`
- `seeds_plus_bbox`: `focus_set_size = 44`
- `seeds_plus_bbox_plus_hd`: `focus_set_size = 75`

这说明：

- `bbox` 和 `high-D neighbors` 不是重复信息
- 联合后确实能得到更完整的局部候选区域

### 10.2 Phase 1b: 对当前数据集，`seeds_plus_hd` 是更好的默认候选

关键结果：

- `seeds_plus_hd`
  - `NP = 9.38`
  - `MRH = 175.8`
  - `Trustworthiness = 32.99`
  - `Continuity = 45.85`
  - `global_drift = 0.0136`
  - `latency = 63.09s`

对比看：

- 直接把 `bbox` 内所有点都硬塞进 `focus_set`，会拉低局部结构指标
- 所以当前数据上，`bbox` 更适合作为意图信号，不适合被一视同仁地硬纳入训练

### 10.3 Phase 2: `bbox-only blending` 非常稳，但偏保守

例如：

- `seeds_plus_hd`: `global_drift 0.01365 -> 0.00019`
- `seeds_plus_bbox_plus_hd`: `global_drift 0.01087 -> 0.00016`

说明：

- `bbox-only blend` 极大降低了全局漂移
- 但也会明显压缩局部变化，尤其是当 `focus_set` 超出 `bbox` 时

### 10.4 Phase 2b: `bbox + focus_set` 是更合理的中间方案

例如：

- `seeds_plus_hd`
  - `bbox-only`: `focus_shift = 0.0164`, `global_drift = 0.00019`
  - `bbox+focus`: `focus_shift = 0.0481`, `global_drift = 0.00221`

- `seeds_plus_bbox_plus_hd`
  - `bbox-only`: `focus_shift = 0.0225`, `global_drift = 0.00016`
  - `bbox+focus`: `focus_shift = 0.0415`, `global_drift = 0.00188`

说明：

- `bbox+focus_set` 明显保住了更多局部变化
- 全局漂移虽然比 `bbox-only` 略大，但仍远小于 direct replace

因此：

- `bbox-only` 太保守
- `direct replace` 太激进
- `bbox + focus_set` 是当前更平衡的方案

### 10.5 Phase 3: 双模型机制已经真正成立

关键检查结果：

- `global_model_unchanged_max_abs = 0.0`
- `local_vs_global_embed_max_abs = 0.0424`

说明：

- refine 之后，全局模型完全没被污染
- 局部模型确实学到了不同的局部表示

这证明当前的 `local visualizer` 不是表面命名，而是真正成立的结构。

---

## 11. 现在如何评估“新训练是否提升了”

答案是：

**可以用已有的 `NP / Trustworthiness / Continuity / MRH`，但不能只看它们。**

当前统一采用三层评价：

### 11.1 局部质量

- `NP`
- `MRH`
- `Trustworthiness`
- `Continuity`

说明：

- 当前产品运行时会优先使用“后端对当前 blended projection 的统一计算结果”
- 不再混用前端 top-k 截断近似的 `Trustworthiness / Continuity`

回答的问题：

- 局部结构有没有更准确
- 高维邻域是否更好地映射到了二维

### 11.2 全局稳定性

- `Global Drift`
- 非 focus 区域平均位移

回答的问题：

- 为了修局部，是否把整张图带偏了

### 11.3 交互成本

- `Latency`

回答的问题：

- 局部训练后，交互还能不能接受

### 11.4 结论标准

一个新方法只有同时满足下面三条，才算真正提升：

1. 局部指标变好
2. 全局 drift 没失控
3. latency 仍可接受

所以不能只说：

- `NP` 升了，所以更好

而应该说：

- `NP / MRH / Trustworthiness / Continuity` 有改善
- `Global Drift` 没明显失控
- 交互延迟仍在可接受范围内

---

## 12. 当前默认推荐配置

基于当前 `backdoor` 数据集实验，推荐：

- `focus_set` 默认推荐：`seeds_plus_hd`
- blending 默认推荐：`bbox + focus_set`
- local training 默认推荐：显式 `local visualizer` 副本

理由：

1. `seeds_plus_hd` 在非 trivial 候选里 direct local quality 最强
2. `bbox + focus_set` 比 `bbox-only` 更能保住局部收益
3. `bbox + focus_set` 的全局漂移仍显著低于 direct replace
4. `local visualizer` 已验证不会污染 `global visualizer`

当前运行时默认值：

- 后端默认 `focus_hd_k = 15`
- 默认 `blend_decay_ratio = 0.35`
- 默认 `focus_mode = balanced`
- 默认 `focus_set_strategy = seeds_plus_hd`

---

## 13. 如何一键复跑 benchmark

参考：

- [REFINE_BENCHMARK_WORKFLOW.md](/home/yilu/workspace/time-travelling-visualizer/refine_docs/REFINE_BENCHMARK_WORKFLOW.md)

配置文件：

- [tests/refine_benchmark_config.example.json](/home/yilu/workspace/time-travelling-visualizer/tests/refine_benchmark_config.example.json)

命令：

```bash
python3 tests/run_refine_benchmark_suite.py \
  --config tests/refine_benchmark_config.example.json
```

可选跳过：

```bash
python3 tests/run_refine_benchmark_suite.py \
  --config tests/refine_benchmark_config.example.json \
  --skip-phase1b \
  --skip-phase3
```

---

## 14. 当前已知限制

1. 第一版只支持 `TimeVis`
- `DVI` 还没有接入这套双模型 refine 方案

2. `blend_decay` 还未开放给前端用户调节
- 当前固定为默认值 `0.35`

3. `blended projection` 只在运行时内存中生成
- 没有作为产品运行时结果落盘
- 这是有意设计，因为不同局部区域的 blended 结果是 query-specific 的

4. 当前 refine 的 `_refined/projection.npy` 采用 subset patch 语义
- 它不是 `local visualizer` 对整张图的 full local projection
- 最终展示语义仍以 runtime blended projection 为准

5. 第一版不做 `instability audit`
- 目前没有给每个点额外估计 refine 结果稳定性

6. 前端构建环境未完全校验
- 当前本地 `web` 目录缺少 `node_modules`
- `pnpm build` 失败的原因是 `tsc: not found`
- 目前无法完成完整前端打包验证

7. 当前推荐结论主要来自 `backdoor` 数据集
- 需要继续在更多数据集上验证稳定性

---

## 15. 后续最自然的继续方向

1. 把 refine 默认参数继续收敛成正式产品配置
- 例如从 `vis_config` 或插件设置面板统一下发

2. 在更多数据集上复跑 benchmark
- 验证 `seeds_plus_hd + bbox+focus blend + local visualizer` 是否稳定成立

3. 逐步把 blend 从“显示层后处理”扩展到“训练目标层约束”
- 例如显式加入 `local/global/stability` 多目标

4. 后续再考虑加入 `instability audit`
- 让 blending 权重与局部结果稳定性联动

---

## 16. 配套文档索引

如果要进一步看设计和调研，可继续参考：

- [REFINE_OPTIMIZATION_NOTES.md](/home/yilu/workspace/time-travelling-visualizer/refine_docs/REFINE_OPTIMIZATION_NOTES.md)
- [REFINE_LITERATURE_REVIEW.md](/home/yilu/workspace/time-travelling-visualizer/refine_docs/REFINE_LITERATURE_REVIEW.md)
- [LOCAL_VISUALIZER_ALGORITHM_PLAN.md](/home/yilu/workspace/time-travelling-visualizer/refine_docs/LOCAL_VISUALIZER_ALGORITHM_PLAN.md)
- [REFINE_VALIDATION_PROTOCOL.md](/home/yilu/workspace/time-travelling-visualizer/refine_docs/REFINE_VALIDATION_PROTOCOL.md)
- [REFINE_BENCHMARK_WORKFLOW.md](/home/yilu/workspace/time-travelling-visualizer/refine_docs/REFINE_BENCHMARK_WORKFLOW.md)
