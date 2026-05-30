# Refine Optimization Notes

更新时间：2026-05-29

本文档整理 `time-travelling-visualizer` 当前项目架构、现有 refine 实现、下一阶段关于局部 refine 的优化目标，以及已核验的相关文献线索，供后续继续做代码设计和实现时参考。

## 1. 项目架构概览

项目目前是一个三层结构：

- `extension/`
  - VS Code 扩展侧入口。
  - 负责读取工作区配置、拼装可视化启动参数、向 webview 发送 `startVisualizing / loadVisualization / syncSession` 消息。
- `web/`
  - React 前端。
  - 负责加载 epoch 数据、展示投影图、接收用户选点、触发 refine、展示 refine 质量指标。
- `tool/`
  - Python 后端。
  - 负责训练 visualizer、生成投影、维护 refine session、写回 refined 结果。

关键代码入口：

- `tool/server/run_visualization.py`
  - `initialize_config(...)`：构建统一配置，推导默认超参数。
  - `init_visualize_component(...)`：组装 `DataProvider + Projector + Strategy + ResultGenerator`。
  - `visualize_run(...)`：首次启动时训练 visualizer 并导出全部 epoch 的 projection/background。
- `tool/server/server.py`
  - `/syncSession`：前端 load 后同步后端 session。
  - `/updateFocusContext`：前端 refine 主入口。
- `web/src/views/plotView.tsx`
  - 加载单个 epoch 数据。
  - 调用 refine 接口。
  - 拉取 refined projection 与 refined low-D neighbors。
  - 展示 refine quality。

## 2. 当前 refine 的实际工作流

### 2.1 前端交互流

当前 refine 交互是：

1. 用户在前端选中一批点。
2. 在 `Precision Control` 中选择 `coarse / balanced / fine`。
3. 点击 `Update Projection`。
4. 前端调用 `/updateFocusContext`。
5. 后端完成局部 refine。
6. 前端重新拉取当前 epoch 的 refined projection 和 refined neighbor。
7. 前端计算与展示位移和邻域质量指标。

当前 UI 位置：

- `web/src/component/function-panel.tsx`
  - 已有 `Focus Mode` 三档。
  - 目前文案是：
    - `coarse`: Visual highlight only.
    - `balanced`: Increase sampling weight.
    - `fine`: Enable LoRA local adaptation.

说明：

- 这三档已经存在于 UI 和 store 中。
- 但对 `DVI / TimeVis` 而言，后端目前并没有真正把这三档变成清晰、稳定、可控的损失权重或区域权重机制。

### 2.2 后端 refine 流

`server.py` 中 `/updateFocusContext` 的流程是：

1. 从 active session 中取出当前 strategy 和 visualizer。
2. 根据选中点构建 focus mask。
3. 调用 `strategy.update_ttav_context(...)` 记录当前上下文。
4. 分方法执行 refine：
   - `DynaVis`：走 `refine_train(...)`，然后全量重生成结果。
   - `DVI / TimeVis`：走 `strategy.refine(...)`。
5. refine 完成后：
   - 清理当前 epoch refined neighbor cache。
   - 将 refined projection 写到 `_refined` 目录。
   - 后台 patch 其他 epoch。
6. 返回 refine 指标：
   - `neighbor_preservation`
   - `mean_rank_hd`
   - `trustworthiness`
   - `continuity`

### 2.3 refined 结果的存储策略

当前 refined 结果不是覆盖原始结果，而是写入并优先读取：

- 原始投影：
  - `visualize/<vis_method>_<vis_id>/epochs/epoch_x/projection.npy`
- refined 投影：
  - `visualize/<vis_method>_<vis_id>_refined/epochs/epoch_x/projection.npy`

读取逻辑：

- 当前端请求 `refine_flag=true` 时，后端优先读 `_refined`。
- 如果 `_refined` 还没准备好，则自动回退到原始投影。

这个设计非常适合后续继续扩展，因为它天然支持：

- 保留原始全局结果；
- 单独维护局部 refine 结果；
- 在前端做新旧结果对比或融合。

## 3. 当前 DVI / TimeVis refine 的真实实现特点

### 3.1 TimeVis refine

`tool/visualize/strategy/timevis_strategy.py` 中的 refine 目前更接近：

- 以旧 projection 为 baseline；
- 自动收集 focus 点的局部邻居；
- 采样 anchor points 约束远处区域稳定；
- 采样 negatives 做局部排斥；
- 只微调 visualizer encoder 的最后两层；
- 目标是让 focus 点与其 high-D 邻居在 2D 中更接近；
- 优化结束后只 patch 局部点，不全量覆盖整张图；
- 随后恢复被微调的参数，保持全局主模型“干净”。

这说明当前 TimeVis refine 是：

- “局部临时优化 + 局部 patch 输出”
- 而不是“训练一个持久化的局部新模型”

### 3.2 DVI refine

`tool/visualize/strategy/dvi_strategy.py` 中的 refine 目前更轻量：

- 从当前 epoch 的 vis model checkpoint 出发；
- 构造局部点集；
- 使用轻量的 `SingleVisLoss` 进行极少步数的局部训练；
- 最后只 patch subset projection。

这同样说明当前 DVI refine 也是：

- “局部快速修补”
- 而不是“显式的新旧模型混合系统”

### 3.3 当前实现的核心优点

现有实现已经具备后续升级所需的几个关键基础：

- 有稳定的 session 机制。
- 有 refined 独立目录，不污染 baseline。
- 有 refined neighbors 独立缓存。
- 前端已经能展示 refine 前后质量差异。
- TimeVis 已经具备 anchor constraint 这种“保留远处稳定、强化局部变化”的雏形。

## 4. 我们下一阶段要做的 refine 优化目标

根据当前讨论，下一阶段 refine 的方向可以整理为下面几条。

### 4.1 从“选点后 patch”升级为“局部 focus-aware 优化”

目标不是单纯 patch 被选中的点，而是：

- 用户快速 zoom in 到某个局部区域后，系统把该区域视为 focus area；
- 在 focus area 内做更强的局部优化；
- 在 focus area 外保留旧结果和全局结构稳定性。

这意味着 refine 的触发条件应逐步从：

- “点选一批种子”

扩展为：

- “点选种子 + 局部视窗/局部区域”

### 4.2 老模型/老投影应作为 seed，而不是被替换

你们想做的是：

- 老模型结果成为二次优化的 seed；
- 新优化建立在旧结果上；
- 老结果保留；
- 新结果只在局部做强化。

这和当前 `_refined` 目录策略完全一致，后续可以进一步明确成两层资产：

- `global visualizer`
- `local focus visualizer`

### 4.3 需要显式的新旧加权关系

后续 refine 不能只是“局部点动了，别的点不动”，而应该支持：

- 新旧 projection 的加权融合；
- 新旧 visualizer 的输出加权融合；
- 权重随与 focus area 的距离变化；
- focus 区域权重大，远处区域权重小。

可以把这个目标理解为：

- 在结果层做 `projection_blend`
- 在模型层做 `visualizer_blend`
- 在训练层做 `loss_weighting`

三层不一定都同时做，但需要统一设计。

### 4.4 focus 程度需要变成真正可调的局部权重机制

当前 `focusMode` 只是一个 UI 档位概念。

后续应考虑把它落成更清晰的参数，例如：

- `seed_weight`
- `focus_radius`
- `distance_decay`
- `anchor_weight`
- `local_train_steps`
- `blend_alpha`

这样 `coarse / balanced / fine` 才能真正对应到：

- 关注区域大小不同；
- 新旧混合比例不同；
- 局部训练强度不同；
- 远处稳定性约束不同。

### 4.5 保留全局模型，同时训练局部新模型

你们的目标更像：

- 保留老 visualizer 作为全局模型；
- 针对当前 focus area 额外训练一个局部新 visualizer；
- 局部模型在 focus area 内拥有更大权重；
- focus area 外逐渐衰减回老模型输出。

这是和当前实现最大的区别：

- 当前：临时微调后 patch，再恢复主模型参数；
- 目标：把局部模型当成可复用的、可解释的对象保留下来。

## 5. 建议的方案表述

为了后续代码工作更清晰，建议把目标方案统一表述成下面这句话：

> 在 baseline global visualizer 的基础上，以用户选中的 seeds 和 zoom-in focus area 为条件，快速训练一个 local visualizer；随后按与 focus area 的距离对 old/new 结果做连续加权融合，从而在保证全局稳定的前提下，显著提高局部结构表达精度。

这个表述的好处是把几个核心点都包含了：

- baseline seed
- local training
- fast refine
- focus-aware weighting
- global stability
- local accuracy

## 6. 后续实现时建议拆成的几个子问题

后续做代码时，建议按下面几个问题逐项落地。

### 6.1 Focus area 如何定义

候选方式：

- 仅由选中 seeds 决定；
- 由当前 zoom 视窗决定；
- 由 seeds 的 high-D 邻居扩展；
- 由 seeds 的 low-D 邻居扩展；
- 混合定义。

建议优先做：

- `selected seeds + 当前 zoom bbox + high-D neighbors`

### 6.2 局部模型如何初始化

候选方式：

- 从 global visualizer 复制参数；
- 只复制 encoder；
- 只复制最后几层；
- 低秩适配；
- 单独小模型蒸馏。

基于现有代码，最自然的第一版是：

- 从 global visualizer 拷贝参数；
- 只允许最后几层在 local refine 中更新；
- 将 local visualizer 持久化保存。

### 6.3 新旧结果如何混合

候选方式：

- 按 low-D 距离衰减；
- 按 high-D 距离衰减；
- 按 anchor graph geodesic 衰减；
- 按用户手动设置区域权重。

建议优先做：

- 先做按 focus seeds 的 high-D 邻域距离衰减；
- 然后再考虑 low-D 几何修正。

### 6.4 前端交互如何升级

建议把 refine 从“单次按钮触发”逐步升级为：

- 快速 zoom in 后出现 focus 区域提示；
- 用户可调 focus strength；
- 用户可调 old/new blend；
- 用户可查看 baseline / refined / blended 三种视图。

## 7. 已核验的相关文献

下面只列出我已核验“真实存在”的工作，并标注与本项目的关系。

### 7.1 直接相关：项目当前方法来源

1. DeepVisualInsight: Time-Travelling Visualization for Spatio-Temporal Causality of Deep Classification Training
   - 来源：arXiv / AAAI 2022
   - 链接：https://arxiv.org/abs/2201.01155
   - 核验点：
     - 标题与作者信息存在。
     - 摘要明确说它是面向 deep classification training 的 time-travelling visualization。
     - 论文强调 spatio-temporal causality 和训练过程分析。
   - 与本项目关系：
     - 本项目 `DVI` 实现的直接方法来源。
     - 后续所有 refine 设计都应明确是否破坏 DVI 原始的空间/时间性质。

2. Temporality Spatialization: A Scalable and Faithful Time-Travelling Visualization for Deep Classifier Training
   - 来源：IJCAI 2022 Proceedings
   - 链接：https://www.ijcai.org/proceedings/2022/0558.pdf
   - 核验点：
     - 论文明确提出 `TimeVis`。
     - 摘要明确声称统一空间关系和时间关系，用单一 visualization model 处理跨 epoch 投影。
     - 文中明确说相较 DVI，效率更高。
   - 与本项目关系：
     - 本项目 `TimeVis` 实现的直接方法来源。
     - 如果后续要做“快速局部 refine”，TimeVis 是最合适的主战场之一。

### 7.2 高相关：可借鉴“参数化快速嵌入/在线投影”

3. Parametric UMAP embeddings for representation and semi-supervised learning
   - 来源：arXiv
   - 链接：https://arxiv.org/abs/2009.12981
   - 核验点：
     - 论文明确提出 parametric UMAP。
     - 摘要明确提到 learned parametric mapping，可用于 fast online embeddings for new data。
   - 与本项目关系：
     - 支持“老模型作为 seed、快速局部适配”的方向。
     - 可作为“局部 visualizer 是一个参数化映射”的理论参考。
   - 注意：
     - 它不是 focus-aware local refine 论文。
     - 它更像为“快速二次投影”提供方法基础。

4. Approximate UMAP allows for high-rate online visualization of high-dimensional data streams
   - 来源：arXiv 2024
   - 链接：https://arxiv.org/abs/2404.04001
   - 核验点：
     - 论文明确提出 approximate UMAP / aUMAP。
     - 摘要明确以实时在线可视化为目标，对比 standard UMAP 与 parametric UMAP。
   - 与本项目关系：
     - 可为“实时 refine”的速度预算设计提供参考。
     - 可参考其在线投影思路。
   - 注意：
     - 它不是局部 focus-aware 方法。
     - 更适合参考实时性，而非局部控制机制。

### 7.3 高相关：可借鉴“局部控制 / 交互式约束”

5. Local Affine Multidimensional Projection (LAMP)
   - 来源：IEEE VIS 2011
   - 链接：https://ieeevis.org/year/2011/paper/infovis/local-affine-multidimensional-projection
   - 核验点：
     - 论文页面存在，标题、作者、摘要可查。
     - 摘要明确强调 local transformations 可按用户知识动态修改。
   - 与本项目关系：
     - 非神经方法，但非常适合借鉴“用户给局部控制点，系统在局部作更灵活投影调整”的思想。
     - 对“old/new result blending”和“focus region 权重控制”有启发意义。
   - 注意：
     - 不能把 LAMP 直接表述成“本项目可直接复现的方法”。
     - 更准确的说法是：提供局部交互式投影控制的经典思想来源。

6. HUMAP: Hierarchical Uniform Manifold Approximation and Projection
   - 来源：arXiv，后续有 TVCG DOI
   - 链接：https://arxiv.org/abs/2106.07718
   - 核验点：
     - 论文明确提出 hierarchical DR。
     - 摘要明确强调 details on demand 和 preserve the mental map。
   - 与本项目关系：
     - 对“先 zoom in，再进入局部更细粒度分析”非常有启发。
     - 对 focus area 交互设计有价值。
   - 注意：
     - 它不是专门针对训练过程的 time-travelling visualizer。
     - 它也不是局部重训 visualizer 的方法。

## 8. 当前不能直接下结论的点

下面这些方向目前只能说“值得调研”，不能在设计文档中写成已有文献直接支持：

- “局部训练一个新 visualizer，再按距离对 global/local visualizer 输出连续加权融合”
  - 这是一个合理且很有潜力的工程/研究方向。
  - 但当前还不能说它就是某篇现成论文的标准做法。

- “以 zoom-in 区域作为 refine 触发条件，再做局部 TimeVis/UMAP”
  - 这是很自然的交互设计。
  - 但需要继续找更贴近的交互式 DR 文献支撑。

- “种子点梯度加权 + 远处区域权重衰减 + 新旧模型双混合”
  - 这是很好的方法构想。
  - 但当前更像你们的项目创新组合，而不是现成文献中的固定模板。

## 9. 面向下一步代码工作的建议

如果接下来要继续做代码，建议按下面顺序推进：

1. 先明确 focus area 的数学定义。
2. 再决定 local visualizer 是“临时 refine”还是“持久化模型”。
3. 再决定融合发生在：
   - loss 层
   - model 输出层
   - projection 结果层
4. 最后再升级前端交互，让 zoom/focus/blend 可视可调。

推荐第一版最务实的落地方向：

1. 保留现有 global visualizer 不变。
2. 从 global visualizer 拷贝出 local visualizer。
3. 只对 focus area 做小步数快速训练。
4. 输出 `local projection`。
5. 按距离生成 `blend weight map`。
6. 在前端或后端生成 `blended projection`。
7. 同时保留：
   - baseline projection
   - refined local projection
   - blended projection

这样最容易验证三件事：

- 局部精度有没有提升；
- 全局漂移有没有变小；
- 用户是否能理解 old/new/blended 三者关系。

## 10. 备注

本文档中的文献部分只保留“已核验真实存在”的条目，并明确区分：

- 直接相关；
- 可借鉴；
- 仅启发，不可直接宣称为本项目现成方法来源。

后续如果继续做文献调研，建议下一版单独新增：

- `REFINE_LITERATURE_REVIEW.md`

并把每篇文献进一步拆成：

- 解决的问题
- 核心机制
- 是否支持实时
- 是否支持局部控制
- 是否支持参数化映射
- 是否适合迁移到 TTAV refine
