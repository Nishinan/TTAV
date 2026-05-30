# Refine Literature Review

更新时间：2026-05-30

本文档面向以下目标方案做文献调研：

> 在 baseline global visualizer 的基础上，以用户选中的 seeds 和 zoom-in focus area 为条件，快速训练一个 local visualizer；随后按与 focus area 的距离对 old/new 结果做连续加权融合，从而在保证全局稳定的前提下，显著提高局部结构表达精度。

调研原则：

- 只记录我已核验“真实存在”的论文或正式论文页。
- 优先使用会议/期刊官网、OpenReview、arXiv 等一手来源。
- 区分：
  - 可直接借鉴的实现思想
  - 可部分借鉴的局部模块思想
  - 更适合做评估与诊断的方法
- 明确哪些部分是文献已有思路，哪些部分仍需要我们自己创新组合。

## 1. 核心判断

目前没有查到一篇与我们的目标方案完全同构的论文，即没有找到“global visualizer + local visualizer + distance-based old/new blending”这一整套标准方法。

但近几年已有工作提供了三类可组合借鉴的思路：

1. 在线/流式更新
   - 研究如何快速更新嵌入函数或嵌入结果。
2. 局部或两阶段优化
   - 研究如何先形成全局骨架，再做局部细化。
3. 局部-全局平衡与稳定性评估
   - 研究如何提升局部结构同时不破坏整体布局，或如何量化不稳定点。

因此，我们的方案更准确地说是：

- 不是单篇论文直接复现；
- 而是把若干成熟子思路进行新的工程/研究组合。

## 2. 最值得重点阅读的 5 篇

### 2.1 A Parallel Framework for Streaming Dimensionality Reduction

- 来源：
  - IEEE VIS 2023
  - IEEE TVCG 2024
- 链接：
  - 论文页：https://virtual.ieeevis.org/year/2023/paper_v-full-1712.html
  - PDF：https://ieeevis.b-cdn.net/vis_2023/pdfs/v-full-1712.pdf
- 我已核验到的真实内容：
  - 论文明确提出一个并行框架，分为：
    - new data embedding
    - embedding function updating
    - embedding result updating
  - 文中明确提到：
    - parametric non-linear embedding
    - incremental learning
    - hybrid local/global update strategy
    - approximate nearest neighbor
    - lazy updating
  - 文中还明确使用 current/updating 两个 embedding function。
- 与我们方案的关系：
  - 这是目前最接近“global model + updating/local model + 稳定更新”的整体系统框架。
  - 它虽然不是 seed/focus-driven local visualizer，但已经明确把“局部更新优先、显著变化时才触发更大更新”的系统思路讲清楚了。
- 我们可直接借鉴的点：
  - `global visualizer` 和 `updating visualizer` 双副本设计。
  - local-first / global-later 的更新调度。
  - lazy update 减少不必要的频繁重训。
- 适配到 TTAV 的建议：
  - 把 local visualizer 视为 updating visualizer。
  - 只有当 local gain 高且 global drift 低时，才将 local 结果纳入 blended 输出。

### 2.2 Recursive SNE: Fast Prototype-Based t-SNE for Large-Scale and Online Data

- 来源：
  - TMLR 2025
- 链接：
  - OpenReview：https://openreview.net/forum?id=7wCPAFMDWM
  - 镜像页面：https://eprints.lancs.ac.uk/id/eprint/232482/
- 我已核验到的真实内容：
  - 论文明确提出：
    - i-RSNE：实时 point-wise update
    - Bi-RSNE：批量更新
  - 核心方法包括：
    - prototype-based initialization
    - localized KL-divergence refinements
- 与我们方案的关系：
  - 这篇很贴近“先用 baseline 作为种子，再做局部 refinement”。
  - 它虽然是 t-SNE 家族，不是 DVI/TimeVis，但“原型初始化 + 局部修正”的思路非常适合 local visualizer 的快速训练。
- 我们可直接借鉴的点：
  - 先从 focus area 提取 representative subset / prototypes。
  - local visualizer 不一定要全区域重新学习，可以从骨架原型启动。
  - 局部优化目标可以局限在 focus neighborhood，而不是整图。

### 2.3 UMATO: Bridging Local and Global Structures for Reliable Visual Analytics with Dimensionality Reduction

- 来源：
  - IEEE TVCG 2025
- 链接：
  - arXiv：https://arxiv.org/abs/2508.16227
  - DOI：https://doi.org/10.1109/TVCG.2025.3602735
- 我已核验到的真实内容：
  - 论文明确使用 two-phase optimization。
  - 第一阶段通过 representative points 构造 skeletal layout。
  - 第二阶段将其余点放回布局中，同时尽量保留区域结构。
- 与我们方案的关系：
  - 这篇非常适合作为我们“global skeleton + local refinement”路线的理论支撑。
  - 我们的 baseline global visualizer 可以视为现成 skeleton。
  - local visualizer 则只负责第二阶段对 focus area 做增强。
- 我们可直接借鉴的点：
  - 不把全图重训当默认路径。
  - 优先维护骨架，再局部增加表达精度。

### 2.4 DREAMS: Preserving both Local and Global Structure in Dimensionality Reduction

- 来源：
  - TMLR 2026 接收
  - arXiv 2025
- 链接：
  - OpenReview：https://openreview.net/forum?id=xpGu3Sichc
  - arXiv：https://arxiv.org/abs/2508.13747
- 我已核验到的真实内容：
  - 论文明确提出一个简单 regularization term。
  - 该正则把 t-SNE 的 local preservation 和 PCA 的 global preservation 联系起来。
  - 可以生成一个 local/global trade-off spectrum。
- 与我们方案的关系：
  - 这是我们设计“old/new 连续融合”时最重要的理论启发之一。
  - 它说明 local gain 与 global stability 本来就是一条连续权衡曲线，而不是二元选择。
- 我们可直接借鉴的点：
  - 不仅在显示层做 blending。
  - 更应该在训练目标层显式建模 local/global trade-off。

### 2.5 GhostUMAP: Measuring Pointwise Instability in Dimensionality Reduction

- 来源：
  - IEEE VIS 2024 Short
- 链接：
  - 论文页：https://content-staging.ieeevis.org/year/2024/paper_v-short-1065.html
- 我已核验到的真实内容：
  - 论文明确提出在 UMAP 优化中加入“ghosts”。
  - ghost 是被动副本，不影响其他点，只被原始点吸引/排斥。
  - ghost spread 可用于估计 pointwise instability。
- 与我们方案的关系：
  - 它不是优化算法，但非常适合作为 refine 后的稳定性审计工具。
  - 尤其适合 seeds/focus area 的不确定性量化。
- 我们可直接借鉴的点：
  - refine 后不要只看 layout 好不好看。
  - 还应估计局部点位移是否稳定、是否只是噪声敏感更新。

## 3. 可直接借鉴的实现思想

这一节只保留我认为可以较直接映射到 TTAV 代码设计中的文献思想。

### 3.1 双 visualizer / 双函数副本

- 主要借鉴：
  - A Parallel Framework for Streaming Dimensionality Reduction
- 可借鉴内容：
  - current embedding function
  - updating embedding function
  - local-first updates
  - lazy switching
- 映射到 TTAV：
  - `global_visualizer`: 现有 baseline model
  - `local_visualizer`: 针对 focus area 启动的轻量副本
  - `blended_projection`: 最终显示层输出

### 3.2 骨架先行、局部再细化

- 主要借鉴：
  - UMATO
  - Recursive SNE
  - Out-of-Core DR
- 可借鉴内容：
  - representative points
  - skeletal layout
  - local refinement
  - out-of-sample style insertion
- 映射到 TTAV：
  - baseline global projection 即 skeletal layout
  - seeds / focus neighbors 构成 representative local subset
  - local visualizer 只在这部分做快速增强

### 3.3 局部-全局多目标平衡

- 主要借鉴：
  - DREAMS
  - Preserving Clusters and Correlations (PCC)
  - Formation-Controlled DR
- 可借鉴内容：
  - local objective
  - global objective
  - regularized trade-off
  - near/far decomposition
- 映射到 TTAV：
  - `L_local`: focus area 的高维邻域保真
  - `L_anchor`: 非 focus anchor 的稳定性约束
  - `L_global_rank` 或 `L_global_corr`: 防止 global drift

### 3.4 交互参数成为方法一部分

- 主要借鉴：
  - Class-constrained t-SNE
  - ModalChorus
- 可借鉴内容：
  - 用户交互参数可直接调节目标函数平衡
  - 支持 set-level 而不只是单点交互
- 映射到 TTAV：
  - `focus_strength`
  - `blend_alpha`
  - `focus_radius`
  - `distance_decay`
  - `stability_guard`

## 4. 可部分借鉴的工作

### 4.1 Approximate UMAP allows for high-rate online visualization of high-dimensional data streams

- 来源：
  - arXiv 2024
- 链接：
  - https://arxiv.org/abs/2404.04001
- 价值：
  - 强调在线可视化速率与近似方法。
  - 对“如何把更新做得更快”有帮助。
- 限制：
  - 它不是局部 focus-aware refine 方法。
  - 更适合作为实时性优化参考，而不是 local/global 融合参考。

### 4.2 Out-of-Core Dimensionality Reduction for Large Data via Out-of-Sample Extensions

- 来源：
  - LDAV 2024
- 链接：
  - https://content-staging.ieeevis.org/year/2024/paper_a-ldav-1003.html
- 价值：
  - 提供 reference projection + out-of-sample extension 思路。
  - 对大规模 focus patch 很有工程价值。
- 限制：
  - 更偏 scalable projection，不是显式的 local training 方案。

### 4.3 ModalChorus

- 来源：
  - IEEE VIS 2024
- 链接：
  - https://content-staging.ieeevis.org/year/2024/paper_v-full-1603.html
- 价值：
  - 交互式 point-set / set-set alignment 思想对 seeds/focus area 很有启发。
- 限制：
  - 核心问题是模态对齐，不是训练过程时序 refine。

## 5. 更适合做评估和诊断的工作

### 5.1 ZADU: A Python Library for Evaluating the Reliability of Dimensionality Reduction Embeddings

- 来源：
  - IEEE VIS 2023 Short
- 链接：
  - https://virtual.ieeevis.org/year/2023/paper_v-short-1036.html
- 价值：
  - 提供多种 distortion measures。
  - 支持分析单点对失真的贡献。
- 适用方式：
  - 用于 refine 前后 benchmark 与 ablation。

### 5.2 Classes are not Clusters: Improving Label-based Evaluation of Dimensionality Reduction

- 来源：
  - IEEE VIS 2023 / IEEE TVCG 2024
- 链接：
  - https://virtual.ieeevis.org/year/2023/paper_v-full-1025.html
- 价值：
  - 提出 Label-Trustworthiness / Label-Continuity。
  - 对类别结构不等同于单簇这一现实更鲁棒。
- 适用方式：
  - 用于训练表示中带 label 的 refine 评估。

## 6. 对我们方案的直接启发整理

### 6.1 文献已明确支持的部分

- `global model + updating model`
  - 由 Streaming DR 相关工作支持。
- `prototype / skeleton + local refinement`
  - 由 Recursive SNE、UMATO、Out-of-Core DR 支持。
- `local/global regularized trade-off`
  - 由 DREAMS、PCC、Formation-Controlled DR 支持。
- `交互参数调节优化目标`
  - 由 Class-constrained t-SNE、ModalChorus 支持。
- `局部稳定性审计`
  - 由 GhostUMAP、ZADU 支持。

### 6.2 文献尚未给出标准答案、需要我们创新组合的部分

- seeds + zoom-in focus area 联合定义 local visualizer 训练区域
- baseline global visualizer 与 local visualizer 的统一产品化接口
- 按与 focus area 的距离做 old/new 连续 blending
- blending 权重与 instability / confidence 联动

因此，下面这几个点应被视为我们的研究创新点，而不是直接说“文献已有”：

- distance-aware old/new blending
- zoom-aware local visualizer triggering
- local visualizer 与 global visualizer 的联合可视化输出

## 7. 建议的精读顺序

如果只精读 5 篇，建议按以下顺序：

1. A Parallel Framework for Streaming Dimensionality Reduction
2. Recursive SNE
3. UMATO
4. DREAMS
5. GhostUMAP

如果要补充实时性工程路线，再加：

6. Approximate UMAP
7. Out-of-Core DR

如果要补充评估体系，再加：

8. ZADU
9. Classes are not Clusters

## 8. 下一步建议

基于以上文献，我建议我们在 TTAV 中采用如下总思路：

1. 以现有 global visualizer 为 skeleton。
2. 以 seeds + focus area 构造 local subset。
3. 从 global visualizer 复制出 local visualizer。
4. local visualizer 只在 focus subset 上快速微调。
5. 训练目标中同时加入：
   - local fidelity
   - anchor stability
   - optional global rank/correlation guard
6. 输出三种结果：
   - baseline projection
   - local projection
   - blended projection
7. 用 instability / trustworthiness / continuity / drift 对结果做评估。

对应的详细算法设计见：

- `LOCAL_VISUALIZER_ALGORITHM_PLAN.md`

## 9. 来源链接汇总

- A Parallel Framework for Streaming Dimensionality Reduction  
  https://ieeevis.b-cdn.net/vis_2023/pdfs/v-full-1712.pdf

- Approximate UMAP allows for high-rate online visualization of high-dimensional data streams  
  https://arxiv.org/abs/2404.04001

- Out-of-Core Dimensionality Reduction for Large Data via Out-of-Sample Extensions  
  https://content-staging.ieeevis.org/year/2024/paper_a-ldav-1003.html

- Recursive SNE  
  https://openreview.net/forum?id=7wCPAFMDWM

- UMATO  
  https://arxiv.org/abs/2508.16227

- DREAMS  
  https://openreview.net/forum?id=xpGu3Sichc

- Preserving Clusters and Correlations (PCC)  
  https://arxiv.org/abs/2503.07609

- Formation-Controlled Dimensionality Reduction  
  https://arxiv.org/abs/2404.06808

- Class-constrained t-SNE  
  https://virtual.ieeevis.org/year/2023/paper_v-full-1507.html

- ModalChorus  
  https://content-staging.ieeevis.org/year/2024/paper_v-full-1603.html

- GhostUMAP  
  https://content-staging.ieeevis.org/year/2024/paper_v-short-1065.html

- ZADU  
  https://virtual.ieeevis.org/year/2023/paper_v-short-1036.html

- Classes are not Clusters  
  https://virtual.ieeevis.org/year/2023/paper_v-full-1025.html
