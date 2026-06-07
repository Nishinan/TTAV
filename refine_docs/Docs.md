# TTAV Refine 相关文献调研与实现建议

更新时间：2026-06-06

本文档基于当前 TTAV `refine` 方案的真实问题重新整理相关文献，并给出面向实现的技术判断。

当前 TTAV 的 refine 已从“局部 patch”升级为：

- `global visualizer`
- `local visualizer`
- `blended projection`

其目标是：

- 提升 `focus_set` 内局部结构精度
- 保持全局布局稳定
- 支持交互式低延迟 refine

但当前双 visualizer 路线存在两个主要问题：

1. `local visualizer` 训练太慢
2. 局部质量提升有限，`NP / MRH / Trustworthiness / Continuity` 仍不够理想

根据现有 benchmark 结果，这不是一个“已有文献里某篇论文可以整套替换”的问题。更准确地说，TTAV 当前的 refine 方案是若干已知思想的工程组合，因此文献调研的重点应该放在：

- 找到更合适的局部更新机制
- 找到更贴近目标指标的优化目标
- 找到更稳的 global-local 结构设计

---

## 1. 核心结论

当前最值得借鉴的不是某一篇“完美替代方案”，而是三条可以组合的路线：

1. `freeze global + focus-set partial re-embedding`
2. `landmark-anchor refinement`
3. `parametric residual adapter`

如果只从“技术贴合度 + 对当前痛点的潜在帮助”来排序，我的结论是：

### 第一优先级

- `coordinate-level partial re-embedding`
- 代表线索：
  - `openTSNE partial embedding / transform`
  - `FIt-SNE / openTSNE` 的高效优化实现

原因：

- 最直接解决“训练太慢”
- 最容易和 TTAV 现有 `focus_set + blended projection` 框架对接
- 不需要保留“复制整个 local visualizer 再训练”的高成本路径

### 第二优先级

- `landmark-anchor refinement`
- 代表线索：
  - `HSNE`
  - `LAMP`
  - `landmark / sparse MDS`

原因：

- 最适合 TTAV 的交互流程：`overview -> zoom-in -> refine`
- 最符合“全局骨架稳定 + 局部增强”的产品目标
- 更适合发展成长期方案

### 第三优先级

- `global frozen + tiny residual adapter`
- 代表线索：
  - `Parametric UMAP`

原因：

- 如果必须保留神经 visualizer 路线，这是最保守、最现实的升级方向
- 但在延迟改善上，通常不如坐标级局部优化直接

一句话总结：

> TTAV 最该从“复制 global model 再局部训练”转向“固定 global skeleton，只对 focus_set 做局部坐标优化或局部插值更新”。

---

## 2. 当前 TTAV refine 的问题抽象

结合现有文档和 benchmark，可以把问题抽象为：

### 2.1 当前系统已经具备的能力

- 已有稳定的 `baseline global projection`
- 已有 `focus_set`
- 已有 `bbox + focus_set` 的 distance-weighted blending
- 已有 `NP / MRH / Trustworthiness / Continuity / Global Drift / Latency` 的 benchmark workflow

### 2.2 当前系统最主要的瓶颈

- `local visualizer` 需要显式复制和局部训练
- 局部训练仍然依赖神经网络微调，交互延迟较高
- 当前 objective 更偏“吸引高维邻居 + 排斥随机负样本 + anchor 约束”
- 当前 blended projection 主要在显示层起作用，训练目标层的 local/global trade-off 仍较弱

### 2.3 我们真正要解决的问题

不是简单地“换一个更好的 DR 算法”，而是同时回答：

1. 有没有比局部网络微调更快的 `focus_set` 更新方式？
2. 有没有更对准 `top-k neighbor fidelity` 的局部优化目标？
3. 有没有更稳的 global-local 结构，让 drift 控制不只依赖显示层 blending？

---

## 3. 比“复制 global model 再局部训练”更快的机制

### 3.1 openTSNE 的 partial embedding / transform

最值得重点借鉴。

核心思想：

- 固定已有 reference embedding
- 只优化新样本或局部样本的位置
- 不重新训练一个完整 embedding model

迁移到 TTAV 的方式：

- 不训练 `local visualizer`
- 直接固定 `global projection`
- 只优化 `focus_set` 的 2D 坐标
- 用 bbox 外一圈 `ring anchors` 控制边界和漂移

为什么贴合 TTAV：

- TTAV 已有 `focus_set`
- TTAV 已有 baseline projection
- TTAV 已有 blending
- 只缺少“局部点直接优化”的机制

潜在收益：

- `Latency` 显著下降
- 更容易把优化目标直接对准局部邻居质量

局限：

- openTSNE 的 transform 更偏“新增点插入已有图”
- TTAV refine 是“已有点的局部重排”，所以需要加上 focus 内部相互作用和 boundary anchors

结论：

> 这是当前最适合 TTAV 的主方向。

### 3.2 FIt-SNE / openTSNE 作为优化底座

价值：

- 提供更高效的局部优化器
- 如果 TTAV 走坐标级局部优化，这类工程底座很适合做 benchmark baseline

结论：

> 不一定是最终方法，但适合作为“局部 re-embedding”的高效实现参考。

### 3.3 Parametric UMAP

核心思想：

- 不直接优化每个点的坐标
- 而是学习一个 parametric mapping

迁移思路：

- 冻结 `global visualizer`
- 只训练一个小的 residual adapter / local head
- 输出变为：

`y = f_global(x) + g(x) * Delta(x)`

其中：

- `g(x)` 是 focus gate
- `Delta(x)` 是小型 residual MLP / adapter

优点：

- 保留 visualizer 作为显式映射函数
- 比复制整个 local visualizer 更轻
- 更容易跨 epoch 泛化

缺点：

- 通常仍然比坐标级局部优化慢
- 更偏保守重构，而不是范式切换

结论：

> 适合“必须保留 parametric visualizer 框架”的场景，但不是 TTAV 当前最优先方向。

### 3.4 LAMP / 局部仿射投影

核心思想：

- 通过 control points / landmarks 构造局部仿射变换
- 不依赖神经网络训练循环

迁移到 TTAV：

1. baseline global projection 给出全局骨架
2. 在 `focus_set` 内选少量 representative points
3. 只精修这些 representative points
4. 其他 focus points 通过 LAMP / RBF / local affine 插值更新

优点：

- 非常快
- 更容易控制全局稳定

缺点：

- 需要先设计 representative selection 和插值稳定性策略
- 对复杂局部非线性结构的表达能力可能不如直接局部优化

结论：

> 很适合做 TTAV 的中期增强方案，尤其适合和 landmark 结构结合。

---

## 4. 更适合局部邻域保真的 objective

### 4.1 NeRV

这篇对 TTAV 很重要。

核心思想：

- 把 DR 明确当作“邻居检索”问题
- 用参数平衡：
  - precision：减少假邻居
  - recall：减少漏邻居

和 TTAV 的关系：

- `Trustworthiness` 对应“别引入假的低维邻居”
- `Continuity` 对应“别漏掉高维真邻居”
- `NP / MRH` 也本质上在考察局部 top-k 邻域的保真度

因此：

> NeRV 的目标函数方向比“只做 attract/repel”更贴近 TTAV 当前的 benchmark 指标。

局限：

- 原版复杂度较高
- 不一定适合全局求解

但对 TTAV 来说：

- refine 只作用于 `focus_set`
- 正好可以把高成本 objective 局限在局部区域

结论：

> 如果 TTAV 要升级 objective，我最推荐优先参考 NeRV 的 precision/recall trade-off。

### 4.2 t-SNE / SNE 类局部邻域目标

价值：

- 经典的邻域概率保真机制
- 对局部重嵌入很自然

适配建议：

- 只在 `focus_set` 内计算高维/低维邻域关系
- 外加 `ring anchor` 和 `drift` 约束，避免局部自发塌缩或过度扭曲

结论：

> 适合作为 coordinate-level partial re-embedding 的基础损失。

### 4.3 UMAP-style fuzzy graph cross-entropy

价值：

- 支持正负采样
- 更适合局部图上的低延迟 SGD

适合 TTAV 的场景：

- 如果你们仍想做局部少步优化
- 或仍保留轻量 adapter / 小模型训练

结论：

> 比当前简单 attract/repel 更成体系，但在“指标直接对齐”上不如 NeRV 直观。

### 4.4 Dynamic t-SNE / 可微稳定项

价值：

- 把“全局不要乱漂”显式写进 loss
- 不再只依赖 refine 后的 drift 度量或显示层 blending

和 TTAV 的关系：

- 当前 `Global Drift` 是 benchmark 指标
- 这类方法提供了把 drift 直接前移到训练目标里的机制

结论：

> 非常值得作为 TTAV refine loss 的配套项，而不是独立主方法。

### 4.5 Parametric UMAP 的全局相关性项

价值：

- 提供了局部邻域项 + 全局结构项 的参数化组合范式

结论：

> 如果继续保留 visualizer 训练路线，这条很适合作为 objective 参考。

---

## 5. 更成熟的 global-local / anchor-based 结构

### 5.1 HSNE

这是 TTAV 非常值得重视的一篇。

核心思想：

- 层级化 landmarks
- overview-first, details-on-demand
- 用户逐步下钻，而不是每次从头重训局部结构

和 TTAV 的贴合点非常强：

- 你们本来就在做交互式 zoom-in
- 你们本来就在构造 `focus_set`
- 你们本来就需要“全局骨架稳定、局部逐步展开”

对 TTAV 的直接启发：

- 可以预先维护 `skeleton / landmark hierarchy`
- refine 时不是临时复制 local model
- 而是基于当前层级展开局部结构，并固定上层 anchors

结论：

> 从长期架构设计上，HSNE 是最适合 TTAV 的 landmark / hierarchy 参考。

### 5.2 LAMP

在 global-local 结构里，LAMP 的角色是：

- 不负责全局层级
- 但非常适合承担“局部快速更新 / 插值传播”的实现层

结论：

> 如果 TTAV 做 landmark hierarchy，LAMP 很适合做局部更新器。

### 5.3 landmark / sparse MDS

价值：

- 强调少量 landmarks 对整体骨架的控制作用

局限：

- 更偏全局 skeleton 参考
- 不是 TTAV refine 的直接替代方法

结论：

> 更适合作为结构灵感，不是首选实现路径。

---

## 6. 适合交互式低延迟 refinement 的方向

### 6.1 A-tSNE

非常值得精读。

核心思想：

- t-SNE 的近似程度可控
- 用户可在分析过程中决定哪些局部值得进一步精修

和 TTAV 的关系：

- 你们的 refine 本来就是交互触发
- 局部区域不需要一开始就最高精度
- 更自然的方式是：
  - 先给出粗但快的结果
  - 再对局部逐步 refine

结论：

> 它不是 TTAV 的最终算法答案，但对“交互式 progressive refine”的产品形态启发很强。

### 6.2 Dynamic t-SNE / S+t-SNE / 时间一致性方法

价值：

- 强调 temporal coherence
- 强调相邻时间步之间的稳定过渡

和 TTAV 的关系：

- TTAV 面向 epoch 序列
- refine 不应只把单个 epoch 当孤立问题

结论：

> 很适合给 TTAV 增加 temporal regularization，而不是作为主 refine 方法单独落地。

### 6.3 TimeVis 本身

最重要的启发不是“继续照着 refine”，而是：

> focus_set 的优化不应只基于当前 epoch，还可以考虑跨 epoch 的局部时间一致性。

结论：

> TimeVis 更像 TTAV refine 的上位约束来源，而不是替代 refine 的新方案。

---

## 7. 文献与 TTAV 的贴合度判断

下面按“技术贴合度 + 预期效果”给出综合判断。

| 路线 | 主要解决 | 对速度帮助 | 对局部精度帮助 | 和 TTAV 贴合度 | 结论 |
| --- | --- | --- | --- | --- | --- |
| openTSNE partial embedding / transform | 只优化局部点 | 很高 | 中高 | 很高 | 最优先 |
| FIt-SNE / openTSNE optimizer | 局部优化加速 | 很高 | 中 | 高 | 适合作为底座 |
| NeRV objective | top-k 邻域保真 | 低 | 很高 | 很高 | 最优先的 objective 参考 |
| HSNE | hierarchy / landmarks / drill-in | 中 | 中高 | 很高 | 长期结构最优 |
| LAMP | control points / local affine | 很高 | 中 | 很高 | 中期很值得做 |
| Dynamic t-SNE | 漂移控制 / 时间一致性 | 中 | 中 | 高 | 配套 loss 很重要 |
| Parametric UMAP | parametric mapping / adapter | 中 | 中 | 高 | 如果保留 visualizer 路线则值得做 |
| landmark MDS / sparse MDS | global skeleton | 中 | 低中 | 中 | 灵感大于直接实现 |

---

## 8. 我建议 TTAV 实现哪些方案

下面不是泛泛推荐，而是按 TTAV 当前代码和风险做的落地排序。

### 方案 A：Coordinate-level Partial Re-embedding

最推荐，优先实现。

核心改动：

- 不再训练 `local visualizer`
- 固定 `baseline global projection`
- 只对 `focus_set` 的 2D 坐标做小规模优化
- bbox 外一圈采样 `ring anchors`
- 继续保留现有 `blended projection`

建议 objective：

- `local neighbor loss`
- `anchor penalty`
- `drift penalty`
- 可选 `temporal penalty`

适合 TTAV 的原因：

- 改动集中在 refine 路径
- 不需要推翻现有 global visualizer
- 最直接降低延迟

预期收益：

- `Latency` 明显下降
- `NP / MRH / Trustworthiness / Continuity` 更容易直接提升

风险：

- 要重新设计 refine 后的 neighbors 与 metrics 计算路径
- 需要小心 focus 外边界过渡

最终判断：

> 这是 TTAV 当前最值得优先尝试的方案。

### 方案 B：Landmark-Anchor Refinement

第二推荐，适合作为中期路线。

核心改动：

- 预计算 global skeleton / representatives / anchors
- refine 时只精修少量 focus representatives
- 其他 focus points 用 LAMP / RBF / local affine 插值传播

适合 TTAV 的原因：

- 与交互流程高度一致
- 更稳地控制全局不漂
- 更容易从产品角度解释给用户

预期收益：

- `Latency` 很可能继续下降
- `Global Drift` 更容易控制
- `focus_set` 规模变大时更稳

风险：

- 实现复杂度高于方案 A
- representative selection 和插值质量是关键

最终判断：

> 这是 TTAV 最值得发展的长期架构方向。

### 方案 C：Global Frozen + Residual Adapter

第三推荐，适合保守升级。

核心改动：

- global visualizer 冻结
- 不再 `deepcopy` 完整 local model
- 只训练一个小 residual adapter / gate

适合 TTAV 的原因：

- 最容易复用现有 visualizer 框架
- 跨 epoch 泛化潜力更好

预期收益：

- 比当前 local visualizer 更轻
- 但速度收益通常不如方案 A/B 直接

风险：

- 仍然要训练网络
- 如果 objective 不改，精度问题未必根本改善

最终判断：

> 如果团队必须保留 parametric visualizer 训练思路，这是一条最现实的升级路径。

---

## 9. 不建议优先做的方向

### 9.1 继续深挖“完整 local visualizer 微调”

原因：

- 它已经暴露出明显的延迟问题
- 即使继续调局部 loss，收益也可能被训练成本吞掉
- 从文献上看，更好的零件都在鼓励“局部点更新 / landmarks / interpolation / partial embedding”

结论：

> 不建议把主要精力继续放在“怎样把 local visualizer 训练得更久/更深”上。

### 9.2 只增强显示层 blending，不动优化目标

原因：

- blending 对全局稳定很重要
- 但它解决不了局部精度本身不够高的问题

结论：

> blending 应保留，但不应再被当作 refine 提升的主手段。

---

## 10. 建议的实现顺序

### Phase A：低风险快速验证

1. 实现 `coordinate-level partial re-embedding`
2. 复用现有 `focus_set + blended projection`
3. 加入 `ring-anchor drift penalty`
4. benchmark 对比当前双 visualizer

目标：

- 先验证是否能把 `Latency` 大幅降下来
- 同时看 `NP / MRH / T / C` 是否至少不差于当前方案

### Phase B：objective 升级

1. 把当前局部 loss 改成更贴近邻域保真的形式
2. 优先参考 `NeRV` 的 precision/recall 取向
3. 同时加入 drift / temporal regularization

目标：

- 把 refine 真正对准 benchmark 指标

### Phase C：结构升级

1. 引入 `landmark / representative / hierarchy`
2. 尝试 `LAMP / local affine / RBF interpolation`

目标：

- 让 TTAV 从“局部小补丁系统”演进成真正稳定的多尺度交互 refine 系统

---

## 11. 最终建议

如果只给一个最现实的判断：

> TTAV 下一步最应该做的不是“把 local visualizer 训练得更好”，而是“减少甚至去掉 local visualizer 的训练，把 refine 改造成固定 global skeleton 的局部重嵌入问题”。

最推荐的组合是：

- `Global skeleton fixed`
- `Focus-set partial re-embedding`
- `Ring-anchor drift control`
- `Distance-weighted blending`

如果你们要保留 parametric visualizer 路线，再考虑：

- `global frozen + residual adapter`

如果要做长期方案，再进一步发展成：

- `hierarchical landmarks + local interpolation refinement`

---

## 12. 文献阅读优先级

建议精读顺序：

1. `openTSNE / partial embedding / transform`
2. `NeRV`
3. `HSNE`
4. `LAMP`
5. `Dynamic t-SNE`
6. `Parametric UMAP`
7. `A-tSNE`

---

## 13. 参考链接

- openTSNE: A Modular Python Library for t-SNE Dimensionality Reduction and Embedding  
  https://file.biolab.si/papers/2024-Policar-JSS.pdf

- FIt-SNE: Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data  
  https://www.nature.com/articles/s41592-018-0308-4

- Parametric UMAP embeddings for representation and semi-supervised learning  
  https://arxiv.org/abs/2009.12981

- Visualizing Data using t-SNE  
  https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

- Class-constrained t-SNE: Combining Data Features and Class Probabilities  
  https://research.tue.nl/en/publications/class-constrained-t-sne-combining-data-features-and-class-probabi

- Local Affine Multidimensional Projection  
  https://www.researchgate.net/publication/51752005_Local_Affine_Multidimensional_Projection

- Sparse multidimensional scaling using landmark points  
  https://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf

- Temporality Spatialization: A Scalable and Faithful Time-Travelling Visualization for Deep Classifier Training  
  https://www.ijcai.org/proceedings/2022/0558.pdf
