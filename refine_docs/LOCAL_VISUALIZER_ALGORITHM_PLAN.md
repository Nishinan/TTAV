# Local Visualizer Algorithm Plan

更新时间：2026-05-30

本文档提出 TTAV 下一阶段 refine 功能的算法设计草案，并明确指出每一部分借鉴了哪些文献思想、哪些部分是我们自己的创新组合。

目标方案：

> 在 baseline global visualizer 的基础上，以用户选中的 seeds 和 zoom-in focus area 为条件，快速训练一个 local visualizer；随后按与 focus area 的距离对 old/new 结果做连续加权融合，从而在保证全局稳定的前提下，显著提高局部结构表达精度。

---

## 1. 设计目标

我们希望同时满足四个目标：

1. 局部表达精度更高
   - focus area 内的高维邻域在 2D 中更清晰。
2. 全局结构稳定
   - focus area 外尽量保持 baseline layout。
3. 响应速度可交互
   - 不允许每次 refine 都全图重训。
4. 结果可解释
   - 用户能区分 baseline、local、blended 三种输出。

---

## 2. 总体思想

### 2.1 我们借鉴了什么

#### A. 双模型 / 更新副本

- 借鉴：
  - A Parallel Framework for Streaming Dimensionality Reduction
- 借鉴点：
  - current / updating embedding function
  - local-first updates

#### B. 骨架先行，再局部细化

- 借鉴：
  - UMATO
  - Recursive SNE
  - Out-of-Core DR
- 借鉴点：
  - representative points
  - skeletal layout
  - local refinement

#### C. 局部-全局多目标平衡

- 借鉴：
  - DREAMS
  - PCC
  - Formation-Controlled DR
- 借鉴点：
  - local/global trade-off
  - near/far decomposition
  - regularized optimization

#### D. 稳定性诊断

- 借鉴：
  - GhostUMAP
  - ZADU
- 借鉴点：
  - pointwise instability
  - refine 后不只看效果，也看可靠性

### 2.2 我们自己的创新组合

下列部分是我们自己的系统性组合，而不是已查到的现成标准方案：

1. seeds + zoom-in focus area 联合决定 local subset
2. baseline global visualizer 派生出 local visualizer
3. local visualizer 训练后不直接替换 baseline，而是输出 blended projection
4. blended weight 按与 focus area 的距离连续衰减
5. blended weight 可进一步与 instability 联动

---

## 3. 算法对象与输入输出

### 3.1 输入

每次 refine 事件的输入定义为：

- `E_cur`
  - 当前查看的 epoch
- `Z_global`
  - baseline global visualizer 在 `E_cur` 上的 2D 投影
- `F_hd`
  - 当前 epoch 的高维表示
- `S`
  - 用户选中的 seed indices
- `B`
  - 当前 zoom-in focus area 的 low-D bounding box
- `cfg`
  - refine 参数配置

其中 `cfg` 暂定包含：

- `focus_mode`
- `focus_radius_hd`
- `focus_radius_ld`
- `seed_weight`
- `anchor_weight`
- `global_guard_weight`
- `local_train_steps`
- `blend_decay`
- `blend_cutoff`

### 3.2 输出

每次 refine 事件输出三种结果：

- `Z_base`
  - baseline projection
- `Z_local`
  - local visualizer 输出的局部强化结果
- `Z_blend`
  - old/new 距离融合后的最终展示结果

同时输出评估指标：

- `focus_trustworthiness`
- `focus_continuity`
- `focus_neighbor_preservation`
- `global_drift`
- `focus_shift`
- `instability_score`

---

## 4. 关键对象定义

### 4.1 Global Visualizer

定义：

- 现有 TTAV 已训练并保存的 `TimeVis` 或 `DVI` visualizer。

职责：

- 提供 baseline skeleton layout。
- 提供稳定的全局参考坐标。
- 作为 local visualizer 的初始化权重来源。

### 4.2 Local Visualizer

定义：

- 由 `global visualizer` 拷贝得到、只针对当前 refine session 训练的小步数更新模型。

职责：

- 在 focus area 内增强局部结构表达。
- 在 focus area 外不追求全面重排，只在必要范围内输出修正。

### 4.3 Focus Area

定义：

- 由 seeds 和 zoom bbox 联合决定的待增强区域。

建议的第一版定义：

- low-D 条件：
  - 点位于 zoom bbox 内
- high-D 条件：
  - 点属于 seeds 的 high-D 近邻
- 合并规则：
  - `focus_set = seeds ∪ ld_box_points ∪ hd_neighbors(seeds)`

这样做的原因：

- 只用 seeds 太稀疏；
- 只用 zoom bbox 容易混入视觉上近但高维无关的点；
- 联合定义更符合用户意图。

借鉴关系：

- “set-level interaction” 参考了 ModalChorus 的 point-set / set-set 交互思路。
- “prototype/local refinement” 参考了 Recursive SNE 与 UMATO。

---

## 5. 算法主流程

### Step 1. 构建 refine session 上下文

输入：

- `S`
- `B`
- `E_cur`
- `F_hd`
- `Z_global`

输出：

- `focus_set`
- `anchor_set`
- `background_set`

定义：

- `focus_set`
  - seeds
  - bbox 内点
  - seeds 的 high-D neighbors
- `anchor_set`
  - 远离 focus_set、但用于稳定全局的代表点
- `background_set`
  - 其余点

建议实现：

1. 从 `Z_global` 中筛出 bbox 内点。
2. 从 `F_hd` 中对 seeds 做 top-k high-D neighbor 扩展。
3. 用并集形成 `focus_set`。
4. 从非 `focus_set` 中采样 `anchor_set`。
   - 可以均匀采样；
   - 也可以按 cluster / density / k-center 采样。

借鉴关系：

- `anchor_set` 思想借鉴当前 TimeVis refine 的 anchor constraint。
- representative/anchor 机制与 UMATO 的 skeleton 思想一致。

### Step 2. 初始化 local visualizer

输入：

- `global_visualizer`

输出：

- `local_visualizer`

建议实现：

第一版采用最保守方案：

- 直接 `deepcopy(global_visualizer)`
- 仅开放最后几层 encoder 参数参与 local training
- decoder 是否训练可做开关

原因：

- 训练更快
- 风险更低
- 最接近当前 TimeVis refine 的已有成功经验

借鉴关系：

- 双副本设计借鉴 Streaming DR。
- 局部小范围参数更新借鉴当前 TTAV TimeVis refine。

### Step 3. 构建 local training subset

输入：

- `focus_set`
- `anchor_set`
- `F_hd`

输出：

- `local_train_subset`
- `local_edges`

建议实现：

将 local training subset 拆成三类样本：

1. focus points
2. focus-highD-neighbors
3. anchors

训练边拆成三类：

1. `E_focus_pos`
   - focus 点与其 high-D neighbors 的正边
2. `E_focus_neg`
   - focus 点与局部或全局 negatives 的负边
3. `E_anchor`
   - anchor 点与其 baseline 邻居的稳定性边

可选第四类：

4. `E_global_guard`
   - 用少量 background representative 组成的全局守护边

借鉴关系：

- local refinement 的局部边训练思路借鉴 Recursive SNE。
- skeleton + representative subset 借鉴 UMATO。
- near/far split 借鉴 Formation-Controlled DR。

### Step 4. 定义 local visualizer 的损失函数

建议的总损失：

`L_total = lambda_local * L_local + lambda_anchor * L_anchor + lambda_global * L_global_guard + lambda_reg * L_reg`

#### 4.1 `L_local`

目标：

- 提高 focus area 的高维邻域保真。

可选形式：

- 继续使用当前 UMAP 风格邻接损失
- 或当前 TTAV refine 中的 attract/repel 结构

推荐第一版：

- 复用现有 `SingleVisLoss` / UMAP-loss 风格
- 只是在 focus 边上提高权重

文献关系：

- 与 Recursive SNE 的 localized refinement 精神一致
- 与当前 TTAV trainer 中 batch-weighting 机制连续

#### 4.2 `L_anchor`

目标：

- 保持 anchor 点接近 baseline global projection。

定义示意：

- `L_anchor = mean(|| z_local(a) - z_global(a) ||^2)`

用途：

- 控制全局区域不要被局部优化拉乱。

文献关系：

- 思想接近当前 TimeVis refine 的 anchor constraint。
- 与 Streaming DR 的 global stability 思路一致。

#### 4.3 `L_global_guard`

目标：

- 限制 refine 导致的 global drift。

推荐第一版：

- 对少量 representative background points 保持 pairwise distance rank 或 correlation。

可选实现：

- rank-based penalty
- correlation penalty

文献关系：

- 借鉴 DREAMS 的 local/global trade-off。
- 借鉴 PCC 的 global correlation preservation。

#### 4.4 `L_reg`

目标：

- 防止 local visualizer 相对 global visualizer 改动过大。

建议第一版：

- 对开放训练的参数加 `||theta_local - theta_global||^2`

用途：

- 防止过拟合 focus 区域。

文献关系：

- 更偏工程 regularization；
- 可视作 DREAMS / Formation-Controlled DR 中“不要极端偏向局部”的工程实现。

### Step 5. 快速训练 local visualizer

建议策略：

- 小步数训练
- 小 batch
- early stopping
- 时间预算上限

建议第一版默认值：

- `local_train_steps = 100 ~ 500`
- `time_budget = 1 ~ 5 sec`

提前停止条件：

- `L_local` 改善不足
- global drift proxy 超阈值
- instability proxy 快速变差

借鉴关系：

- “快速更新、交互式响应”借鉴 Streaming DR 和 Approximate UMAP 的实时化思路。

### Step 6. 生成 `Z_local`

建议两版：

#### 第一版

- 仅对：
  - focus_set
  - focus 近邻
  - anchor_set
  做 local_visualizer 前向

其余点保留 baseline。

优点：

- 成本低
- 与当前 TTAV patch 风格兼容

#### 第二版

- 对全图跑一次 local_visualizer 前向，得到完整 `Z_local`

优点：

- 融合更自然

缺点：

- 成本更高

建议：

- 第一阶段先做第一版。

### Step 7. 构造 continuous blend weight

这是我们方案中最关键、也最具创新性的步骤。

定义每个点的 blending 权重：

- `w_i in [0, 1]`

最终输出：

- `Z_blend(i) = (1 - w_i) * Z_base(i) + w_i * Z_local(i)`

#### 7.1 权重输入因素

建议综合以下量：

1. 与 focus area 的 low-D 距离
2. 与 seeds 的 high-D 距离
3. 是否属于 focus_set
4. instability 估计

建议第一版先只用：

1. low-D distance to focus hull / bbox
2. hard focus membership boost

#### 7.2 第一版权重函数

建议定义：

- 若 `i in focus_set`，则 `w_i = 1`
- 否则：
  - `d_i = distance(z_base(i), focus_region)`
  - `w_i = exp(- d_i^2 / sigma^2)`
  - 超出 cutoff 后置 0

其中：

- `sigma` 对应 `blend_decay`
- `cutoff` 对应 `blend_cutoff`

优点：

- 平滑
- 可解释
- 易调参

#### 7.3 第二版扩展

在第一版基础上再加 instability 修正：

- `w_i = w_dist(i) * (1 - instability_i)`

解释：

- 若 local result 在某点不稳定，则减少其对 blended result 的影响。

借鉴关系：

- “local/global continuous trade-off” 借鉴 DREAMS。
- instability 修正借鉴 GhostUMAP。

### Step 8. 结果评估与守门

建议 refine 后做两层评估：

#### 8.1 必做评估

- `focus_neighbor_preservation`
- `focus_trustworthiness`
- `focus_continuity`
- `focus_shift`
- `global_drift`

#### 8.2 可选评估

- pointwise instability
- label-aware trustworthiness / continuity

#### 8.3 结果接受规则

建议设置 acceptance gate：

- 若 `focus_quality_gain > threshold`
- 且 `global_drift < threshold`
- 则接受 `Z_blend`
- 否则回退到 `Z_base`

借鉴关系：

- 评估思路借鉴当前 TTAV 已有 metrics。
- instability 审计借鉴 GhostUMAP。
- 更多 embedding distortion 分析可借鉴 ZADU。

---

## 6. 与当前 TTAV 架构的映射

### 6.1 可直接复用的现有能力

1. baseline global visualizer
   - 当前 DVI / TimeVis 已有
2. refined 独立目录
   - 已有 `_refined`
3. front-end refine workflow
   - 已有 `/updateFocusContext`
4. focus selection
   - 已有 selected indices
5. metrics display
   - 已有 refine quality panel
6. TimeVis anchor-constrained local refine
   - 已有雏形

### 6.2 需要新增的关键能力

1. zoom bbox 传到后端
2. local visualizer 副本管理
3. focus_set 联合构建
4. blended projection 保存与读取
5. blend weight 计算与可视化
6. instability 估计

---

## 7. 分阶段实现计划

下面按最稳妥的顺序设计代码改造路线。

### Phase 0. 术语与接口重构

目标：

- 不立刻改算法，先把现有 refine 术语和接口变得适合扩展。

任务：

1. 把当前 `focusMode` 文案从：
   - coarse / balanced / fine
   扩展为更明确的参数结构
2. 后端 refine 请求结构增加：
   - `zoom_bbox`
   - `focus_strategy`
   - `blend_config`
3. 统一输出：
   - `baseline_projection`
   - `local_projection`
   - `blended_projection`

代码触点：

- `web/src/views/plotView.tsx`
- `web/src/component/function-panel.tsx`
- `web/src/communication/backend.ts`
- `tool/server/server.py`

### Phase 1. Focus area 定义与 blended result 最小闭环

目标：

- 先不训练真正的 persistent local visualizer；
- 先让系统支持：
  - seeds + zoom bbox
  - 基于现有 refine 结果做 blended projection

任务：

1. 前端把当前 zoom bbox 传给后端。
2. 后端构建：
   - bbox points
   - seeds high-D neighbors
   - `focus_set`
3. 基于现有 refine 生成 `Z_local`
4. 根据 low-D distance 生成 `blend weights`
5. 输出 `Z_blend`

价值：

- 先验证“distance-based blending”本身有没有价值。
- 风险最低。

### Phase 2. Local visualizer 副本化

目标：

- 不再把 refine 仅看作当前模型临时 patch。
- 引入真正的 `local_visualizer` 对象。

任务：

1. 从 `global_visualizer` 复制模型。
2. 只开放最后几层训练。
3. 保持 local visualizer 生命周期仅绑定当前 refine session。
4. `Z_local` 改为 local visualizer 的输出。

价值：

- 完成从“局部 patch”到“局部模型”的真正转型。

### Phase 3. 多目标 local training

目标：

- local training 不再只是现有的局部吸引/排斥；
- 加入 anchor / global guard / reg。

任务：

1. 加入 `L_anchor`
2. 加入 `L_global_guard`
3. 加入 `L_reg`
4. 做一轮 ablation：
   - 只有 local
   - local + anchor
   - local + anchor + global_guard

价值：

- 明确 local gain 与 global stability 的 trade-off。

### Phase 4. Instability-aware blending

目标：

- blended result 不再只由距离决定；
- 加入点级稳定性估计。

任务：

1. 设计轻量 instability proxy
   - 第一版不必完整复现 GhostUMAP
2. 将 instability 纳入 `w_i`
3. 可视化 unstable points

价值：

- 让 refined result 更可信，也更适合做论文表达。

### Phase 5. Benchmark 与前端交互升级

目标：

- 让方案可评估、可展示、可写论文。

任务：

1. 前端支持切换：
   - baseline
   - local
   - blended
2. 前端支持调：
   - blend decay
   - focus radius
   - stability guard
3. 离线 benchmark：
   - latency
   - focus quality gain
   - global drift
   - instability

---

## 8. 建议的第一版默认实现

为了尽快开始改代码，我建议第一版先做下面这个保守版本：

### 第一版定义

- `focus_set = seeds ∪ bbox_points ∪ highD_neighbors(seeds)`
- `local_visualizer = deepcopy(global_visualizer)`
- 只训练最后两层 encoder
- `L_total = L_local + lambda_anchor * L_anchor + lambda_reg * L_reg`
- 不先上 `L_global_guard`
- `Z_local` 只 patch focus_set 与 anchors
- `w_i` 只基于 low-D distance to focus region
- 先不引入 instability-aware weighting

### 原因

- 改动最小
- 与现有 TimeVis refine 最连续
- 最容易验证有效性

---

## 9. 当前仍需拍板的设计点

以下细节我建议在真正开始改代码前与你确认：

1. `focus_set` 是否第一版就纳入 bbox 内所有点？
   - 还是只纳入 bbox 内且接近 seeds 的点？

2. `local_visualizer` 第一版是否只支持 TimeVis？
   - 还是 DVI 和 TimeVis 一起改？

3. blended result 是否要作为单独文件落盘？
   - 例如：
     - `_local`
     - `_blended`
   - 还是先只存在内存里？

4. 前端第一版是否需要用户可调 `blend_decay`？
   - 还是先固定默认值？

5. instability 审计是否放到第二阶段？
   - 我建议先放第二阶段。

---

## 10. 我建议的下一步

建议我们接下来按下面顺序开始实际代码改造：

1. 先实现 `zoom_bbox` 从前端传到后端
2. 再实现 `focus_set` 联合构建
3. 再基于现有 refine 结果做 `Z_blend`
4. 再把 local visualizer 副本化

原因：

- 这条路径最不容易把现有系统打坏；
- 可以尽快验证：
  - focus area 定义是否合理
  - distance-based blending 是否有价值

如果这一版验证有效，再进入多目标 local training 和 instability-aware blending。
