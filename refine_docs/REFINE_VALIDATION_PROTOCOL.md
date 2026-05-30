# Refine Validation Protocol

更新时间：2026-05-30

本文档规定 TTAV 下一阶段 `global visualizer + local visualizer + blended projection` 方案的验证方式，目标是让每一步实现都能：

- 有对应测试脚本
- 有可保存的数值结果
- 有可保存的图像结果
- 有可比较的性能指标
- 最终能按论文式实验组织材料

## 1. 最终双模型设计

### 1.1 Global Visualizer 应该是什么

最终建议：

- `global visualizer` 继续使用现有 TTAV 中已经训练好的 `TimeVis` 或 `DVI` 模型
- 它的定位是：
  - 全局稳定骨架
  - baseline projection 的唯一来源
  - local visualizer 的初始化权重来源

设计原则：

- 不应该因为一次局部 refine 被永久污染
- 它负责“快”和“稳”
- 它不追求局部极致精度

### 1.2 Local Visualizer 应该是什么

最终建议：

- `local visualizer` 是从 `global visualizer` 拷贝出来的 session-level 副本
- 它只针对当前 epoch、当前 focus area 做快速小步训练
- 它的定位是：
  - 局部增强器
  - 不替代 global visualizer
  - 只负责“局部更清楚”

设计原则：

- 只开放少量参数层训练，优先最后两层 encoder
- 训练步数短、时间预算短
- 它负责“精”
- 不要求它自己单独承担全局布局质量

### 1.3 最终显示结果是什么

最终展示不直接显示 `local projection`，而是显示：

- `blended projection`

其中：

- focus area 内更多使用 `local projection`
- focus area 外更多使用 `baseline projection`

所以最终职责分工是：

- `global visualizer`: 全局快、稳
- `local visualizer`: 局部精
- `blended projection`: 给用户看的最终结果

## 2. 为什么先分阶段验证

我们当前不应该一步到位直接做完整 local visualizer 系统。

原因：

1. 不确定 `bbox + seeds` 是否真的比只用 seeds 更好
2. 不确定 `blended projection` 是否真的比“直接替换 refine 结果”更好
3. 不确定 local visualizer 的收益是否值得额外复杂度

因此建议按三阶段验证：

### Phase 1

验证：

- `bbox + seeds` 是否能更合理定义 `focus_set`

### Phase 2

验证：

- `blended projection` 是否比直接替换更优

### Phase 3

验证：

- 真正的 `local visualizer` 是否比“现有 patch refine + blending”更优

## 3. 关键词解释

### 3.1 bbox

- `bbox = bounding box`
- 指当前 zoom-in 视图对应的二维矩形框
- 包括：
  - `x_min`
  - `x_max`
  - `y_min`
  - `y_max`

### 3.2 seeds

- 用户主动选中的点
- 是 refine 的起点

### 3.3 focus area

- 用户当前真正关心的局部区域
- 由：
  - `seeds`
  - `bbox`
  - seeds 的 high-D neighbors
  联合决定

### 3.4 focus_set

- 所有被认为属于 focus area 的点

### 3.5 baseline projection

- 原始全局模型给出的 2D 坐标

### 3.6 local projection

- 局部模型给出的 2D 坐标

### 3.7 blended projection

- `baseline projection` 与 `local projection` 的加权融合结果
- 是最终展示给用户的结果

### 3.8 global drift

- refine 后非 focus 区域整体偏移程度
- 越小越好

### 3.9 focus quality

- focus area 内的结构质量指标
- 常见包括：
  - trustworthiness
  - continuity
  - neighbor preservation

## 4. 每个阶段怎么测

### 4.1 Phase 1: bbox + seeds 验证

目标：

- 证明联合定义的 `focus_set` 比只用 seeds 更合理

比较对象：

1. `seeds_only`
2. `seeds + hd_neighbors`
3. `seeds + bbox`
4. `seeds + bbox + hd_neighbors`

建议指标：

- `focus_size`
- `focus_hd_coverage`
- `focus_ld_coverage`
- `focus_neighbor_preservation_after_refine`
- `focus_trustworthiness_after_refine`
- `global_drift_after_refine`

建议图像：

- focus area 叠加图
- 不同 focus_set 定义的点高亮图
- refine 前后局部放大图

测试脚本：

- `tests/test_refine_phase1_focus_set.py`

结果保存：

- `refine_validation_results/phase1_focus_set/`

### 4.2 Phase 2: blended projection 验证

目标：

- 证明 `blended projection` 比直接替换 `local projection` 更稳

比较对象：

1. `baseline`
2. `local_direct_replace`
3. `blended_projection`

建议指标：

- `focus_neighbor_preservation`
- `focus_trustworthiness`
- `focus_continuity`
- `focus_shift`
- `global_drift`
- `stability_ratio = focus_shift / global_drift`

建议图像：

- baseline / local / blended 三联图
- focus area 局部放大对比图
- drift heatmap
- blend weight map

测试脚本：

- `tests/test_refine_phase2_blending.py`

结果保存：

- `refine_validation_results/phase2_blending/`

### 4.3 Phase 3: local visualizer 验证

目标：

- 证明真正的 `local visualizer` 比现有 patch refine 更值得保留

比较对象：

1. `baseline`
2. `existing_patch_refine`
3. `patch_refine + blending`
4. `local_visualizer + blending`

建议指标：

- 所有 Phase 2 指标
- `latency_total`
- `latency_train`
- `latency_blend`
- `peak_memory`
- `acceptance_rate`

建议图像：

- 四联图
- 训练时间条形图
- 质量-延迟散点图

测试脚本：

- `tests/test_refine_phase3_local_visualizer.py`

结果保存：

- `refine_validation_results/phase3_local_visualizer/`

## 5. 论文式结果组织方式

建议所有结果按如下结构保存：

- `refine_validation_results/`
  - `phase1_focus_set/`
    - `metrics.json`
    - `summary.csv`
    - `fig_focus_set_examples.png`
    - `fig_focus_set_comparison.png`
  - `phase2_blending/`
    - `metrics.json`
    - `summary.csv`
    - `fig_baseline_local_blended.png`
    - `fig_blend_weight_map.png`
    - `fig_drift_heatmap.png`
  - `phase3_local_visualizer/`
    - `metrics.json`
    - `summary.csv`
    - `fig_quality_vs_latency.png`
    - `fig_local_visualizer_comparison.png`
  - `figures_for_paper/`
    - 用于最终论文或汇报的精选图

## 6. 对比应该看哪些方面

参考近年论文的通用比较方式，建议至少从下面四个维度比：

1. 局部质量
   - trustworthiness
   - continuity
   - neighbor preservation

2. 全局稳定
   - global drift
   - anchor displacement

3. 实时性
   - total latency
   - train latency
   - post-process latency

4. 可解释性
   - blend weight map 是否合理
   - unstable points 是否集中在局部边界

## 7. 我建议的当前决策

在真正开始写代码前，我推荐：

1. 第一版只做 `TimeVis`
2. `focus_set` 第一版就使用：
   - `seeds ∪ bbox内点 ∪ seeds高维近邻`
3. 第一版先做 `blended projection`
4. 第一版先不做真正的 persistent `local visualizer`
5. 每做一步就把结果写入 `refine_validation_results/`

原因：

- 这是风险最小、信息量最大的路线
- 最容易产出可比较实验
- 最接近论文里常见的 incremental validation 方式
