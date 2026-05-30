# Refine Benchmark Workflow

更新时间：2026-05-30

本文档描述如何用一套固定流程评估 TTAV 的 refine 设计是否真正提升效果。

## 1. 评估目标

我们不是只问：

- `NP` 有没有升

而是同时问：

- 局部结构是否更好
- 全局是否仍然稳定
- 交互时间是否还能接受

因此 benchmark 统一分三类指标：

### 1.1 局部质量

- `NP`
- `MRH`
- `Trustworthiness`
- `Continuity`

### 1.2 全局稳定性

- `Global Drift`
- 非 focus 区域平均位移

### 1.3 交互成本

- `Latency`

只有三者一起看，才能判断“是否提升”。

## 2. 当前默认推荐

基于 `backdoor / TimeVis_1 / epoch_10` 的现有实验：

- `focus_set` 默认推荐：`seeds_plus_hd`
- blending 默认推荐：`bbox + focus_set`
- local training 默认推荐：显式 `local visualizer` 副本

原因：

- `seeds_plus_hd` 在非 trivial 候选里 direct local quality 最强
- `bbox + focus_set` 比 `bbox-only` 更能保住局部变化
- 同时它的 drift 仍显著低于 direct replace
- `local visualizer` 已验证不会污染 `global visualizer`

## 3. 一键运行方式

### 3.1 配置文件

复制并修改：

- `tests/refine_benchmark_config.example.json`

需要填写：

- `content_path`
- `baseline_vis_id`
- `epoch`
- `seeds`
- `bbox`
- `hd_k`
- `focus_mode`
- `blend_decay_ratio`

当前运行时默认值：

- 后端默认 `focus_hd_k=15`
- 前后端默认 `blend_decay_ratio=0.35`
- 默认 `focus_mode=balanced`

### 3.2 总控脚本

运行：

```bash
python3 tests/run_refine_benchmark_suite.py \
  --config tests/refine_benchmark_config.example.json
```

它会顺序执行：

1. `test_refine_phase1_focus_set.py`
2. `test_refine_phase1b_refine_compare.py`
3. `test_refine_phase2_blending.py`
4. `test_refine_phase2b_focus_aware_blending.py`
5. `test_refine_phase3_local_visualizer.py`
6. `build_refine_benchmark_summary.py`

如果只想跳过耗时阶段：

```bash
python3 tests/run_refine_benchmark_suite.py \
  --config tests/refine_benchmark_config.example.json \
  --skip-phase1b \
  --skip-phase3
```

## 4. 输出位置

所有实验输出都在：

- `refine_validation_results/`

关键子目录：

- `phase1_focus_set/`
- `phase1b_refine_compare/`
- `phase2_blending/`
- `phase2b_focus_aware_blending/`
- `phase3_local_visualizer/`
- `benchmark_summary/`

## 5. 如何判定“新方法更好”

建议使用下面这套判断规则：

### 5.1 对 local visualizer / focus_set 的判断

优先看：

- `NP` 更高
- `MRH` 更低
- `Trustworthiness` 更高
- `Continuity` 更高

### 5.2 对 blending 的判断

优先看：

- `focus_shift` 不要被压得过小
- `global_drift` 相比 direct replace 明显下降

### 5.3 对整体方案的判断

一个方案只有在下面三条同时成立时，才算真正提升：

1. 局部指标变好
2. 全局 drift 没失控
3. latency 仍能接受

## 6. 当前已知限制

- 当前前端构建环境缺 `node_modules`，所以还没有完成正式的前端打包校验
- blended 指标目前主要在运行时计算，适合交互评估；若后续需要论文级复现，建议再补一条离线评估路径
- 当前推荐结论来自 `backdoor` 数据集，需要后续在更多数据集上复核
