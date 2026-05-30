# Refine Benchmark Summary

Generated: 2026-05-30T10:51:19

## Inputs
- Phase 1 focus-set source: `metrics_20260530_093731.json`
- Phase 1b refine compare: `summary_20260530_095238.csv`
- Phase 2 bbox-only blending: `summary_20260530_100118.csv`
- Phase 2b focus-aware blending: `summary_20260530_103218.csv`
- Phase 3 local visualizer check: `local_visualizer_check_20260530_101636.json`

## Current Recommendation
- Default focus-set candidate on the current `backdoor` experiment: `seeds_plus_hd`
- Reason: it gives the strongest non-trivial direct local quality (`NP=9.38`, `MRH=175.8`) while focus-aware blending keeps global drift to `0.0022`.
- Local visualizer validation: global model unchanged max abs diff = `0.0`; local-vs-global embed max abs diff = `0.0424`.

## How To Evaluate Improvement
- Local quality: `NP`, `MRH`, `Trustworthiness`, `Continuity`.
- Global stability: `Global Drift` after blending, plus non-focus displacement.
- Interaction cost: `Latency`.
- Use all three together. A method is better only if local quality improves without global drift or latency becoming unacceptable.

## Combined Table

| candidate | focus_set_size | NP direct | MRH direct | focus_shift direct | drift direct | focus_shift bbox+blend | drift bbox+blend | focus_shift focus+blend | drift focus+blend | latency(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seeds_only | 3 | 10.00 | 186.4 | 0.0258 | 0.0145 | 0.0258 | 0.0002 | 0.0258 | 0.0002 | 19.35 |
| seeds_plus_bbox | 44 | 7.27 | 216.8 | 0.0585 | 0.0100 | 0.0585 | 0.0002 | 0.0585 | 0.0002 | 63.12 |
| seeds_plus_bbox_plus_hd | 75 | 8.00 | 202.7 | 0.0415 | 0.0109 | 0.0225 | 0.0002 | 0.0415 | 0.0019 | 64.33 |
| seeds_plus_hd | 48 | 9.38 | 175.8 | 0.0481 | 0.0136 | 0.0164 | 0.0002 | 0.0481 | 0.0022 | 63.09 |

## Notes
- Phase 1 selected seeds: `[10975, 11490, 17685]`
- Phase 1 bbox: `{'x_min': 3.577545, 'x_max': 3.653369, 'y_min': 1.164337, 'y_max': 1.236651}`
- `bbox-only blend` is the most conservative option.
- `bbox+focus blend` better preserves local changes for larger focus sets while still reducing drift far below direct replacement.