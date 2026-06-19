# HCI Improvement Directions — Time-Travelling Visualizer

> Based on literature review of CHI / IEEE VIS 2023–2025 works.  
> Six concrete feature directions, ordered by implementation cost.

---

## Direction 1 — Density Contour Overlay on Scatter Plot

**Source:** WizMap (ACL 2023) — scalable interactive embedding visualization with map-like interaction design.

**Problem:** The scatter plot is a raw point cloud. In dense regions, cluster boundaries are invisible — especially when zoomed out. Users can't grasp the overall distribution structure without individually inspecting points.

**Feature:** Add a `DensityContourOverlay` SVG layer (following the existing `NeighborOverlay` pattern in `chart.tsx`) that renders kernel density estimation (KDE) contours over the scatter plot:
- Compute KDE on-the-fly in the browser using `d3-contour` / `d3.contourDensity()` from the current epoch's `projection` array.
- Render as translucent filled `<path>` elements in the chart's SVG overlay, colored per class (same color palette as the points, lower opacity).
- Contours update on every epoch switch and stay in sync with user pan/zoom via the embedding-atlas `proxy` coordinate transform.

**Value:** Users instantly see which classes are tightly clustered vs. spread, whether classes overlap, and whether a given refine made the separation clearer — without counting individual dots.

**Backend change needed:** No.

---

## Direction 2 — LIVE / FINAL State Badge on Chart

**Source:** A Survey on Progressive Visualization (IEEE TVCG 2024) — design principle: intermediate results and final results must be visually distinct or users will misjudge data quality.

**Problem:** During refine (`isRefining.current`, `plotView.tsx:889`) and after refine completes, the scatter plot looks identical. Users cannot tell whether they are viewing a mid-computation snapshot or the finished result.

**Feature:** An absolute-positioned status badge in the top-right corner of the chart canvas:

| State | Badge |
|-------|-------|
| Idle | (hidden) |
| Running | `● Refining… 150 steps` — blue pulsing dot animation |
| Done | `✓ Refined` — green, fades out after 3 s |

Additional visual cue: a 2 px colored `outline` on the chart container while refining (blue → removed on completion).

**Implementation:** Promote `isRefining.current` to a Zustand store field `refineStatus: 'idle' | 'running' | 'done'`; `steps_completed` is already available from the existing real-time polling loop — just pipe it into the badge text.

**Backend change needed:** No.

---

## Direction 3 — HD Feature Discriminability Panel for Selected Cluster

**Source:** DimBridge (IEEE VIS 2024) — interactive explanation of visual patterns in dimensionality reductions via predicate logic.

**Problem:** After box-selecting a group of points, users know *which* points they selected but not *why* those points cluster together in the HD space. They refine blindly, without understanding the underlying feature structure.

**Feature:** When `selectedIndices` is non-empty, show a "Cluster Analysis" section inside the "Selected" FunctionalBlock in the sidebar:

- **Lightweight frontend version:** Compute HD-neighbor density ratio from `originalNeighbors` already cached on the client — report a "cluster cohesion" score (high / medium / low) instantly.
- **Full backend version:** New API endpoint `/analyzeCluster` accepts `selectedIndices`; backend runs ANOVA F-score or mutual information over `full_feat` to find the top-5 feature dimensions that best discriminate the selection from the rest, returning:

```json
{
  "top_dims": [
    { "dim": 42, "coverage": 0.78, "range": [0.3, 0.8] },
    { "dim":  7, "coverage": 0.65, "range": [-0.1, 0.4] }
  ]
}
```

- Frontend renders a compact horizontal bar list: `Dim 42 ████░░  78%`.

**Value:** Answers "why do these points cluster together?" before the user commits to a refine — enables hypothesis-driven, not guess-driven, interaction.

**Backend change needed:** Yes (new `/analyzeCluster` endpoint + feature-space ANOVA).

---

## Direction 4 — Drag a Point on the Scatter Plot to Trigger Refine

**Source:** ParamsDrag (IEEE VIS 2024) — users drag structure-level features directly in a visualization to steer parameter space exploration, instead of adjusting parameters in a side panel.

**Problem:** The current refine workflow is 4 separate steps across two UI regions:  
① Click "Box Select" in sidebar → ② Draw box on canvas → ③ Click "Update Projection" in sidebar → ④ Wait.  
Steps ① and ③ are in the sidebar; ② is on the canvas — the user's attention bounces back and forth.

**Feature:** A new "Drag Refine" toggle in the sidebar. When active:

1. Hovering over a point shows a grab cursor and a highlight ring (reuses `NeighborOverlay`).
2. User **click-drags** the point to a target position; during the drag the canvas shows:
   - Origin: semi-transparent grey circle ("from")
   - Path: dashed arrow line
   - Target: solid colored circle ("to")
3. On mouse-up: the dragged point (plus its HD neighbors) is automatically set as `selectedIndices`; the from→to offset vector is passed as a soft directional constraint to refine; `onUpdateProjection()` fires immediately.

This collapses 4 steps into **1 gesture** — canonical direct manipulation.

**Implementation:** `EmbeddingView` in embedding-atlas provides `onHover` giving the hovered data point; add `onMouseDown` / `onMouseMove` / `onMouseUp` handlers to the chart overlay div (already exists for box select), record drag vector, auto-populate `selectedIndices`, call refine.

**Backend change needed:** No (reuses existing refine API protocol).

---

## Direction 5 — Refine History Provenance Panel

**Source:** AGDebugger (CHI 2025) — interactive debugging of AI systems; user study finding: users' top need is a **revertible step history**, not more automation.

**Problem:** After multiple refine operations, users have no record of what was done, to which points, on which epoch, or what the quality was. There is no way to compare "was refine #2 better than refine #3?" or to roll back.

**Feature:** A "Refine History" FunctionalBlock in the sidebar (default collapsed). After each successful refine, a `RefineRecord` is pushed to a `refineHistory` array in the Zustand store:

```typescript
type RefineRecord = {
  id: number;
  timestamp: string;            // "14:32:05"
  epoch: number;
  focusCount: number;
  metrics: {
    trustworthiness: number;
    continuity: number;
    meanRankHD: number;
  };
  snapshotProjection: number[][];  // saved 2D coordinates for restore
};
```

UI: a compact timeline list —

```
● #3  Epoch 5  14:32:05  12 pts  T=82.1%  C=79.3%  [Restore]
● #2  Epoch 5  14:31:10   8 pts  T=74.6%  C=71.2%
● #1  Epoch 3  14:28:43   3 pts  T=68.3%  C=65.7%
```

Clicking **[Restore]** writes the saved `snapshotProjection` back into the displayed epoch data, allowing side-by-side comparison between refine iterations.

**Backend change needed:** No (snapshot is saved client-side from the data already returned by the refine polling loop).

---

## Direction 6 — Guidance Arrows on Focus Points Before Refine

**Source:** SpaceEditing (IUI 2024) — the interface shows recommended movement directions on selected points based on HD neighbor structure, reducing the cognitive burden of "where should I move these?".

**Problem:** After selecting focus points, users must guess where to "push" them — they only see the 2D projection and have no signal about where the HD neighborhood structure suggests they should go.

**Feature:** After box selection, for each selected point, render a guidance arrow in the `NeighborOverlay` SVG layer:

1. Retrieve the point's `originalNeighbors` (HD top-k, already cached client-side).
2. Compute the 2D centroid of those HD neighbors in the *current projection* (screen coordinates via embedding-atlas `proxy`).
3. Draw an SVG dashed arrow from the selected point toward that centroid.
4. Arrow color = the dominant class color of the HD neighbors (e.g., green if the HD neighbors mostly belong to class A).

**Meaning to user:** *"Your high-dim neighbors are in this direction in 2D — but you're currently separated from them. Refine will pull you toward here."*

Users see the expected movement direction *before* pressing Update Projection, eliminating blind triggering.

**Backend change needed:** No (uses `originalNeighbors` and `projection` data already available on the client).

---

## Summary

| # | Feature | Core Value | Files to Change | Backend? |
|---|---------|-----------|----------------|---------|
| 1 | Density contour overlay | See overall distribution structure at a glance | `chart.tsx` | No |
| 2 | LIVE / FINAL state badge | Distinguish intermediate vs. final results | `chart.tsx`, `plotView.tsx` | No |
| 3 | HD feature discriminability panel | Understand *why* selected points cluster | `function-panel.tsx`, new `/analyzeCluster` | Yes |
| 4 | Drag-to-refine gesture | 4-step workflow → 1 drag | `chart.tsx`, `plotView.tsx` | No |
| 5 | Refine history provenance panel | Revertible history, iteration comparison | `function-panel.tsx`, store | No |
| 6 | Guidance arrows on focus points | Know expected movement direction before refining | `chart.tsx` NeighborOverlay | No |

### Recommended implementation order (by effort vs. impact)

1. **Direction 2** — State badge: lowest cost, eliminates a real user confusion point.
2. **Direction 6** — Guidance arrows: pure frontend, high HCI value, reuses existing overlay.
3. **Direction 1** — Density contours: adds a new visual layer, needs `d3-contour` dependency.
4. **Direction 5** — Provenance panel: medium effort, high research value (good for user study).
5. **Direction 4** — Drag-to-refine: most impactful UX change, moderate frontend effort.
6. **Direction 3** — HD feature panel: highest research contribution, requires backend work.

---

## Evaluation Plan (if doing a user study)

Based on practices from CHI / IEEE VIS papers in this domain:

| Method | What it measures |
|--------|----------------|
| **Think-aloud protocol** | Where users get confused, what insights they discover |
| **Semi-structured interview** | Intent, workflow, improvement suggestions |
| **SUS questionnaire** | Overall usability score (target: > 68) |
| **NASA-TLX** | Cognitive load (lower = better) |
| **Insight count** | Number of meaningful training phenomena discovered per session |
| **Task completion time** | Time to identify a predefined anomaly in training |
| **Baseline comparison** | vs. only using loss curves / accuracy plots |
