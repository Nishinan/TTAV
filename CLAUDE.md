# CLAUDE.md

## 🧠 Role
You are Claude Code, working as a **reliable teammate**, not a one-shot assistant.

Your goal is to produce **correct, minimal, and verifiable changes**.

---

## 📂 Project Context

### What is TTAV?
**Time-Travelling Adapter Visualization** (TTAV): An interactive real-time refinement system where users select focus points in a 2D visualization and the system dynamically refines projections for those points without breaking global topology. Uses test-time adaptation (LoRA injection in fine mode) for efficient local micro-adjustments.

### Repository Structure
```
time-travelling-visualizer/
├── tool/
│   ├── visualize/          # Core visualization strategies (DVI, TimeVis, DynaVis, UMAP)
│   │   ├── strategy/       # Strategy implementations + loss functions + trainers
│   │   └── visualize_model.py  # VisModel (encoder/decoder) & LoRA injection
│   ├── server/             # Flask backend (port 5050)
│   │   ├── server.py       # Main API endpoints
│   │   ├── server_utils.py # Metric computation & caching
│   │   └── run_visualization.py  # Entry point, strategy/visualizer orchestration
│   └── [data/epoch processing scripts]
├── web/                    # React frontend (TypeScript)
│   ├── src/
│   │   ├── views/plotView.tsx  # Main 2D plot + interaction handling
│   │   ├── communication/backend.ts  # API client
│   │   └── state/          # Global state (Zustand)
├── tests/test_core.py      # Unit & integration tests (23 tests, all passing)
└── CLAUDE.md               # This file
```

### How to Run

**Prerequisites:**
```bash
conda activate visualizer  # Assumes environment set up with PyTorch, TensorFlow (optional), etc.
cd /home/shinan/time-travelling-visualizer
```

**Backend (Flask API, port 5050):**
```bash
python tool/server/server.py
```

**Frontend (React, port 3000):**
```bash
cd web
npm install  # First time only
npm start
```

**Tests (all 23 tests, ~10s runtime):**
```bash
python tests/test_core.py
```

### Build & Visualization Entry Point
```bash
python tool/server/run_visualization.py --config <dataset.json> --method <DVI|TimeVis|DynaVis|UMAP>
```
Outputs: `visualize/{method}_{id}/epochs/epoch_N/projection.npy` (per-epoch 2D embeddings)

---

## 🎯 Task Execution Rules

### Always follow this workflow:

1. **Understand** – Restate the goal in your own words
2. **Gather** – Read relevant files, check errors, review docs
3. **Propose** – Outline a clear plan before coding
4. **Confirm** – Wait for user approval on non-trivial tasks
5. **Implement** – Make minimal, focused changes
6. **Verify** – Run tests or confirm correctness

---

## 📌 Prompt Interpretation

When receiving a task, always identify:

- **Goal** – What needs to be done
- **Context** – Which files or information matter
- **Constraints** – What must NOT be changed
- **Done Criteria** – What defines success

If anything is unclear → ask clarifying questions before proceeding.

---

## 🚫 Constraints

- Do NOT refactor large portions of code
- Do NOT modify files unrelated to the task
- Do NOT introduce new dependencies unless explicitly requested
- Do NOT change public interfaces unless asked

---

## 📏 Scope Control

- Only modify files explicitly mentioned or directly relevant
- If scope is ambiguous → ask before acting

---

## 📤 Output Rules

- Prefer showing diffs over full file contents
- Show only necessary changes
- Do NOT output entire files unless specifically requested
- Use concise explanations

---

## 🔍 Verification

Before finishing a task, always:

- Ensure the code runs without syntax errors
- Confirm no unintended regressions
- Run existing tests if available
- Check edge cases (e.g., empty inputs, errors)

---

## 🧪 Testing

- Add tests if they are missing and the change requires them
- Do NOT remove or disable existing tests

---

## 🔎 Code Review Awareness

When reviewing your own or others' changes:

- Look for bugs or regressions
- Identify risky changes (e.g., performance, security)
- Ensure alignment with project constraints

---

## 🔌 External Tools (MCP)

Use external tools (filesystem, shell, web fetch) only when:

- Required data is outside the repository
- Data is dynamic or too large to reason about manually
- Tool usage improves accuracy or efficiency

---

## 🧩 Skill Reuse

If you perform the same type of task repeatedly:

- Suggest turning the pattern into a reusable skill
- Keep skills focused and well-scoped

---

## 🔁 Continuous Improvement

If mistakes happen repeatedly:

- Suggest improvements to CLAUDE.md
- Keep rules short, actionable, and practical

---

## ⚠️ Safety

- If uncertain → ask for clarification
- If the change is risky (e.g., data loss, breaking change) → explain the risk first and wait for confirmation

---

## 🏗️ Architecture & Data Flow

### Three-Tier Stack
```
VS Code Extension ↔ React Frontend (port 3000) ↔ Flask Backend (port 5050)
     (optional)           TypeScript               Python
```

### Model Output Format (Unified)
All models now return **4-tuple**: `(emb_to, emb_from, recon_to, recon_from)`
- `VisModel` (TimeVis, DVI): encoder/decoder pair
- All loss functions (`SingleVisLoss`, `DVILoss`, `HybridLoss`, `BoundaryAwareLoss`) expect this format

### Training Flow
1. **static training** (`train_vis_model()`):
   - Constructs high-D edges (spatial + temporal for TimeVis, per-epoch for DVI)
   - Runs dataloader + trainer
   - Saves model to `visualize/{method}_{id}/` or `epochs/epoch_N/vis_model.pth` (DVI per-epoch)

2. **dynamic refine** (TTAV):
   - User selects focus points via frontend
   - `updateFocusContext` API → `strategy.refine(focus_indices=...)`
   - Fine-tunes with lightweight loss (no temporal) for ~1.5s
   - Saves refined projections to `visualize/{method}_{id}_refined/epochs/epoch_N/projection.npy`
   - Frontend loads with `refine_flag=True` query parameter

### Key Concepts

**Focus Modes (for fine-tuning):**
- `"coarse"`: weight=1.0, full model trainable
- `"balanced"`: weight=2.0, moderate update
- `"fine"`: weight=5.0 + LoRA injection (only LoRA params trainable)

**Metrics:**
- **Trustworthiness (T)**: % of k-NN neighbors in low-D that were also k-NN in high-D
- **Continuity (C)**: % of k-NN neighbors in high-D that are also k-NN in low-D
- **Computation**: `_compute_trustworthiness_continuity(high_neighbors, low_neighbors)` (shared core)
- **Caching**: Metrics cached to `visualize/{method}_{id}/metrics_cache.json` to avoid re-computation

---

## 🔌 API Endpoints (Flask)

### GET `/getProjectionNeighbors`
**Purpose:** Load 2D embedding and neighborhood information for a given epoch.

**Query Parameters:**
- `content_path`: path to dataset root
- `vis_method`: "DVI" | "TimeVis" | "DynaVis" | "UMAP"
- `vis_id`: visualization id (e.g., "0")
- `epoch`: epoch number
- `refine_flag`: boolean (default: false) — if true, loads from `{method}_{id}_refined/` directory

**Response:**
```json
{
  "projections": [[x1, y1], [x2, y2], ...],
  "neighbors": [
    { "high_neighbors": [...], "low_neighbors": [...] },
    ...
  ],
  "trustworthiness": 0.85,
  "continuity": 0.82
}
```

### POST `/updateFocusContext`
**Purpose:** Trigger TTAV refinement for selected points.

**Body:**
```json
{
  "content_path": "/path/to/dataset",
  "selected_indices": [42, 71, 99],
  "focus_mode": "fine",
  "vis_method": "DVI",
  "vis_id": "0"
}
```

**Response:**
```json
{ "status": "success" }
```

**Side Effect:** Saves refined projections to `visualize/{method}_{id}_refined/epochs/...` on disk.

### GET `/`
**Purpose:** Serve React SPA (index.html from `web/dist/`).

---

## 🎯 Key Files for Common Tasks

| Task | Primary Files |
|------|---|
| Add a new loss function | `tool/visualize/strategy/losses.py` (inherit `nn.Module`) |
| Modify training flow | `tool/visualize/strategy/{dvi,timevis,dynavis}_strategy.py:train_vis_model()` |
| Add TTAV fine-tuning feature | `strategy.refine()` + `server.py:updateFocusContext` + `plotView.tsx:handleUpdate` |
| Compute new metrics | `tool/server/server_utils.py:_compute_trustworthiness_continuity()` (core) |
| Change API response format | `tool/server/server.py` + `web/src/communication/backend.ts` (sync both) |
| Debug model loading | `tool/visualize/strategy/projector.py` (DVIProjector, TimeVisProjector, etc.) |

---

## ⚡ Performance Notes

- **Metric caching**: `calculate_visualize_metrics()` caches T&C to `metrics_cache.json` — avoids re-computation on repeated API calls
- **TTAV time budget**: `refine()` runs for ~1.5s max, fine-tunes only on local neighborhood (default ~15 neighbors × num_focus_points)
- **LoRA efficiency**: Fine mode injects low-rank adapters into decoder layers only — fewer trainable params than full model
- **Data handler**: Both `DataHandler` (TimeVis) and `DVIDataHandler` (DVI) return 6-element tuples: `(edge_to, edge_from, a_to, a_from, idx_to, idx_from)` for TTAV index tracking

---

## 🧪 Testing

All tests pass (23/23):
```bash
conda run -n visualizer python tests/test_core.py
```

Tests cover:
- Model forward formats (VisModel 4-tuple, SingleVisualizationModel dict)
- Loss function correctness (DVILoss + VisModel, SingleVisLoss with TTAV weights)
- Dataset handlers (6-element return for index tracking)
- Trainer mechanics (early stopping, LoRA injection)
- Metric computation (core T&C algorithm)
- Utilities (projector classes, metric caching, strategy_abstract torch.save fix)