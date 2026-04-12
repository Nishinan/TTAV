
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Time-Travelling Visualizer** is a VS Code extension for visualizing and debugging ML model training. It lets users step through training epochs, inspect how the high-dimensional representation space evolves, and analyze influence functions.

## Architecture

Three-tier system communicating over HTTP:

```
VS Code Extension (TypeScript)   ←IPC→   Web Frontend (React/TS)   ←HTTP:5050→   Python Backend (Flask)
extension/                                web/                                      tool/
```

- **Extension** (`extension/src/`) — registers commands, tree views, and a webview panel. `extension.ts` activates, `control.ts` drives config/workflows, `views.ts` registers all UI panels.
- **Web** (`web/src/`) — React SPA served in the webview. `plotView.tsx` is the main visualization view; Zustand manages state (`state/state.unified.ts`); `communication/backend.ts` makes axios calls to the Flask server.
- **Backend** (`tool/`) — Flask server at port 5050. `server/server.py` is the entry; `server/run_visualization.py` initializes configs and launches the visualization pipeline; `visualize/` contains all algorithm implementations.

### Visualization Strategy Pattern

Algorithms are implemented as strategies under `tool/visualize/strategy/`:
- **DVI** (`dvi_strategy.py`) — reconstruction + UMAP loss
- **TimeVis** (`timevis_strategy.py`) — adds temporal continuity constraints
- **DynaVis** (`tool/visualize/dynavis/runner.py`) — separate runner with motion/velocity loss
- **UMAP** — standard UMAP via `projector.py`

Training event detection (PredictionFlip, ConfidenceChange, SignificantMovement, etc.) lives in `tool/visualize/training_event.py`.

### Data Flow

1. User configures session in the extension → extension POSTs `/syncSession` + `/startVisualize` to backend
2. Backend runs the selected visualization strategy and writes projection results
3. Web frontend fetches epoch data and renders the scatter plot
4. User interactions (focus context, time-travel) send updates back to the backend via `/updateFocusContext`

## Setup

```bash
# Python backend (one-time)
bash setup.sh        # creates conda env "visualizer" (Python 3.10)

# Frontend dependencies
cd web && npm install
cd ../extension && npm install
```

## Development

```bash
# Terminal 1 – Backend (Flask, port 5050)
conda activate visualizer
cd tool/server && python server.py

# Terminal 2 – Web dev server (Vite, port 5173)
cd web && npm run dev

# Terminal 3 – Extension
# VS Code: Run and Debug → "Run Extension" (F5)
```

VS Code tasks in `.vscode/tasks.json` automate starting/stopping these processes; launch configs are in `.vscode/launch.json`.

## Build

```bash
cd web && npm run build           # outputs to web/dist/
cd extension && npm run compile   # outputs to extension/out/
```

## Lint

```bash
cd web && npm run lint
cd extension && npm run lint
```

## Key Config Details

- Backend port is hardcoded to **5050**; Vite dev server uses **5173**
- GPU selection via `gpu_id` setting (`-1` = CPU)
- Supported vis methods: `DVI`, `TimeVis`, `DynaVis`, `UMAP`
- Method hyperparameters (lambda, n_neighbors, max_epochs, etc.) are declared in `extension/package.json` under `contributes.configuration`
