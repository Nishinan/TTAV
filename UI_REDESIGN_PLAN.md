# UI Redesign Plan — Time-Travelling Visualizer

> Senior product design + engineering perspective.  
> Goal: minimal, professional, high-end feel. Inspired by Linear / Vercel Dashboard / Raycast design language.

---

## Part 1 — Panel-Level Redesigns (function-panel.tsx)

### Panel 1 — Precision Control

**Problem:** A purple card (`background: #faf5ff`, `color: #6b21a8`, `border: #e5e7eb`) sits inside a grey VS Code sidebar — the most visually jarring element in the entire UI. The purple accent (`#7c3aed`) is completely inconsistent with the rest of the app's blue accent (`var(--accent-blue, #3278F0)`).

**Before:**
```
PRECISION CONTROL
  Focus Mode          [ Balanced ▾ ]

  "Increase sampling weight."

  ╔══════════════════════════════════╗  ← purple card #faf5ff
  ║  Focus Type       [ All Focus ▾ ]║
  ║  [ ⊡ Box Select: ON ]            ║  ← purple button #7c3aed
  ║                                   ║
  ║  ● 3 primary points               ║
  ║  ◦ 0 secondary                    ║
  ╚══════════════════════════════════╝

  [ ↺ Update Projection ]
```

**After:**
```
PRECISION CONTROL
  Focus Mode          [ Balanced ▾ ]
  Focus Type          [ All Focus ▾ ]
  ──────────────────────────────────
  [ ⊡  Start Box Select ]            ← blue accent when active, outline default
  ──────────────────────────────────
  ● 3 primary  ·  ◦ 0 secondary      ← single line, muted text

  [ ↺  Update Projection ]
```

**Key changes:**
- Remove purple card entirely; all rows inline at the same level
- `Focus Mode` and `Focus Type` are sibling rows (both with `Select` dropdowns)
- Box Select button: active state → `var(--accent-blue)` replaces `#7c3aed`
- Primary/secondary point counts merged into one muted line
- Thin `border-top` dividers replace the card border for grouping

**Files:** `function-panel.tsx`  
**Backend change:** None

---

### Panel 2 — Refine Quality

**Problem:** Six numbers laid out flat with no visual grouping or quality feedback. Users must memorize thresholds (e.g. "T > 85% = excellent") to interpret results. Hardcoded colors `#666`, `#1a1a1a`, `#aaa`, `#f0f0f0`, `linear-gradient(#52c41a, #1890ff)` break dark theme.

**Before:**
```
REFINE QUALITY
  Focus Displacement   0.0234
  Global Drift         0.0008
  NP (k=10)           18.3%
  HD-Nbr Rank          12.4
  Trustworthiness      82.1%
  Continuity           79.6%

  [======= Focus / Drift bar =======]
```

**After:**
```
REFINE QUALITY

  POSITION ───────────────────────────
  Displacement        0.0234
  Global Drift        0.0008

  STRUCTURE ──────────────────────────
  NP (k=10)          18.3%    ●       ← orange dot (<85%)
  HD Rank             12.4    ●       ← green dot (≤20)
  Trustworthiness    82.1%    ●       ← green dot (>80%)
  Continuity         79.6%    ●       ← orange dot (<80%)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  focus 18%  ████░░░░░░░  drift 82%
```

**Status dot thresholds:**
| Metric | Green ✓ | Orange ~ | Red ✗ |
|--------|---------|----------|-------|
| NP (k=10) | ≥ 25% | ≥ 10% | < 10% |
| HD Rank | ≤ 10 | ≤ 25 | > 25 |
| Trustworthiness | ≥ 85% | ≥ 70% | < 70% |
| Continuity | ≥ 85% | ≥ 70% | < 70% |

**Key changes:**
- Two section headers `POSITION` / `STRUCTURE` (10px uppercase, `--text-muted`)
- Thin `border-top` dividers between sections
- 8px colored dot (●) per structure metric — green / orange / red based on thresholds
- Metric values: `font-family: var(--metric-value-font)` (monospace), tabular-nums
- Progress bar: replace `#f0f0f0` → `var(--layout-border-color)`, replace hardcoded gradient → `var(--color-success)` to `var(--accent-blue)`
- All text colors → CSS variables

**Files:** `function-panel.tsx`, `index.css`  
**Backend change:** None

---

### Panel 3 — Settings

**Problem:** The `Display` sub-section is wrapped in a white card (`background: #ffffff`, `border: 1px solid #d9d9d9`) nested inside the grey sidebar — a card-inside-card-inside-card pattern. On dark themes the white background is jarring.

**Before:**
```
SETTINGS
  Point Size    ●━━━━  3
  Mode          [ Points ▾ ]
  Neighbors     [ None ▾ ]
  Display   ╔═══════════════════╗
            ║ Show Label     ○  ║  ← white card, hardcoded #ffffff
            ║ Show Index     ○  ║
            ║ Show Trail     ○  ║
            ║ Show Background ● ║
            ╚═══════════════════╝
```

**After:**
```
SETTINGS
  Point Size    ●━━━━  3
  ────────────────────────────
  Mode          [ Points ▾ ]
  Neighbors     [ None ▾ ]
  ────────────────────────────
  Show Label              ○
  Show Index              ○
  Show Trail              ○
  Show Background         ●
```

**Key changes:**
- Remove the nested `border/background` card from Display section entirely
- All 4 Switch rows laid out as flat inline rows (same pattern as Mode/Neighbors)
- `span` min-width unified at 90px so labels and controls form clean two-column alignment
- `border-top` thin dividers replace the visual card grouping

**Files:** `function-panel.tsx`  
**Backend change:** None

---

### Panel 4 — Highlight

**Problem:** `❌` and `🔄` emoji icons are inconsistent cross-platform, look unprofessional in a research tool, and don't respect the VS Code theme color system.

**Before:**
```
HIGHLIGHT
  ❌  Prediction Error        ○
  🔄  Prediction Flip         ○
```

**After (using lucide-react, already installed):**
```
HIGHLIGHT
  ✕  Prediction Error         ○    ← XCircle, color: var(--color-error)
  ↻  Prediction Flip          ○    ← RefreshCw, color: var(--color-warning)
```

**Key changes:**
- `import { XCircle, RefreshCw } from 'lucide-react'`
- Icon size: 13px, vertically aligned with label text
- Colors use semantic CSS variables (not hardcoded red/orange)

**Files:** `function-panel.tsx`  
**Backend change:** None

---

### Global CSS Changes (index.css)

**New semantic tokens:**
```css
html {
  --color-success:       #22c55e;
  --color-warning:       #f59e0b;
  --color-error:         #ef4444;
  --metric-value-font:   'SF Mono', 'Cascadia Code', 'Consolas', monospace;
}
```

**Fix asymmetric block margin:**
```css
/* Before */
.functional-block {
    margin-right: 2em;   /* ← causes content to sit left-biased */
}

/* After */
.functional-block {
    /* remove margin-right: 2em */
    padding-bottom: 0.5em;
    border-bottom: 1px solid var(--layout-border-color);
}
```

**Base font size increase:**
```css
/* Before */
body { font-size: 10px; }

/* After */
body { font-size: 11px; }   /* +1px, significantly improves readability */
```

---

## Part 2 — Overall Layout Redesign

### Current Layout

```
┌─────────────────────────────────────────────────────────────┐
│  RootLayout (horizontal PanelGroup)                          │
│  ┌──────────┬────────────────────────────────────────────┐  │
│  │WebSideBar│  AppCombinedView (78%)                      │  │
│  │  22%     │  ┌──────────────────────────────────────┐  │  │
│  │          │  │  Top Panel (76% height)               │  │  │
│  │Content   │  │  ┌─────────────────┬──────────────┐  │  │  │
│  │Path      │  │  │  Canvas +       │  Functions   │  │  │  │
│  │Method    │  │  │  Timeline  70%  │  Panel  30%  │  │  │  │
│  │DataType  │  │  │                 │              │  │  │  │
│  │TaskType  │  │  └─────────────────┴──────────────┘  │  │  │
│  │          │  ├──────────────────────────────────────┤  │  │
│  │[Start]   │  │  Bottom Dock (24% height)             │  │  │
│  │[Load]    │  │  Influence │ Tokens                   │  │  │
│  │[Sync]    │  │  "Select a training event..."         │  │  │
│  └──────────┴──┴──────────────────────────────────────┴──┘  │
└─────────────────────────────────────────────────────────────┘
```

**Space utilization problems:**
- Canvas actual area: `78% × 76%` ≈ **59% of total screen** (should be ~70%)
- WebSideBar: used only at startup, then idle for the entire session
- Bottom Dock: 24% height permanently occupied, 99% of the time showing empty placeholder

---

### Layout Option A — Merge Bottom Dock into Right Panel (Recommended)

Move `Influence` and `Tokens` from the bottom dock into the right panel's tab bar. Delete `BottomDock` entirely.

**Proposed layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  RootLayout (horizontal PanelGroup)                          │
│  ┌──────────┬────────────────────────────────────────────┐  │
│  │WebSideBar│  AppCombinedView (78%)                      │  │
│  │  22%     │  PanelGroup (horizontal only, no vertical)  │  │
│  │          │  ┌────────────────────┬───────────────────┐ │  │
│  │          │  │  Canvas +          │ Functions         │ │  │
│  │          │  │  Timeline          │ Training Events   │ │  │
│  │          │  │  (70%)             │ Influence  ← new  │ │  │
│  │          │  │                    │ Tokens     ← new  │ │  │
│  │          │  │                    │  (30%)            │ │  │
│  └──────────┴──┴────────────────────┴───────────────────┘ │  │
└─────────────────────────────────────────────────────────────┘
```

**Impact:**
- Canvas height: `76%` → **`100%`** (gains 24% vertical space)
- All analysis panels consolidated in one place (right sidebar)
- Code change: `plotView.tsx` — remove `BottomDock`, add 2 tabs to `FunctionViewPanels`

**Trade-off:** Influence and Canvas can no longer be visible simultaneously. Since Influence requires selecting a training event first (not a continuous parallel workflow), this is acceptable.

**Files:** `web/src/views/plotView.tsx`  
**Backend change:** None

---

### Layout Option B — Auto-Collapse WebSideBar in VS Code Context

In VS Code Webview, `window.vscode` exists and the extension's own sidebar handles all configuration. The `WebSideBar` component is only needed in web dev mode (`npm run dev` / `localhost:5173`).

**Change in `main.tsx`:**
```tsx
const isVSCode = typeof (window as any).vscode !== 'undefined';

<Panel
  defaultSize={isVSCode ? 0 : 22}
  collapsed={isVSCode}
  collapsible
  collapsedSize={0}
  ...
>
  <WebSideBar />
</Panel>
```

**Impact:**
- In VS Code: Canvas width: `78%` → **`100%`** (gains 22% horizontal space)
- In browser dev mode: WebSideBar still visible at 22% as before
- Zero UX regression — VS Code users never needed this panel

**Files:** `web/src/main.tsx`  
**Backend change:** None

---

### Combined A + B Result

```
┌────────────────────────────────────────────────────────┐
│  (VS Code context — WebSideBar auto-collapsed)          │
│  ┌───────────────────────────────────┬───────────────┐ │
│  │  Canvas + Timeline                │ Functions     │ │
│  │                                   │ Training Evts │ │
│  │         (70%)                     │ Influence     │ │
│  │                                   │ Tokens        │ │
│  │                                   │  (30%)        │ │
│  └───────────────────────────────────┴───────────────┘ │
└────────────────────────────────────────────────────────┘
```

Canvas area: from original `78% × 76%` ≈ **59%** → new `100% × 100% × 70%` = **70%**.  
A net gain of ~11 percentage points of absolute screen area for the primary visualization.

---

## Part 3 — Tab Bar Position Optimization

### Current State

The UI has two separate tab systems with inconsistent positioning:

| Location | Tabs | Position |
|----------|------|----------|
| Right panel top | Functions / Training Events | Horizontal, top |
| Bottom dock right edge | Influence / Tokens | Vertical, right side |

This creates visual fragmentation — a horizontal tab system and a vertical tab system coexisting.

### Proposed: Unified Horizontal Top Tabs in Right Panel

After merging (Option A), all 4 panels share one horizontal tab bar at the top of the right panel:

```
┌─ Functions ──── Events ──── Influence ──── Tokens ──────────┐
│                                                               │
│   Content area for the active tab                            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Design details:**
- Tab bar: `size="small"`, `tabBarGutter={0}` (already used)
- Tab labels: 11px, `var(--text-muted)` default, `var(--text-primary)` active
- Active indicator: bottom border `2px solid var(--accent-blue)` (Ant Design default)
- No icon in tabs — text-only, clean

**Why this beats the current vertical right-side tabs:**
- Vertical tabs on the right edge are a non-standard pattern (unusual affordance)
- The labels "Influence" and "Tokens" appear as vertical text which is harder to read
- Horizontal tabs at top is the universal convention (VS Code itself uses this for editor tabs)

---

## Implementation Summary

| # | Change | Files | Effort |
|---|--------|-------|--------|
| 1 | CSS semantic tokens + margin fix + font size | `index.css` | 10 min |
| 2 | Precision Control: remove purple card, flat layout | `function-panel.tsx` | 15 min |
| 3 | Refine Quality: grouping + status dots + mono font | `function-panel.tsx` | 20 min |
| 4 | Settings: remove nested white card | `function-panel.tsx` | 10 min |
| 5 | Highlight: emoji → lucide-react icons | `function-panel.tsx` | 5 min |
| 6 | Layout A: merge bottom dock into right panel tabs | `plotView.tsx` | 20 min |
| 7 | Layout B: auto-collapse WebSideBar in VS Code | `main.tsx` | 5 min |
| **Total** | | | **~85 min** |

### Recommended Implementation Order

1. **#1 CSS tokens** first — foundational, every subsequent change depends on it
2. **#6 Layout A** + **#7 Layout B** — biggest visual impact, frees screen space
3. **#2 Precision Control** — most jarring visual bug, fix it early
4. **#3 Refine Quality** — adds the most UX value (status dots = instant quality feedback)
5. **#4 Settings** + **#5 Highlight** — polish pass

---

## Evaluation (if doing a user study)

These changes directly map to measurable HCI metrics:

| Change | Metric | Method |
|--------|--------|--------|
| Status dots on Refine Quality | Time to assess projection quality | Task timing |
| Layout A (canvas size) | Insight count per session | Think-aloud |
| Precision Control flat layout | Error rate in box-select workflow | Error logging |
| Overall | SUS score (target > 68) | Questionnaire |
| Overall | NASA-TLX cognitive load | Questionnaire |
