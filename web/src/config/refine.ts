export const REFINE_DEFAULTS = {
    blendDecayRatio: 0.35,
    progressPollMs: 1500,
} as const;

// Neighborhood size (top-k) the refine objective and the HD/LD neighbor panels
// operate on. HD neighbors are fetched from the backend at REFINE_TOP_K_MAX depth
// so the Focus Neighbors panel and refine overlay honor any k up to this cap;
// keep the selector clamp and the fetch depth pinned to the same constant.
export const REFINE_TOP_K_MIN = 3;
export const REFINE_TOP_K_MAX = 20;
