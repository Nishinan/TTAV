"""Runtime defaults for TTAV refine behavior.

Keep these values centralized so the frontend, backend, and benchmark docs
can evolve against one named configuration instead of scattered literals.
"""

REFINE_RUNTIME_DEFAULTS = {
    "focus_hd_k": 15,
    "blend_decay_ratio": 0.35,
    "focus_mode": "balanced",
    "focus_set_strategy": "seeds_plus_hd",
    "refine_max_steps": 20000,
    "refine_time_budget_seconds": None,
    "refine_log_every_steps": 100,
    "refine_min_steps": 300,
    "refine_loss_window": 50,
    "refine_loss_rel_tol": 1e-4,
}


def get_refine_runtime_defaults() -> dict:
    return dict(REFINE_RUNTIME_DEFAULTS)
