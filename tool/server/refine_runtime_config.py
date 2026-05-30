"""Runtime defaults for TTAV refine behavior.

Keep these values centralized so the frontend, backend, and benchmark docs
can evolve against one named configuration instead of scattered literals.
"""

REFINE_RUNTIME_DEFAULTS = {
    "focus_hd_k": 15,
    "blend_decay_ratio": 0.35,
    "focus_mode": "balanced",
}


def get_refine_runtime_defaults() -> dict:
    return dict(REFINE_RUNTIME_DEFAULTS)
