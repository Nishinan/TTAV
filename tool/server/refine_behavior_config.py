"""Behavior controls for TTAV refine execution.

This file centralizes two concerns that were previously scattered as ad-hoc
flat keys inside ``vis_config``:
1. progressive intermediate projection updates for the frontend
2. stopping-condition policy for refine training

Callers may override these defaults by passing ``vis_config["refine_behavior"]``
with a partial nested dict, while older flat keys remain supported for
backwards compatibility.
"""

from copy import deepcopy


REFINE_BEHAVIOR_DEFAULTS = {
    "progressive_updates": {
        "enabled": True,
        "snapshot_every_steps": 30,
        "poll_interval_ms": 1500,
        "enable_sampled_metrics": True,
        "sample_metrics_every_steps": 200,
    },
    "stopping": {
        "enable_max_steps": True,
        "max_steps": 5000,
        "enable_time_budget": False,
        "time_budget_seconds": None,
        "enable_loss_converged": True,
        "min_steps": 200,
        "loss_window": 30,
        "loss_rel_tol": 5e-4,
        "priority": ["loss_converged", "time_budget", "max_steps"],
        "safety_loop_cap": 200000,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_refine_behavior_config(vis_config=None) -> dict:
    vis_config = vis_config or {}
    cfg = deepcopy(REFINE_BEHAVIOR_DEFAULTS)

    nested_override = vis_config.get("refine_behavior")
    if isinstance(nested_override, dict):
        cfg = _deep_merge(cfg, nested_override)

    progress_cfg = cfg["progressive_updates"]
    stop_cfg = cfg["stopping"]

    if "refine_progressive_updates_enabled" in vis_config:
        progress_cfg["enabled"] = bool(vis_config.get("refine_progressive_updates_enabled"))
    if "refine_snapshot_every_steps" in vis_config:
        progress_cfg["snapshot_every_steps"] = max(1, int(vis_config.get("refine_snapshot_every_steps", 30)))
    if "refine_progress_poll_interval_ms" in vis_config:
        progress_cfg["poll_interval_ms"] = max(50, int(vis_config.get("refine_progress_poll_interval_ms", 1500)))
    if "refine_enable_sampled_metrics" in vis_config:
        progress_cfg["enable_sampled_metrics"] = bool(vis_config.get("refine_enable_sampled_metrics"))
    if "refine_sample_metrics_every_steps" in vis_config:
        progress_cfg["sample_metrics_every_steps"] = max(1, int(vis_config.get("refine_sample_metrics_every_steps", 200)))

    if "refine_max_steps" in vis_config:
        stop_cfg["max_steps"] = max(1, int(vis_config.get("refine_max_steps", stop_cfg["max_steps"])))
        stop_cfg["enable_max_steps"] = True
    if "refine_enable_max_steps" in vis_config:
        stop_cfg["enable_max_steps"] = bool(vis_config.get("refine_enable_max_steps"))

    if "refine_time_budget_seconds" in vis_config:
        raw = vis_config.get("refine_time_budget_seconds")
        if raw in (None, "", False):
            stop_cfg["time_budget_seconds"] = None
            stop_cfg["enable_time_budget"] = False
        else:
            stop_cfg["time_budget_seconds"] = float(raw)
            stop_cfg["enable_time_budget"] = True
    if "refine_enable_time_budget" in vis_config:
        stop_cfg["enable_time_budget"] = bool(vis_config.get("refine_enable_time_budget"))

    if "refine_enable_loss_converged" in vis_config:
        stop_cfg["enable_loss_converged"] = bool(vis_config.get("refine_enable_loss_converged"))
    if "refine_min_steps" in vis_config:
        stop_cfg["min_steps"] = max(1, int(vis_config.get("refine_min_steps", stop_cfg["min_steps"])))
    if "refine_loss_window" in vis_config:
        stop_cfg["loss_window"] = max(2, int(vis_config.get("refine_loss_window", stop_cfg["loss_window"])))
    if "refine_loss_rel_tol" in vis_config:
        stop_cfg["loss_rel_tol"] = float(vis_config.get("refine_loss_rel_tol", stop_cfg["loss_rel_tol"]))
    if "refine_stop_priority" in vis_config and isinstance(vis_config.get("refine_stop_priority"), list):
        stop_cfg["priority"] = list(vis_config.get("refine_stop_priority"))

    return cfg


def get_refine_behavior_defaults() -> dict:
    return deepcopy(REFINE_BEHAVIOR_DEFAULTS)
