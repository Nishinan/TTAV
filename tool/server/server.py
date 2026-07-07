import os
import sys
import shutil
import json
import uuid
import time
import traceback
import threading
from pathlib import Path
import numpy as np
import torch
# from llm_agent import call_llm_agent
from run_visualization import visualize_run, init_visualize_component

from flask import request, Flask, jsonify, make_response, send_file,send_from_directory
from flask_cors import CORS, cross_origin
from run_visualization import initialize_config
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../visualize')

from server_utils import *
from refine_runtime_config import REFINE_RUNTIME_DEFAULTS
from refine_behavior_config import resolve_refine_behavior_config

# flask for API server
app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

# Check for "--dev" argument
is_dev_mode = "--dev" in sys.argv


# ttav_context = {
#     "focus_mode": "coarse",
#     "selected_indices": None,
#     "mask": None # Step 3: 向量化布尔掩码
# }


# Global session to keep objects alive for the single active scene
active_session = {
    "strategy": None,
    "visualizer": None,
    "content_path": None,
    "vis_id": None,
    "vis_method": None,
    "vis_config": {},
    "eif_session_info": None,
}

EIF_BUNDLE_ROOT = Path("/root/project/Dataset/eif_bundles")
EIF_STATIC_SESSION = "EIF_STATIC_BUNDLE"
EIF_SESSION_STATUS_FILE = "eif_session_status.json"
EIF_BUILD_TASKS = {}

# Cache of fully-initialized strategy objects, keyed by (content_path, vis_method, vis_id).
# Avoids re-creating TimeVis and re-loading vis_model.pth weights when the user
# switches back to a previously visited sample.
_EIF_STRATEGY_CACHE: dict[tuple, dict] = {}
_EIF_STRATEGY_CACHE_LOCK = threading.Lock()


def _strategy_cache_key(content_path: str, vis_method: str, vis_id: str) -> tuple:
    return (str(Path(content_path).resolve()), str(vis_method), str(vis_id))


def _get_cached_strategy(content_path: str, vis_method: str, vis_id: str) -> dict | None:
    key = _strategy_cache_key(content_path, vis_method, vis_id)
    with _EIF_STRATEGY_CACHE_LOCK:
        return _EIF_STRATEGY_CACHE.get(key)


def _put_cached_strategy(content_path: str, vis_method: str, vis_id: str, strategy, visualizer, config: dict):
    key = _strategy_cache_key(content_path, vis_method, vis_id)
    with _EIF_STRATEGY_CACHE_LOCK:
        _EIF_STRATEGY_CACHE[key] = {"strategy": strategy, "visualizer": visualizer, "config": config}
    print(f"[StrategyCache] Cached strategy for {Path(content_path).name} {vis_method}/{vis_id}", flush=True)


def _invalidate_strategy_cache(content_path: str, vis_method: str, vis_id: str):
    key = _strategy_cache_key(content_path, vis_method, vis_id)
    with _EIF_STRATEGY_CACHE_LOCK:
        removed = _EIF_STRATEGY_CACHE.pop(key, None)
    if removed is not None:
        print(f"[StrategyCache] Invalidated cache for {Path(content_path).name} {vis_method}/{vis_id}", flush=True)


def _status_path_for_content(content_path):
    return Path(content_path) / EIF_SESSION_STATUS_FILE


def _normalize_eif_session_status(payload, *, content_path=None, sample_id=None, vis_method=None, vis_id=None):
    now_ms = int(time.time() * 1000)
    normalized = dict(payload or {})
    normalized.setdefault("sample_id", sample_id or "")
    normalized.setdefault("content_path", str(content_path) if content_path is not None else "")
    normalized.setdefault("vis_method", vis_method or "TimeVis")
    normalized.setdefault("vis_id", str(vis_id or "1"))
    normalized.setdefault("eif_bundle", True)
    normalized.setdefault("trainable_session_status", "registered")
    normalized["refine_ready"] = normalized.get("trainable_session_status") == "ready"
    normalized.setdefault("message", "EIF bundle registered.")
    normalized.setdefault("updated_at", now_ms)
    return normalized


def _read_eif_session_status(content_path, *, sample_id=None, vis_method=None, vis_id=None):
    status_path = _status_path_for_content(content_path)
    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                return _normalize_eif_session_status(
                    json.load(f),
                    content_path=content_path,
                    sample_id=sample_id,
                    vis_method=vis_method,
                    vis_id=vis_id,
                )
        except Exception:
            traceback.print_exc()
    return _normalize_eif_session_status(
        {},
        content_path=content_path,
        sample_id=sample_id,
        vis_method=vis_method,
        vis_id=vis_id,
    )


def _write_eif_session_status(content_path, status_payload):
    normalized = _normalize_eif_session_status(status_payload, content_path=content_path)
    status_path = _status_path_for_content(content_path)
    status_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_info_path = Path(content_path) / "dataset" / "info.json"
    if dataset_info_path.exists():
        try:
            dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
            dataset_info["eif_bundle"] = True
            dataset_info["refine_capable"] = True
            dataset_info["trainable_session_status"] = normalized["trainable_session_status"]
            dataset_info["refine_ready"] = normalized["refine_ready"]
            dataset_info["trainable_session_message"] = normalized["message"]
            dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            traceback.print_exc()

    vis_info_path = Path(content_path) / "visualize" / f"{normalized['vis_method']}_{normalized['vis_id']}" / "info.json"
    if vis_info_path.exists():
        try:
            vis_info = json.loads(vis_info_path.read_text(encoding="utf-8"))
            vis_info["eif_bundle"] = True
            vis_info["trainable_session_status"] = normalized["trainable_session_status"]
            vis_info["refine_ready"] = normalized["refine_ready"]
            vis_info["trainable_session_message"] = normalized["message"]
            vis_info_path.write_text(json.dumps(vis_info, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            traceback.print_exc()

    return normalized


def _is_trainable_session_ready(content_path, vis_method, vis_id):
    status = _read_eif_session_status(content_path, vis_method=vis_method, vis_id=vis_id)
    model_path = Path(content_path) / "visualize" / f"{vis_method}_{vis_id}" / "vis_model.pth"
    return status.get("trainable_session_status") == "ready" and model_path.exists()


def _build_eif_task_key(content_path, vis_method, vis_id):
    return f"{content_path}::{vis_method}::{vis_id}"


def _build_fast_fit_vis_config(embedding_dim, vis_config):
    next_config = dict(vis_config or {})
    width_1 = min(128, max(32, embedding_dim // 16))
    width_2 = min(32, max(8, width_1 // 4))
    encoder_dims = [embedding_dim, width_1, width_2, 2]
    decoder_dims = [2, width_2, width_1, embedding_dim]
    next_config["dimension"] = embedding_dim
    next_config["encoder_dims"] = encoder_dims
    next_config["decoder_dims"] = decoder_dims
    next_config.setdefault("gpu_id", -1)
    next_config.setdefault("resolution", [300, 300])
    next_config.setdefault("fast_fit_steps", 240)
    next_config.setdefault("fast_fit_lr", 0.01)
    next_config.setdefault("fast_fit_patience", 30)
    next_config.setdefault("fast_fit_recon_weight", 0.05)
    return next_config


def _fit_fast_trainable_session(content_path, sample_id, vis_method, vis_id, vis_config, *, data_type="Text", task_type="Alignment"):
    from run_visualization import initialize_config
    from visualize_model import VisModel

    info_path = Path(content_path) / "visualize" / f"{vis_method}_{vis_id}" / "info.json"
    existing_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    available_epochs = []
    epochs_root = Path(content_path) / "epochs"
    for child in epochs_root.iterdir():
        if child.is_dir() and child.name.startswith("epoch_"):
            try:
                available_epochs.append(int(child.name.split("_")[1]))
            except Exception:
                pass
    available_epochs.sort()
    if not available_epochs:
        raise ValueError("No available epochs found for EIF bundle")

    first_epoch = available_epochs[0]
    embedding_path = Path(content_path) / "epochs" / f"epoch_{first_epoch}" / "embeddings.npy"
    projection_path = Path(content_path) / "visualize" / f"{vis_method}_{vis_id}" / "epochs" / f"epoch_{first_epoch}" / "projection.npy"
    embeddings = np.load(embedding_path).astype(np.float32)
    target_projection = np.load(projection_path).astype(np.float32)
    if embeddings.ndim != 2 or target_projection.ndim != 2 or target_projection.shape[1] != 2:
        raise ValueError("EIF fast-fit requires 2D projection targets and 2D embedding arrays")
    if len(embeddings) != len(target_projection):
        raise ValueError("Embeddings and projection must have the same number of points")

    fit_vis_config = _build_fast_fit_vis_config(int(embeddings.shape[1]), vis_config)
    config = initialize_config(content_path, vis_method, vis_id, data_type, task_type, fit_vis_config)
    device = torch.device("cuda:{}".format(config['vis_config']['gpu_id']) if torch.cuda.is_available() and config['vis_config']['gpu_id'] != -1 else "cpu")
    model = VisModel(config['vis_config']['encoder_dims'], config['vis_config']['decoder_dims']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['vis_config'].get('fast_fit_lr', 0.01)))
    projection_weight = 1.0
    recon_weight = float(config['vis_config'].get('fast_fit_recon_weight', 0.05))
    max_steps = int(config['vis_config'].get('fast_fit_steps', 240))
    patience = int(config['vis_config'].get('fast_fit_patience', 30))

    feat_t = torch.from_numpy(embeddings).to(dtype=torch.float32, device=device)
    proj_t = torch.from_numpy(target_projection).to(dtype=torch.float32, device=device)

    # Build HD k-NN graph on embeddings for neighbor-aware training.
    # Using both a warmup phase (regress to initial UMAP projection) and a
    # neighbor-attraction phase keeps the global layout stable while making the
    # encoder weights consistent with the attract/repel losses used in refine().
    from sklearn.neighbors import NearestNeighbors as _SkNNS
    _k_hd = min(10, len(embeddings) - 1)
    _nbrs_fit = _SkNNS(n_neighbors=_k_hd + 1, algorithm='auto').fit(embeddings)
    _, _nn_idx = _nbrs_fit.kneighbors(embeddings)
    hd_neighbor_idx = _nn_idx[:, 1:]  # [N, k_hd], exclude self
    hd_nbr_t = torch.from_numpy(embeddings[hd_neighbor_idx.flatten()]).to(dtype=torch.float32, device=device)
    hd_nbr_t = hd_nbr_t.view(len(embeddings), _k_hd, -1)  # [N, k_hd, D]

    rng_seed = np.random.default_rng(42)
    _neg_k = min(20, len(embeddings) - 1)

    best_loss = float('inf')
    best_state = None
    stale_steps = 0

    # Warmup steps: pure projection regression to anchor global layout.
    warmup_steps = min(max_steps // 4, 60)
    attract_weight = 1.0
    anchor_weight = 2.0  # keep global layout stable via regression anchor

    model.train()
    for step in range(max_steps):
        optimizer.zero_grad()
        pred_proj = model.encoder(feat_t)        # [N, 2]
        recon = model.decoder(pred_proj)

        # Reconstruction loss (light regulariser)
        loss_recon = torch.mean((recon - feat_t) ** 2)

        # Phase 1 (warmup): regress to initial UMAP projection for global stability
        loss_proj = torch.mean((pred_proj - proj_t) ** 2)

        if step < warmup_steps:
            loss = loss_proj + recon_weight * loss_recon
        else:
            # Phase 2: neighbor attraction + global anchor
            pred_nbr = model.encoder(hd_nbr_t.view(-1, embeddings.shape[1]))  # [N*k, 2]
            pred_nbr = pred_nbr.view(len(embeddings), _k_hd, 2)               # [N, k, 2]
            diff = pred_proj.unsqueeze(1) - pred_nbr                          # [N, k, 2]
            loss_attract = diff.pow(2).sum(dim=-1).mean()

            # Random negatives repulsion (sampled each step for diversity)
            neg_idx = rng_seed.choice(len(embeddings), size=min(_neg_k, len(embeddings)), replace=False)
            neg_t = feat_t[neg_idx]
            pred_neg = model.encoder(neg_t)  # [neg_k, 2]
            dist_neg = (pred_proj.unsqueeze(1) - pred_neg.unsqueeze(0)).pow(2).sum(dim=-1).sqrt()
            margin = 1.0
            loss_repel = torch.clamp(margin - dist_neg, min=0.0).pow(2).mean()

            loss = (attract_weight * loss_attract
                    + 0.3 * loss_repel
                    + anchor_weight * loss_proj
                    + recon_weight * loss_recon)

        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().cpu().item())
        if current_loss + 1e-7 < best_loss:
            best_loss = current_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_steps = 0
        else:
            stale_steps += 1

        if step % 40 == 0:
            print(f"[EIF-FastFit] sample={sample_id} step={step} loss={current_loss:.6f} proj={float(loss_proj.detach().cpu().item()):.6f} recon={float(loss_recon.detach().cpu().item()):.6f}", flush=True)
        if stale_steps >= patience:
            print(f"[EIF-FastFit] Early stop at step {step} best_loss={best_loss:.6f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        for epoch in available_epochs:
            epoch_embedding_path = Path(content_path) / "epochs" / f"epoch_{epoch}" / "embeddings.npy"
            epoch_embeddings = np.load(epoch_embedding_path).astype(np.float32)
            epoch_t = torch.from_numpy(epoch_embeddings).to(dtype=torch.float32, device=device)
            fitted_projection = model.encoder(epoch_t).cpu().numpy().astype(np.float32)
            epoch_projection_dir = Path(content_path) / "visualize" / f"{vis_method}_{vis_id}" / "epochs" / f"epoch_{epoch}"
            epoch_projection_dir.mkdir(parents=True, exist_ok=True)
            np.save(epoch_projection_dir / "projection.npy", fitted_projection)

    model_save_path = Path(content_path) / "visualize" / f"{vis_method}_{vis_id}" / "vis_model.pth"
    torch.save({
        "loss": best_loss,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, model_save_path)

    info_payload = {
        **existing_info,
        "content_path": str(content_path),
        "vis_method": vis_method,
        "vis_id": vis_id,
        "data_type": data_type,
        "task_type": task_type,
        "sample_id": sample_id,
        "eif_bundle": True,
        "fit_mode": "eif_projection_regression",
        "vis_config": config['vis_config'],
    }
    info_path.write_text(json.dumps(info_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_trainable_session(content_path, sample_id, vis_method, vis_id, vis_config, *, data_type="Text", task_type="Alignment"):
    _write_eif_session_status(content_path, {
        "sample_id": sample_id,
        "content_path": content_path,
        "vis_method": vis_method,
        "vis_id": vis_id,
        "trainable_session_status": "building",
        "message": "Fitting adaptive refine session to EIF projection...",
    })
    _fit_fast_trainable_session(content_path, sample_id, vis_method, vis_id, vis_config, data_type=data_type, task_type=task_type)
    return _write_eif_session_status(content_path, {
        "sample_id": sample_id,
        "content_path": content_path,
        "vis_method": vis_method,
        "vis_id": vis_id,
        "trainable_session_status": "ready",
        "message": "Adaptive refine session is ready.",
    })


def _start_trainable_session_build(content_path, sample_id, vis_method, vis_id, vis_config, *, data_type="Text", task_type="Alignment"):
    task_key = _build_eif_task_key(content_path, vis_method, vis_id)
    with _refine_lock:
        existing = EIF_BUILD_TASKS.get(task_key)
        if existing is not None and existing.is_alive():
            return False

        def _runner():
            try:
                _build_trainable_session(
                    content_path,
                    sample_id,
                    vis_method,
                    vis_id,
                    vis_config,
                    data_type=data_type,
                    task_type=task_type,
                )
            except Exception as exc:
                traceback.print_exc()
                _write_eif_session_status(content_path, {
                    "sample_id": sample_id,
                    "content_path": content_path,
                    "vis_method": vis_method,
                    "vis_id": vis_id,
                    "trainable_session_status": "error",
                    "message": f"Adaptive refine session build failed: {exc}",
                    "error": str(exc),
                })
            finally:
                with _refine_lock:
                    EIF_BUILD_TASKS.pop(task_key, None)

        import threading
        thread = threading.Thread(target=_runner, daemon=True)
        EIF_BUILD_TASKS[task_key] = thread
        thread.start()
        return True


def update_active_session(config, visualizer, strategy):
    """统一更新 Session 的工具函数"""
    global active_session
    active_session.update({
        "strategy": strategy,
        "visualizer": visualizer,
        "content_path": config.get("content_path"),
        "vis_id": config.get("visualizationID") or config.get("vis_id"),
        "vis_method": config.get("vis_method"),
        "vis_config": config.get("vis_config", {}),
        "eif_session_info": None,
    })


@app.route('/syncSession', methods=['POST'])
def sync_session():
    """新接口：允许前端 Load 时同步 Session"""
    req = request.get_json()
    try:
        info_path = os.path.join(req['content_path'], 'dataset', 'info.json')
        dataset_info = read_file_as_json(info_path) or {}
        if dataset_info.get("eif_bundle"):
            vis_method = req.get("vis_method", "TimeVis")
            vis_id = str(req.get("vis_id") or req.get("visualizationID", "0"))
            eif_session_info = _read_eif_session_status(
                req["content_path"],
                sample_id=dataset_info.get("sample_id"),
                vis_method=vis_method,
                vis_id=vis_id,
            )

            if _is_trainable_session_ready(req["content_path"], vis_method, vis_id):
                cached = _get_cached_strategy(req["content_path"], vis_method, vis_id)
                if cached is not None:
                    strategy = cached["strategy"]
                    visualizer = cached["visualizer"]
                    config = cached["config"]
                    print(f"[syncSession] Reusing cached strategy for {Path(req['content_path']).name}", flush=True)
                else:
                    config = initialize_config(
                        req['content_path'],
                        vis_method,
                        vis_id,
                        req['data_type'],
                        req['task_type'],
                        req['vis_config']
                    )
                    visualizer, strategy = init_visualize_component(config)
                    _put_cached_strategy(req["content_path"], vis_method, vis_id, strategy, visualizer, config)
                update_active_session(config, visualizer, strategy)
                active_session["eif_session_info"] = eif_session_info
                return jsonify({
                    "status": "success",
                    "message": "EIF adaptive refine session synced",
                    "eifBundle": True,
                    "trainableSessionStatus": eif_session_info["trainable_session_status"],
                    "refineReady": True,
                    "eifSessionInfo": eif_session_info,
                })

            active_session.update({
                "strategy": EIF_STATIC_SESSION,
                "visualizer": None,
                "content_path": req.get("content_path"),
                "vis_id": vis_id,
                "vis_method": vis_method,
                "vis_config": req.get("vis_config", {}),
                "eif_session_info": eif_session_info,
            })
            return jsonify({
                "status": "success",
                "message": eif_session_info.get("message", "EIF bundle session synced"),
                "eifBundle": True,
                "trainableSessionStatus": eif_session_info["trainable_session_status"],
                "refineReady": False,
                "eifSessionInfo": eif_session_info,
            })

        config = initialize_config(
            req['content_path'],
            req['vis_method'],
            req.get('vis_id') or req.get('visualizationID', "0"),
            req['data_type'],
            req['task_type'],
            req['vis_config']
        )

        # 即使是 Load，我们也调用 init 来准备好 strategy 对象（比如加载模型）
        visualizer, strategy = init_visualize_component(config)
        update_active_session(config, visualizer, strategy)
        return jsonify({
            "status": "success",
            "message": "Session synced on server",
            "eifBundle": False,
            "refineReady": True,
            "eifSessionInfo": None,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# Global lock: only one refine() may run at a time (strategy objects are not thread-safe).
_refine_lock = threading.Lock()

@app.route('/updateFocusContext', methods=['POST'])
@cross_origin()
def update_focus_context():
    """
    Endpoint to receive user selection and trigger dynamic refinement.
    """
    req = request.get_json()
    content_path = req.get("content_path")
    selected_indices = req.get("selected_indices", [])
    focus_mode = req.get("focus_mode", "balanced")
    current_epoch = req.get("current_epoch", None)  # epoch currently viewed by user
    zoom_bbox = req.get("zoom_bbox")
    secondary_indices = [int(i) for i in req.get("secondary_indices", [])]

    # Check if a session is active
    if active_session["strategy"] is None:
        print("No active session, strategy:", active_session["strategy"],
              ", path:", active_session["content_path"], "content path:", content_path)
        return jsonify({"status": "error", "message": "No active session"}), 400

    if active_session["strategy"] == EIF_STATIC_SESSION:
        eif_info = active_session.get("eif_session_info") or {}
        return jsonify({
            "status": "error",
            "message": eif_info.get("message", "Adaptive refine session is still preparing."),
            "trainableSessionStatus": eif_info.get("trainable_session_status", "registered"),
            "refineReady": False,
        }), 400

    # Reject concurrent refine requests immediately rather than queueing them.
    if not _refine_lock.acquire(blocking=False):
        return jsonify({"status": "error", "message": "Refinement already in progress"}), 429

    strategy = active_session["strategy"]
    visualizer = active_session["visualizer"]

    try:
        print(f"Starting refinement: mode={focus_mode}, selected_points={selected_indices}")

        vis_method = active_session["vis_method"]
        focus_indices = selected_indices
        focus_summary = {
            "seed_count": len(selected_indices),
            "bbox_count": 0,
            "hd_neighbor_count": 0,
            "focus_set_size": len(selected_indices),
            "used_bbox": False,
        }

        if vis_method == "TimeVis":
            hd_k = int(active_session.get("vis_config", {}).get("refine_hd_k", REFINE_RUNTIME_DEFAULTS["focus_hd_k"]))
            effective_bbox = None if selected_indices else zoom_bbox
            focus_indices, focus_summary = build_focus_set(
                content_path=content_path,
                vis_method=vis_method,
                vis_id=active_session["vis_id"],
                epoch=current_epoch,
                seed_indices=selected_indices,
                zoom_bbox=effective_bbox,
                hd_k=hd_k,
            )
            print(f"[server] build_focus_set: seeds={len(selected_indices)}, bbox_used={effective_bbox is not None}, focus_size={len(focus_indices)}")

        # Option A: seeds are the attract/ranking targets; expansion → context.
        refine_focus_indices, secondary_indices = _split_focus_targets(
            selected_indices, focus_indices, secondary_indices)
        print(f"[server] refine targets={len(refine_focus_indices)} (seeds), "
              f"context/secondary={len(secondary_indices)}")

        mask = strategy.get_focus_mask(focus_indices)
        strategy.update_ttav_context(focus_indices, focus_mode, mask)

        if vis_method == "DynaVis":
            strategy.refine_train(focus_mode=focus_mode)
            print("Start generating DynaVis visualization results...")
            visualizer.visualize_all_epochs()
            print("DynaVis visualization results generated.")
        elif vis_method in ("DVI", "TimeVis"):
            print("Start refining visualization model...")
            strategy.refine(
                focus_indices=refine_focus_indices,
                neighbor_indices=[],
                current_epoch=current_epoch,
                epochs_to_update=10,
                secondary_indices=secondary_indices if secondary_indices else None,
                top_k=req.get("refine_top_k"),
                priority=req.get("refine_priority"),
            )
            if current_epoch is not None:
                vis_id = active_session["vis_id"]
                patched_indices = getattr(strategy, "_last_patch_indices", None)
                if patched_indices:
                    try:
                        update_projection_neighbors_incremental(
                            content_path, vis_method, vis_id, current_epoch, patched_indices,
                        )
                    except Exception as cache_ex:
                        print(f"[TimeVis] Incremental neighbor cache update failed: {cache_ex}")
                        invalidate_projection_neighbors_cache(
                            content_path, vis_method, vis_id, current_epoch
                        )
                else:
                    invalidate_projection_neighbors_cache(
                        content_path, vis_method, vis_id, current_epoch
                    )
            print("Refinement finished. Refined projections saved to _refined directory.")
            # Patch remaining epochs in the background so switching epochs also shows refined results.
            strategy.patch_other_epochs(skip_epoch=current_epoch)
        else:
            visualizer.visualize_all_epochs()

        # Return backend-computed metrics (full-dataset exact computation)
        return jsonify({
            "status": "success",
            "neighbor_preservation": getattr(strategy, '_last_refine_np',    None),
            "mean_rank_hd":          getattr(strategy, '_last_refine_mrh',   None),
            "trustworthiness":       getattr(strategy, '_last_refine_trust',  None),
            "continuity":            getattr(strategy, '_last_refine_cont',   None),
            "focus_set_size":        focus_summary["focus_set_size"],
            "focus_seed_count":      focus_summary["seed_count"],
            "focus_bbox_count":      focus_summary["bbox_count"],
            "focus_hd_neighbor_count": focus_summary["hd_neighbor_count"],
            "focus_indices":         focus_indices,
            "refine_status":         getattr(strategy, "_last_refine_status", None),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        _refine_lock.release()


# ── Session-based refine (async, with progress streaming) ─────────────────────
_refine_sessions: dict = {}
_refine_sessions_lock = threading.Lock()


def normalize_content_path(content_path) -> str:
    if not content_path:
        content_path = active_session.get("content_path")
    if not content_path:
        raise ValueError("content_path is required and no active session is loaded")
    return str(content_path).strip()


def _split_focus_targets(selected_indices, focus_indices, secondary_indices):
    """Option A: when the user explicitly selects points, ONLY those seeds are
    ranking/attract targets — so a single click becomes a genuine single-focus
    refine (len==1) and the escalation path can guarantee 100% top-10. The
    build_focus_set expansion (seeds' HD neighbors, etc.) is demoted to context
    and merged into secondary_indices (shape/context, not attract targets).

    When there is no explicit selection (bbox/region refine), the whole focus
    set stays as targets → multi-focus best-effort, unchanged.

    Returns (refine_focus_indices, merged_secondary_indices).
    """
    seed_set = {int(i) for i in (selected_indices or [])}
    if not seed_set:
        return list(focus_indices), list(secondary_indices or [])
    refine_focus = sorted(seed_set)
    context = {int(i) for i in focus_indices} - seed_set
    merged_secondary = sorted({int(i) for i in (secondary_indices or [])} | context)
    return refine_focus, merged_secondary


def _prepare_refine_request(req):
    content_path = normalize_content_path(req.get("content_path"))
    selected_indices = req.get("selected_indices", [])
    focus_mode = req.get("focus_mode", "balanced")
    current_epoch = req.get("current_epoch", None)
    zoom_bbox = req.get("zoom_bbox")
    secondary_indices = [int(i) for i in req.get("secondary_indices", [])]

    if active_session["strategy"] is None:
        raise ValueError("No active session")
    if active_session["strategy"] == EIF_STATIC_SESSION:
        raise ValueError("EIF static bundles do not support refinement yet")

    strategy = active_session["strategy"]
    visualizer = active_session["visualizer"]
    vis_method = active_session["vis_method"]
    vis_id = active_session["vis_id"]
    vis_config = active_session.get("vis_config", {})
    resolved_refine_behavior = resolve_refine_behavior_config(vis_config)

    focus_indices = selected_indices
    focus_summary = {
        "seed_count": len(selected_indices),
        "bbox_count": 0,
        "bbox_indices": [],
        "hd_neighbor_count": 0,
        "focus_set_size": len(selected_indices),
        "used_bbox": False,
    }

    if vis_method == "TimeVis":
        hd_k = int(vis_config.get("refine_hd_k", REFINE_RUNTIME_DEFAULTS["focus_hd_k"]))
        # Only use zoom_bbox when the user hasn't explicitly selected specific points.
        # If selected_indices is non-empty, the user wants to refine those points only;
        # adding the full viewport bbox would expand focus to all visible points.
        effective_bbox = None if selected_indices else zoom_bbox
        focus_indices, focus_summary = build_focus_set(
            content_path=content_path,
            vis_method=vis_method,
            vis_id=vis_id,
            epoch=current_epoch,
            seed_indices=selected_indices,
            zoom_bbox=effective_bbox,
            hd_k=hd_k,
        )
        print(f"[server _prepare] build_focus_set: seeds={len(selected_indices)}, bbox_used={effective_bbox is not None}, focus_size={len(focus_indices)}")

    # Option A: seeds are the attract/ranking targets; expansion → context.
    refine_focus_indices, merged_secondary = _split_focus_targets(
        selected_indices, focus_indices, secondary_indices)
    print(f"[server _prepare] refine targets={len(refine_focus_indices)} "
          f"(seeds), context/secondary={len(merged_secondary)}")

    return {
        "content_path": content_path,
        "selected_indices": selected_indices,
        "focus_mode": focus_mode,
        "current_epoch": current_epoch,
        "zoom_bbox": zoom_bbox,
        "strategy": strategy,
        "visualizer": visualizer,
        "vis_method": vis_method,
        "vis_id": vis_id,
        "focus_indices": focus_indices,            # expanded set — for mask/UI highlight
        "refine_focus_indices": refine_focus_indices,  # seeds only — refine attract targets
        "focus_summary": focus_summary,
        "bbox_indices": focus_summary.get("bbox_indices", []),
        "resolved_refine_behavior": resolved_refine_behavior,
        "secondary_indices": merged_secondary,
        "top_k": req.get("refine_top_k"),          # C3: neighborhood size (None → session default)
        "priority": req.get("refine_priority"),    # B1: accuracy↔layout tradeoff (None → default)
    }


def _run_refine_request(prepared_req, progress_callback=None):
    content_path = prepared_req["content_path"]
    focus_mode = prepared_req["focus_mode"]
    current_epoch = prepared_req["current_epoch"]
    strategy = prepared_req["strategy"]
    visualizer = prepared_req["visualizer"]
    vis_method = prepared_req["vis_method"]
    vis_id = prepared_req["vis_id"]
    focus_indices = prepared_req["focus_indices"]
    refine_focus_indices = prepared_req.get("refine_focus_indices", focus_indices)
    focus_summary = prepared_req["focus_summary"]
    bbox_indices = prepared_req.get("bbox_indices", [])
    secondary_indices = prepared_req.get("secondary_indices", [])
    resolved_refine_behavior = prepared_req.get("resolved_refine_behavior")

    print(f"Starting refinement: mode={focus_mode}, selected_points={prepared_req['selected_indices']}")

    mask = strategy.get_focus_mask(focus_indices)
    strategy.update_ttav_context(focus_indices, focus_mode, mask)

    if vis_method == "DynaVis":
        strategy.refine_train(focus_mode=focus_mode)
        print("Start generating DynaVis visualization results...")
        visualizer.visualize_all_epochs()
        print("DynaVis visualization results generated.")
    elif vis_method in ("DVI", "TimeVis"):
        print("Start refining visualization model...")
        strategy.refine(
            focus_indices=refine_focus_indices,
            neighbor_indices=[],
            current_epoch=current_epoch,
            epochs_to_update=10,
            progress_callback=progress_callback,
            should_stop_callback=prepared_req.get("should_stop_callback"),
            progress_refresh_indices=bbox_indices,
            secondary_indices=secondary_indices if secondary_indices else None,
            top_k=prepared_req.get("top_k"),
            priority=prepared_req.get("priority"),
        )
        if current_epoch is not None:
            patched_indices = getattr(strategy, "_last_patch_indices", None)
            if patched_indices:
                try:
                    update_projection_neighbors_incremental(
                        content_path, vis_method, vis_id, current_epoch, patched_indices,
                    )
                except Exception as cache_ex:
                    print(f"[TimeVis] Incremental neighbor cache update failed: {cache_ex}")
                    invalidate_projection_neighbors_cache(content_path, vis_method, vis_id, current_epoch)
            else:
                invalidate_projection_neighbors_cache(content_path, vis_method, vis_id, current_epoch)
        print("Refinement finished. Refined projections saved to _refined directory.")
        strategy.patch_other_epochs(skip_epoch=current_epoch)
    else:
        visualizer.visualize_all_epochs()

    return {
        "status": "success",
        "neighbor_preservation": getattr(strategy, '_last_refine_np',    None),
        "mean_rank_hd":          getattr(strategy, '_last_refine_mrh',   None),
        "trustworthiness":       getattr(strategy, '_last_refine_trust',  None),
        "continuity":            getattr(strategy, '_last_refine_cont',   None),
        "focus_set_size":        focus_summary["focus_set_size"],
        "focus_seed_count":      focus_summary["seed_count"],
        "focus_bbox_count":      focus_summary["bbox_count"],
        "focus_hd_neighbor_count": focus_summary["hd_neighbor_count"],
        "focus_indices":         focus_indices,
        "bbox_indices":          bbox_indices,
        "training_context_indices": getattr(strategy, "_last_training_context_indices", focus_indices),
        "patch_indices":         getattr(strategy, "_last_patch_indices", focus_indices),
        "refine_status":         getattr(strategy, "_last_refine_status", None),
        "resolved_refine_behavior": resolved_refine_behavior,
    }


def _update_refine_session(session_id, **fields):
    with _refine_sessions_lock:
        session = _refine_sessions.get(session_id)
        if session is None:
            return
        session.update(fields)
        if "projection" in fields:
            session["projection_version"] = session.get("projection_version", 0) + 1


def _run_refine_session_worker(session_id, prepared_req):
    try:
        def _should_stop_callback():
            with _refine_sessions_lock:
                session = _refine_sessions.get(session_id)
                return bool(session and session.get("stop_requested", False))

        def _progress_callback(payload):
            session_fields = {
                "status": "stopping" if _should_stop_callback() else "running",
                "steps_completed": payload.get("steps_completed", 0),
                "focus_indices": payload.get("focus_indices", prepared_req["focus_indices"]),
                "training_context_indices": payload.get("training_context_indices", []),
                "patch_indices": payload.get("patch_indices", []),
                "bbox_indices": payload.get("bbox_indices", prepared_req.get("bbox_indices", [])),
            }
            if "projection" in payload:
                session_fields["projection"] = payload.get("projection")
            if "sampled_metrics" in payload:
                session_fields["sampled_metrics"] = payload.get("sampled_metrics")
            if "refine_live" in payload:
                session_fields["refine_live"] = payload.get("refine_live")
            _update_refine_session(session_id, **session_fields)

        prepared_req["should_stop_callback"] = _should_stop_callback
        result = _run_refine_request(prepared_req, progress_callback=_progress_callback)
        _update_refine_session(
            session_id,
            status="completed",
            steps_completed=0,
            focus_indices=result.get("focus_indices", prepared_req["focus_indices"]),
            training_context_indices=result.get("training_context_indices", []),
            patch_indices=result.get("patch_indices", []),
            bbox_indices=result.get("bbox_indices", prepared_req.get("bbox_indices", [])),
            sampled_metrics={
                "neighbor_preservation": result.get("neighbor_preservation"),
                "mean_rank_hd": result.get("mean_rank_hd"),
                "trustworthiness": result.get("trustworthiness"),
                "continuity": result.get("continuity"),
            },
            result=result,
            resolved_refine_behavior=prepared_req.get("resolved_refine_behavior"),
        )
    except Exception as ex:
        traceback.print_exc()
        _update_refine_session(session_id, status="failed", error=str(ex))
    finally:
        _refine_lock.release()


@app.route('/startRefineSession', methods=['POST'])
@cross_origin()
def start_refine_session():
    req = request.get_json()
    if not _refine_lock.acquire(blocking=False):
        return jsonify({"status": "error", "message": "Refinement already in progress"}), 429

    try:
        prepared_req = _prepare_refine_request(req)
    except Exception as e:
        _refine_lock.release()
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400

    session_id = str(uuid.uuid4())
    with _refine_sessions_lock:
        _refine_sessions[session_id] = {
            "status": "queued",
            "projection": None,
            "projection_version": 0,
            "steps_completed": 0,
            "focus_indices": prepared_req["focus_indices"],
            "training_context_indices": [],
            "patch_indices": [],
            "bbox_indices": prepared_req.get("bbox_indices", []),
            "sampled_metrics": None,
            "stop_requested": False,
            "result": None,
            "resolved_refine_behavior": prepared_req.get("resolved_refine_behavior"),
            "error": None,
        }

    worker = threading.Thread(
        target=_run_refine_session_worker,
        args=(session_id, prepared_req),
        daemon=True,
    )
    worker.start()

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "focus_indices": prepared_req["focus_indices"],
        "bbox_indices": prepared_req.get("bbox_indices", []),
        "resolved_refine_behavior": prepared_req.get("resolved_refine_behavior"),
    })


@app.route('/getRefineSessionProgress', methods=['POST'])
@cross_origin()
def get_refine_session_progress():
    req = request.get_json()
    session_id = req.get("session_id")
    since_version = int(req.get("since_version", -1))

    with _refine_sessions_lock:
        session = _refine_sessions.get(session_id)
        if session is None:
            return jsonify({"status": "error", "message": "Unknown refine session"}), 404
        projection = session.get("projection")
        projection_version = session.get("projection_version", 0)
        response = {
            "status": session.get("status", "unknown"),
            "steps_completed": session.get("steps_completed", 0),
            "projection_version": projection_version,
            "focus_indices": session.get("focus_indices", []),
            "training_context_indices": session.get("training_context_indices", []),
            "patch_indices": session.get("patch_indices", []),
            "bbox_indices": session.get("bbox_indices", []),
            "sampled_metrics": session.get("sampled_metrics"),
            "refine_live": session.get("refine_live"),
            "stop_requested": bool(session.get("stop_requested", False)),
            "resolved_refine_behavior": session.get("resolved_refine_behavior"),
            "result": session.get("result"),
            "error": session.get("error"),
        }
        if projection is not None and projection_version != since_version:
            response["projection"] = projection.tolist() if isinstance(projection, np.ndarray) else projection

    return jsonify(response)


@app.route('/stopRefineSession', methods=['POST'])
@cross_origin()
def stop_refine_session():
    req = request.get_json()
    session_id = req.get("session_id")

    with _refine_sessions_lock:
        session = _refine_sessions.get(session_id)
        if session is None:
            return jsonify({"status": "error", "message": "Unknown refine session"}), 404
        session["stop_requested"] = True
        if session.get("status") in ("queued", "running"):
            session["status"] = "stopping"

    return jsonify({"status": "success", "session_id": session_id, "stop_requested": True})


@app.route('/discardRefine', methods=['POST'])
@cross_origin()
def discard_refine():
    """B3 Undo: revert refinement by deleting the _refined projection(s) so the
    graceful fallback in load_projection() serves the original baseline again.

    epoch omitted → discard ALL refined epochs (full revert to baseline);
    epoch given  → discard just that epoch. Refined neighbor caches are
    invalidated for each removed epoch.
    """
    req = request.get_json() or {}
    content_path = req.get("content_path")
    vis_method   = req.get("vis_method")
    vis_id       = req.get("vis_id")
    epoch        = req.get("epoch", None)

    if not (content_path and vis_method and vis_id is not None):
        return jsonify({"status": "error", "message": "content_path, vis_method, vis_id required"}), 400

    refined_root = os.path.join(content_path, 'visualize', f"{vis_method}_{vis_id}_refined", 'epochs')
    removed = []
    if os.path.isdir(refined_root):
        if epoch is not None:
            epoch_dirs = [f'epoch_{epoch}']
        else:
            epoch_dirs = [d for d in os.listdir(refined_root) if d.startswith('epoch_')]
        for d in epoch_dirs:
            proj_path = os.path.join(refined_root, d, 'projection.npy')
            if os.path.exists(proj_path):
                try:
                    os.remove(proj_path)
                    _ep = d[len('epoch_'):]
                    removed.append(_ep)
                    try:
                        invalidate_projection_neighbors_cache(content_path, vis_method, vis_id, _ep)
                    except Exception as _ce:
                        print(f"[discardRefine] neighbor cache invalidate failed for {d}: {_ce}")
                except OSError as _oe:
                    print(f"[discardRefine] could not remove {proj_path}: {_oe}")

    print(f"[discardRefine] reverted {len(removed)} epoch(s) to baseline: {removed}")
    return jsonify({"status": "success", "removed_epochs": removed})


@app.route('/refinedEpochs', methods=['POST'])
@cross_origin()
def refined_epochs():
    """C2: list epochs that currently have a refined projection on disk, so the
    timeline can mark refined vs pending (background patch fills the rest in)."""
    req = request.get_json() or {}
    content_path = req.get("content_path")
    vis_method   = req.get("vis_method")
    vis_id       = req.get("vis_id")
    if not (content_path and vis_method and vis_id is not None):
        return jsonify({"status": "error", "message": "content_path, vis_method, vis_id required"}), 400

    root = os.path.join(content_path, 'visualize', f"{vis_method}_{vis_id}_refined", 'epochs')
    eps = []
    if os.path.isdir(root):
        for d in os.listdir(root):
            if d.startswith('epoch_') and os.path.exists(os.path.join(root, d, 'projection.npy')):
                try:
                    eps.append(int(d[len('epoch_'):]))
                except ValueError:
                    pass
    return jsonify({"status": "success", "refined_epochs": sorted(eps)})


@app.route('/startVisualizing', methods = ["POST"])
def start_visualizing():
    """
    Modified start endpoint to register the active session.
    """
    req = request.get_json()
    content_path = req['content_path']
    # ... other params ...
    vis_method = req['vis_method']
    vis_id = req['vis_id'] or "0"
    data_type = req['data_type']
    task_type = req['task_type']
    vis_config = req['vis_config']
    
    # 构造预期的文件夹名称
    folder_name = f"{vis_method}_{vis_id}"
    target_dir = os.path.join(content_path, "visualize", folder_name)

    if os.path.exists(target_dir):
        # 409 Conflict 是处理此类逻辑的标准 HTTP 状态码
        return jsonify({
            "status": "error",
            "message": f"Session ID '{vis_id}' already exists for {vis_method}. Please use a different ID or delete the old folder."
        }), 409
    
    visualizer, strategy = visualize_run(content_path, vis_method, vis_id, data_type, task_type, vis_config)
    
     # Store in global session for subsequent refinement calls
    # 同步更新 Session
    update_active_session(req, visualizer, strategy)
    
    return make_response(jsonify({"status": "initialized"}), 200)
    
@app.route("/", methods=["GET", "POST"])
def GUI():
    return send_from_directory('../../web/dist/configs/plotView', 'index.html')


"""
Api: get training process info

Request:
    content_path (str)
Response:
    color_list (list): list of colors
    label_text_list (list): list of label text
"""
@app.route('/getTrainingProcessInfo', methods=["GET"])
@cross_origin()
def get_training_process_info():
    content_path = request.args.get('content_path')
    
    epochs_dir = os.path.join(content_path, 'epochs')
    available_epochs = []

    if os.path.exists(epochs_dir) and os.path.isdir(epochs_dir):
        try:
            for item in os.listdir(epochs_dir):
                if item.startswith('epoch_'):
                    full_path = os.path.join(epochs_dir, item)
                    if os.path.isdir(full_path):
                        epoch_num_str = item[len('epoch_'):]
                        if epoch_num_str.isdigit():
                            available_epochs.append(int(epoch_num_str))
            
            available_epochs.sort()
        except Exception as e:
            print(f"Error scanning epochs directory: {e}")
            available_epochs = []

    config = read_file_as_json(os.path.join(content_path, 'dataset', 'info.json'))
    
    if config == None or 'classes' not in config:
        # infer from labels.npy
        label_file = os.path.join(content_path, 'dataset', 'labels.npy')
        labels = np.load(label_file, allow_pickle=True)
        class_num = len(np.unique(labels))
        color_list  = get_coloring_list(class_num)
        label_text_list = [str(i) for i in range(class_num)]
    else:
        color_list  = get_coloring_list(len(config['classes']))
        label_text_list = config['classes']
    
    result = jsonify({
        'color_list': color_list,
        'label_text_list': label_text_list,
        'available_epochs': available_epochs
    })
    return make_response(result, 200)


"""
Api: get minimum info of one epoch

Request:
    content_path (str)
    vis_id (str)
    epoch (str): epoch number
Response:
    config (dict)
    project (list)
    label_list (list): label list of samples in projection
"""
@app.route('/updateProjection', methods = ["POST"])
@cross_origin()
def update_projection():
    req = request.get_json()
    content_path = req['content_path']
    vis_id = req['vis_id']
    epoch = int(req['epoch'])
    vis_method = req['vis_method']
    # refine_flag is optional; when True load from the _refined directory
    refine_flag = bool(req.get('refine_flag', False))
    print(f"[updateProjection] content_path={content_path!r} vis_method={vis_method!r} vis_id={vis_id!r} epoch={epoch} refine_flag={refine_flag}")

    projection = load_projection(content_path, vis_method, vis_id, epoch, refine_flag)

    result = jsonify({
        'projection': projection,
    })
    return make_response(result, 200)


"""
Api: start training visualization model and get visualization result

Request:
    content_path (str)
    vis_method (str)
    task_type (str): "classification", "regression"
    vis_config (dict): visualization config
Response:
    None
# """

"""
Api: get text data of all samples

Request:
    content_path (str)
Response:
    text_list (lsit of str)
"""
@app.route('/getAllText', methods = ["POST"])
def get_all_text():
    req = request.get_json()
    content_path = req['content_path']

    text_list = get_all_texts(content_path)
    token_list_path = os.path.join(content_path, 'dataset', 'token_list.json')
    text_data_path = os.path.join(content_path, 'dataset', 'text_data.json')
    token_list = read_file_as_json(token_list_path) if os.path.exists(token_list_path) else text_list
    text_data = read_file_as_json(text_data_path) if os.path.exists(text_data_path) else text_list

    if text_list is None:
        return make_response(jsonify({'error_message': "getting all texts failed"}), 400)

    result = jsonify({
        'text_list': text_list,
        'text_data': text_data,
        'token_list': token_list,
    })
    return make_response(result, 200)


@app.route('/registerEIFBundle', methods=['POST'])
@cross_origin()
def register_eif_bundle():
    req = request.get_json()
    if not req:
        return jsonify({"status": "error", "message": "Missing JSON body"}), 400

    sample_id = str(req.get("sample_id", "")).strip()
    bundle = req.get("bundle")
    vis_method = str(req.get("vis_method", "TimeVis")).strip() or "TimeVis"
    vis_id = str(req.get("vis_id", "1")).strip() or "1"
    overwrite = bool(req.get("overwrite", True))
    build_trainable_session = bool(req.get("build_trainable_session", False))
    wait_until_ready = bool(req.get("wait_until_ready", False))
    data_type = str(req.get("data_type", "Text")).strip() or "Text"
    task_type = str(req.get("task_type", "Alignment")).strip() or "Alignment"
    vis_config = req.get("vis_config", {"gpu_id": -1}) or {"gpu_id": -1}

    if not sample_id:
        return jsonify({"status": "error", "message": "sample_id is required"}), 400
    if not isinstance(bundle, dict):
        return jsonify({"status": "error", "message": "bundle must be an object"}), 400

    labels = bundle.get("labels")
    text_list = bundle.get("text_list")
    embeddings = bundle.get("embeddings")
    projection = bundle.get("projection")

    if not isinstance(labels, list) or not isinstance(text_list, list):
        return jsonify({"status": "error", "message": "bundle.labels and bundle.text_list must be lists"}), 400
    if not isinstance(embeddings, list) or not isinstance(projection, list):
        return jsonify({"status": "error", "message": "bundle.embeddings and bundle.projection must be lists"}), 400

    num_points = len(labels)
    if len(text_list) != num_points or len(embeddings) != num_points or len(projection) != num_points:
        return jsonify({"status": "error", "message": "All bundle arrays must have the same length"}), 400

    target_dir = EIF_BUNDLE_ROOT / sample_id
    method_dir = target_dir / "visualize" / f"{vis_method}_{vis_id}"
    refined_method_dir = target_dir / "visualize" / f"{vis_method}_{vis_id}_refined"

    current_status = _read_eif_session_status(str(target_dir), sample_id=sample_id, vis_method=vis_method, vis_id=vis_id)
    if method_dir.exists() and not overwrite:
        if build_trainable_session and not _is_trainable_session_ready(str(target_dir), vis_method, vis_id):
            if wait_until_ready:
                current_status = _build_trainable_session(
                    str(target_dir),
                    sample_id,
                    vis_method,
                    vis_id,
                    vis_config,
                    data_type=data_type,
                    task_type=task_type,
                )
            elif current_status.get("trainable_session_status") != "building":
                _write_eif_session_status(str(target_dir), {
                    "sample_id": sample_id,
                    "content_path": str(target_dir),
                    "vis_method": vis_method,
                    "vis_id": vis_id,
                    "trainable_session_status": "building",
                    "message": "Fitting adaptive refine session to EIF projection...",
                })
                _start_trainable_session_build(
                    str(target_dir),
                    sample_id,
                    vis_method,
                    vis_id,
                    vis_config,
                    data_type=data_type,
                    task_type=task_type,
                )
            current_status = _read_eif_session_status(str(target_dir), sample_id=sample_id, vis_method=vis_method, vis_id=vis_id)

        return jsonify({
            "status": "success",
            "sample_id": sample_id,
            "content_path": str(target_dir),
            "num_points": num_points,
            "vis_method": vis_method,
            "vis_id": vis_id,
            "cached": True,
            "trainableSessionStatus": current_status.get("trainable_session_status", "registered"),
            "refineReady": bool(current_status.get("refine_ready", False)),
            "statusMessage": current_status.get("message"),
        })

    target_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        invalidate_bundle_neighbor_caches(str(target_dir))
        _invalidate_strategy_cache(str(target_dir), vis_method, vis_id)
        if method_dir.exists():
            shutil.rmtree(method_dir)
        if refined_method_dir.exists():
            shutil.rmtree(refined_method_dir)

    dataset_dir = target_dir / "dataset"
    epoch_dir = target_dir / "epochs" / "epoch_1"
    vis_dir = target_dir / "visualize" / f"{vis_method}_{vis_id}" / "epochs" / "epoch_1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    classes = bundle.get("classes") or ["prompt", "output"]
    dataset_info = {
        "model": bundle.get("model", "EIFTokenBundle"),
        "classes": classes,
        "eif_bundle": True,
        "refine_capable": True,
        "sample_id": sample_id,
        "prompt_len": bundle.get("prompt_len"),
        "trainable_session_status": "registered",
        "refine_ready": False,
    }

    with open(dataset_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    np.save(dataset_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    with open(dataset_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(bundle.get("index", {"train": list(range(num_points)), "test": []}), f, indent=2)

    with open(dataset_dir / "text.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in text_list))

    token_list = bundle.get("token_list", text_list)
    text_data = bundle.get("text_data", text_list)
    with open(dataset_dir / "token_list.json", "w", encoding="utf-8") as f:
        json.dump(token_list, f, ensure_ascii=False)
    with open(dataset_dir / "text_data.json", "w", encoding="utf-8") as f:
        json.dump(text_data, f, ensure_ascii=False)

    align = bundle.get("align")
    if align is not None:
        with open(dataset_dir / "align.json", "w", encoding="utf-8") as f:
            json.dump(align, f, indent=2, ensure_ascii=False)

    predictions = bundle.get("predictions")
    if predictions is not None:
        np.save(epoch_dir / "predictions.npy", np.asarray(predictions, dtype=np.float32))

    np.save(epoch_dir / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    np.save(vis_dir / "projection.npy", np.asarray(projection, dtype=np.float32))

    vis_info = {
        "content_path": str(target_dir),
        "vis_method": vis_method,
        "vis_id": vis_id,
        "data_type": data_type,
        "task_type": task_type,
        "vis_config": vis_config,
        "sample_id": sample_id,
        "eif_bundle": True,
        "trainable_session_status": "registered",
        "refine_ready": False,
    }
    with open(target_dir / "visualize" / f"{vis_method}_{vis_id}" / "info.json", "w", encoding="utf-8") as f:
        json.dump(vis_info, f, indent=2, ensure_ascii=False)

    current_status = _write_eif_session_status(str(target_dir), {
        "sample_id": sample_id,
        "content_path": str(target_dir),
        "vis_method": vis_method,
        "vis_id": vis_id,
        "trainable_session_status": "registered",
        "message": "EIF bundle uploaded successfully.",
    })

    if build_trainable_session:
        if wait_until_ready:
            current_status = _build_trainable_session(
                str(target_dir),
                sample_id,
                vis_method,
                vis_id,
                vis_config,
                data_type=data_type,
                task_type=task_type,
            )
        else:
            current_status = _write_eif_session_status(str(target_dir), {
                "sample_id": sample_id,
                "content_path": str(target_dir),
                "vis_method": vis_method,
                "vis_id": vis_id,
                "trainable_session_status": "building",
                "message": "Fitting adaptive refine session to EIF projection...",
            })
            _start_trainable_session_build(
                str(target_dir),
                sample_id,
                vis_method,
                vis_id,
                vis_config,
                data_type=data_type,
                task_type=task_type,
            )

    return jsonify({
        "status": "success",
        "sample_id": sample_id,
        "content_path": str(target_dir),
        "num_points": num_points,
        "vis_method": vis_method,
        "vis_id": vis_id,
        "trainableSessionStatus": current_status.get("trainable_session_status", "registered"),
        "refineReady": bool(current_status.get("refine_ready", False)),
        "statusMessage": current_status.get("message"),
    })


@app.route('/getEIFBundleStatus', methods=['POST'])
@cross_origin()
def get_eif_bundle_status():
    req = request.get_json() or {}
    content_path = str(req.get("content_path", "")).strip()
    if not content_path:
        return jsonify({"status": "error", "message": "content_path is required"}), 400

    vis_method = str(req.get("vis_method", "TimeVis")).strip() or "TimeVis"
    vis_id = str(req.get("vis_id", "1")).strip() or "1"
    status_payload = _read_eif_session_status(content_path, vis_method=vis_method, vis_id=vis_id)
    status_payload["refine_ready"] = _is_trainable_session_ready(content_path, vis_method, vis_id)
    status_payload["trainable_session_status"] = "ready" if status_payload["refine_ready"] else status_payload.get("trainable_session_status", "registered")
    return jsonify({
        "status": "success",
        "eifBundle": True,
        "trainableSessionStatus": status_payload["trainable_session_status"],
        "refineReady": bool(status_payload["refine_ready"]),
        "eifSessionInfo": status_payload,
        "message": status_payload.get("message", "EIF bundle status fetched."),
    })

@app.route('/getAlignment', methods = ["POST"])
def get_alignment():
    req = request.get_json()
    content_path = req['content_path']

    alignment = get_alignment_data(content_path)

    if alignment is None:
        return make_response(jsonify({'error_message': "getting alignment failed"}), 400)

    result = jsonify({
        'alignment': alignment
    })
    return make_response(result, 200)

"""
Api: get selected attributes of the dataset

Request:
    content_path (str)
    epoch (str): epoch number
    attributes (list): selected attributes
Response:
    attribute1 (object)
    attribute2 (object)
    ...
"""
@app.route('/getAttributes', methods = ["POST"])
@cross_origin()
def get_attributes():
    req = request.get_json()
    content_path = req['content_path']
    epoch = req['epoch']
    attributes = req['attributes']

    result = {}
    for attribute in attributes:
        result[attribute] = load_single_attribute(content_path, epoch, attribute)

    result = jsonify(result)
    return make_response(result, 200)


"""
Api: get simple filter result

Request:
    content_path (str)
    epoch (str)
    filter_type (str): "label", "prediction", "train", "test"
    filter_data (str): label name
Response:
    indices (list of int): indeices of samples that satisfy the filter
"""
@app.route('/getSimpleFilterResult', methods = ["POST"])
@cross_origin()
def get_simple_filter_result():
    req = request.get_json()
    content_path = req['content_path']
    epoch = int(req['epoch'])
    filters = req['filters']

    config = read_file_as_json(os.path.join(content_path, 'config.json'))
    indices, error_message = get_filter_result(config, content_path, epoch, filters)

    if indices is None:
        return make_response(jsonify({'error_message': error_message}), 400)

    result = jsonify({
        'indices': indices
    })
    return make_response(result, 200)


"""
Api: get background image

Request:
    content_path (str)
    vis_id (str)
    width (int)
    height (int)
    scale (list of float)
Response:
    background_image_base64 (str): base64 encoded im
"""    
@app.route('/getBackground', methods = ["POST"])
@cross_origin()
def get_background():
    req = request.get_json()
    content_path = req['content_path']
    vis_id = req['vis_id']
    epoch = int(req['epoch'])
    vis_method = req['vis_method']

    try:
        base64_image = load_background(content_path,vis_method, vis_id, epoch)
        result = jsonify({
            'background_image_base64': base64_image
        })
        return make_response(result, 200)
    except Exception as e:
        return make_response(jsonify({'error_message': 'Error in loading background'}), 400)

"""
Api: get image data of one sample

Request:
    content_path (str)
    index (str): sample index
Response:
    image_base64 (str): base64 encoded image
"""
@app.route('/getImageData', methods = ["POST"])
@cross_origin()
def get_image_data():
    req = request.get_json()
    content_path = req['content_path']
    if('index' not in req):
        return make_response(jsonify({'image_base64': ''}), 200)
    
    index = req['index']

    try:
        base64_image = load_one_image(content_path, index)
        result = jsonify({
            'image_base64': base64_image
        })
        return make_response(result, 200)
    except Exception as e:
        result = jsonify({
            'image_base64': ''
        })
        return make_response(result, 200)


"""
Api: get text data of one sample

Request:
    content_path (str)
    index (str): sample index
Response:
    text (str): text data
"""
@app.route('/getTextData', methods = ["POST"])
@cross_origin()
def get_text_data():
    req = request.get_json()
    content_path = req['content_path']
    if('index' not in req):
        return make_response(jsonify({'text': ''}), 200)
    
    index = req['index']

    try:
        text = load_one_text(content_path, index)
        result = jsonify({
            'text': text
        })
        return make_response(result, 200)
    except Exception as e:
        result = jsonify({
            'text': ''
        })
        return make_response(result, 200)


"""
Api: get high dimensional neighbors of one sample

Request:
    content_path (str)
    epoch (str)
Response:
    neighbors (array[][])
"""
@app.route('/getOriginalNeighbors', methods = ["POST"])
@cross_origin()
def get_original_neighbors():
    req = request.get_json()
    content_path = req['content_path']
    epoch = int(req['epoch'])
    
    try:
        neighbors = calculate_high_dimensional_neighbors(content_path, epoch)
        result = jsonify({
            'neighbors': neighbors,
        })
        return make_response(result, 200)
    except Exception as e:
        print(e)
        return make_response(jsonify({'error_message': 'Error in calculating neighbors'}), 400)

"""
Api: get projection neighbors of one sample

Request:
    content_path (str)
    vis_id (str)
    epoch (str)
Response:
    neighbors (array[][])
"""
@app.route('/getProjectionNeighbors', methods = ["POST"])
@cross_origin()
def get_projection_neighbors():
    req = request.get_json()
    content_path = req['content_path']
    vis_id = req['vis_id']
    epoch = int(req['epoch'])
    vis_method = req['vis_method']
    # Support refine_flag so the caller can request neighbors from the refined projection
    refine_flag = bool(req.get('refine_flag', False))
    blend_bbox = req.get('blend_bbox')
    blend_decay_ratio = float(req.get('blend_decay_ratio', REFINE_RUNTIME_DEFAULTS['blend_decay_ratio']))
    blend_focus_indices = req.get('blend_focus_indices') or []

    projection_data = req.get('projection_data')  # pre-blended projection from frontend

    try:
        if projection_data is not None:
            # Use the exact projection the frontend is displaying — guarantees that
            # neighbor lines are computed from the same coordinates as the visual.
            neighbors, index_list = calculate_projection_neighbors_for_projection(
                content_path,
                projection_data,
            )
        elif blend_bbox is not None or blend_focus_indices:
            blended_projection = build_runtime_blended_projection(
                content_path,
                vis_method,
                vis_id,
                epoch,
                blend_bbox,
                decay_ratio=blend_decay_ratio,
                focus_indices=blend_focus_indices,
            )
            neighbors, index_list = calculate_projection_neighbors_for_projection(
                content_path,
                blended_projection,
            )
        else:
            neighbors, index_list = calculate_projection_neighbors(content_path, vis_method, vis_id, epoch, refine_flag=refine_flag)
        result = jsonify({
            'neighbors': neighbors,
            'index_list': index_list,
        })
        return make_response(result, 200)
    except Exception as e:
        print(e)
        return make_response(jsonify({'error_message': 'Error in calculating neighbors'}), 400)

    
@app.route('/getVisualizeMetrics', methods = ["POST"])
@cross_origin()
def get_visualize_metrics():
    req = request.get_json()
    content_path = req['content_path']
    vis_id = req['vis_id']
    epoch = int(req['epoch'])
    vis_method = req['vis_method']
    try:
        metrics = calculate_visualize_metrics(content_path, vis_method, vis_id, epoch)
        result = jsonify(metrics)
        return make_response(result, 200)
    except Exception as e:
        print(e)
        return make_response(jsonify({'error_message': 'Error in calculating metrics'}), 400)


@app.route('/getInfluenceSamples', methods=["POST"])
@cross_origin()
def get_influence_samples():
    req = request.get_json()
    content_path = req['content_path']
    epoch = int(req['epoch'])
    training_event = req['training_event']
    num_samples = int(req['num_samples'])

    try:
        if training_event['type'] == 'InconsistentMovement': 
            # attribution of closeness or separation between a pair of samples
            print("Tracing InconsistentMovement")
            influence_samples = movement_attribution(content_path, epoch, training_event, num_samples)
        else: 
            # atribution of a particular prediction
            print("Tracing PredictionError")
            influence_samples = prediction_attribution(content_path, epoch, training_event, num_samples)
        
        result = jsonify({
            "influence_samples": influence_samples,
        })
        return make_response(result, 200)
    except Exception as e:
        print(e)
        return make_response(jsonify({'error_message': 'Error in calculating influence samples'}), 400)


@app.route('/calculateTrainingEvents', methods=["POST"])
@cross_origin()
def calculate_training_events():
    req = request.get_json()
    content_path = req['content_path']
    epoch = int(req['epoch'])
    event_types = req['event_types']

    try:
        training_events = compute_training_events(content_path, epoch, event_types)
        result = jsonify({
            "training_events": training_events,
        })
        return make_response(result, 200)
    except Exception as e:
        print(e)
        return make_response(jsonify({'error_message': 'Error in calculating training events'}), 400)


def check_port_inuse(port, host):
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, port))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()

# for contrast
if __name__ == "__main__":
    host = '0.0.0.0'
    port = 5050
    while check_port_inuse(port, host):
        port = port + 1

    if not is_dev_mode:
        # use_reloader=True: werkzeug auto-restarts on any .py file change (no extra deps)
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    else:
        from livereload import Server
        from flask_debugtoolbar import DebugToolbarExtension

        app.debug = True
        app.threaded = True
        app.config['SECRET_KEY'] = 'a-random-secret-key'
        toolbar = DebugToolbarExtension(app)

        server = Server(app.wsgi_app)

        server.watch('../frontend/**/*.css')
        server.watch('../frontend/**/*.html')
        server.watch('../frontend/**/*.js')
        server.serve(host=host, port=port)
