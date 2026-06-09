import os
import sys
import shutil
import json
import uuid
from pathlib import Path
import numpy as np
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
    "vis_config": {}
}

EIF_BUNDLE_ROOT = Path("/root/project/Dataset/eif_bundles")
EIF_STATIC_SESSION = "EIF_STATIC_BUNDLE"

def update_active_session(config, visualizer, strategy):
    """统一更新 Session 的工具函数"""
    global active_session
    normalized_content_path = normalize_content_path(config.get("content_path"))
    active_session.update({
        "strategy": strategy,
        "visualizer": visualizer,
        "content_path": normalized_content_path,
        "vis_id": config.get("visualizationID") or config.get("vis_id"),
        "vis_method": config.get("vis_method"),
        "vis_config": config.get("vis_config", {})
    })

@app.route('/syncSession', methods=['POST'])
def sync_session():
    """新接口：允许前端 Load 时同步 Session"""
    req = request.get_json()
    try:
        content_path = normalize_content_path(req['content_path'])
        req['content_path'] = content_path
        info_path = os.path.join(content_path, 'dataset', 'info.json')
        dataset_info = read_file_as_json(info_path) or {}
        if dataset_info.get("eif_bundle"):
            active_session.update({
                "strategy": EIF_STATIC_SESSION,
                "visualizer": None,
                "content_path": content_path,
                "vis_id": req.get("vis_id", "0"),
                "vis_method": req.get("vis_method"),
                "vis_config": req.get("vis_config", {}),
            })
            return jsonify({"status": "success", "message": "EIF static bundle session synced"})

        config = initialize_config(
            req['content_path'], 
            req['vis_method'], 
            req.get('vis_id', "0"),
            req['data_type'], 
            req['task_type'], 
            req['vis_config']
        )
        
        # 即使是 Load，我们也调用 init 来准备好 strategy 对象（比如加载模型）
        visualizer, strategy = init_visualize_component(config)
        update_active_session(config, visualizer, strategy)
        return jsonify({"status": "success", "message": "Session synced on server"})
    except Exception as e:
        import traceback
        traceback.print_exc() 
        return jsonify({"status": "error", "message": str(e)}), 500

        
import threading

# Global lock: only one refine() may run at a time (strategy objects are not thread-safe).
_refine_lock = threading.Lock()
_refine_sessions = {}
_refine_sessions_lock = threading.Lock()


def _prepare_refine_request(req):
    content_path = normalize_content_path(req.get("content_path"))
    selected_indices = req.get("selected_indices", [])
    focus_mode = req.get("focus_mode", "balanced")
    current_epoch = req.get("current_epoch", None)
    zoom_bbox = req.get("zoom_bbox")

    if active_session["strategy"] is None:
        print("No active session, strategy:", active_session["strategy"],
              ", path:", active_session["content_path"], "content path:", content_path)
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
        focus_set_strategy = vis_config.get(
            "refine_focus_set_strategy",
            REFINE_RUNTIME_DEFAULTS["focus_set_strategy"],
        )
        focus_indices, focus_summary = build_focus_set(
            content_path=content_path,
            vis_method=vis_method,
            vis_id=vis_id,
            epoch=current_epoch,
            seed_indices=selected_indices,
            zoom_bbox=zoom_bbox,
            hd_k=hd_k,
            strategy=focus_set_strategy,
        )

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
        "focus_indices": focus_indices,
        "focus_summary": focus_summary,
        "bbox_indices": focus_summary.get("bbox_indices", []),
        "resolved_refine_behavior": resolved_refine_behavior,
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
    focus_summary = prepared_req["focus_summary"]
    bbox_indices = prepared_req.get("bbox_indices", [])
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
            focus_indices=focus_indices,
            neighbor_indices=[],
            current_epoch=current_epoch,
            epochs_to_update=10,
            progress_callback=progress_callback,
            should_stop_callback=prepared_req.get("should_stop_callback"),
            progress_refresh_indices=bbox_indices,
        )
        if current_epoch is not None:
            patched_indices = getattr(strategy, "_last_patch_indices", None)
            if patched_indices:
                try:
                    update_projection_neighbors_incremental(
                        content_path,
                        vis_method,
                        vis_id,
                        current_epoch,
                        patched_indices,
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
        "focus_set_strategy":    focus_summary.get("focus_set_strategy"),
        "bbox_indices":          bbox_indices,
        "focus_indices":         focus_indices,
        "training_context_indices": getattr(strategy, "_last_training_context_indices", focus_indices),
        "patch_indices":         getattr(strategy, "_last_patch_indices", focus_indices),
        "resolved_refine_behavior": resolved_refine_behavior,
        "timings":               getattr(strategy, "_last_refine_timing", None),
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
            _update_refine_session(session_id, **session_fields)

        prepared_req["should_stop_callback"] = _should_stop_callback
        result = _run_refine_request(prepared_req, progress_callback=_progress_callback)
        _update_refine_session(
            session_id,
            status="completed",
            steps_completed=(result.get("timings") or {}).get("steps_completed", 0),
            focus_indices=result.get("focus_indices", prepared_req["focus_indices"]),
            training_context_indices=result.get("training_context_indices", []),
            patch_indices=result.get("patch_indices", []),
            bbox_indices=result.get("bbox_indices", prepared_req.get("bbox_indices", [])),
            timings=result.get("timings"),
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
        import traceback
        traceback.print_exc()
        _update_refine_session(session_id, status="failed", error=str(ex))
    finally:
        _refine_lock.release()

@app.route('/updateFocusContext', methods=['POST'])
@cross_origin()
def update_focus_context():
    """
    Endpoint to receive user selection and trigger dynamic refinement.
    """
    req = request.get_json()
    if not _refine_lock.acquire(blocking=False):
        return jsonify({"status": "error", "message": "Refinement already in progress"}), 429

    try:
        prepared_req = _prepare_refine_request(req)
        return jsonify(_run_refine_request(prepared_req))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
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
            "timings": None,
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
        "focus_set_strategy": prepared_req["focus_summary"].get("focus_set_strategy"),
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
            "timings": session.get("timings"),
            "sampled_metrics": session.get("sampled_metrics"),
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


@app.route('/startVisualizing', methods = ["POST"])
def start_visualizing():
    """
    Modified start endpoint to register the active session.
    """
    req = request.get_json()
    content_path = normalize_content_path(req['content_path'])
    req['content_path'] = content_path
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
    content_path = normalize_content_path(request.args.get('content_path'))
    
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
    content_path = normalize_content_path(req['content_path'])
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
    content_path = normalize_content_path(req['content_path'])

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

    if method_dir.exists() and not overwrite:
        return jsonify({
            "status": "success",
            "sample_id": sample_id,
            "content_path": str(target_dir),
            "num_points": num_points,
            "vis_method": vis_method,
            "vis_id": vis_id,
            "cached": True,
        })
    target_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        invalidate_bundle_neighbor_caches(str(target_dir))
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
        "sample_id": sample_id,
        "prompt_len": bundle.get("prompt_len"),
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
        "data_type": "Text",
        "task_type": "Alignment",
        "vis_config": req.get("vis_config", {"gpu_id": -1}),
        "sample_id": sample_id,
        "eif_bundle": True,
    }
    with open(target_dir / "visualize" / f"{vis_method}_{vis_id}" / "info.json", "w", encoding="utf-8") as f:
        json.dump(vis_info, f, indent=2, ensure_ascii=False)

    return jsonify({
        "status": "success",
        "sample_id": sample_id,
        "content_path": str(target_dir),
        "num_points": num_points,
        "vis_method": vis_method,
        "vis_id": vis_id,
    })

@app.route('/getAlignment', methods = ["POST"])
def get_alignment():
    req = request.get_json()
    content_path = normalize_content_path(req['content_path'])

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
    content_path = normalize_content_path(req['content_path'])
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
    content_path = normalize_content_path(req['content_path'])
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
    content_path = normalize_content_path(req['content_path'])
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
    content_path = normalize_content_path(req['content_path'])
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
    content_path = normalize_content_path(req['content_path'])
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
    content_path = normalize_content_path(req['content_path'])
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
    content_path = normalize_content_path(req['content_path'])
    vis_id = req['vis_id']
    epoch = int(req['epoch'])
    vis_method = req['vis_method']
    # Support refine_flag so the caller can request neighbors from the refined projection
    refine_flag = bool(req.get('refine_flag', False))
    blend_bbox = req.get('blend_bbox')
    blend_decay_ratio = float(req.get('blend_decay_ratio', REFINE_RUNTIME_DEFAULTS['blend_decay_ratio']))
    blend_focus_indices = req.get('blend_focus_indices') or []

    try:
        if blend_bbox is not None or blend_focus_indices:
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
    content_path = normalize_content_path(req['content_path'])
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


@app.route('/getRefineMetrics', methods=["POST"])
@cross_origin()
def get_refine_metrics():
    req = request.get_json()
    content_path = normalize_content_path(req['content_path'])
    vis_id = req['vis_id']
    epoch = int(req['epoch'])
    vis_method = req['vis_method']
    refine_flag = bool(req.get('refine_flag', False))
    blend_bbox = req.get('blend_bbox')
    blend_decay_ratio = float(req.get('blend_decay_ratio', REFINE_RUNTIME_DEFAULTS['blend_decay_ratio']))
    blend_focus_indices = req.get('blend_focus_indices') or []
    focus_indices = req.get('focus_indices') or []

    try:
        if blend_bbox is not None or blend_focus_indices:
            projection = build_runtime_blended_projection(
                content_path,
                vis_method,
                vis_id,
                epoch,
                blend_bbox,
                decay_ratio=blend_decay_ratio,
                focus_indices=blend_focus_indices,
            )
        else:
            projection = load_projection(content_path, vis_method, vis_id, epoch, refine_flag=refine_flag)
        metrics = calculate_refine_metrics_for_projection(
            content_path,
            epoch,
            projection,
            focus_indices,
        )
        return make_response(jsonify(metrics), 200)
    except Exception as e:
        print(e)
        return make_response(jsonify({'error_message': 'Error in calculating refine metrics'}), 400)


@app.route('/getInfluenceSamples', methods=["POST"])
@cross_origin()
def get_influence_samples():
    req = request.get_json()
    content_path = normalize_content_path(req['content_path'])
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
    content_path = normalize_content_path(req['content_path'])
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

    s = None
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
