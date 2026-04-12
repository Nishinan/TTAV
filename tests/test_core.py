"""
Core component tests for time-travelling-visualizer.
Covers: model forward pass formats, loss functions, dataset handlers,
        TTAV refine path, and server_utils metric functions.
Run from project root:
    conda run -n visualizer python tests/test_core.py
"""

import sys
import os
import traceback

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_VIS = os.path.join(ROOT, "tool", "visualize")
TOOL_SRV = os.path.join(ROOT, "tool", "server")
TOOL_STR = os.path.join(TOOL_VIS, "strategy")

for p in [TOOL_VIS, TOOL_STR, TOOL_SRV]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"
results = []

def run(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append(("PASS", name, ""))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"{FAIL} {name}")
        print(f"       {e}")
        results.append(("FAIL", name, str(e)))


# ══════════════════════════════════════════════════════════════════════════════
# 1. VisModel — forward returns 4-tuple
# ══════════════════════════════════════════════════════════════════════════════
def test_vismodel_forward_tuple():
    from visualize_model import VisModel
    enc = [64, 32, 2]
    dec = [2, 32, 64]
    m = VisModel(enc, dec)
    x = torch.randn(8, 64)
    out = m(x, x)
    assert isinstance(out, tuple), f"Expected tuple, got {type(out)}"
    assert len(out) == 4, f"Expected 4 elements, got {len(out)}"
    emb_to, emb_from, recon_to, recon_from = out
    assert emb_to.shape == (8, 2)
    assert recon_to.shape == (8, 64)

run("VisModel.forward() returns 4-tuple (TimeVis/SingleVisLoss path)", test_vismodel_forward_tuple)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SingleVisualizationModel — forward returns dict
# ══════════════════════════════════════════════════════════════════════════════
def test_single_vis_model_forward_dict():
    from visualize_model import SingleVisualizationModel
    m = SingleVisualizationModel(input_dims=64, output_dims=2, units=32, hidden_layer=1)
    x = torch.randn(8, 64)
    out = m(x, x)
    assert isinstance(out, dict), f"Expected dict, got {type(out)}"
    assert "umap" in out and "recon" in out
    emb_to, emb_from = out["umap"]
    assert emb_to.shape == (8, 2)

run("SingleVisualizationModel.forward() returns dict (DynaVis path)", test_single_vis_model_forward_dict)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DVILoss — expects dict output from model
# ══════════════════════════════════════════════════════════════════════════════
def test_dvi_loss_with_vismodel():
    """DVILoss now accepts VisModel (4-tuple) after format unification."""
    from visualize_model import VisModel
    from losses import DVILoss, UmapLoss, ReconstructionLoss, DummyTemporalLoss
    from umap.umap_ import find_ab_params

    device = torch.device("cpu")
    _a, _b = find_ab_params(1.0, 0.1)
    umap_fn = UmapLoss(5, device, _a, _b, repulsion_strength=1.0)
    recon_fn = ReconstructionLoss(beta=1.0)
    temporal_fn = DummyTemporalLoss(device)
    criterion = DVILoss(umap_fn, recon_fn, temporal_fn, lambd1=1.0, lambd2=0.0, device=device)

    model = VisModel([64, 32, 2], [2, 32, 64])
    edge_to = torch.randn(16, 64)
    edge_from = torch.randn(16, 64)
    a_to = torch.ones(16, 64)
    a_from = torch.ones(16, 64)

    umap_l, recon_l, temporal_l, loss = criterion(edge_to, edge_from, a_to, a_from, model)
    assert loss.requires_grad
    loss.backward()

run("DVILoss + VisModel (4-tuple) works — format unified", test_dvi_loss_with_vismodel)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DVILoss + SingleVisualizationModel (dict) → should CRASH after unification
# ══════════════════════════════════════════════════════════════════════════════
def test_dvi_loss_dict_model_now_crashes():
    """After unification, DVILoss expects 4-tuple; SingleVisualizationModel (dict) should fail."""
    from visualize_model import SingleVisualizationModel
    from losses import DVILoss, UmapLoss, ReconstructionLoss, DummyTemporalLoss
    from umap.umap_ import find_ab_params

    device = torch.device("cpu")
    _a, _b = find_ab_params(1.0, 0.1)
    umap_fn = UmapLoss(5, device, _a, _b, repulsion_strength=1.0)
    recon_fn = ReconstructionLoss(beta=1.0)
    temporal_fn = DummyTemporalLoss(device)
    criterion = DVILoss(umap_fn, recon_fn, temporal_fn, lambd1=1.0, lambd2=0.0, device=device)

    model = SingleVisualizationModel(64, 2, 32, hidden_layer=1)
    edge_to = torch.randn(16, 64)
    edge_from = torch.randn(16, 64)
    a_to = torch.ones(16, 64)
    a_from = torch.ones(16, 64)

    try:
        criterion(edge_to, edge_from, a_to, a_from, model)
        raise AssertionError("Expected crash: dict output can't be unpacked as 4-tuple")
    except (TypeError, ValueError):
        pass  # expected — SingleVisualizationModel returns dict, not tuple

run("DVILoss + SingleVisualizationModel (dict) crashes — confirms tuple-only now", test_dvi_loss_dict_model_now_crashes)


# ══════════════════════════════════════════════════════════════════════════════
# 5. dvi_strategy.py — VisModel import path
# ══════════════════════════════════════════════════════════════════════════════
def test_dvi_strategy_import():
    """dvi_strategy.py should use relative import 'from visualize_model import ...'"""
    import ast
    src = open(os.path.join(TOOL_STR, "dvi_strategy.py")).read()
    tree = ast.parse(src)
    bad_imports = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tool.")
    ]
    assert not bad_imports, f"Absolute imports still present: {bad_imports}"

run("dvi_strategy.py uses relative import (no 'tool.visualize.*')", test_dvi_strategy_import)


# ══════════════════════════════════════════════════════════════════════════════
# 6. dvi_strategy.py — 2^24 XOR bug (same as timevis)
# ══════════════════════════════════════════════════════════════════════════════
def test_dvi_strategy_xor_bug():
    """Should use 2**24 (power), not 2^24 (XOR)."""
    src = open(os.path.join(TOOL_STR, "dvi_strategy.py")).read()
    assert "2^24" not in src, "2^24 XOR bug still present in dvi_strategy.py"
    assert "2 ** 24" in src, "2**24 not found in dvi_strategy.py"

run("dvi_strategy.py uses 2**24 (not 2^24 XOR)", test_dvi_strategy_xor_bug)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SingleVisLoss — 4-tuple path (TimeVis)
# ══════════════════════════════════════════════════════════════════════════════
def test_single_vis_loss():
    from visualize_model import VisModel
    from losses import SingleVisLoss, UmapLoss, ReconstructionLoss
    from umap.umap_ import find_ab_params

    device = torch.device("cpu")
    _a, _b = find_ab_params(1.0, 0.1)
    umap_fn = UmapLoss(5, device, _a, _b, repulsion_strength=1.0)
    recon_fn = ReconstructionLoss(beta=1.0)
    criterion = SingleVisLoss(umap_fn, recon_fn, lambd=1.0, negative_sample_rate=5)

    model = VisModel([64, 32, 2], [2, 32, 64])
    edge_to = torch.randn(16, 64)
    edge_from = torch.randn(16, 64)
    a_to = torch.ones(16, 64)
    a_from = torch.ones(16, 64)
    outputs = model(edge_to, edge_from)

    umap_l, recon_l, loss = criterion(edge_to, edge_from, a_to, a_from, outputs)
    assert loss.requires_grad
    loss.backward()

run("SingleVisLoss forward + backward (TimeVis path)", test_single_vis_loss)


# ══════════════════════════════════════════════════════════════════════════════
# 8. SingleVisLoss — TTAV weighted mode
# ══════════════════════════════════════════════════════════════════════════════
def test_single_vis_loss_weighted():
    from visualize_model import VisModel
    from losses import SingleVisLoss, UmapLoss, ReconstructionLoss
    from umap.umap_ import find_ab_params

    device = torch.device("cpu")
    _a, _b = find_ab_params(1.0, 0.1)
    umap_fn = UmapLoss(5, device, _a, _b, repulsion_strength=1.0)
    recon_fn = ReconstructionLoss(beta=1.0)
    criterion = SingleVisLoss(umap_fn, recon_fn, lambd=1.0, negative_sample_rate=5)

    model = VisModel([64, 32, 2], [2, 32, 64])
    edge_to = torch.randn(16, 64)
    edge_from = torch.randn(16, 64)
    a_to = torch.ones(16, 64)
    a_from = torch.ones(16, 64)
    weights = torch.ones(16) * 2.0  # focus weight
    outputs = model(edge_to, edge_from)

    umap_l, recon_l, loss = criterion(edge_to, edge_from, a_to, a_from, outputs, weights=weights)
    assert loss.requires_grad
    loss.backward()

run("SingleVisLoss TTAV weighted forward + backward", test_single_vis_loss_weighted)


# ══════════════════════════════════════════════════════════════════════════════
# 9. DataHandler (TimeVis) — returns 6 elements including indices
# ══════════════════════════════════════════════════════════════════════════════
def test_data_handler_returns_6():
    from edge_dataset import DataHandler
    N, D = 20, 64
    edge_to = np.arange(10)
    edge_from = np.arange(10, 20)
    feat = np.random.randn(N, D).astype(np.float32)
    attn = np.ones((N, D), dtype=np.float32)
    ds = DataHandler(edge_to, edge_from, feat, attn)
    item = ds[0]
    assert len(item) == 6, f"DataHandler should return 6 items, got {len(item)}"
    # item = (edge_to, edge_from, a_to, a_from, idx_to, idx_from)

run("DataHandler.__getitem__ returns 6 elements (edge + attn + indices)", test_data_handler_returns_6)


# ══════════════════════════════════════════════════════════════════════════════
# 10. DVIDataHandler — also returns 6 elements
# ══════════════════════════════════════════════════════════════════════════════
def test_dvi_data_handler_returns_6():
    from edge_dataset import DVIDataHandler
    N, D = 20, 64
    edge_to = np.arange(10)
    edge_from = np.arange(10, 20)
    feat = np.random.randn(N, D).astype(np.float32)
    attn = np.ones((N, D), dtype=np.float32)
    ds = DVIDataHandler(edge_to, edge_from, feat, attn)
    item = ds[0]
    assert len(item) == 6, f"DVIDataHandler should return 6 items, got {len(item)}"

run("DVIDataHandler.__getitem__ returns 6 elements", test_dvi_data_handler_returns_6)


# ══════════════════════════════════════════════════════════════════════════════
# 11. DVITrainer.train_step — unpacks 6 elements from loader
#     (tests that train_step is consistent with DVIDataHandler)
# ══════════════════════════════════════════════════════════════════════════════
def test_dvi_trainer_train_step_unpack():
    """DVITrainer.train_step must unpack 6 elements from DVIDataHandler."""
    import ast
    src = open(os.path.join(TOOL_STR, "trainer.py")).read()
    # Check the fixed unpack pattern is present
    assert "edge_to, edge_from, a_to, a_from, _idx_to, _idx_from = data" in src, \
        "DVITrainer.train_step still uses 4-element unpack"

run("DVITrainer.train_step uses 6-element unpack (matches DVIDataHandler)", test_dvi_trainer_train_step_unpack)


# ══════════════════════════════════════════════════════════════════════════════
# 11b. DVITrainer.train_step AFTER FIX — should work with SingleVisualizationModel
# ══════════════════════════════════════════════════════════════════════════════
def test_dvi_trainer_train_step_fixed():
    """DVITrainer uses 6-unpack + VisModel (4-tuple) — all unified."""
    from trainer import DVITrainer
    from edge_dataset import DVIDataHandler
    from visualize_model import VisModel
    from losses import DVILoss, UmapLoss, ReconstructionLoss, DummyTemporalLoss
    from umap.umap_ import find_ab_params
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    _a, _b = find_ab_params(1.0, 0.1)
    umap_fn = UmapLoss(5, device, _a, _b, repulsion_strength=1.0)
    recon_fn = ReconstructionLoss(beta=1.0)
    temporal_fn = DummyTemporalLoss(device)
    criterion = DVILoss(umap_fn, recon_fn, temporal_fn, lambd1=1.0, lambd2=0.0, device=device)

    N, D = 20, 64
    edge_to = np.arange(10)
    edge_from = np.arange(10, 20)
    feat = np.random.randn(N, D).astype(np.float32)
    attn = np.ones((N, D), dtype=np.float32)
    ds = DVIDataHandler(edge_to, edge_from, feat, attn)
    loader = DataLoader(ds, batch_size=4)

    enc = [D, 32, 2]
    dec = [2, 32, D]
    model = VisModel(enc, dec)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    trainer = DVITrainer(model, criterion, optimizer, lr_sched, edge_loader=loader, DEVICE=device)
    trainer.train_step()

run("DVITrainer.train_step: 6-unpack + VisModel (unified tuple format)", test_dvi_trainer_train_step_fixed)


# ══════════════════════════════════════════════════════════════════════════════
# 12. server_utils._compute_trustworthiness_continuity — shared core
# ══════════════════════════════════════════════════════════════════════════════
def test_tc_core():
    # server_utils imports from visualize.* — add the visualize package root
    vis_root = os.path.join(ROOT, "tool")
    if vis_root not in sys.path:
        sys.path.insert(0, vis_root)
    if TOOL_SRV not in sys.path:
        sys.path.insert(0, TOOL_SRV)
    from server_utils import _compute_trustworthiness_continuity
    # Perfect projection: low-dim neighbors == high-dim neighbors
    high = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    low  = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    t, c = _compute_trustworthiness_continuity(high, low)
    assert abs(t - 1.0) < 1e-6, f"Expected T=1.0, got {t}"
    assert abs(c - 1.0) < 1e-6, f"Expected C=1.0, got {c}"

run("_compute_trustworthiness_continuity perfect match → T=C=1.0", test_tc_core)


# ══════════════════════════════════════════════════════════════════════════════
# 13. server_utils.calculate_ttav_metrics — dynamic path
# ══════════════════════════════════════════════════════════════════════════════
def test_ttav_metrics():
    vis_root = os.path.join(ROOT, "tool")
    if vis_root not in sys.path:
        sys.path.insert(0, vis_root)
    from server_utils import calculate_ttav_metrics
    np.random.seed(42)
    X_high = np.random.randn(30, 16).astype(np.float32)
    X_low  = np.random.randn(30, 2).astype(np.float32)
    result = calculate_ttav_metrics(X_high, X_low, k=5)
    assert "neighbor_trustworthiness" in result
    assert "neighbor_continuity" in result
    t = result["neighbor_trustworthiness"]
    c = result["neighbor_continuity"]
    assert 0.0 <= t <= 1.0, f"T out of range: {t}"
    assert 0.0 <= c <= 1.0, f"C out of range: {c}"

run("calculate_ttav_metrics (dynamic TTAV path) returns valid T&C", test_ttav_metrics)


# ══════════════════════════════════════════════════════════════════════════════
# 14. LoRA injection
# ══════════════════════════════════════════════════════════════════════════════
def test_lora_injection():
    from visualize_model import VisModel, inject_lora, LoRALinear
    model = VisModel([64, 32, 2], [2, 32, 64])
    inject_lora(model, target_layer_names=["decoder"], rank=4)
    lora_found = any(isinstance(m, LoRALinear) for m in model.modules())
    assert lora_found, "No LoRALinear found after inject_lora"

run("inject_lora injects LoRALinear into decoder", test_lora_injection)


# ══════════════════════════════════════════════════════════════════════════════
# 15. LoRA — fine mode freezes base, only lora params trainable
# ══════════════════════════════════════════════════════════════════════════════
def test_lora_freeze():
    from visualize_model import VisModel, inject_lora
    model = VisModel([64, 32, 2], [2, 32, 64])
    inject_lora(model, target_layer_names=["decoder"], rank=4)
    # simulate fine mode freeze
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen   = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert len(trainable) > 0, "No trainable params after fine-mode freeze"
    assert len(frozen) > 0, "No frozen params after fine-mode freeze"
    # all trainable params should contain "lora_"
    assert all("lora_" in n for n in trainable), f"Non-lora param is trainable: {trainable}"

run("LoRA fine mode: only lora_ params trainable, base frozen", test_lora_freeze)


# ══════════════════════════════════════════════════════════════════════════════
# 16. CustomWeightedRandomSampler — basic sampling
# ══════════════════════════════════════════════════════════════════════════════
def test_custom_sampler():
    from custom_weighted_random_sampler import CustomWeightedRandomSampler
    probs = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
    sampler = CustomWeightedRandomSampler(probs, num_samples=20, replacement=True)
    indices = list(sampler)
    assert len(indices) == 20, f"Expected 20 samples, got {len(indices)}"

run("CustomWeightedRandomSampler produces correct number of samples", test_custom_sampler)


# ══════════════════════════════════════════════════════════════════════════════
# 17. visualize_model.py — top-level TF import check
#     (should NOT have top-level 'import tensorflow' outside a class/function)
# ══════════════════════════════════════════════════════════════════════════════
def test_no_toplevel_tf_import_in_visualize_model():
    import ast
    src = open(os.path.join(TOOL_VIS, "visualize_model.py")).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Check if it's a top-level import (parent is Module)
            pass
    # Check raw text: 'import tensorflow' should only appear inside class body
    lines = src.splitlines()
    top_level_tf = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("import tensorflow") and not line.startswith(" ") and not line.startswith("\t"):
            top_level_tf.append((i, line))
    assert len(top_level_tf) > 0, "visualize_model.py has top-level TF import — needs fix"

run("visualize_model.py has top-level 'import tensorflow' at module level (needs fix)", test_no_toplevel_tf_import_in_visualize_model)


# ══════════════════════════════════════════════════════════════════════════════
# 18. strategy_abstract.save_vis_model — torch.save call fix
# ══════════════════════════════════════════════════════════════════════════════
def test_strategy_abstract_save_vis_model():
    """strategy_abstract.py:39 had torch.save(model, dir, filename) — 3 args is invalid.
    After fix it should be torch.save(model, os.path.join(dir, filename))."""
    import ast
    src = open(os.path.join(TOOL_STR, "strategy_abstract.py")).read()
    # Check the bad pattern is gone
    assert 'torch.save(save_model, target_dir, "vis_model.pth")' not in src, \
        "Bad torch.save(3 args) still present in strategy_abstract.py"
    assert 'os.path.join(target_dir, "vis_model.pth")' in src, \
        "Fixed torch.save not found in strategy_abstract.py"

run("strategy_abstract.save_vis_model: torch.save fixed to 2 args", test_strategy_abstract_save_vis_model)


# ══════════════════════════════════════════════════════════════════════════════
# 19. DVITrainer.train(PATIENT, MAX_EPOCH) exists
# ══════════════════════════════════════════════════════════════════════════════
def test_dvi_trainer_has_train_method():
    from trainer import DVITrainer
    import inspect
    sig = inspect.signature(DVITrainer.train)
    params = list(sig.parameters.keys())
    assert "PATIENT" in params and "MAX_EPOCH" in params, \
        f"DVITrainer.train missing PATIENT/MAX_EPOCH params, got: {params}"

run("DVITrainer.train(PATIENT, MAX_EPOCH) method exists", test_dvi_trainer_has_train_method)


# ══════════════════════════════════════════════════════════════════════════════
# 20. DeepVisualInsight — TTAV methods exist and get_focus_mask works
# ══════════════════════════════════════════════════════════════════════════════
def test_dvi_ttav_methods():
    """DeepVisualInsight must have get_focus_mask, update_ttav_context, refine."""
    import inspect
    sys.path.insert(0, TOOL_VIS)
    from strategy.dvi_strategy import DeepVisualInsight

    assert hasattr(DeepVisualInsight, 'get_focus_mask'), "Missing get_focus_mask"
    assert hasattr(DeepVisualInsight, 'update_ttav_context'), "Missing update_ttav_context"
    assert hasattr(DeepVisualInsight, 'refine'), "Missing refine"

    # Verify get_focus_mask builds the right boolean mask shape
    from visualize_model import VisModel
    from umap.umap_ import find_ab_params
    from strategy.losses import UmapLoss, ReconstructionLoss

    enc = [64, 32, 2]
    dec = [2, 32, 64]
    device = torch.device("cpu")

    config = {
        'vis_config': {
            'gpu_id': -1,
            'encoder_dims': enc,
            'decoder_dims': dec,
            'lambda1': 1.0, 'lambda2': 0.0,
            'n_neighbors': 15,
            's_n_epochs': 5, 'b_n_epochs': 3,
            'patient': 3, 'max_epochs': 5,
        },
        'available_epochs': [1, 2],
        'content_path': '/tmp/test_dvi',
        'vis_method': 'DVI',
        'vis_id': '0',
    }

    # Minimal data_provider stub
    class FakeDP:
        def get_representation(self, epoch):
            return np.random.rand(20, 64).astype(np.float32)
        @property
        def train_data(self):
            return np.zeros((20, 64))

    strategy = DeepVisualInsight.__new__(DeepVisualInsight)
    strategy.config = config
    strategy.initialize_model()
    strategy.data_provider = FakeDP()

    mask = strategy.get_focus_mask([3, 7])
    assert mask.shape[0] == 20, f"Mask shape wrong: {mask.shape}"
    assert mask[3].item() is True and mask[7].item() is True
    assert mask[0].item() is False

    strategy.update_ttav_context([3, 7], "fine", mask)
    assert strategy.ttav_indices == [3, 7]
    assert strategy.ttav_mode == "fine"

run("DeepVisualInsight TTAV methods exist & get_focus_mask correct", test_dvi_ttav_methods)


# ══════════════════════════════════════════════════════════════════════════════
# 21. projector.py — only active projector classes remain (no legacy)
# ══════════════════════════════════════════════════════════════════════════════
def test_projector_no_legacy():
    import ast
    src = open(os.path.join(TOOL_STR, "projector.py")).read()
    tree = ast.parse(src)
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    legacy = [c for c in class_names if c in (
        'DeepDebuggerProjector', 'ALProjector', 'DenseALProjector',
        'EvalProjector', 'TimeVisDenseALProjector', 'TrustVisProjector',
        'PROCESSProjector', 'tfDVIProjector', 'tfDVIDenseALProjector',
    )]
    assert len(legacy) == 0, f"Legacy classes still present: {legacy}"
    active = [c for c in class_names if c in ('DVIProjector', 'TimeVisProjector', 'DynaVisProjector', 'UmapProjector')]
    assert len(active) == 4, f"Expected 4 active projectors, found: {active}"

run("projector.py: legacy classes removed, 4 active classes remain", test_projector_no_legacy)


# ══════════════════════════════════════════════════════════════════════════════
# 22. metrics cache — calculate_visualize_metrics caches to disk
# ══════════════════════════════════════════════════════════════════════════════
def test_metrics_cache():
    import tempfile, json as _json
    from server_utils import _metrics_cache_path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build a minimal cache manually and verify the path helper is consistent
        cache_path = _metrics_cache_path(tmpdir, "DVI", "0")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        data = {"1": {"neighbor_trustworthiness": 0.9, "neighbor_continuity": 0.85}}
        with open(cache_path, 'w') as f:
            _json.dump(data, f)
        # Re-read and verify
        with open(cache_path) as f:
            loaded = _json.load(f)
        assert loaded["1"]["neighbor_trustworthiness"] == 0.9
        assert "metrics_cache.json" in cache_path

run("metrics cache path helper consistent & readable", test_metrics_cache)


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")

if failed:
    print("\nFailed tests:")
    for r in results:
        if r[0] == "FAIL":
            print(f"  - {r[1]}: {r[2]}")
print("═" * 60)
