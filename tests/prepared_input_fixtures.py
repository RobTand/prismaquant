"""Pure prepared-weight files and finite geometry shared by CPU fixtures."""
import hashlib
import pickle
import torch
from prismaquant.production_weight_cache import ProductionWeightCache
LAYERS = [0, 1, 2, 3]
FMT = "NVFP4"
SHAPE = (16, 32)
STATISTICS_CAP = 5000
WINDOWS_PER_LAYER = 2


def _qnames(layer):
    return [f"model.layers.{layer}.mlp.gate_proj",
            f"model.layers.{layer}.mlp.up_proj"]


def _render_files(tmp_path):
    files = {}
    for layer in LAYERS:
        for index, name in enumerate(_qnames(layer)):
            path = tmp_path / "renders" / f"layer{layer}-{index}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((torch.arange(512, dtype=torch.float32).reshape(
                *SHAPE) + layer * 100 + index), path)
            raw = path.read_bytes()
            files[(name, FMT)] = {
                "qname": name, "fmt": FMT, "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest()}
    return files


def _production_pkl(tmp_path, files):
    weights = {pair: spec["path"] for pair, spec in files.items()}
    verified = {
        pair: {"rendered_weight": {
            "shape": list(SHAPE),
            "logical_bytes": SHAPE[0] * SHAPE[1] * 4},
            "render_file_sha256": spec["sha256"]}
        for pair, spec in files.items()}
    cache = ProductionWeightCache(
        weights=weights, levers={},
        metadata={"verified_cells": verified})
    path = tmp_path / "production.pkl"
    path.write_bytes(pickle.dumps(cache))
    return path


def _execution(n_probes=2):
    from prismaquant.joint_retained_window_plan import (
        EXECUTION_SCHEMA, RetainedWindowBudget)
    budget = RetainedWindowBudget(
        50 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20, 1 << 20,
        1 << 20, 1 << 20, 1 << 20, 1 << 20, STATISTICS_CAP,
        4 << 20, 4)
    return {
        "n_probes": n_probes,
        "operator_windows": {
            "schema": "prismaquant.joint_operator_windows.v1",
            "max_statistics_bytes": 1 << 20,
            "max_candidate_bytes": 1 << 20,
            "max_render_resident_bytes": 1 << 20,
            "max_load_buffer_bytes": 1 << 20,
            "workspace_reserve_bytes": 1 << 20,
            "max_replay_cotangent_bytes": 1 << 20,
            "prefetch_workers": 1},
        "retained_operator_windows": {
            "schema": EXECUTION_SCHEMA,
            "budget": budget.as_dict(),
            "source_reserve_bytes": 1 << 20,
            "source_loading_reserve_bytes": 2 << 20},
        "boundary_storage": {
            "schema": "prismaquant.aura.boundary_storage.v2",
            "capture_order": "layer_major",
            "directory": "/tmp/bridge-boundary",
            "max_resident_bytes": 1 << 20,
            "max_auxiliary_bytes": 1 << 20,
            "max_artifact_bytes": 1 << 20,
            "prefetch_batches": 2},
    }


