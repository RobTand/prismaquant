"""pq309 driver reaches the GLM TP>1 preparation (RobTand/prismaquant#681)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_path = Path(__file__).parents[1] / 'experiments/pq309_native_moe_panel.py'
_spec = importlib.util.spec_from_file_location('pq309_panel', _path)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


def glm_shape(tp):
    return dict(geometry_version=1, geometry_id='glm53_next_routed_stack_v1',
                source_id='glm5_next', n_routed_experts=288, top_k=8,
                hidden_size=4096, intermediate_size=2048, shared_experts=1,
                n_group=1, topk_group=1, topk_method='noaux_tc',
                scoring_func='sigmoid', norm_topk_prob=True,
                routed_scaling_factor=2.5, swiglu_limit=10.0, gated=True,
                tensor_parallel=tp, tensor_parallel_cut_axis='intermediate')


def lfm_shape():
    return dict(experts=32, hidden_size=512, intermediate_size=1024, top_k=4)


def args(**overrides):
    base = dict(quality_prepared=None, quality_prepared_sha256=None,
                quality_source_model=None, quality_source_model_sha256=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_world_of_one_binds_no_quality_preparation():
    assert driver.quality_binding(args(), {}, shape=lfm_shape()) == (None, None)
    assert driver.quality_binding(args(), {}, shape=glm_shape(1)) == (None, None)


def test_rank_local_without_binding_is_refused():
    with pytest.raises(ValueError, match='quality-prepared'):
        driver.quality_binding(args(), {}, shape=glm_shape(2))


def test_rank_local_reads_plan_binding_and_cli_overrides(tmp_path):
    blob = tmp_path/'prepared.json'
    blob.write_text('{"schema": "ok"}')
    from hashlib import sha256
    binding = {'path': str(blob), 'sha256': sha256(blob.read_bytes()).hexdigest()}
    identity = {'model': 'glm5-next', 'digest': '0'*64}
    plan = {'quality_prepared': binding, 'quality_source_model': identity}
    assert driver.quality_binding(args(), plan, shape=glm_shape(2)) == (binding, identity)
    other = tmp_path/'other.json'
    other.write_text('{"schema": "other"}')
    other_binding = {'path': str(other), 'sha256': sha256(other.read_bytes()).hexdigest()}
    other_identity = {'model': 'glm5-next', 'digest': '1'*64}
    identity_path = tmp_path/'identity.json'
    identity_path.write_text(json.dumps(other_identity))
    got = driver.quality_binding(
        args(quality_prepared=other, quality_prepared_sha256=other_binding['sha256'],
             quality_source_model=identity_path,
             quality_source_model_sha256=sha256(identity_path.read_bytes()).hexdigest()),
        plan, shape=glm_shape(2))
    assert got == (other_binding, other_identity)


def test_cli_binding_without_digest_is_refused(tmp_path):
    blob = tmp_path/'prepared.json'
    blob.write_text('{}')
    with pytest.raises(ValueError, match='quality-prepared-sha256'):
        driver.quality_binding(args(quality_prepared=blob), {}, shape=glm_shape(2))


def test_roster_walks_the_geometry_own_expert_count():
    from prismaquant.native_moe_panel import _shape_for_roster
    assert _shape_for_roster(lfm_shape())['experts'] == 32
    assert _shape_for_roster(glm_shape(2))['experts'] == 288
