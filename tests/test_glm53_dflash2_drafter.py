"""CPU tests for the glm53_dflash2_drafter source candidate (PQ #1617).

The base image cannot load the DFlash2 drafter for two reasons, both read
off the base image's own bytes (upstream vLLM fd4a15126, verified
byte-identical to the kpool-tail image for every file this set touches):

1. The V2 runner enables aux hidden states for method ``dflash`` and calls
   ``set_eagle3_aux_hidden_state_layers``, which raises ``Model does not
   support EAGLE3 interface`` unless the target implements
   ``SupportsEagle3``. Neither GLM5-next wrapper does.
2. ``_get_kv_cache_groups_glm5_next`` returns None when any non-Mamba,
   non-tail spec is not exactly ``MLAAttentionSpec``; the drafter's five
   sliding-window layers trip the guard.

The fixtures below are exact excerpts of those base bytes. The defect
tests assert the base exhibits both refusals; the fix tests apply the
real ``patched_source`` of each patch script and assert the refusal is
gone. No test imports or vendors a serving runtime.
"""

import hashlib
import importlib.util
from pathlib import Path

PATCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "prismaquant"
    / "serving_runtime_patches"
    / "glm53_dflash2_drafter"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PATCH_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EAGLE = _load("patch_glm5next_eagle3")
KV = _load("patch_glm5_drafter_kv_group")

# Exact excerpts of the base image's model.py (sha 4436911e...).
BASE_IMPORT = (
    "from vllm.model_executor.models.interfaces import (\n"
    "    HasInnerState,\n"
    "    IsHybrid,\n"
    "    MixtureOfExperts,\n"
    "    SupportsPP,\n"
    ")\n"
)
BASE_CAUSAL = (
    "class Glm5NextForCausalLM(\n"
    "    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid\n"
    "):\n"
)
BASE_MODEL_CLASS = "class Glm5NextModel(nn.Module):\n"
BASE_ACTIVE = (
    "        self._active_layers = self.layers[self.start_layer : self.end_layer]\n"
    "\n"
    "        if get_pp_group().is_last_rank:\n"
)
BASE_LOOP_HEAD = (
    "        full_num_tokens = positions.shape[0]\n"
    "        if self.is_sequence_parallel:\n"
    "            hidden_states = sp_shard(hidden_states)\n"
    "\n"
)
BASE_LOOP = (
    "        for layer in self._active_layers:\n"
    "            hidden_states, residual, post, comb = layer(\n"
    "                positions, hidden_states, residual, post, comb\n"
    "            )\n"
)
BASE_MLLM = (
    "class Glm5NextForConditionalGeneration(\n"
    "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid\n"
    "):\n"
)
BASE_RETURN = (
    "        hidden_states = self.norm(hidden_states)\n"
    "        return hidden_states\n"
)

# Exact excerpts of the base image's kv_cache_utils.py (sha 7d299419...).
BASE_GUARD = """\
    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
    }
    if not mamba_specs or not all(
        type(spec) is MLAAttentionSpec for spec in attn_specs.values()
    ):
        return None
"""
BASE_GROUPS_RETURN = """\
    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )
"""
BASE_HELPER_DEF = "def _glm5_next_tensor_layout(\n"
BASE_RECOGNIZE = """\
    for group in uniform_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
            attn_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
            tail_group = group
    if attn_group is None or not mamba_groups:
        return None
"""
BASE_ANNOTATION = """\
    tuple[
        KVCacheGroupSpec,
        list[KVCacheGroupSpec],
        list[str],
        list[str],
        int,
        int,
        list[str],
        int,
    ]
"""
BASE_TAIL_FILL = """\
        tail_page = tail_pages.pop()
        if tail_page > idx_page:
            return None
"""
BASE_LAYOUT_RETURN = """\
    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
    )
"""
BASE_UNPACK = """\
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = glm5_layout
"""
BASE_BYTES_RETURN = """\
        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout
        return len(mla_names) * mla_page + len(idx_names) * idx_page
"""
BASE_CONFIG_BYTES = """\
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
"""
BASE_CONFIG_TENSORS = """\
                add_tensor(tail_name, tail_specs[tail_name], offset)

        return KVCacheConfig(
"""
BASE_MEMORY_RETURN = """\
        if tail_names:
            total_blocks += 1
        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)
"""
KV_FIXTURE = "".join([
    BASE_GUARD, BASE_GROUPS_RETURN, BASE_HELPER_DEF, BASE_RECOGNIZE,
    BASE_ANNOTATION, BASE_TAIL_FILL, BASE_LAYOUT_RETURN, BASE_UNPACK,
    BASE_UNPACK, BASE_BYTES_RETURN, BASE_CONFIG_BYTES, BASE_CONFIG_TENSORS,
    BASE_MEMORY_RETURN,
])


def _runner_refuses_without_interface(bases: tuple) -> bool:
    """The runner's rule: SupportsEagle3 must be in the target's bases."""
    return "SupportsEagle3" not in bases


def _base_guard_returns_none(spec_types: tuple) -> bool:
    """The base guard: every non-Mamba, non-tail spec must be MLA exactly."""
    attn = [t for t in spec_types if t not in ("MambaSpec", "KpoolTailSpec")]
    return not all(t == "MLAAttentionSpec" for t in attn)


def test_base_target_lacks_the_eagle3_interface():
    """Defect 1, red evidence: the base wrappers name no Eagle3 interface."""
    assert _runner_refuses_without_interface(
        ("Module", "HasInnerState", "SupportsPP", "MixtureOfExperts", "IsHybrid")
    )
    assert "SupportsEagle3" not in BASE_IMPORT
    assert "SupportsEagle3" not in BASE_CAUSAL
    assert "aux_hidden_states" not in BASE_LOOP + BASE_RETURN


def test_base_guard_drops_a_model_with_drafter_layers():
    """Defect 2, red evidence: five SWA layers trip the fast-path guard."""
    spec_types = (
        ("MambaSpec",) * 33
        + ("MLAAttentionSpec",) * 12
        + ("KpoolTailSpec",)
        + ("SlidingWindowSpec",) * 5
    )
    assert _base_guard_returns_none(spec_types)
    assert "SlidingWindowSpec" not in BASE_GUARD


def test_eagle3_patch_adds_the_interface_and_aux_plumbing():
    fixed = EAGLE.patched_source(
        (BASE_IMPORT + BASE_MODEL_CLASS + BASE_ACTIVE + BASE_LOOP_HEAD
         + BASE_LOOP + BASE_RETURN + BASE_CAUSAL + BASE_MLLM).encode()
    ).decode()
    assert "SupportsEagle3" in fixed
    assert "EagleModelMixin" in fixed
    assert "aux_hidden_states" in fixed
    assert not _runner_refuses_without_interface(
        ("Module", "HasInnerState", "SupportsPP", "MixtureOfExperts",
         "IsHybrid", "SupportsEagle3")
    )
    # A rerun completes instead of doubling the edits.
    assert EAGLE.patched_source(fixed.encode()).decode() == fixed


def test_kv_patch_partitions_the_drafter_and_keeps_the_guard():
    fixed = KV.patched_source(KV_FIXTURE.encode()).decode()
    assert "type(spec) is SlidingWindowSpec" in fixed
    assert "type(spec) is MLAAttentionSpec" in fixed
    assert "draft_group" in fixed
    assert "MambaSpec, KpoolTailSpec" in fixed
    # A rerun completes instead of doubling the edits.
    assert KV.patched_source(fixed.encode()).decode() == fixed


def test_patch_scripts_are_the_recorded_bytes():
    # The edits are the artifact; a test that names a different file than
    # the build runs is the image-tag problem again. Re-record on change.
    for name, digest in (
        ("patch_glm5next_eagle3.py",
         "fc1afc2e5fde1c12a587022fa48859dfea65659cc35521200d4195b5f8b97b76"),
        ("patch_glm5_drafter_kv_group.py",
         "86663be2310167dd448d8d270aaa8d19a75cfd3d044f428ca34df76d9a3ed009"),
        ("selfcheck_glm53_dflash2_drafter.py",
         "631c90d7f79f34e374ed8952d38eced23491810556e20a5f1c9faef6fcf4a96a"),
    ):
        body = (PATCH_DIR / name).read_bytes()
        assert hashlib.sha256(body).hexdigest() == digest, name


def test_patches_refuse_foreign_bytes():
    for patch in (EAGLE, KV):
        try:
            patch.patched_source(b"unrelated serving runtime")
        except ValueError:
            continue
        raise AssertionError(f"{patch.__name__} accepted foreign bytes")


