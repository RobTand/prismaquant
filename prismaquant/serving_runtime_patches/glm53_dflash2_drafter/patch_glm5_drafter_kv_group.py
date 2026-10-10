#!/usr/bin/env python3
"""Keep the DFlash2 drafter's layers out of the GLM5-next fast-path guard.

``_get_kv_cache_groups_glm5_next`` (``vllm/v1/core/kv_cache_utils.py``)
returns None when any layer spec other than Mamba or the kpool tail is not
exactly ``MLAAttentionSpec``. The DFlash2 drafter registers five plain
sliding-window layers, so the model falls back to the generic page path,
whose page unification rescales the kpool tail's block away from its pool
size and dies at warmup on the tail-shape assert.

The fix partitions exact-type ``SlidingWindowSpec`` layers out of the guard
(exact type: ``KpoolTailSpec`` subclasses ``SlidingWindowSpec``, so the
tail stays where it is), keeps every byte of the base-model math, and
appends the drafter as one extra standalone group LAST, with native pages
and ``num_blocks``-sized per-layer tensors, so existing group ids and
offsets do not move. ``_glm5_next_tensor_layout`` recognizes the extra
group as a ninth element; without draft layers every consumer computes
exactly what it computed before.

Standalone native pages are deliberate: the draft page divides neither the
MLA page nor vice versa, so slot-sharing would need preconditions this
patch does not claim. The pool grows by the draft block; the eager serve
measures the cost.

Twelve edits. Each edit carries an already-applied marker that is absent
in the base file, so a rerun completes a partial run and a foreign file
is refused instead of double-applied. The file is refused unless it is
byte-identical to the base image's.
"""

import hashlib
from pathlib import Path

#: v1/core/kv_cache_utils.py as the base image ships it (upstream vLLM
#: fd4a15126, untouched by the mtp-mapper and kpool-tail sets). The edit
#: refuses any other bytes, so a moved base fails the build instead of
#: receiving an edit written for another file.
BASE_KV_SHA256 = (
    "7d299419aece21423962748d14101f60441a0f1c844c1bb37d53f6541d50da3a"
)

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = "v1/core/kv_cache_utils.py"

OLD_PARTITION = """\
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
NEW_PARTITION = """\
    # GLM53_DFLASH2_DRAFTER_GROUP partition: a spec-decode drafter
    # (DFlash2) adds plain exact-type SlidingWindowSpec layers on top of
    # the GLM-5-Next hybrid. Partition them out (exact type: KpoolTailSpec
    # subclasses SlidingWindowSpec) so they do not disqualify the model
    # from this fast path; they are appended as one extra group below.
    draft_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if type(spec) is SlidingWindowSpec
    }
    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
        and type(spec) is not SlidingWindowSpec
    }
    if not mamba_specs or not all(
        type(spec) is MLAAttentionSpec for spec in attn_specs.values()
    ):
        return None
"""

OLD_RETURN = """\
    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )
"""
NEW_RETURN = """\
    # GLM53_DFLASH2_DRAFTER_GROUP append: standalone native-page group LAST.
    draft_group: KVCacheGroupSpec | None = None
    if draft_specs:
        draft_uniform = UniformTypeKVCacheSpecs.from_specs(draft_specs)
        assert draft_uniform is not None
        draft_group = KVCacheGroupSpec(list(draft_specs), draft_uniform)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
        + ([draft_group] if draft_group is not None else [])
    )
"""

OLD_HELPER_ANCHOR = "def _glm5_next_tensor_layout(\n"
NEW_HELPER = """\
def _glm5_next_draft_specs(
    kv_cache_groups: list[KVCacheGroupSpec],
    draft_names: list[str],
) -> dict[str, KVCacheSpec]:
    \"\"\"The drafter's standalone specs, keyed by layer name (empty if none).\"\"\"
    if not draft_names:
        return {}
    draft_group = next(
        group for group in kv_cache_groups if draft_names[0] in group.layer_names
    )
    inner = cast(UniformTypeKVCacheSpecs, draft_group.kv_cache_spec).kv_cache_specs
    return dict(inner)


def _glm5_next_tensor_layout(
"""

OLD_RECOGNIZE = """\
    for group in uniform_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
            attn_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
            tail_group = group
    if attn_group is None or not mamba_groups:
        return None
"""
NEW_RECOGNIZE = """\
    # GLM53_DFLASH2_DRAFTER_GROUP recognize: the standalone drafter group.
    draft_group: KVCacheGroupSpec | None = None
    for group in uniform_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
            attn_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
            tail_group = group
        elif all(type(spec) is SlidingWindowSpec for spec in inner.values()):
            assert draft_group is None
            draft_group = group
    if attn_group is None or not mamba_groups:
        return None
"""

OLD_ANNOTATION = """\
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
NEW_ANNOTATION = """\
    tuple[
        KVCacheGroupSpec,
        list[KVCacheGroupSpec],
        list[str],
        list[str],
        int,
        int,
        list[str],
        int,
        list[str],
    ]
"""

OLD_TAIL_FILL = """\
        tail_page = tail_pages.pop()
        if tail_page > idx_page:
            return None
"""
NEW_TAIL_FILL = """\
        tail_page = tail_pages.pop()
        if tail_page > idx_page:
            return None

    draft_names: list[str] = []
    if draft_group is not None:
        draft_names = list(draft_group.layer_names)
"""

OLD_LAYOUT_RETURN = """\
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
NEW_LAYOUT_RETURN = """\
    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
        draft_names,
    )
"""

OLD_UNPACK = """\
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
NEW_UNPACK = """\
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
            draft_names,
        ) = glm5_layout
"""

OLD_BYTES_RETURN = """\
        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout
        return len(mla_names) * mla_page + len(idx_names) * idx_page
"""
NEW_BYTES_RETURN = """\
        _, _, mla_names, idx_names, mla_page, idx_page, _, _, draft_names = (
            glm5_layout
        )
        draft_block = sum(
            spec.page_size_bytes
            for spec in _glm5_next_draft_specs(
                kv_cache_groups, draft_names
            ).values()
        )
        return len(mla_names) * mla_page + len(idx_names) * idx_page + draft_block
"""

OLD_CONFIG_BYTES = """\
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
"""
NEW_CONFIG_BYTES = """\
        draft_specs = _glm5_next_draft_specs(kv_cache_groups, draft_names)
        draft_block = sum(
            spec.page_size_bytes for spec in draft_specs.values()
        )
        bytes_per_block = (
            len(mla_names) * mla_page + len(idx_names) * idx_page + draft_block
        )
"""

OLD_CONFIG_TENSORS = """\
                add_tensor(tail_name, tail_specs[tail_name], offset)

        return KVCacheConfig(
"""
NEW_CONFIG_TENSORS = """\
                add_tensor(tail_name, tail_specs[tail_name], offset)

        draft_base = idx_base + len(idx_names) * idx_page * num_blocks
        for index, draft_name in enumerate(draft_names):
            offset = (
                draft_base
                + index * draft_specs[draft_name].page_size_bytes * num_blocks
            )
            add_tensor(draft_name, draft_specs[draft_name], offset)

        return KVCacheConfig(
"""
OLD_MEMORY_RETURN = """\
        if tail_names:
            total_blocks += 1
        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)
"""
NEW_MEMORY_RETURN = """\
        if tail_names:
            total_blocks += 1
        draft_total = sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in _glm5_next_draft_specs(
                kv_cache_groups, draft_names
            ).values()
        )
        return (
            total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)
            + draft_total
        )
"""

EDITS = (
    # (tag, old, new, already-applied marker, expected old count).
    ("draft partition", OLD_PARTITION, NEW_PARTITION,
     "GLM53_DFLASH2_DRAFTER_GROUP partition", 1),
    ("draft group append", OLD_RETURN, NEW_RETURN,
     "GLM53_DFLASH2_DRAFTER_GROUP append", 1),
    ("draft helper", OLD_HELPER_ANCHOR, NEW_HELPER,
     "def _glm5_next_draft_specs(", 1),
    ("draft recognize", OLD_RECOGNIZE, NEW_RECOGNIZE,
     "GLM53_DFLASH2_DRAFTER_GROUP recognize", 1),
    ("layout annotation", OLD_ANNOTATION, NEW_ANNOTATION,
     "        int,\n        list[str],\n    ]", 1),
    ("draft names fill", OLD_TAIL_FILL, NEW_TAIL_FILL,
     "draft_names: list[str] = []", 1),
    ("pool block bytes", OLD_BYTES_RETURN, NEW_BYTES_RETURN,
     "for spec in _glm5_next_draft_specs(", 1),
    ("config block bytes", OLD_CONFIG_BYTES, NEW_CONFIG_BYTES,
     "for spec in draft_specs.values()", 1),
    ("config draft tensors", OLD_CONFIG_TENSORS, NEW_CONFIG_TENSORS,
     "draft_base = idx_base", 1),
    ("memory draft usage", OLD_MEMORY_RETURN, NEW_MEMORY_RETURN,
     "draft_total = sum(", 1),
)


def patched_source(data: bytes) -> bytes:
    """Apply the twelve edits; refuse foreign bytes, complete partial runs."""
    text = data.decode()
    for tag, old, new, marker, count in EDITS:
        if marker in text:
            continue
        found = text.count(old)
        if found != count:
            raise ValueError(
                f"expected {count} {tag} target(s), found {found}"
            )
        text = text.replace(old, new)
    return text.encode()


def main() -> None:
    target = SITE / TARGET
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BASE_KV_SHA256:
        raise SystemExit(f"{TARGET} is not the base image's (sha256 {digest})")
    target.write_text(patched_source(raw).decode())
    compile(target.read_text(), str(target), "exec")
    print("applied glm53_dflash2_drafter kv_group")


if __name__ == "__main__":
    main()
