"""Focused production-shaped tests for fixed-scale product LDLQ atoms."""

from __future__ import annotations

import pytest
import torch

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import subtable_bit_widths
from prismaquant.cb_ldlq_atoms import (
    LDLQ_CANDIDATE_WORKSPACE_DEFAULT,
    _forward_substitute_product_atom,
    assign_product_atom,
    candidate_chunk_size,
    candidate_workspace_bound_bytes,
    prepare_upper_inverse_cholesky,
    product_spec,
    reassign_product_2d,
    reassign_product_3d_batched,
)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _raw_fields(
    grid: str,
    *,
    weight: torch.Tensor | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    generator = _generator(seed)
    if weight is None:
        weight = torch.randn(3, 256, generator=generator) * 0.25
    col_weights = torch.rand(
        int(weight.shape[-1]), generator=generator
    ) + 0.05
    fields = cb.nvfp4_cb_fields(
        weight,
        12,
        grid=grid,
        mode="product",
        col_weights=col_weights,
        scale_sweep=False,
        encode_tier="max",
    )
    return weight, col_weights, fields


def test_product_atom_sizes_are_serialization_specific():
    fp4 = product_spec(grid="fp4", mode="product")
    fp8 = product_spec(grid="fp8", mode="product")
    assert (fp4.atom_size, fp4.subtables_per_vector) == (4, 2)
    assert (fp8.atom_size, fp8.subtables_per_vector) == (2, 4)


def test_k24_expert_batch_candidate_workspace_is_bounded():
    """K24 NVFP4 means a real 4096-entry table, not a tiny toy K."""
    rows = 16 * 2048  # current default expert batch times representative rows
    chunk = candidate_chunk_size(
        row_instances=rows,
        atom_size=4,
        codebook_entries=4096,
        element_size=4,
        workspace_bytes=LDLQ_CANDIDATE_WORKSPACE_DEFAULT,
    )
    bound = candidate_workspace_bound_bytes(
        row_instances=rows,
        atom_size=4,
        candidate_chunk=chunk,
        element_size=4,
    )
    assert 1 <= chunk < 4096
    assert bound <= LDLQ_CANDIDATE_WORKSPACE_DEFAULT
    # The current conservative formula resolves to 60 candidates: the search
    # makes multiple bounded passes instead of allocating E*R*4096*4.
    assert chunk == 60


@pytest.mark.parametrize(
    "grid,total_bits,n_sub,expected_widths,expected_chunks,expected_passes",
    [
        ("fp4", 18, 2, (9, 9), (60, 60), 18),
        ("fp8", 38, 4, (10, 10, 9, 9), (113, 113, 113, 113), 30),
    ],
)
def test_real_k18_k38_search_pass_estimates_are_explicit(
    grid: str,
    total_bits: int,
    n_sub: int,
    expected_widths: tuple[int, ...],
    expected_chunks: tuple[int, ...],
    expected_passes: int,
):
    """Document the exhaustive-search cost at E16 x R2048.

    K18 is NVFP4 with two 9-bit tables.  K38 is FP8 with four physical
    subtables (10/10/9/9), not a two-way 19/19 split.  Both stay within the
    candidate budget and have a tractable number of bounded passes.
    """
    spec = product_spec(grid=grid, mode="product")
    assert spec.subtables_per_vector == n_sub
    widths = subtable_bit_widths(total_bits, "product", n_sub)
    assert widths == expected_widths
    rows = 16 * 2048
    chunks = tuple(
        candidate_chunk_size(
            row_instances=rows,
            atom_size=spec.atom_size,
            codebook_entries=1 << width,
            element_size=4,
            workspace_bytes=LDLQ_CANDIDATE_WORKSPACE_DEFAULT,
        )
        for width in widths
    )
    assert chunks == expected_chunks
    assert sum(
        ((1 << width) + chunk - 1) // chunk
        for width, chunk in zip(widths, chunks)
    ) == expected_passes
    for chunk in chunks:
        assert candidate_workspace_bound_bytes(
            row_instances=rows,
            atom_size=spec.atom_size,
            candidate_chunk=chunk,
            element_size=4,
        ) <= LDLQ_CANDIDATE_WORKSPACE_DEFAULT


def test_chunked_atom_search_matches_whole_table_and_first_tie():
    generator = _generator(10)
    rows, atom, entries = 5, 4, 257
    target = torch.randn(rows, atom, generator=generator, dtype=torch.float64)
    scales = 0.1 + torch.rand(
        rows, atom, generator=generator, dtype=torch.float64
    )
    table = torch.randn(
        entries, atom, generator=generator, dtype=torch.float64
    )
    table[200] = table[3]
    upper = torch.triu(
        torch.randn(atom, atom, generator=generator, dtype=torch.float64) * 0.1
    )
    upper.diagonal().add_(1.0)

    whole = assign_product_atom(
        target,
        scales,
        table,
        upper,
        workspace_bytes=1 << 30,
    )
    # Force many small candidate chunks.
    per_candidate = candidate_workspace_bound_bytes(
        row_instances=rows,
        atom_size=atom,
        candidate_chunk=1,
        element_size=target.element_size(),
    )
    chunked = assign_product_atom(
        target,
        scales,
        table,
        upper,
        workspace_bytes=per_candidate * 7,
    )
    assert torch.equal(chunked[1], whole[1])
    torch.testing.assert_close(chunked[0], whole[0], rtol=0, atol=0)
    torch.testing.assert_close(chunked[2], whole[2], rtol=1e-13, atol=1e-13)

    tie_target = scales[:1] * table[3]
    tie = assign_product_atom(
        tie_target,
        scales[:1],
        table,
        torch.eye(atom, dtype=torch.float64),
        workspace_bytes=per_candidate * 7,
    )
    assert int(tie[1]) == 3


@pytest.mark.parametrize("atom", [2, 4])
def test_explicit_2d_atom_forward_substitution_matches_triangular_reference(
    atom: int,
):
    generator = _generator(100 + atom)
    rows, candidates = 7, 13
    residual = torch.randn(
        rows,
        candidates,
        atom,
        generator=generator,
        dtype=torch.float64,
    )
    upper = torch.triu(
        torch.randn(atom, atom, generator=generator, dtype=torch.float64) * 0.1
    )
    upper.diagonal().add_(1.0)
    rhs = residual.reshape(rows * candidates, atom).T.contiguous()
    reference = torch.linalg.solve_triangular(
        upper.T, rhs, upper=False
    ).T.reshape(rows, candidates, atom)
    explicit = _forward_substitute_product_atom(residual, upper)
    torch.testing.assert_close(explicit, reference, rtol=2e-15, atol=2e-15)


def test_r4096_fp8_workspace_no_longer_needs_a_wide_triangular_rhs():
    """Pin the audited dense shape whose old CUDA solve fell off a cliff."""
    rows, entries, atom = 4096, 1024, 2
    chunk = candidate_chunk_size(
        row_instances=rows,
        atom_size=atom,
        codebook_entries=entries,
        element_size=4,
        workspace_bytes=LDLQ_CANDIDATE_WORKSPACE_DEFAULT,
    )
    assert chunk == 910
    assert rows * chunk == 3_727_360


@pytest.mark.parametrize("grid,atom", [("fp4", 4), ("fp8", 2)])
def test_outer_tile_64_is_only_a_buffer_boundary(grid: str, atom: int):
    weight, _col_weights, fields = _raw_fields(grid, seed=20 + atom)
    generator = _generator(30 + atom)
    activations = torch.randn(48, 256, generator=generator)
    upper = prepare_upper_inverse_cholesky(
        activations,
        device=weight.device,
        damping_fraction=0.01,
    ).upper_inverse_cholesky
    kwargs = dict(
        weight=weight,
        scales=fields["scales"],
        codebooks=fields["codebook"],
        upper_inverse_cholesky=upper,
        grid=grid,
        mode="product",
        candidate_workspace_bytes=4 * 1024 * 1024,
    )
    natural = reassign_product_2d(**kwargs, outer_tile_columns=atom)
    tile64 = reassign_product_2d(**kwargs, outer_tile_columns=64)
    whole = reassign_product_2d(**kwargs, outer_tile_columns=256)
    assert torch.equal(natural.indices, tile64.indices)
    assert torch.equal(natural.indices, whole.indices)
    assert torch.equal(natural.reconstructed, tile64.reconstructed)
    assert torch.equal(natural.reconstructed, whole.reconstructed)


@pytest.mark.parametrize(
    "grid,index_tail",
    [("fp4", (32, 2)), ("fp8", (32, 4))],
)
def test_2d_callsite_preserves_field_geometry_and_fixed_values(
    grid: str,
    index_tail: tuple[int, int],
):
    weight, col_weights, fields = _raw_fields(grid, seed=40)
    activations = torch.randn(48, 256, generator=_generator(41))
    updated = cb.ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        activations,
        grid=grid,
        mode="product",
        block_size=64,
    )
    assert tuple(updated["indices"].shape) == (3, *index_tail)
    assert tuple(updated["indices"].shape) == tuple(fields["indices"].shape)
    assert updated["scales"] is fields["scales"]
    assert updated["codebook"] is fields["codebook"]
    assert updated["shape"] is fields["shape"]
    reconstruction = cb.nvfp4_cb_reconstruct(
        updated, 12, grid=grid, mode="product"
    )
    assert reconstruction.shape == weight.shape
    assert torch.isfinite(reconstruction).all()


def test_col_weights_do_not_reenter_fixed_field_ldlq_metric():
    weight, col_weights, fields = _raw_fields("fp4", seed=50)
    activations = torch.randn(48, 256, generator=_generator(51))
    first = cb.ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        activations,
        grid="fp4",
        mode="product",
    )
    second = cb.ldlq_reassign_cb_fields(
        weight,
        fields,
        torch.linspace(1000.0, 0.001, 256),
        activations,
        grid="fp4",
        mode="product",
    )
    assert torch.equal(first["indices"], second["indices"])


def test_hessian_failure_keeps_raw_fields_byte_geometry():
    weight, col_weights, fields = _raw_fields("fp4", seed=60)
    malformed = torch.randn(32, 256, generator=_generator(61))
    malformed[0, 0] = float("nan")
    updated = cb.ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        malformed,
        grid="fp4",
        mode="product",
    )
    assert updated["indices"] is fields["indices"]
    assert updated["scales"] is fields["scales"]
    assert updated["codebook"] is fields["codebook"]


def test_all_dead_hessian_keeps_raw_fields():
    weight, col_weights, fields = _raw_fields("fp4", seed=62)
    updated = cb.ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        torch.zeros(32, 256),
        grid="fp4",
        mode="product",
    )
    assert updated["indices"] is fields["indices"]
    assert updated["scales"] is fields["scales"]
    assert updated["codebook"] is fields["codebook"]


def test_partial_dead_hessian_keeps_raw_fields_for_coupled_atoms():
    weight, col_weights, fields = _raw_fields("fp4", seed=63)
    activations = torch.randn(48, 256, generator=_generator(64))
    activations[:, 0] = 0
    updated = cb.ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        activations,
        grid="fp4",
        mode="product",
    )
    assert updated["indices"] is fields["indices"]
    assert updated["scales"] is fields["scales"]
    assert updated["codebook"] is fields["codebook"]


def test_unsupported_mode_fails_closed_with_gate_telemetry():
    """LDLQ supports ONLY product mode; anything else must pass through.

    This used to be exercised on ``signed``, which was deleted 2026-08-17.
    Retargeted to ``full`` rather than removed: the property under test is
    LDLQ's fail-closed behaviour on a mode whose assignment atoms it cannot
    reassign, and that property outlived the particular mode that motivated
    it. ``full`` is now the only such mode, so it is the only remaining way to
    reach this branch at all.
    """
    generator = _generator(70)
    weight = torch.randn(2, 256, generator=generator) * 0.2
    col_weights = torch.rand(256, generator=generator) + 0.1
    fields = cb.nvfp4_cb_fields(
        weight,
        13,
        grid="fp4",
        mode="full",
        col_weights=col_weights,
        scale_sweep=False,
        encode_tier="max",
    )
    activations = torch.randn(32, 256, generator=generator)
    direct = cb.ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        activations,
        grid="fp4",
        mode="full",
    )
    assert direct["indices"] is fields["indices"]
    gated, info = cb.ldlq_reassign_cb_fields_gated(
        weight,
        fields,
        col_weights,
        activations,
        grid="fp4",
        mode="full",
        k=13,
    )
    assert gated is fields
    assert info["gate"] == "raw_unsupported_ldlq_mode"
    assert info["kept_ldlq"] is False


def test_packed_artifact_route_refuses_serial_and_nondivisible_experts(
    monkeypatch,
):
    generator = _generator(80)
    experts, rows, columns = 3, 2, 256
    weight = torch.randn(
        experts, rows, columns, generator=generator
    ) * 0.2
    col_weights = torch.rand(columns, generator=generator) + 0.1
    fields = cb.nvfp4_cb_fields(
        weight,
        12,
        grid="fp4",
        mode="product",
        col_weights=col_weights,
        scale_sweep=False,
        encode_tier="max",
    )
    activations = tuple(
        torch.randn(32 + expert, columns, generator=generator)
        for expert in range(experts)
    )
    with pytest.raises(RuntimeError, match="canonical E16 batching"):
        cb.ldlq_reassign_cb_fields(
            weight,
            fields,
            col_weights,
            activations,
            grid="fp4",
            mode="product",
            batch_experts=False,
        )
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ_EXPERT_BATCH", "16")
    with pytest.raises(RuntimeError, match="not divisible"):
        cb.ldlq_reassign_cb_fields(
            weight,
            fields,
            col_weights,
            activations,
            grid="fp4",
            mode="product",
            batch_experts=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_batched_matches_serial_atoms_and_outer_tile_64():
    device = torch.device("cuda")
    generator = _generator(90)
    experts, rows, columns = 2, 3, 256
    weight_cpu = torch.randn(
        experts, rows, columns, generator=generator
    )
    scales_cpu = 0.1 + torch.rand(
        experts, rows, columns // 16, generator=generator
    )
    tables_cpu = tuple(
        torch.randn(64, 4, generator=generator) for _ in range(2)
    )
    activations = tuple(
        torch.randn(64, columns, generator=generator).to(device)
        for _ in range(experts)
    )
    upper = torch.stack(
        [
            prepare_upper_inverse_cholesky(
                rows_, device=device, damping_fraction=0.01
            ).upper_inverse_cholesky
            for rows_ in activations
        ]
    )
    weight = weight_cpu.to(device)
    scales = scales_cpu.to(device)
    tables = tuple(table.to(device) for table in tables_cpu)
    batched = reassign_product_3d_batched(
        weight,
        scales,
        tables,
        upper,
        grid="fp4",
        mode="product",
        outer_tile_columns=64,
        candidate_workspace_bytes=8 * 1024 * 1024,
    )
    serial = [
        reassign_product_2d(
            weight[expert],
            scales[expert],
            tables,
            upper[expert],
            grid="fp4",
            mode="product",
            outer_tile_columns=64,
            candidate_workspace_bytes=8 * 1024 * 1024,
        )
        for expert in range(experts)
    ]
    atom_outer = reassign_product_3d_batched(
        weight,
        scales,
        tables,
        upper,
        grid="fp4",
        mode="product",
        outer_tile_columns=4,
        candidate_workspace_bytes=8 * 1024 * 1024,
    )
    serial_indices = torch.stack([part.indices for part in serial])
    serial_reconstruction = torch.stack(
        [part.reconstructed for part in serial]
    )
    assert torch.equal(batched.indices, serial_indices)
    assert torch.equal(batched.reconstructed, serial_reconstruction)
    assert torch.equal(batched.indices, atom_outer.indices)
    assert torch.equal(batched.reconstructed, atom_outer.reconstructed)
