"""Bound requests for campaign preparation, capture and loaded execution."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class CaptureJoinRequest:
    calibration_census: str
    capture_calibration_out: str


@dataclass(frozen=True, slots=True)
class CapturePrepRequest:
    model: str
    nsamples: int
    seqlen: int
    seed: int
    layer_stride: int
    calibration_input_receipt: Mapping | None
    calibration_census: str
    units: str | None
    research_exact_member: str | None
    allow_pinned: str | None
    pinned_roster_only: bool
    max_act_rows: int
    attention_implementation: Literal['eager', 'sdpa']
    capture_chain_ranges: Sequence[tuple[int, int]]
    capture_chain_boundary_storage: Mapping
    capture_calibration_out: str


def capture_chain_bookend_request(args) -> CapturePrepRequest | CaptureJoinRequest:
    """Bind a phase after the CLI's ordered acquisition/draw/source refusals."""
    if args.capture_chain == 'join':
        return CaptureJoinRequest(calibration_census=args.calibration_census,
                                  capture_calibration_out=args.capture_calibration_out)
    if args.capture_chain != 'prep':
        raise ValueError('capture-chain bookend request requires prep or join')
    return CapturePrepRequest(
        model=args.model, nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed,
        layer_stride=args.layer_stride,
        calibration_input_receipt=args.calibration_input_receipt,
        calibration_census=args.calibration_census, units=args.units,
        research_exact_member=args.research_exact_member,
        allow_pinned=args.allow_pinned, pinned_roster_only=args.pinned_roster_only,
        max_act_rows=args.max_act_rows,
        attention_implementation=args.attention_implementation,
        capture_chain_ranges=args.capture_chain_ranges,
        capture_chain_boundary_storage=args.capture_chain_boundary_storage,
        capture_calibration_out=args.capture_calibration_out)


@dataclass(frozen=True, slots=True)
class CensusRequest:
    model: str
    nsamples: int
    seqlen: int
    seed: int
    layer_stride: int
    calibration_input_receipt: Mapping | None
    cache_dir: str
    census_out: str
    tp_degree: int
    family_restriction: Mapping | None


@dataclass(frozen=True, slots=True)
class CaptureMonolith:
    pass


@dataclass(frozen=True, slots=True)
class CaptureQuantum:
    layer_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    model: str
    nsamples: int
    seqlen: int
    seed: int
    calibration_input_receipt: Mapping | None
    cache_dir: str
    tp_degree: int
    family_restriction: Mapping | None
    calibration_census: str
    capture_calibration_out: str
    units: str | None
    phase: CaptureMonolith | CaptureQuantum
    max_act_rows: int
    streaming_capture_policy: Literal['legacy', 'shared-inputs-release-v1', 'shared-inputs-bounded-v1']
    capture_load_policy: Mapping | None
    streaming_cache_slots: int
    streaming_prefetch_workers: int
    streaming_cache_headroom_gb: float


def census_request(args) -> CensusRequest:
    """Bind the draw/scope and destination consumed by census publication."""
    if not args.census_out:
        raise ValueError('census request requires its output')
    return CensusRequest(model=args.model, nsamples=args.nsamples, seqlen=args.seqlen,
        seed=args.seed, layer_stride=args.layer_stride,
        calibration_input_receipt=args.calibration_input_receipt,
        cache_dir=args.cache_dir, census_out=args.census_out,
        tp_degree=args.tp_degree, family_restriction=args.family_restriction)


def streamed_calibration_request(args) -> CensusRequest | CaptureRequest:
    """Bind a model-running phase after prep/join and source admission."""
    if args.census_out:
        return census_request(args)
    if not args.capture_calibration_out or args.capture_chain not in (None, 'quantum'):
        raise ValueError('streamed calibration requires census, monolith or quantum')
    phase = (CaptureMonolith() if args.capture_chain is None
             else CaptureQuantum(args.capture_layer_range))
    return CaptureRequest(model=args.model, nsamples=args.nsamples, seqlen=args.seqlen,
        seed=args.seed, calibration_input_receipt=args.calibration_input_receipt,
        cache_dir=args.cache_dir, tp_degree=args.tp_degree,
        family_restriction=args.family_restriction,
        calibration_census=args.calibration_census,
        capture_calibration_out=args.capture_calibration_out, units=args.units,
        phase=phase, max_act_rows=args.max_act_rows,
        streaming_capture_policy=args.streaming_capture_policy,
        capture_load_policy=args.capture_load_policy,
        streaming_cache_slots=args.streaming_cache_slots,
        streaming_prefetch_workers=args.streaming_prefetch_workers,
        streaming_cache_headroom_gb=args.streaming_cache_headroom_gb)


@dataclass(frozen=True, slots=True)
class CampaignScopeRequest:
    """Controls read by loaded-model scope, census and topology admission."""
    model: str
    nsamples: int
    seqlen: int
    seed: int
    calibration_input_receipt: Mapping | None
    layer_stride: int
    allow_pinned: str | None
    pinned_roster_only: bool
    census_out: str | None
    calibration_census: str | None
    units: str | None
    research_exact_member: str | None
    rate_band: str | None
    max_rounds: int
    seed_checkpoint: str | None
    seed_wire_dir: str | None
    capture_calibration_out: str | None
    calibration_cache: str | None
    family_restriction: Mapping | None


def campaign_scope_request(args) -> CampaignScopeRequest:
    """Bind scope admission after the CLI's phase and source gates."""
    return CampaignScopeRequest(
        model=args.model,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        calibration_input_receipt=args.calibration_input_receipt,
        layer_stride=args.layer_stride,
        allow_pinned=args.allow_pinned,
        pinned_roster_only=args.pinned_roster_only,
        census_out=args.census_out,
        calibration_census=args.calibration_census,
        units=args.units,
        research_exact_member=args.research_exact_member,
        rate_band=args.rate_band,
        max_rounds=args.max_rounds,
        seed_checkpoint=args.seed_checkpoint,
        seed_wire_dir=args.seed_wire_dir,
        capture_calibration_out=args.capture_calibration_out,
        calibration_cache=args.calibration_cache,
        family_restriction=args.family_restriction,
    )


@dataclass(frozen=True, slots=True)
class CampaignEncoding:
    mode: Literal['attested', 'readable', 'research']
    tp_degree: int
    family_restriction: Mapping | None
    hessian: Literal['require', 'off']
    identity_bytes: int


@dataclass(frozen=True, slots=True)
class AnchorPlacement:
    anchors: int
    rate_band: str | None
    exhaustive_rate_grid: bool
    max_rounds: int


def anchor_placement(args) -> AnchorPlacement:
    """Bind the actual round-one plan shared by priming and execution."""
    return AnchorPlacement(anchors=args.anchors, rate_band=args.rate_band,
        exhaustive_rate_grid=args.exhaustive_rate_grid, max_rounds=args.max_rounds)


@dataclass(frozen=True, slots=True)
class CampaignRefinement:
    placement: AnchorPlacement
    anchor_budget: int
    loo_gate: float
    max_artifact_bpp: float


@dataclass(frozen=True, slots=True)
class CampaignPublication:
    batch_size: int
    overlap_bytes: int
    deadline_seconds: float
    requested_identity_threads: int


@dataclass(frozen=True, slots=True)
class CampaignOutputs:
    out: str
    wire_dir: Path
    checkpoint: Path
    hessian_reference_policy: Mapping | None


@dataclass(frozen=True, slots=True)
class CampaignDraw:
    model: str
    nsamples: int
    seqlen: int
    max_act_rows: int
    layer_stride: int


@dataclass(frozen=True, slots=True)
class CampaignSeed:
    checkpoint: str
    wire_dir: str | None


@dataclass(frozen=True, slots=True)
class ResidentCampaignWeights:
    pass


@dataclass(frozen=True, slots=True)
class SelectedCampaignWeights:
    capture: str
    capture_sha256: str
    census: str
    load_policy: Mapping | None
    snapshot_policy: Literal['whole-layer-v1', 'selected-tensors-v1']
    head: Literal['stream', 'load-all']


@dataclass(frozen=True, slots=True)
class LoadedCensusRequest:
    census: CensusRequest
    menu_mode: Literal['attested', 'readable', 'research']


@dataclass(frozen=True, slots=True)
class LoadedPricingRequest:
    encoding: CampaignEncoding
    draw: CampaignDraw
    source: ResidentCampaignWeights | SelectedCampaignWeights
    recipe: CampaignRefinement
    publication: CampaignPublication
    outputs: CampaignOutputs
    seed: CampaignSeed | None
    selection_requested: bool


def loaded_campaign_request(args, *, mode, selected_source, streaming_head,
                            wire_dir, checkpoint,
                            identity_threads) -> LoadedCensusRequest | LoadedPricingRequest:
    """Bind one admitted loaded phase; raw CLI settings remain a separate wire."""
    if args.census_out:
        return LoadedCensusRequest(census_request(args), mode)
    encoding = CampaignEncoding(mode=mode, tp_degree=args.tp_degree,
        family_restriction=args.family_restriction, hessian=args.hessian,
        identity_bytes=args.campaign_identity_bytes)
    source = (SelectedCampaignWeights(capture=args.calibration_cache,
        capture_sha256=args.calibration_cache_sha256, census=args.calibration_census,
        load_policy=args.capture_load_policy, snapshot_policy=args.source_snapshot_policy,
        head='stream' if streaming_head else 'load-all')
        if selected_source else ResidentCampaignWeights())
    return LoadedPricingRequest(encoding=encoding,
        draw=CampaignDraw(model=args.model, nsamples=args.nsamples,
            seqlen=args.seqlen, max_act_rows=args.max_act_rows, layer_stride=args.layer_stride),
        source=source,
        recipe=CampaignRefinement(placement=anchor_placement(args),
            anchor_budget=args.anchor_budget, loo_gate=args.loo_gate,
            max_artifact_bpp=args.max_artifact_bpp),
        publication=CampaignPublication(batch_size=args.anchor_batch_size,
            overlap_bytes=args.publication_overlap_bytes, deadline_seconds=args.deadline_seconds,
            requested_identity_threads=identity_threads),
        outputs=CampaignOutputs(out=args.out, wire_dir=wire_dir,
            checkpoint=checkpoint, hessian_reference_policy=args.export_hessian_reference_policy),
        seed=(None if not args.seed_checkpoint else
              CampaignSeed(args.seed_checkpoint, args.seed_wire_dir)),
        selection_requested=bool(args.units))
