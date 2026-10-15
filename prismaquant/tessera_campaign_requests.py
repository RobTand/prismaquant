"""Bound requests for campaign census and calibration-capture operations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
