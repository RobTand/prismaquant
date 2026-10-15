"""Bound requests for capture-chain operations that never load a model."""
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
