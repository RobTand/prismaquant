"""Cold command schema for Tessera campaign execution and planning.

The parser is the CLI authority. Numerical admission, immutable-source proof
and execution remain with the campaign operations.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .tessera_campaign_table import parse_family_restriction
from .tessera_menu import MENU_MODES
from .tessera_row_stream import ROW_HEAD_STREAM, ROW_HEADS
from .tessera_serving_scope import add_serving_scope_arguments

# Both endpoints and a midpoint give leave-one-out one interior anchor.
ROUND_ONE_ANCHORS = 3


def campaign_argument_parser(*, description: str | None = None) -> argparse.ArgumentParser:
    """Build the exact execution parser without initializing model runtimes."""
    ap = argparse.ArgumentParser(description=description)
    add_serving_scope_arguments(ap)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="cost payload (.pkl)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="identity-bound JSON manifest with sibling .parts "
                         "unit shards; defaults beside --out")
    ap.add_argument("--menu-mode", default=None, choices=sorted(MENU_MODES))
    ap.add_argument("--family-restriction", default=None, type=parse_family_restriction,
                    help="Opt-in JSON prismaquant.tessera_campaign_family_restriction.v1 "
                         "with explicit dense and routed_moe canonical family lists. "
                         "Narrows pricing only; grants no serving support. Restricted "
                         "seed imports refuse incompatible families or rate-band anchors.")
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration-input", default=None,
                    help="Optional immutable calibration_ids safetensors draw; requires its SHA256 and real corpus.")
    ap.add_argument("--calibration-input-sha256", default=None,
                    help="Independent SHA256 of --calibration-input, binding its draw provenance.")
    ap.add_argument("--calibration-corpus", default=None,
                    help="Actual UTF-8 corpus file matching the pinned draw's provenance.text_sha256.")
    ap.add_argument("--max-act-rows", type=int, default=512)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="price every Nth decoder layer (1 = every Linear)")
    ap.add_argument("--anchors", type=int, default=ROUND_ONE_ANCHORS)
    ap.add_argument("--anchor-batch-size", type=int, default=1,
                    help="maximum compatible expert anchors in one producer "
                         "batch within this action (1 = scalar). Does not "
                         "change the anchor schedule or PB placement.")
    ap.add_argument("--campaign-identity-bytes", type=int, default=0,
                    help="experimental closed-roster producer identity hold reservation. "
                         "0 keeps the feature off; a positive value is charged by "
                         "selected-source PB admission and runtime refuses a larger plan.")
    ap.add_argument("--campaign-identity-threads", type=int, default=None,
                    help="threads that read, verify and receipt selected capture "
                         "entries (the streaming row head), or that build the "
                         "closed-roster identity hold and verify resumed wire "
                         "receipts (the load-all head). Default: the CPUs this row "
                         "was admitted with. The selected-source plan charges N "
                         "in-flight host copies.")
    ap.add_argument("--row-head", default=ROW_HEAD_STREAM, choices=ROW_HEADS,
                    help="how a selected-source row gets its X and H. 'stream' "
                         "(default) starts encoding once the first batch's capture "
                         "entries are verified and resident, keeps two batches "
                         "resident, and writes the six identity-bound outputs at "
                         "finalize; a row that needs the whole set first runs "
                         "'load-all' instead and prints the dependency. 'load-all' "
                         "makes every selected X and H resident before the first encode.")
    ap.add_argument("--publication-overlap-bytes", type=int, default=0,
                    help="stage up to N bytes of already-encoded render/wire "
                         "artifacts on one writer thread so the next batch "
                         "encodes while they are written (0 = publish "
                         "synchronously on the encode thread, the default and "
                         "byte-for-byte the historical path). An anchor is "
                         "still journalled only after its own files exist.")
    ap.add_argument("--max-rounds", type=int, default=0,
                    help="hard stop on adaptive rounds (0 = governed by "
                         "--anchor-budget instead). Rounds are not the "
                         "budget: a round adds ONE anchor to each surface "
                         "that is still failing its gate, so capping rounds "
                         "caps how far the worst surface can be improved, "
                         "which is the opposite of what the adaptive loop is "
                         "for.")
    ap.add_argument("--rate-band", default=None,
                    help="'lo,hi' in q256 body-rate units. Round one puts a "
                         "window family's two anchors at the ends of this "
                         "band instead of at the ends of its whole realisable "
                         "range, and an audited unit gets a third inside it. "
                         "Unset reproduces every artifact built before "
                         "2026-09-06.")
    ap.add_argument("--exhaustive-rate-grid", action="store_true",
                    help="Research-only: in the declared --rate-band, measure every "
                         "legal family rung in round one. This changes neither the "
                         "family's legal domain nor serving/export admission.")
    ap.add_argument("--anchor-budget", type=int, default=12,
                    help="max anchors per (fused group, family) surface. The "
                         "adaptive loop keeps splitting the worst-predicted "
                         "interval until every member's LOO clears "
                         "--loo-gate or this budget is spent, and the payload "
                         "records which of the two stopped each surface.")
    ap.add_argument("--loo-gate", type=float, default=0.25,
                    help="max |log2 error| an interpolated surface may carry")
    ap.add_argument("--tp-degree", type=int, default=1)
    ap.add_argument("--max-artifact-bpp", type=float, default=0.0,
                    help="cap the MEASURED envelope at this artifact bpp "
                         "(0 = each family's whole legal range). A budget "
                         "decision, not a menu one: rungs above the envelope "
                         "stay in the menu and are simply not priced, because "
                         "the surface refuses to extrapolate.")
    ap.add_argument("--hessian", default="require", choices=("require", "off"),
                    help="'require' (default) prices every rung with the "
                         "per-unit XtX from this run's calibration "
                         "activations, which is what the encoder's shipping "
                         "default consumes; it REFUSES if the pinned encoder "
                         "cannot take one. 'off' prices weights-only "
                         "deliberately and stamps hessian.supplied=false on "
                         "every row, so a weights-only price can never be "
                         "read as a shipping one.")
    ap.add_argument("--deadline-seconds", type=float, default=0.0,
                    help="stop starting new anchors after this much wall time")
    ap.add_argument("--units", default=None,
                    help="JSON selection of fused anchor groups to MEASURE "
                         "(prismaquant.tessera_campaign_units.v1). The scope "
                         "resolution, the census and the menus are unchanged; "
                         "only the encoding work narrows, and the checkpoint "
                         "identity narrows with it, so two invocations over "
                         "disjoint selections never contend. Omitted: measure "
                         "every group in scope, exactly as before.")
    ap.add_argument("--research-exact-member", default=None,
                    help="Research-only scalar endpoint for one named member of a "
                         "fully validated --units group. Requires a complete "
                         "calibration census/cache and one rate-band; carries no "
                         "allocator or stack-estimate licence.")
    ap.add_argument("--calibration-census", default=None,
                    help="JSON per-unit calibration row counts for the WHOLE "
                         "priced scope (prismaquant.tessera_campaign_census.v1, "
                         "written by --census-out). Every unit this run hooks "
                         "is checked against it; the run then stamps the "
                         "scope's fit_tokens rather than its own selection's, "
                         "which is what makes a sharded campaign's Hessian "
                         "identity the monolith's.")
    ap.add_argument("--seed-checkpoint", default=None,
                    help="another campaign's checkpoint manifest whose already "
                         "measured anchors this run may adopt for the units it "
                         "prices. Every adopted row goes through the SAME "
                         "per-row gates a resume does -- the producer's input "
                         "identity recomputed from this run's weights, menu, "
                         "Hessian and static scale, and the cached wire "
                         "re-verified against it -- so a row is adopted only "
                         "when its bytes are the bytes this run would have "
                         "encoded. The run-level identities need not match: "
                         "that is the point, and the difference is stamped.")
    ap.add_argument("--seed-wire-dir", default=None,
                    help="the seed checkpoint's wire cache; its blobs are "
                         "linked into this run's wire dir before verification. "
                         "Defaults to <seed cache>/wire beside the manifest.")
    ap.add_argument("--census-out", default=None,
                    help="collect the calibration census over the whole scope, "
                         "write it here and exit. No Hessians, no retained "
                         "scoring rows, no encodes.")
    ap.add_argument("--attention-implementation", choices=("eager", "sdpa"), default=None,
                    help="Explicit HF attention backend required for canonical census/capture.")
    ap.add_argument("--streaming", action="store_true",
                    help="Use the source layer cache for census/capture or selected anchors from a complete capture.")
    ap.add_argument("--streaming-cache-slots", type=int, default=2)
    ap.add_argument("--source-snapshot-policy", default="whole-layer-v1",
                    choices=("whole-layer-v1", "selected-tensors-v1"),
                    help="selected-tensors-v1 loads authenticated weight dependencies only; requires selected streaming capture reuse")
    ap.add_argument("--source-scope", default=None,
                    help="A profile-declared source outside the decoder body (ModelProfile.source_scope, "
                         "e.g. GLM's 'mtp'); requires --source-snapshot-policy selected-tensors-v1 and a "
                         "census carrying the scope's load contract (PQ #1316).")
    ap.add_argument("--allow-pinned", default=None,
                    help="Comma-separated tokens: a profile-pinned Linear whose name contains one "
                         "enters the roster (the allocator's --allow-pinned grammar). A token that "
                         "lifts nothing refuses. The census records the lift (PQ #1843).")
    ap.add_argument("--pinned-roster-only", action="store_true",
                    help="Price only the Linears --allow-pinned lifts: the unpinned body and the "
                         "expert population leave the roster (a scoped campaign, e.g. GLM "
                         "attention; PQ #1843). Requires --allow-pinned.")
    ap.add_argument("--streaming-prefetch-workers", type=int, default=1)
    ap.add_argument("--streaming-cache-headroom-gb", type=float, default=24)
    ap.add_argument("--streaming-capture-policy", default="legacy",
                    choices=("legacy", "shared-inputs-release-v1", "shared-inputs-bounded-v1"),
                    help="Opt-in shared capture; bounded also checks phase ownership and physical memory.")
    ap.add_argument("--capture-load-policy", type=json.loads, default=None,
                    help="Explicit verified activation load v1 JSON policy; requires bounded capture or hash-bound selected streaming reuse.")
    ap.add_argument('--export-hessian-reference-policy', type=json.loads, default=None,
                    help='Opt-in canonical H reference load-policy JSON; requires selected reuse of a complete capture.')
    ap.add_argument("--capture-calibration-out", default=None,
                    help="Capture full-census float32 prefix X and uncapped H once, then exit.")
    ap.add_argument("--capture-chain", default=None, choices=("prep", "quantum", "join"),
                    help="Run the streamed capture as layer-chain quanta (PQ #1885): prep seals "
                         "the identity and ranges once, each quantum captures one layer range "
                         "from its predecessor's boundary, and join publishes the manifest.")
    ap.add_argument("--capture-layer-range", default=None,
                    help="A capture chain quantum's source layers A:B.")
    ap.add_argument("--capture-chain-ranges", default=None,
                    help="A capture chain prep's layer ranges A:B,B:C,... tiling every source layer.")
    ap.add_argument("--capture-chain-boundary-storage", type=json.loads, default=None,
                    help="A capture chain prep's boundary storage policy JSON: where the quanta "
                         "pass hidden states along, and its byte ceilings.")
    ap.add_argument("--calibration-cache", default=None,
                    help="Verified capture manifest; prefetch selected X/H before encoding.")
    ap.add_argument("--calibration-cache-sha256", default=None,
                    help="Expected capture manifest hash, sealed by the campaign planner.")
    ap.add_argument("--source-identity-cache", default=None,
                    help="A prismaquant.streamed_model.identity_cache.v1 proof of every source "
                         "shard's full-file SHA256. A selected-source row adopts it before any "
                         "payload read instead of hashing the shards it reads; a proof that "
                         "refuses leaves every read to hash fresh (PQ #1497).")
    ap.add_argument("--source-identity-cache-sha256", default=None,
                    help="Expected SHA256 of the --source-identity-cache file, bound by the planner.")
    ap.add_argument("--acquisition-request", default=None,
                    help="Opt-in authenticated joint full-domain acquisition request; schedules exact research render/wire work only.")
    ap.add_argument("--acquisition-request-sha256", default=None,
                    help="Independent SHA256 of --acquisition-request; both flags are required together.")
    return ap


def parse_campaign_arguments(argv: Sequence[str] | None = None, *,
                             description: str | None = None) -> argparse.Namespace:
    """Parse external CLI values; execution normalizes and binds requests next."""
    return campaign_argument_parser(description=description).parse_args(argv)


def campaign_action_arguments(argv: Sequence[str]) -> list[str]:
    """The campaign arguments after an action's last Python module invocation."""
    items = [str(token) for token in argv]
    if '-m' not in items:
        raise ValueError('row argv runs no python module, so its demand cannot be derived')
    index = len(items) - 1 - items[::-1].index('-m')
    return items[index + 2:]


def explicit_campaign_options(argv: Sequence[str]) -> dict[str, object]:
    """Parse partial planning argv using execution's syntax and conversions.

    Missing fields stay absent, including false/zero-valued explicit options.
    The planner supplies model/output/cache paths outside shared campaign argv;
    only those execution-only requirements are relaxed here.
    """
    parser = campaign_argument_parser()
    for action in parser._actions:
        if action.dest in {'model', 'out', 'cache_dir'}:
            action.required = False
        action.default = argparse.SUPPRESS
    items = [str(token) for token in argv]
    if items and not items[0].startswith('-'):
        items = campaign_action_arguments(items)
    return vars(parser.parse_args(items))
