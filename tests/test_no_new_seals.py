"""No new run-gate seals on the campaign path (PQ #1147).

Rob, 2026-09-24: "All sealing should be disabled until further notice." A run
seal compares a recorded identity with the running one and refuses. Those go
through ``prismaquant.dev_mode.seal_check``, which refuses only under
``PRISMAQUANT_DEV_MODE=0``.

This lint finds every ``if`` whose test compares two non-literal values with
``==`` or ``!=``, where a name on either side mentions ``sha256``,
``identity`` or ``digest``, and whose body raises. Every such site on the
campaign path is on the allowlist below with its reason. Most are byte
integrity: bytes that do not hash to the digest they were stored under, or a
record that does not reproduce its own seal. A new site fails this test until
it goes through ``seal_check`` or earns an allowlist entry with a reason.

The allowlist keys on (file, enclosing function) with an exact count, so a new
seal inside an allowlisted function also fails. Converting a site to
``seal_check`` lowers the count and fails the test too, until the entry is
updated.

The lint reads ``if``-then-``raise`` sites only. Helpers that compare and raise
inside a call (``_same(...)``) are outside it.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The campaign path: Stage A, Stage B, the join, the dispatchers, the catalog
# extension, the qualification, and the modules they call for identities.
MODULES = (
    "prismaquant/aura_cost.py",
    "prismaquant/cost_currency.py",
    "prismaquant/cost_streaming.py",
    "prismaquant/dev_mode.py",
    "prismaquant/joint_adjoint_checkpoints.py",
    "prismaquant/joint_catalog_extension.py",
    "prismaquant/joint_cost_quantum.py",
    "prismaquant/joint_cost_stage_a.py",
    "prismaquant/joint_forward_resume.py",
    "prismaquant/joint_layer_quanta.py",
    "prismaquant/joint_projection_backend.py",
    "prismaquant/joint_quanta_join.py",
    "prismaquant/joint_replay_spill.py",
    "prismaquant/joint_stage_b_head.py",
    "prismaquant/stage_a_chain_resume.py",
    "prismaquant/stage_a_chain_seed.py",
    "prismaquant/stage_a_chain_split.py",
    "prismaquant/tessera_joint_aura.py",
    "tools/dispatch_joint_quanta.py",
    "tools/dispatch_tessera_campaign.py",
    "tools/regenerate_joint_quanta.py",
)

INTEGRITY = "integrity"
STRUCTURE = "structure"
WALL = "wall"
AMBIGUOUS = "ambiguous"

# (file, enclosing function) -> (count, kind, reason). ``wall`` is a data or
# layout identity (calibration, roster, probes) that dev mode must not cross.
# ``ambiguous`` was listed in the PQ #1147 report; the ruling (2026-09-24)
# keeps each such site a refusal in both modes, as on main.
ALLOWLIST = {
    ("prismaquant/aura_cost.py", "_load_aura_unit_checkpoint"): (
        1, INTEGRITY, "a unit checkpoint's envelope digest against its payload bytes"),
    ("prismaquant/aura_cost.py", "compute_aura_cost_streamed"): (
        4, AMBIGUOUS, "render coverage against the roster and the render tensor proof "
        "(integrity); a restored row's probe and activation identity (ambiguous, #1147 "
        "report)"),
    ("prismaquant/aura_cost.py", "compute_aura_cost_streamed.commit_streamed_units"): (
        1, AMBIGUOUS, "the source execution backend changed during one measurement"),
    ("prismaquant/aura_cost.py", "compute_aura_cost_streamed._record_joint_operator"): (
        1, INTEGRITY, "the operator measured is the render the cache holds, within one run"),
    ("prismaquant/cost_streaming.py", "StreamedBoundaryArtifacts.rebind"): (
        1, WALL, "the bind identity digest covers calibration; a dev resume compares key "
        "by key first and rebinds the stored identity"),
    ("prismaquant/cost_streaming.py", "StreamedBoundaryArtifacts.authorize_resume_inputs"): (
        1, STRUCTURE, "a resumed boundary entry belongs to this generation"),
    ("prismaquant/cost_streaming.py", "StreamedBoundaryArtifacts.attach"): (
        1, STRUCTURE, "the attached forward recovery is the session it records"),
    ("prismaquant/cost_streaming.py", "StreamedBoundaryArtifacts._entry_identity"): (
        1, STRUCTURE, "a boundary reference names a live generation"),
    ("prismaquant/cost_streaming.py", "_hash_one_source_shard"): (
        1, INTEGRITY, "a source shard changed while it was hashed"),
    ("prismaquant/cost_streaming.py", "build_source_checkpoint_identity"): (
        1, INTEGRITY, "source metadata changed while it was hashed"),
    ("prismaquant/cost_streaming.py", "build_streamed_model_identity"): (
        1, INTEGRITY, "a source shard changed while it was hashed"),
    ("prismaquant/cost_streaming.py", "_read_streamed_model_identity_cache"): (
        1, INTEGRITY, "an identity cache for another source is not this source's hashes"),
    ("prismaquant/cost_streaming.py", "validate_streamed_model_identity"): (
        2, INTEGRITY, "the identity schema, and content_sha256 against its shard digests"),
    ("prismaquant/cost_streaming.py", "validate_cached_streamed_model_identity"): (
        1, INTEGRITY, "a cached identity whose tensor-to-shard map is not this checkpoint's"),
    ("prismaquant/joint_adjoint_checkpoints.py", "unpack_shared_states"): (
        1, INTEGRITY, "a pack member against its recorded digest"),
    ("prismaquant/joint_adjoint_checkpoints.py", "AdjointCheckpointAttempt.reference_activation"): (
        1, STRUCTURE, "a checkpoint names only its owner's cotangent at its boundary"),
    ("prismaquant/joint_adjoint_checkpoints.py", "_load_checkpoint_shared_states"): (
        1, INTEGRITY, "a shared-state entry against its recorded digest and size"),
    ("prismaquant/joint_adjoint_checkpoints.py", "read_shared_state_pack"): (
        1, INTEGRITY, "a shared-state entry against its recorded digest and size"),
    ("prismaquant/joint_adjoint_checkpoints.py", "load_adjoint_receipt"): (
        1, INTEGRITY, "receipt bytes against the pinned digest"),
    ("prismaquant/joint_cost_quantum.py", "verify_quantum_identity"): (
        4, AMBIGUOUS, "record bytes, the record's own seal and each input file against its "
        "digest (integrity); the Stage A slice binding (ambiguous, #1147 report)"),
    ("prismaquant/joint_cost_quantum.py", "require_chain_readset"): (
        1, INTEGRITY, "the staged data manifest is the readset the record names"),
    ("prismaquant/joint_cost_quantum.py", "bind_joint_served_quantizer"): (
        1, AMBIGUOUS, "a format override against the served quantizer binding"),
    ("prismaquant/joint_cost_quantum.py", "run_layer_quantum_core"): (
        1, AMBIGUOUS, "the Stage A slice handed in against the one the record binds"),
    ("prismaquant/joint_cost_quantum.py", "run_layer_quantum_core._record_joint_operator"): (
        1, INTEGRITY, "the operator measured is the render the cache holds, within one run"),
    ("prismaquant/joint_cost_quantum.py", "run_layer_quantum_core.commit_streamed_units"): (
        1, AMBIGUOUS, "the source execution backend changed during one measurement"),
    ("prismaquant/joint_cost_quantum.py", "main"): (
        1, STRUCTURE, "a path and its digest are supplied together"),
    ("prismaquant/joint_forward_resume.py", "_read"): (
        2, INTEGRITY, "proof bytes against the pinned digest"),
    ("prismaquant/joint_forward_resume.py", "_checked_group"): (
        3, INTEGRITY, "the export manifest, its action and descriptor bytes against their "
        "digests"),
    ("prismaquant/joint_forward_resume.py", "validate_forward_state"): (
        1, INTEGRITY, "the stored session digest reproduces from the stored bind identity"),
    ("prismaquant/joint_forward_resume.py", "build_forward_recovery"): (
        1, INTEGRITY, "a local export manifest changed after it was written"),
    ("prismaquant/joint_layer_quanta.py", "check_quantum_for_campaign"): (
        2, AMBIGUOUS, "the record reproduces its own seal (integrity); the Stage A slice "
        "binding (ambiguous, #1147 report)"),
    ("prismaquant/joint_layer_quanta.py", "covered_slices"): (
        1, WALL, "Stage A proofs from mixed runs cover no one chain"),
    ("prismaquant/joint_layer_quanta.py", "bind_quantum_boundary_readset"): (
        2, AMBIGUOUS, "the readset wire against its digest (integrity); the Stage A slice "
        "binding (ambiguous, #1147 report)"),
    ("prismaquant/joint_layer_quanta.py", "check_head_slice_binding"): (
        2, AMBIGUOUS, "a head slice's prepared and production-pickle bindings, one "
        "executable's internal agreement"),
    ("prismaquant/joint_layer_quanta.py", "build_quantum_executable_manifest"): (
        2, AMBIGUOUS, "prepared inputs against the render prerequisite and the campaign, one "
        "executable's internal agreement"),
    ("prismaquant/joint_layer_quanta.py", "build_quantum_executable_manifest._take"): (
        1, INTEGRITY, "two entries for one offset disagree on bytes"),
    ("prismaquant/joint_layer_quanta.py", "bind_quantum_executable"): (
        2, AMBIGUOUS, "the executable wire against its digest (integrity); the Stage A slice "
        "binding (ambiguous, #1147 report)"),
    ("prismaquant/joint_projection_backend.py", "normalize_projection_backend"): (
        1, INTEGRITY, "the kernel binary against its packaged qualification"),
    ("prismaquant/joint_projection_backend.py", "validate_projection_backend_identity"): (
        1, AMBIGUOUS, "a serialized arithmetic record must name a qualified backend "
        "(numerics; the runtime comparison is a seal_check)"),
    ("prismaquant/joint_projection_backend.py", "prewarm_projection_backend"): (
        3, INTEGRITY, "the compiled source, flags and loaded binary against the "
        "qualification's bytes"),
    ("prismaquant/joint_quanta_join.py", "_check_record"): (
        3, INTEGRITY, "the record's own seal, a retargeted receipt, and the unit roster "
        "(wall)"),
    ("prismaquant/joint_quanta_join.py", "_read_status"): (
        1, INTEGRITY, "a status receipt belongs to its quantum"),
    ("prismaquant/joint_quanta_join.py", "_load_cost_payload"): (
        2, INTEGRITY, "a payload answers for its own record and Stage A slice (retargeted "
        "receipt)"),
    ("prismaquant/joint_quanta_join.py", "_preserve_allocation_payload"): (
        1, WALL, "quanta joined into one table share their probe and measurement shape"),
    ("prismaquant/stage_a_chain_resume.py", "load_chain_state"): (
        2, INTEGRITY, "chain state bytes against the pinned digest and its own seal"),
    ("prismaquant/stage_a_chain_resume.py", "read_chain_state"): (
        1, INTEGRITY, "chain state bytes reproduce their own seal (a split join's reader)"),
    ("prismaquant/stage_a_chain_resume.py", "resume_records"): (
        1, INTEGRITY, "a resume record reproduces its own seal"),
    ("prismaquant/stage_a_chain_resume.py", "plan_chain_resume"): (
        1, INTEGRITY, "the tail checkpoint the chain state names is on disk and sealed"),
    ("prismaquant/stage_a_chain_seed.py", "load_seed_spec"): (
        1, INTEGRITY, "seed spec bytes against the pinned digest"),
    ("prismaquant/stage_a_chain_seed.py", "_sealed_checkpoint"): (
        2, INTEGRITY, "checkpoint bytes against the pinned digest and its own seal"),
    ("prismaquant/stage_a_chain_seed.py", "plan_chain_seed"): (
        1, WALL, "the bind identity digest covers calibration and probes; the checkpoint "
        "records nothing finer"),
    ("prismaquant/stage_a_chain_seed.py", "compare_seed_plane"): (
        1, STRUCTURE, "the seed rolled a whole plane at the compare boundary"),
    ("prismaquant/stage_a_chain_seed.py", "checkpoint_plane_distance"): (
        1, WALL, "two checkpoints compared as arms of one run share a bind identity"),
    ("prismaquant/stage_a_chain_split.py", "read_partial"): (
        2, INTEGRITY, "a partial's manifest reproduces its own bytes and seal; its pack "
        "against the size and digest its row names"),
    ("prismaquant/tessera_joint_aura.py", "_read_verified_wire_blob"): (
        1, INTEGRITY, "staged wire bytes against the receipt digest"),
    ("prismaquant/tessera_joint_aura.py", "_read_wire_bytes"): (
        1, INTEGRITY, "staged wire bytes against the receipt digest"),
    ("tools/dispatch_joint_quanta.py", "_executable_prepared_input"): (
        3, AMBIGUOUS, "the manifest wire against its digest (integrity); the prepared "
        "payload and Stage A slice bindings (ambiguous, #1147 report)"),
    ("tools/dispatch_joint_quanta.py", "quantum_argv"): (
        1, INTEGRITY, "slice bytes against the digest"),
    ("tools/dispatch_joint_quanta.py", "check_stage_a_proofs"): (
        1, AMBIGUOUS, "a record's Stage A slice against the proof for its layer"),
    ("tools/dispatch_joint_quanta.py", "_read_bound_slice"): (
        1, INTEGRITY, "slice bytes against the digest"),
    ("tools/dispatch_tessera_campaign.py", "_bound_json"): (
        1, INTEGRITY, "a plan input's bytes against its bound digest"),
    ("tools/dispatch_tessera_campaign.py", "_bound_digest"): (
        1, INTEGRITY, "a file's bytes against its bound digest"),
    ("tools/dispatch_tessera_campaign.py", "_bound_pickle"): (
        1, INTEGRITY, "a pickle's bytes against its bound digest"),
    ("tools/dispatch_tessera_campaign.py", "verify_joint_campaign_scope"): (
        1, STRUCTURE, "the campaign identity has the expected schema"),
    ("tools/dispatch_tessera_campaign.py", "_calibration_cache_binding"): (
        1, AMBIGUOUS, "the capture binds this census (calibration data)"),
    ("tools/dispatch_tessera_campaign.py", "_merge_export_hessian_references.accepted_rows"): (
        1, INTEGRITY, "reference commitments bind the priced row they merge"),
    ("tools/dispatch_tessera_campaign.py", "merge_checkpoint"): (
        2, AMBIGUOUS, "rows merged into one checkpoint agree on identity and unit inputs"),
    ("tools/regenerate_joint_quanta.py", "_load_json"): (
        1, INTEGRITY, "bytes against the pinned digest"),
    ("tools/regenerate_joint_quanta.py", "_check_authorized_diff"): (
        1, INTEGRITY, "a regenerated record reproduces its own seal"),
    ("tools/regenerate_joint_quanta.py", "_prior_slice_manifest"): (
        1, INTEGRITY, "manifest bytes against the digest the prior record stored"),
    ("tools/regenerate_joint_quanta.py", "_check_authorized_metadata_diff"): (
        2, INTEGRITY, "the new manifest digest hashes the new slice; the record reproduces "
        "its own seal"),
    ("tools/regenerate_joint_quanta.py", "_load_production_cache"): (
        1, INTEGRITY, "production cache bytes against the pinned digest"),
    ("tools/regenerate_joint_quanta.py", "main"): (
        1, STRUCTURE, "a path and its digest are supplied together"),
}

_NAMES = re.compile(r"sha256|identity|digest", re.IGNORECASE)
_LITERALS = (ast.Constant, ast.Set, ast.Dict, ast.List, ast.Tuple)


def _texts(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            yield child.id
        elif isinstance(child, ast.Attribute):
            yield child.attr
        elif isinstance(child, ast.Constant) and isinstance(child.value, str) \
                and len(child.value) < 80:
            yield child.value


def _is_identity_compare(test) -> bool:
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if (isinstance(op, (ast.Eq, ast.NotEq))
                    and not isinstance(left, _LITERALS)
                    and not isinstance(right, _LITERALS)
                    and any(_NAMES.search(text) for side in (left, right)
                            for text in _texts(side))):
                return True
            left = right
    return False


class _Sites(ast.NodeVisitor):
    def __init__(self):
        self.stack: list[str] = []
        self.found: list[tuple[str, int]] = []

    def _scope(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

    def visit_If(self, node):
        if _is_identity_compare(node.test) and any(
                isinstance(statement, ast.Raise) for statement in node.body):
            self.found.append((".".join(self.stack) or "<module>", node.lineno))
        self.generic_visit(node)


def seal_sites(source: str) -> list[tuple[str, int]]:
    """(enclosing function, line) of each identity if-then-raise in ``source``."""
    visitor = _Sites()
    visitor.visit(ast.parse(source))
    return visitor.found


def violations(sources: dict[str, str]) -> list[str]:
    """Every difference between the scanned sites and the allowlist."""
    counts: Counter = Counter()
    lines: dict = {}
    for path, source in sources.items():
        for scope, line in seal_sites(source):
            counts[(path, scope)] += 1
            lines.setdefault((path, scope), []).append(line)
    problems = []
    for key in sorted(set(counts) | set(ALLOWLIST)):
        allowed = ALLOWLIST.get(key, (0, None, None))[0]
        if counts[key] > allowed:
            problems.append(
                f"{key[0]}:{lines[key]} {key[1]}: {counts[key]} identity if-raise site(s), "
                f"{allowed} allowed. A run seal goes through dev_mode.seal_check; "
                "a byte-integrity check needs an ALLOWLIST entry with its reason")
        elif counts[key] < allowed:
            problems.append(
                f"{key[0]} {key[1]}: the allowlist names {allowed} site(s), {counts[key]} "
                "remain; lower the entry")
    return problems


def _campaign_sources() -> dict[str, str]:
    return {path: (ROOT / path).read_text() for path in MODULES}


def test_every_campaign_seal_goes_through_seal_check():
    assert violations(_campaign_sources()) == []


def test_every_allowlist_entry_gives_its_kind_and_reason():
    for key, (count, kind, reason) in ALLOWLIST.items():
        assert key[0] in MODULES, key
        assert count >= 1, key
        assert kind in {INTEGRITY, STRUCTURE, WALL, AMBIGUOUS}, key
        assert isinstance(reason, str) and len(reason) > 20, key


def test_the_lint_does_not_flag_seal_check():
    sources = _campaign_sources()
    assert not [scope for scope, _ in seal_sites(sources["prismaquant/dev_mode.py"])
                if scope == "seal_check"]


NEW_SEAL = '''

def load_run(record, running):
    if record["plan_sha256"] != running.plan_sha256:
        raise RuntimeError("the run binds another plan")
'''

INSIDE = "        _require_hex(quantum_sha256, \"--quantum-sha256\")\n"
NEW_SEAL_IN_AN_ALLOWED_FUNCTION = INSIDE + (
    "        if record_prepared_identity != running_identity:\n"
    "            raise RuntimeError(\"the record binds another prepared payload\")\n")


@pytest.mark.parametrize("allowed", [False, True], ids=["new-function", "allowlisted-function"])
def test_a_new_seal_fails_the_lint(allowed):
    sources = _campaign_sources()
    path = "prismaquant/joint_cost_quantum.py"
    if allowed:
        assert INSIDE in sources[path]
        sources[path] = sources[path].replace(INSIDE, NEW_SEAL_IN_AN_ALLOWED_FUNCTION, 1)
    else:
        sources[path] += NEW_SEAL
    problems = violations(sources)
    assert len(problems) == 1
    assert "seal_check" in problems[0]
    assert ("verify_quantum_identity: 5" if allowed else "load_run: 1") in problems[0]


def test_converting_a_site_requires_lowering_its_entry():
    sources = _campaign_sources()
    path = "tools/regenerate_joint_quanta.py"
    source = sources[path]
    line = "    if bool(args.catalog_extension) != bool(args.catalog_extension_sha256):"
    assert line in source
    sources[path] = source.replace(line, "    if False:", 1)
    problems = violations(sources)
    assert problems == [f"{path} main: the allowlist names 1 site(s), 0 remain; "
                        "lower the entry"]
