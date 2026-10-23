# Completed architecture-refactor tooling

The one-shot source movers and proof drivers for the October 14–16 refactor
are retained here with their original bytes. The codemod-only Unicode and
idempotence tests move with their mutator. These files describe a completed
migration, not supported commands against today's package layout.

Reproduction uses the recorded source commit and admitted PrismaBuild action
from `docs/reports/prismaquant-refactor-verification-2026-10-14.md`, rather than
running an old mutator on the current tree. The original complete tool set is
available at commit `5af6092384f4567472c5165b15e2e716cb87f39a`.

Three reusable checks retain live owners in `tools/`:

- `freeze_refactor_checkout.py` freezes edited sources before PrismaBuild
  submission; `tests/test_refactor_source_capture.py` verifies its refusal of
  conflicted source. The Claude-fix qualification uses this mechanism.
- `verify_campaign_operation_globals.py` checks the actual loaded campaign
  operation, and remains a dependency of the campaign execution tests.
- `verify_tessera_calibration_forward.py` checks the live calibration boundary
  and remains a dependency of the campaign scope and partition tests.

Current pin provisioning belongs to `tools/provision_tessera_pin.py`, not the
archived migration-specific environment helper.

The native-render comparison driver moved here after the codec review fixes.
Its archived bytes are from `dfad64a185`; that version uses the explicit lever
and public codec interfaces. The original pre-review driver remains available
at `5af6092384`. Permanent native-render tests now cover the codec boundary,
recorded fused globals and exported-cache parity directly.
