"""Call the Tessera public serving witness verifier.

Tessera owns the producer contract. It publishes one versioned witness
(schema ``tessera.endpoint_runtime_witness.v1``) and one standalone CPU
verifier CLI (``tools/verify_endpoint_witness.py``). This module consumes
that public CLI as a subprocess. It never imports Tessera code, starts no
rank, and seals no identity. The CLI verdict is the release authority. The
standard-library recheck in ``served_task_backend`` stays a fast local
pre-check only. It never overrules a public CLI refusal.

Pinned producer contract (Tessera master ``b2875875a4``; CLI logic last
changed in ``f34305799d``; Tessera #1056 and #1128 are closed):

- witness schema ``tessera.endpoint_runtime_witness.v1``
- verdict schema ``tessera.endpoint_witness_verdict.v1``
- offline proof scope ``recorded_runtime_byte_binding``
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

#: Tessera master with the complete approved CLI.
TESSERA_VERIFIER_COMMIT = "b2875875a40d2b594e99f43b63cfe503d73ec2e7"

#: Change that completed the CLI contract.
TESSERA_VERIFIER_CHANGE = "f34305799d3ebf72cbb99f34acf0b716df6d23ef"

#: CLI path inside a Tessera checkout at the pinned commit.
TESSERA_VERIFIER_RELATIVE = "tools/verify_endpoint_witness.py"

#: Machine verdict of a passed offline check.
PUBLIC_VERDICT_VALID = "valid"

#: Proof scope of a passed offline check.
PUBLIC_PROOF_SCOPE_RECORDED = "recorded_runtime_byte_binding"

#: Verdict schema the pinned CLI prints.
PUBLIC_VERDICT_SCHEMA = "tessera.endpoint_witness_verdict.v1"


def verifier_cli(root: str | Path) -> Path:
    """Return the pinned CLI inside ``root``. Refuse when it is absent."""
    cli = Path(root) / TESSERA_VERIFIER_RELATIVE
    if not cli.is_file():
        raise ValueError(f"public verifier CLI is absent: {cli}")
    return cli


def split_expected(expected: dict[str, Any]) -> tuple[dict[str, Any],
                                                      dict[str, Any]]:
    """Split a combined expectation into CLI artifact and tokenizer files."""
    try:
        artifacts = expected["artifacts"]
        tokenizer = expected["tokenizer"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"expected facts lack CLI inputs: {exc}") from exc
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("expected artifacts lack CLI inputs")
    if not isinstance(tokenizer, dict) or not tokenizer:
        raise ValueError("expected tokenizer lacks CLI inputs")
    return artifacts, tokenizer


def run_public_verifier(*, cli: str | Path, witness: str | Path,
                        expected: dict[str, Any], served_dir: str | Path,
                        tokenizer_dir: str | Path | None = None,
                        python: str = "python3",
                        timeout: int = 300) -> dict[str, Any]:
    """Run the pinned CLI offline. Return its valid verdict or refuse.

    The CLI proves the recorded launch and the served byte content. It
    never contacts the endpoint in offline mode.
    """
    cli_path = Path(cli)
    witness_path = Path(witness)
    served_path = Path(served_dir)
    if not cli_path.is_file():
        raise ValueError(f"public verifier CLI is absent: {cli_path}")
    if not witness_path.is_file():
        raise ValueError(f"served witness is absent: {witness_path}")
    if not served_path.is_dir():
        raise ValueError(f"served artifact dir is absent: {served_path}")
    ranks = expected.get("ranks") if isinstance(expected, dict) else None
    if (not isinstance(ranks, list) or not ranks
            or any(type(rank) is not int for rank in ranks)):
        raise ValueError("expected ranks do not cover the complete world")
    for key in ("endpoint", "served_alias", "attempt_id"):
        fact = expected.get(key) if isinstance(expected, dict) else None
        if not isinstance(fact, str) or not fact:
            raise ValueError(f"expected {key} is not explicit text")
    artifacts, tokenizer = split_expected(expected)
    token_path = Path(tokenizer_dir) if tokenizer_dir is not None else None
    if token_path is not None and not token_path.is_dir():
        raise ValueError(f"server tokenizer dir is absent: {token_path}")
    argv = [python, str(cli_path), str(witness_path), "--served-dir",
            str(served_path), "--offline", "--expect-endpoint",
            expected["endpoint"], "--expect-alias",
            expected["served_alias"], "--expect-attempt",
            expected["attempt_id"], "--expect-ranks",
            ",".join(str(rank) for rank in ranks)]
    if token_path is not None:
        argv += ["--tokenizer-dir", str(token_path)]
    import tempfile
    with tempfile.TemporaryDirectory(prefix="pq-public-verifier-") as tmp:
        artifacts_path = Path(tmp) / "expect_artifacts.json"
        tokenizer_path = Path(tmp) / "expect_tokenizer.json"
        artifacts_path.write_text(json.dumps(artifacts, sort_keys=True),
                                  encoding="utf-8")
        tokenizer_path.write_text(json.dumps(tokenizer, sort_keys=True),
                                  encoding="utf-8")
        argv += ["--expect-artifacts", str(artifacts_path),
                 "--expect-tokenizer", str(tokenizer_path)]
        try:
            completed = subprocess.run(argv, capture_output=True, text=True,
                                       timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(
                f"public verifier did not run: {exc}") from exc
    try:
        verdict = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError("public verifier printed no verdict: "
                         + (completed.stderr.strip() or str(exc))) from exc
    if not isinstance(verdict, dict):
        raise ValueError("public verifier printed no verdict object")
    if verdict.get("schema") != PUBLIC_VERDICT_SCHEMA:
        raise ValueError("public verifier verdict has an unknown schema")
    if verdict.get("mode") != "offline":
        raise ValueError("public verifier did not run the offline check")
    if verdict.get("verdict") != PUBLIC_VERDICT_VALID:
        raise ValueError("public verifier refused: "
                         + str(verdict.get("reason")))
    if verdict.get("proof_scope") != PUBLIC_PROOF_SCOPE_RECORDED:
        raise ValueError("public verifier overstated its proof scope")
    if verdict.get("current_endpoint_verified") is not False:
        raise ValueError("public verifier claimed a live endpoint proof")
    return verdict
