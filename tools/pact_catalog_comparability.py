"""Identify the retained PR #2277 artifact; never execute, restamp or promote it.

Adapted from the retained historical pact_catalog_comparability.py. This portable
identity lane preserves its inventory/three-owner/accepted-owner closure and
create-only report, but deliberately does not run its pickle/producer-based
numerical lane. See docs/ARCHITECTURE.md for the qualification boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ALLOWED_PACKAGE_CHANGE = frozenset({
    "joint_catalog_extension.py", "joint_served_activation.py", "joint_stageb_resources.py",
})
PACKET_SCHEMAS = {
    "source_closure": "pact.catalog_source_comparability.v2",
    "package_attestation": "pact.test_only_head_package_attestation.v1",
    "source_default_control_readiness": "pact.companion_source_default_control_readiness.v1",
    "exact_review_handoff": "pact.exact_review_handoff.v1",
}
CLOSURE_STATEMENT = (
    "Every durable original PrismaQuant package input outside the three explicitly "
    "authorized metadata/admission owners remains byte-identical, including encoder "
    "invocation, loss/probe, calibration, model/profile and quantizer owners. The "
    "accepted b522 catalog/proof owner is itself unchanged. Changed resource/policy "
    "metadata and exact new head still require independent review; file hashes alone "
    "are not numerical or full-price qualification."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def binding(path):
    return {"path": str(path), **identity(path)}


def content_identity(value):
    return {key: value[key] for key in ("sha256", "bytes")}


def relative_name(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and ".." not in p.parts
            and p.as_posix() == name and ".git" not in p.parts
            and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"},
            f"Invalid durable relative path: {name}")
    return p


def inventory(root):
    """Read all durable bytes, including untracked files and non-Python data."""
    require(root.is_dir(), f"Source root is not a directory: {root}")
    result = {}

    def walk_error(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__"})
        for name in dirs + sorted(files):
            p = Path(directory) / name
            if name == ".git" or p.suffix in {".pyc", ".pyo"}:
                continue
            require(not p.is_symlink(), f"Symlink is not a durable regular source file: {p}")
            if p.is_dir():
                continue
            require(p.is_file(), f"Not a regular source file: {p}")
            rel = p.relative_to(root).as_posix()
            relative_name(rel)
            result[rel] = identity(p)
    return dict(sorted(result.items()))


def package_hash(root, files):
    """Retained ProductionWeightCache framing; no producer import is needed."""
    h = hashlib.sha256()
    for name in sorted(files):
        path = root / name
        encoded = name.encode("utf-8")
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
        h.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                h.update(block)
    return h.hexdigest()


def source_identity(root, declared, observed_files):
    observed_commit = None
    # A source export inside some other checkout must not inherit its parent's HEAD.
    if os.path.lexists(root / ".git"):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_OPTIONAL_LOCKS="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        observed_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True, capture_output=True, text=True, env=env,
        ).stdout.strip()
        require(observed_commit == declared["commit"],
                f"Source commit mismatch at {root}: {observed_commit} != {declared['commit']}")
    package_files = {k.removeprefix("prismaquant/"): v for k, v in observed_files.items()
                     if k.startswith("prismaquant/")}
    package_sha256 = package_hash(root / "prismaquant", package_files)
    require(package_sha256 == declared["package_sha256"], f"Package hash mismatch at {root}")
    return {"root": str(root), "commit": declared["commit"],
            "observed_git_head": observed_commit,
            "identification": "git_head_and_complete_inventory" if observed_commit else "complete_inventory_export",
            "package_sha256": package_sha256}, package_files


def compare_inventory(actual, expected, context):
    changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
    require(not changed, f"{context} file inventory mismatch: {', '.join(changed)}")


def evidence_file(root, spec, context):
    name = relative_name(spec["path"])
    path = root.joinpath(*name.parts)
    require(path.resolve().is_relative_to(root.resolve()), f"{context}: evidence escapes root")
    require(path.is_file(), f"{context}: missing evidence {path}")
    require(identity(path) == content_identity(spec), f"{context}: evidence digest/bytes mismatch: {path}")
    return path


def verify_diff(path, spec, changed, roots):
    require(spec["paths"] == changed, "Source diff paths do not reconcile complete source inventories")
    text = path.read_bytes().decode("utf-8")
    sections = re.split(r"(?m)^diff --git ", text)
    require(sections[0] == "" and len(sections) > 1, "Invalid source diff text")
    paths = []
    for section in sections[1:]:
        header, _, body = section.partition("\n")
        match = re.fullmatch(r"a/(.+) b/(.+)", header)
        require(match is not None and match[1] == match[2], "Source diff must use no-renames paths")
        name = match[1]
        relative_name(name)
        paths.append(name)
        index = re.search(r"(?m)^index ([0-9a-f]{40})\.\.([0-9a-f]{40})(?: [0-7]{6})?$", body)
        require(index is not None, f"Source diff requires full blob identities: {name}")
        for side, blob in zip(("original", "bridge"), index.groups()):
            p = roots[side] / name
            if not p.exists():
                expected_blob = "0" * 40
            else:
                raw = p.read_bytes()
                expected_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            require(blob == expected_blob, f"Source diff {side} blob mismatch: {name}")
    require(sorted(paths) == changed, "Source diff text paths do not match complete diff inventory")
    return {**binding(path), "paths": changed, "text": text}


def check_fields(packet, expected, context, *, optional=False):
    for key, value in expected.items():
        if optional and key not in packet:
            continue
        require(key in packet and packet[key] == value, f"{context}: contradictory {key} claim")


def verify_packets(manifest, evidence_root, sources, package_files, file_identities, catalog):
    specs = manifest["evidence"]
    require(PACKET_SCHEMAS.keys() <= specs.keys(), "Missing required evidence packet roles")
    packets, bindings = {}, {}
    final_commit = sources["bridge"]["commit"]
    for role, spec in specs.items():
        path = evidence_file(evidence_root, spec, role)
        packet = json.loads(path.read_bytes())
        if role in PACKET_SCHEMAS:
            require(packet["schema"] == PACKET_SCHEMAS[role], f"{role}: wrong packet schema")
        if "schema" in spec:
            require(packet["schema"] == spec["schema"], f"{role}: wrong declared schema")
        producer = spec.get("producer_commit", final_commit)
        # Companion packets can describe an earlier head with the identical
        # package; final attestation/handoff must describe the final source head.
        if role in {"package_attestation", "exact_review_handoff", "qualification_packet"}:
            require(producer == final_commit, f"{role}: producer head differs from final head")
        head_key = "head_sha" if role == "source_default_control_readiness" else "head"
        if role != "source_closure":
            check_fields(packet, {head_key: producer}, role)
        packets[role] = packet
        bindings[role] = {**binding(path), "schema": packet["schema"], "producer_commit": producer,
                          "verification": "bytes_and_cross_bindings"}

    closure = packets["source_closure"]
    for side in ("original", "bridge"):
        declared = closure[side + "_source"]
        check_fields(declared, {"package_sha256": sources[side]["package_sha256"]}, "closure package")
        producer = (sources[side]["commit"] if side == "original" else
                    bindings["source_closure"]["producer_commit"])
        check_fields(declared, {"commit": producer}, "closure commit", optional=True)
    check_fields(closure, {"head": bindings["source_closure"]["producer_commit"]},
                 "closure head", optional=True)
    check_fields(closure, {
        "package_files": len(package_files["original"]),
        "changed_package_paths": sorted(ALLOWED_PACKAGE_CHANGE), "file_identities": file_identities,
        "non_admission_package_files_byte_identical": True,
        "accepted_b522_catalog_owner_byte_identical": True,
        "bridge_same_encoder_control": None, "canary_comparability_proved": False,
        "accepted_as_comparable": False, "new_joint_prices_admitted": 0,
        "source_hash_restamp": False, "retained_rows_modified": False, "review_required": True,
    }, "closure")
    check_fields(closure["accepted_b522_catalog_owner"], content_identity(catalog), "closure catalog")
    check_fields(packets["package_attestation"], {
        "actual_package_sha256": sources["bridge"]["package_sha256"], "package_equal_companion": True,
    }, "attestation package")
    readiness = packets["source_default_control_readiness"]
    require(bindings["source_default_control_readiness"]["producer_commit"] ==
            bindings["source_closure"]["producer_commit"], "Readiness/closure producer head mismatch")
    check_fields(readiness["actual_closure"], {
        "sha256": bindings["source_closure"]["sha256"],
        "package_sha256": sources["bridge"]["package_sha256"],
        "package_files": len(package_files["original"]),
        "changed_package_paths": sorted(ALLOWED_PACKAGE_CHANGE),
        "accepted_b522_catalog_owner_sha256": catalog["sha256"],
    }, "readiness closure")
    check_fields(readiness["actual_closure"], {
        "bytes": bindings["source_closure"]["bytes"],
        "other323_files_byte_identical": True, "accepted_b522_catalog_owner_byte_identical": True,
        "numerical_encoder_loss_probe_calibration_model_quantizer_owners_unchanged": True,
    }, "readiness closure", optional=True)
    if "accepted_b522_source_commit" in manifest:
        check_fields(readiness.get("accepted_b522", {}),
                     {"head": manifest["accepted_b522_source_commit"]}, "accepted catalog head")
    handoff = packets["exact_review_handoff"]
    check_fields(handoff, {"table_joint_pin_default_source_promotion": False}, "handoff")
    dependencies = {}
    link = handoff["qualification_packet"]
    if "qualification_packet" in bindings:
        target = bindings["qualification_packet"]
        check_fields(link, {"sha256": target["sha256"]}, "handoff qualification")
        check_fields(link, {"bytes": target["bytes"]}, "handoff qualification", optional=True)
        dependencies["qualification_packet"] = {**target, "verification": "explicitly_located_matching_evidence"}
    else:
        # The production fixture intentionally omits external CAS/numerical
        # payloads. Pin their retained reference, without claiming observed bytes.
        reference = manifest.get("retained_external_bindings", {}).get("qualification_packet")
        require(reference is not None, "Missing explicitly bound qualification dependency")
        check_fields(link, reference, "handoff qualification")
        dependencies["qualification_packet"] = {**reference, "verification": "retained_reference_only_not_observed"}
    return bindings, dependencies


def verify(args):
    require(not args.output.exists() and not args.output.is_symlink(), f"Output already exists: {args.output}")
    manifest = json.loads(args.manifest.read_bytes())
    require(manifest["schema"] == "pact.catalog_source_manifest.v1", "Wrong manifest schema")
    allowed = manifest["allowed_package_changes"]
    require(len(allowed) == 3 and set(allowed) == ALLOWED_PACKAGE_CHANGE,
            "Manifest allowlist cannot change the fixed three-owner authority")
    roots = {"original": args.original_source.resolve(), "bridge": args.bridge_source.resolve()}
    actual, sources, packages = {}, {}, {}
    shared = manifest["source_inventory"]["shared"]
    for side, root in roots.items():
        unique = manifest["source_inventory"][side]
        require(not shared.keys() & unique.keys(), f"{side}: overlapping source inventory entries")
        expected = {name: content_identity(value) for name, value in {**shared, **unique}.items()}
        for name in expected:
            relative_name(name)
        actual[side] = inventory(root)
        compare_inventory(actual[side], expected, side)
        sources[side], packages[side] = source_identity(root, manifest[side + "_source"], actual[side])
    changed = sorted(k for k in actual["original"].keys() | actual["bridge"].keys()
                     if actual["original"].get(k) != actual["bridge"].get(k))
    file_identities = {k: {side: packages[side].get(k) for side in roots}
                       for k in sorted(packages["original"].keys() | packages["bridge"].keys())}
    package_changed = sorted(k for k, values in file_identities.items() if values["original"] != values["bridge"])
    require(set(package_changed) == ALLOWED_PACKAGE_CHANGE, f"Unauthorized package delta: {package_changed}")
    require(packages["original"].keys() == packages["bridge"].keys(), "Package file roster changed")
    catalog = binding(roots["bridge"] / "prismaquant/joint_catalog_extension.py")
    require(catalog["sha256"] == manifest["accepted_b522_catalog_owner_sha256"],
            "Accepted b522 catalog/proof owner digest mismatch")
    diff_path = evidence_file(args.evidence_root, manifest["source_diff"], "source diff")
    source_diff = verify_diff(diff_path, manifest["source_diff"], changed, roots)
    evidence, dependencies = verify_packets(manifest, args.evidence_root, sources, packages, file_identities, catalog)
    regressions = {}
    for name, spec in manifest.get("historical_regressions", {}).items():
        require(spec["producer_commit"] == sources["bridge"]["commit"], f"Historical regression producer head: {name}")
        path = evidence_file(args.evidence_root, spec, f"historical regression {name}")
        # Snapshots are inert evidence; never load them as Python or count them
        # as tests run by this verifier.
        compare_inventory({name: identity(path)}, {name: actual["bridge"]["tests/" + name]}, "Historical regression")
        regressions[name] = {**binding(path), "producer_commit": spec["producer_commit"]}
    require(manifest.get("historical_regressions_are_fresh_acceptance", False) is False,
            "Historical regressions cannot be fresh acceptance")
    return {
        "schema": "pact.catalog_source_comparability.v3", "identity_verified": True,
        "original_source": sources["original"], "bridge_source": sources["bridge"],
        "manifest": binding(args.manifest), "source_diff": source_diff,
        "source_inventory": actual, "file_identities": file_identities,
        "package_files": len(packages["original"]), "changed_package_paths": package_changed,
        "non_admission_package_files_byte_identical": True,
        "accepted_b522_catalog_owner": catalog, "accepted_b522_catalog_owner_byte_identical": True,
        "evidence": evidence, "dependencies": dependencies,
        "historical_regressions": regressions, "historical_regressions_are_fresh_acceptance": False,
        "numerical_closure_statement": CLOSURE_STATEMENT,
        "numerical_controls": {"status": "not_run", "historical_claims_are_not_fresh_measurements": True},
        "bridge_same_encoder_control": None, "canary_comparability_proved": False,
        "accepted_as_comparable": False, "new_joint_prices_admitted": 0,
        "retained_rows_modified": False, "source_hash_restamp": False, "review_required": True,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--identity-only", action="store_true", required=True)
    for name in ("original-source", "bridge-source", "manifest", "evidence-root", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    try:
        # Refuse outputs inside inputs, even when they would currently be new:
        # identity verification must not change a supplied tree or evidence set.
        output = args.output.resolve()
        require(all(not output.is_relative_to(p.resolve()) for p in
                    (args.original_source, args.bridge_source, args.evidence_root)),
                "Output must be outside source and evidence inputs")
        report = verify(args)
        raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(raw).hexdigest(),
                          "identity_verified": True, "accepted_as_comparable": False}))
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"Identity verification refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
