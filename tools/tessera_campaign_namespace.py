"""Package-free ownership contract, not a scheduler or an input-content qualifier."""
from __future__ import annotations

import copy
import json
import os
import re
import stat
from contextlib import suppress
from pathlib import Path, PurePosixPath

from tools.pq_profile_digest import canonical_json_bytes, canonical_json_sha256

SCHEMA = "prismaquant.tessera_campaign_namespace.v1"
RECONCILIATION_SCHEMA = "prismaquant.tessera_namespace_reconciliation.v1"
OUTPUTS = {"--out": "cost.pkl", "--cache-dir": "cache", "--checkpoint": "cost.anchors.json"}
# Only these cache/temp destinations are supported by this opt-in CPU slice.
WRITABLE_ENV = ("TMPDIR", "TMP", "TEMP", "HF_HOME", "HF_HUB_CACHE",
                "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME",
                "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                "CUDA_CACHE_PATH", "NUMBA_CACHE_DIR", "PYTHONPYCACHEPREFIX",
                "PRISMAQUANT_TMPDIR")


def refuse_path_symlinks(value: str, *, directory: bool = True) -> None:
    """Inspect existing ancestors without creating or resolving the destination."""
    path = Path(value)
    for ancestor in (*reversed(path.parents), path):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot inspect scratch path {ancestor}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"scratch path contains a symlink: {ancestor}")
        if (ancestor != path or directory) and not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"scratch path is not a directory: {ancestor}")


def namespace_absolute_path(value: object) -> Path:
    """Validate path syntax without opening or authenticating input artifacts."""
    if (not isinstance(value, str) or not value.startswith("/") or value.startswith("//")
            or "\x00" in value or str(PurePosixPath(value)) != value
            or ".." in PurePosixPath(value).parts):
        raise RuntimeError("namespace paths must be canonical absolute paths")
    return Path(value)


def namespace_path(value: object, *, directory: bool = True) -> Path:
    path = namespace_absolute_path(value)
    refuse_path_symlinks(str(path), directory=directory)
    return path


def namespace_request_parts(row: dict) -> tuple[int, dict, int]:
    """Accept only the existing single container-wrapper campaign command shape."""
    argv = row.get("argv", [])
    if (not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv)
            or argv[:4] != ["python3", "-m", "tools.tessera_campaign_container", "--spec"]
            or len(argv) < 10 or argv[5] != "--"):
        raise RuntimeError("namespace requires an unambiguous container campaign request")
    try:
        spec = json.loads(argv[4])
    except (ValueError, TypeError) as exc:
        raise RuntimeError("namespace container spec is not valid JSON") from exc
    module_index = 8 if argv[7:8] == ["-u"] else 7
    if (not isinstance(spec, dict)
            or argv[module_index:module_index + 2] != ["-m", "prismaquant.tessera_campaign"]):
        raise RuntimeError("namespace requires a direct campaign command")
    inner_start = module_index + 2
    if not isinstance(row.get("env"), dict):
        raise RuntimeError("namespace requires an explicit environment mapping")
    if row.get("env") != spec.get("env"):
        raise RuntimeError("namespace outer/spec environment differs")
    for flag in OUTPUTS:
        if sum(arg == flag or arg.startswith(flag + "=") for arg in argv[inner_start:]) != 1:
            raise RuntimeError(f"namespace requires one explicit {flag}")
        index = argv.index(flag, inner_start) if flag in argv[inner_start:] else -1
        if index < 0 or index + 1 == len(argv) or argv[index + 1].startswith("--"):
            raise RuntimeError(f"namespace requires a separate value for {flag}")
    if any(name.startswith("PRISMAQUANT_") and ("SCRATCH" in name or "SPILL" in name
               or "CONTAINER_CACHE" in name) for name in row["env"]):
        raise RuntimeError("namespace slice does not admit bounded-local scratch overrides")
    return 4, spec, inner_start


def namespace_unbound_request(row: dict) -> dict:
    result = copy.deepcopy(row)
    index, spec, _ = namespace_request_parts(result)
    spec.pop("namespace_binding", None)
    result["argv"][index] = canonical_json_bytes(spec, where="namespace spec").decode()
    return result


def namespace_retarget(row: dict, destination: str) -> dict:
    result = namespace_unbound_request(row)
    index, spec, inner_start = namespace_request_parts(result)
    for flag, filename in OUTPUTS.items():
        result["argv"][result["argv"].index(flag, inner_start) + 1] = destination + "/" + filename
    for name in WRITABLE_ENV:
        # Explicit cache/temp roots prevent image defaults writing shared legacy paths.
        result["env"][name] = destination + "/environment/" + name
    spec["env"] = result["env"]
    result["argv"][index] = canonical_json_bytes(spec, where="namespace spec").decode()
    return result


def namespace_reference(record: object) -> None:
    if (not isinstance(record, dict) or set(record) != {"path", "sha256"}
            or not isinstance(record["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
        raise RuntimeError("namespace needs an explicit path/SHA256 input reference")
    # Input references are syntax-checked only: no historical artifact reads.
    namespace_absolute_path(record["path"])


def prepare_namespace_requests(*, requests: list[dict], selected: list[str],
                               reconciliation: dict, expected_evidence: dict,
                               readsets: dict, provenance: dict, expected_provenance: dict,
                               reviewed_commit: str, executed_commit: str,
                               checkout: str, root: str) -> list[dict]:
    """Prepare metadata only; all authority expectations come from the caller.

    Complete reconciliation describes the supplied roster, not a historical READY
    label. Input digests and runtime provenance are independently supplied bindings,
    NOT proof of content, compatibility, completion or resource qualification.
    """
    namespace_path(root)
    namespace_path(checkout)
    if root == "/":
        raise RuntimeError("namespace root cannot own the filesystem")
    for commit in (reviewed_commit, executed_commit):
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise RuntimeError("namespace requires full source commits")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError("namespace requires a complete request roster")
    hashes = [canonical_json_sha256(row, where="namespace original request") for row in requests]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("namespace duplicate original request identities")
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or not set(selected) <= set(hashes)):
        raise RuntimeError("namespace selected identities are missing or duplicate")
    roster_sha256 = canonical_json_sha256(requests, where="namespace original roster")
    if (not isinstance(reconciliation, dict)
            or set(reconciliation) != {"schema", "requests_sha256", "evidence", "status"}
            or reconciliation["schema"] != RECONCILIATION_SCHEMA
            or not isinstance(reconciliation["status"], dict)
            or set(reconciliation["status"]) != set(hashes)
            or any(value not in ("completed", "unfinished") for value in reconciliation["status"].values())):
        raise RuntimeError("namespace needs complete explicit reconciliation")
    if reconciliation["requests_sha256"] != roster_sha256:
        raise RuntimeError("namespace reconciliation roster digest differs")
    namespace_reference(expected_evidence)
    if reconciliation["evidence"] != expected_evidence:
        raise RuntimeError("namespace independent reconciliation evidence differs")
    if any(reconciliation["status"][key] != "unfinished" for key in selected):
        raise RuntimeError("namespace may select only reconciled unfinished requests")
    if provenance != expected_provenance:
        raise RuntimeError("namespace independent dependency provenance differs")
    if (set(provenance) != {"tessera_commit", "prismabuild_commit", "container_content_sha256"}
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{" + str(size) + r"}", value) is None
                   for name, size in (("tessera_commit", 40), ("prismabuild_commit", 40),
                                      ("container_content_sha256", 64)) for value in (provenance[name],))):
        raise RuntimeError("namespace requires exact dependency provenance")
    if set(readsets) != set(selected):
        raise RuntimeError("namespace selected readset roster differs")
    result = []
    for key in sorted(selected):
        original = requests[hashes.index(key)]
        _, spec, _ = namespace_request_parts(original)
        if "namespace_binding" in spec:
            raise RuntimeError("namespace cannot retarget an already bound request")
        namespace_reference(readsets[key])
        if original.get("data_manifest") != readsets[key]["path"]:
            raise RuntimeError("namespace independent readset path differs")
        if spec["container"].get("content_sha256") != provenance["container_content_sha256"]:
            raise RuntimeError("namespace container provenance differs")
        normalized = namespace_retarget(original, "{namespace-row}")
        normalized["cwd"] = checkout
        base = {"schema": SCHEMA, "root": root, "reviewed_commit": reviewed_commit,
                "executed_commit": executed_commit, "original_request_sha256": key,
                "roster_sha256": roster_sha256, "selected": sorted(selected), "reconciliation": reconciliation,
                "readset": readsets[key], "provenance": provenance,
                "normalized_request": normalized}
        request_key = canonical_json_sha256(base, where="namespace row binding")
        request = namespace_retarget(normalized, root + "/" + request_key)
        binding = {**base, "request_key": request_key, "request": request,
                   "request_sha256": canonical_json_sha256(request, where="namespace request")}
        index, output_spec, _ = namespace_request_parts(request)
        row = copy.deepcopy(request)
        output_spec["namespace_binding"] = binding
        row["argv"][index] = canonical_json_bytes(output_spec, where="namespace bound spec").decode()
        validate_namespace_request(row)
        result.append(row)
    return result


def namespace_destinations(request: dict, binding: dict) -> list[tuple[Path, bool]]:
    """The same owned destinations drive path and mount validation."""
    directory = Path(binding["root"]) / binding["request_key"]
    return [(directory / filename, flag == "--cache-dir") for flag, filename in OUTPUTS.items()] + [
        (Path(request["env"][name]), True) for name in WRITABLE_ENV]


def namespace_mounts(spec: dict) -> list[tuple[Path, bool]]:
    """One mount view for namespace containment and writable identity coverage."""
    return [(Path(mount["target"]),
             mount["source"] == mount["target"] and not mount.get("readonly", False))
            for mount in spec["container"].get("mounts", [])]


def validate_namespace_request(row: dict, *, executed_commit: str | None = None) -> dict:
    """Reproduce immutable ownership and request bytes, without admitting inputs."""
    _, spec, _ = namespace_request_parts(row)
    binding = spec.get("namespace_binding")
    fields = {"schema", "root", "reviewed_commit", "executed_commit", "original_request_sha256",
              "roster_sha256", "selected", "reconciliation", "readset", "provenance", "normalized_request",
              "request_key", "request", "request_sha256"}
    if not isinstance(binding, dict) or set(binding) != fields or binding["schema"] != SCHEMA:
        raise RuntimeError("namespace binding is missing or malformed")
    namespace_path(binding["root"])
    for field in ("reviewed_commit", "executed_commit"):
        if not isinstance(binding[field], str) or re.fullmatch(r"[0-9a-f]{40}", binding[field]) is None:
            raise RuntimeError("namespace requires full source commits")
    base = {key: value for key, value in binding.items() if key not in ("request_key", "request", "request_sha256")}
    request_key = canonical_json_sha256(base, where="namespace row binding")
    if binding["request_key"] != request_key:
        raise RuntimeError("namespace row binding digest differs")
    if executed_commit is not None and binding["executed_commit"] != executed_commit:
        raise RuntimeError("namespace executed source commit differs")
    request = namespace_unbound_request(row)
    request_sha256 = canonical_json_sha256(request, where="namespace request")
    if binding["request_sha256"] != request_sha256:
        raise RuntimeError("namespace request digest differs")
    if binding["request"] != request:
        raise RuntimeError("namespace published request bytes differ")
    directory = namespace_path(binding["root"] + "/" + request_key)
    expected = namespace_retarget(binding["normalized_request"], str(directory))
    if request != expected:
        raise RuntimeError("namespace destinations or request differ from owned row")
    for path, is_directory in namespace_destinations(request, binding):
        namespace_path(str(path), directory=is_directory)
    # No input or mount may place read-only evidence inside owned writable space.
    for reference in (binding["readset"], binding["reconciliation"]["evidence"]):
        namespace_reference(reference)
        if Path(reference["path"]).is_relative_to(Path(binding["root"])):
            raise RuntimeError("namespace input evidence overlaps owned outputs")
    _, _, inner_start = namespace_request_parts(request)
    argv = request["argv"][inner_start:]
    for flag in ("--model", "--units", "--calibration-census", "--calibration-cache",
                 "--source-identity-cache", "--seed-checkpoint", "--seed-wire-dir"):
        for index, argument in enumerate(argv):
            if argument == flag and index + 1 < len(argv):
                value = argv[index + 1]
            elif argument.startswith(flag + "="):
                value = argument.split("=", 1)[1]
            else:
                continue
            input_path = namespace_absolute_path(value)
            if input_path.is_relative_to(Path(binding["root"])):
                raise RuntimeError("namespace input overlaps owned outputs")
    mounts = namespace_mounts(spec)
    for target, writable_identity in mounts:
        if target.is_relative_to(Path(binding["root"])):
            raise RuntimeError("namespace output is hidden by a declared mount")
        if directory.is_relative_to(target) and not writable_identity:
            raise RuntimeError("namespace needs writable identity-mapped output mounts")
    for destination, _ in namespace_destinations(request, binding):
        if not any(destination.is_relative_to(target) and writable_identity
                   for target, writable_identity in mounts):
            raise RuntimeError("namespace destination lacks a writable identity-mapped mount")
    return binding


def namespace_publication_record(rows: list[dict]) -> dict:
    bindings = [validate_namespace_request(row) for row in rows]
    if not bindings or len({binding["root"] for binding in bindings}) != 1:
        raise RuntimeError("namespace publication needs one nonempty owner root")
    if len({binding["request_key"] for binding in bindings}) != len(bindings):
        raise RuntimeError("namespace publication has duplicate request identities")
    if any(binding["selected"] != bindings[0]["selected"] for binding in bindings):
        raise RuntimeError("namespace publication selected roster differs")
    if sorted(binding["original_request_sha256"] for binding in bindings) != bindings[0]["selected"]:
        raise RuntimeError("namespace publication is missing selected requests")
    return {"schema": SCHEMA, "root": bindings[0]["root"],
            "bindings": {binding["request_key"]: canonical_json_sha256(binding, where="namespace ownership")
                         for binding in bindings}}


def namespace_adapter_request(spec: dict, command: list[str], environ) -> dict:
    binding = spec.get("namespace_binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("request"), dict):
        raise RuntimeError("namespace binding is missing or malformed")
    row = copy.deepcopy(binding["request"])
    namespace_request_parts(row)
    row["env"] = spec.get("env")
    row["argv"][4] = canonical_json_bytes(spec, where="namespace adapter spec").decode()
    row["argv"][6:] = command
    validate_namespace_request(row)
    for name, expected in row["env"].items():
        if environ.get(name) != expected:
            raise RuntimeError(f"namespace outer/spec environment differs for {name}")
    require_namespace_publication(row)
    return row


def require_namespace_publication(row: dict) -> dict:
    binding = validate_namespace_request(row)
    root = namespace_path(binding["root"])
    record_path = namespace_path(str(root / "namespace.json"), directory=False)
    try:
        record = json.loads(record_path.read_bytes())
        row_root = root / binding["request_key"]
        for filename, expected in (("binding.json", binding), ("request.json", row)):
            path = namespace_path(str(row_root / filename), directory=False)
            if path.read_bytes() != canonical_json_bytes(expected, where="namespace publication"):
                raise RuntimeError("namespace published ownership/request bytes differ")
    except (OSError, ValueError) as exc:
        raise RuntimeError("namespace published ownership is unavailable") from exc
    binding_sha256 = canonical_json_sha256(binding, where="namespace ownership")
    if (not isinstance(record, dict) or set(record) != {"schema", "root", "bindings"}
            or record["schema"] != SCHEMA or record["root"] != str(root)
            or not isinstance(record["bindings"], dict)):
        raise RuntimeError("namespace owner record is malformed")
    if record["bindings"].get(binding["request_key"]) != binding_sha256:
        raise RuntimeError("namespace published binding digest differs")
    return binding


def establish_namespace_temporaries(row: dict) -> None:
    """Create writable temps only after ownership; never follow raced symlinks.

    Directory descriptors anchor each mkdir/open to the admitted owner tree.
    Concurrent same-binding callers may reuse directories, never replace them.
    """
    binding = require_namespace_publication(row)
    root = namespace_absolute_path(binding["root"]) / binding["request_key"]
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)

    def open_directory(name: str, parent_fd: int, *, create: bool = False) -> int:
        if create:
            with suppress(FileExistsError):
                os.mkdir(name, dir_fd=parent_fd)
        return os.open(name, flags, dir_fd=parent_fd)

    try:
        steps = [(component, False) for component in root.parts[1:]] + [("environment", True)]
        for component, create in steps:
            next_fd = open_directory(component, fd, create=create)
            os.close(fd)
            fd = next_fd
        for name in ("TMPDIR", "TMP", "TEMP"):
            temporary_fd = open_directory(name, fd, create=True)
            try:
                if not os.access(".", os.W_OK, dir_fd=temporary_fd, effective_ids=True):
                    raise RuntimeError("namespace owned temporary directory is not writable")
            finally:
                os.close(temporary_fd)
    except OSError as exc:
        raise RuntimeError("namespace cannot establish owned temporary directories") from exc
    finally:
        os.close(fd)
