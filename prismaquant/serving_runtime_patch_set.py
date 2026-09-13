"""Reader for a recorded serving-runtime patch set.

A patch set is what you get when the pinned serving image does not serve a
model and a derived image does. It is a *second* runtime identity, and the
whole reason this module exists is that an image tag is not a pin: a tag on one
box is unreadable by a gate, a reviewer or a future session, and the edits it
carries have no home in the repository.

Two rules the type enforces, both of them principle consequences:

*   **A patch set is RECORDED, never attested.** Principle 14 says a claim
    about what a serving runtime *does* is derived from a machine-readable
    table that runtime publishes, or refused. A patch set is the opposite
    shape: a producer-side statement about a runtime that was *modified* here.
    So ``attested`` must be ``false``, and nothing in this module hands a
    caller a route, an activation contract, or an eligibility answer. It
    records a build and a measurement, and it makes the scope of that
    measurement unavoidable.
*   **Qualification carries its own scope.** A recorded capability claim
    inherits the scope of the artifact it was measured on. A patch set
    measured on a 4-layer stub qualifies a 4-layer stub;
    :meth:`ServingRuntimePatchSet.require_qualified_for` refuses anything
    wider rather than letting "READY" quietly become "serves".

Digest grammar: ``derived_image``, when it is not ``null``, must be an exact
``repository@sha256:<64 lowercase hex>`` reference under the same regex the
lane-eligibility reader uses. Registry ports are not part of that repository
charset, so a private LAN registry has to publish on port 80 for its images to
be pinnable here -- measured 2026-09-13 against Tessera's
``runtime_image.require_runtime_image``, and the reason this module reuses one
grammar instead of writing a looser second one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from prismaquant.lane_eligibility import _DIGEST_IMAGE

__all__ = [
    "ServingRuntimePatchSetError",
    "ServingRuntimePatchSet",
    "SERVING_RUNTIME_PATCH_SETS_DIR",
    "load_serving_runtime_patch_set",
]

SCHEMA = "prismaquant.serving_runtime_patch_set.v1"

SERVING_RUNTIME_PATCH_SETS_DIR = Path(__file__).resolve().parent / "serving_runtime_patches"

#: A local docker image id is a config digest, not a registry manifest digest.
#: It is recorded so the built image is identifiable on the box that holds it,
#: and it is kept in a DIFFERENT field from ``derived_image`` because it cannot
#: be pulled: conflating the two is how "we have a digest" becomes "anyone can
#: reproduce this".
_LOCAL_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


class ServingRuntimePatchSetError(RuntimeError):
    """A patch-set manifest is missing, malformed, or claims too much."""


@dataclass(frozen=True)
class ServingRuntimePatchSet:
    """One recorded patch set, as read off its ``MANIFEST.json``."""

    name: str
    directory: Path
    base_image: str
    derived_image: "str | None"
    derived_image_local_id: "str | None"
    derived_image_local_tag: "str | None"
    qualification: Mapping[str, Any]
    patches: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any]

    @property
    def qualified_model_layers(self) -> int:
        return int(self.qualification["model_layers"])

    @property
    def full_model_layers(self) -> int:
        return int(self.qualification["full_model_layers"])

    @property
    def is_fully_qualified(self) -> bool:
        """Has the patch set been measured on the whole body, or a slice of it?"""
        return self.qualified_model_layers >= self.full_model_layers

    def scope_sentence(self) -> str:
        """The one line any consumer must print beside a claim from this set."""
        q = self.qualification
        return (
            f"{self.name}: measured on {q['model']} "
            f"({self.qualified_model_layers} of {self.full_model_layers} layers, "
            f"{q['dtype']}, TP{q['tensor_parallel']}, modes "
            f"{'+'.join(q['modes_ready'])}) on {q['measured_on']} "
            f"{q['measured_at']}. RECORDED, not attested; "
            f"{len(q['not_qualified'])} stated limits."
        )

    def require_qualified_for(self, *, model_layers: int) -> None:
        """Refuse a body the recorded measurement does not cover.

        Four layers READY is not a served 45-layer model, and the refusal is
        the deliverable: a caller that wants to quote this patch set for the
        full body has to change the *measurement*, not the sentence.
        """
        if model_layers > self.qualified_model_layers:
            raise ServingRuntimePatchSetError(
                f"patch set {self.name!r} is qualified on "
                f"{self.qualified_model_layers} layers and was asked for "
                f"{model_layers}. A patch set inherits the scope of the "
                f"artifact it was measured on. Unresolved limits: "
                + "; ".join(self.qualification["not_qualified"]))

    def serve_image_reference(self) -> str:
        """The pullable derived image, or a refusal naming what is missing."""
        if self.derived_image is None:
            raise ServingRuntimePatchSetError(
                f"patch set {self.name!r} has no registry-pinned derived image. "
                f"It exists as local id {self.derived_image_local_id} "
                f"(tag {self.derived_image_local_tag!r}) on "
                f"{self.raw.get('derived_image_built_on')!r}, which is a docker "
                "config digest and cannot be pulled. Push it to a registry that "
                "serves on port 80 -- the digest grammar's repository charset "
                "has no ':' -- and record the manifest digest in derived_image.")
        return self.derived_image


def _require(manifest: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in manifest:
        raise ServingRuntimePatchSetError(f"{where} is missing required key {key!r}")
    return manifest[key]


def load_serving_runtime_patch_set(
    name: str,
    *,
    root: "Path | None" = None,
) -> ServingRuntimePatchSet:
    """Load and validate one patch set by directory name."""
    base = SERVING_RUNTIME_PATCH_SETS_DIR if root is None else Path(root)
    directory = base / name
    path = directory / "MANIFEST.json"
    if not path.is_file():
        raise ServingRuntimePatchSetError(f"no patch-set manifest at {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ServingRuntimePatchSetError(f"{path} is not valid JSON: {exc}") from exc
    where = str(path)
    schema = _require(manifest, "schema", where)
    if schema != SCHEMA:
        raise ServingRuntimePatchSetError(
            f"{where}.schema is {schema!r}, expected {SCHEMA!r}")
    if _require(manifest, "attested", where) is not False:
        raise ServingRuntimePatchSetError(
            f"{where}.attested must be false. A patch set is a producer-side "
            "statement about a runtime this repository MODIFIED; nothing here "
            "is derived from a table the runtime publishes, so no gate may "
            "read it as a route claim (principle 14)")

    base_image = _require(manifest, "base_image", where)
    if not isinstance(base_image, str) or _DIGEST_IMAGE.fullmatch(base_image) is None:
        raise ServingRuntimePatchSetError(
            f"{where}.base_image must be an exact repository@sha256:<64 hex> "
            f"reference, got {base_image!r}. A tag is not a pin: it is what "
            "this whole file exists to replace")

    derived = _require(manifest, "derived_image", where)
    if derived is not None and (
            not isinstance(derived, str)
            or _DIGEST_IMAGE.fullmatch(derived) is None):
        raise ServingRuntimePatchSetError(
            f"{where}.derived_image must be a digest-pinned image or null, got "
            f"{derived!r}. Registry ports are not in the repository charset, so "
            "a LAN registry must publish on port 80 to be pinnable")

    local_id = manifest.get("derived_image_local_id")
    if local_id is not None and (
            not isinstance(local_id, str)
            or _LOCAL_IMAGE_ID.fullmatch(local_id) is None):
        raise ServingRuntimePatchSetError(
            f"{where}.derived_image_local_id must be sha256:<64 hex> or null, "
            f"got {local_id!r}")
    if derived is None and local_id is None:
        raise ServingRuntimePatchSetError(
            f"{where} names neither a derived image nor a local image id. A "
            "patch set that cannot identify what it built records nothing")

    qualification = _require(manifest, "qualification", where)
    if not isinstance(qualification, Mapping):
        raise ServingRuntimePatchSetError(f"{where}.qualification must be an object")
    for key in ("status", "model", "model_layers", "full_model_layers", "dtype",
                "tensor_parallel", "modes_ready", "checks_passed",
                "not_qualified", "measured_on", "measured_at"):
        _require(qualification, key, f"{where}.qualification")
    if not qualification["not_qualified"]:
        raise ServingRuntimePatchSetError(
            f"{where}.qualification.not_qualified is empty. A patch set with no "
            "stated limits is a patch set whose limits were not looked for; "
            "state them or do not record the claim")
    if int(qualification["model_layers"]) > int(qualification["full_model_layers"]):
        raise ServingRuntimePatchSetError(
            f"{where}.qualification measures more layers than the model has")

    patches = _require(manifest, "patches", where)
    if not isinstance(patches, list) or not patches:
        raise ServingRuntimePatchSetError(f"{where}.patches must be a non-empty list")
    for index, patch in enumerate(patches):
        at = f"{where}.patches[{index}]"
        if not isinstance(patch, Mapping):
            raise ServingRuntimePatchSetError(f"{at} must be an object")
        script = _require(patch, "script", at)
        if not (directory / str(script)).is_file():
            raise ServingRuntimePatchSetError(
                f"{at}.script {script!r} is not in {directory}. The edits are "
                "the artifact; a manifest that names a file it does not carry "
                "is the image tag problem again")
        digest = _require(patch, "sha256", at)
        if not isinstance(digest, str) or _LOCAL_IMAGE_ID.fullmatch(
                f"sha256:{digest}") is None:
            raise ServingRuntimePatchSetError(f"{at}.sha256 must be 64 lowercase hex")
        edits = _require(patch, "edits", at)
        if not isinstance(edits, list) or not edits:
            raise ServingRuntimePatchSetError(f"{at}.edits must be a non-empty list")

    return ServingRuntimePatchSet(
        name=str(_require(manifest, "name", where)),
        directory=directory,
        base_image=base_image,
        derived_image=derived,
        derived_image_local_id=local_id,
        derived_image_local_tag=manifest.get("derived_image_local_tag"),
        qualification=qualification,
        patches=tuple(patches),
        raw=manifest,
    )
