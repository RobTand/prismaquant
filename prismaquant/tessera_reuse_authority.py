"""PrismaQuant's reuse authority for Tessera's cached-unit and Hessian intake.

Tessera stands alone (tessera#599): it defines the reuse-authority protocol
(``tessera.cached_unit.ReuseAuthority``) and names no PrismaQuant record.
PrismaQuant writes the records a rooted cached-unit bundle
(``tessera.cached_units.v2``) binds, so PrismaQuant ships their reader here:

- the catalog extension (``joint_catalog_extension``, v1, v2 and v3) and the
  T4 candidate overlay (``tools/build_t4_overlay_catalog.py``);
- each encoder-source adoption (``joint_catalog_extension.ADOPTION_SCHEMA``);
- the reseal proof bundle (``tools/reseal_campaign_identity.py``);
- the served activation policy (``joint_served_activation``, v1 and v2);
- the canonical calibration cache a Hessian reference binds
  (``tessera_calibration_cache.SCHEMA`` and ``SOURCE``).

These checks moved here from Tessera ``src/tessera/cached_unit.py`` and
``src/tessera/hessian_capture.py`` without change: the same documents accept
and refuse, with the same reasons, in the same order.  A new PrismaQuant record
version is now a change to this file alone, with no Tessera release.

Supplying it:

- in process, pass ``PRODUCER_AUTHORITY`` as ``CachedUnitBundle(authority=...)``
  and ``CANONICAL_CAPTURE`` as ``canonical_capture=`` to
  ``ReferenceHessians``, ``ReferenceHessianCollection`` and
  ``ActivationSource.from_capture``;
- to Tessera's exporter, pass this file's path as ``--producer-authority``
  (:data:`AUTHORITY_PATH`).

The exporter loads the file by path, in a process where the ``prismaquant``
package may not be importable, so this module imports only the standard
library.  ``tests/test_tessera_reuse_authority.py`` holds each constant equal
to its writer's.
"""
from __future__ import annotations

from pathlib import Path
import re

#: The catalog-extension documents a rooted bundle's reuse authority may bind.
#: v1 bound one completed Stage A receipt; v2 (PQ #993) binds the Stage A run
#: header, so an extension can exist from the first sealed band; v3 (PQ #1126)
#: is v2 plus the campaign scope derived when the run header sealed none
#: (tessera#670).  PrismaQuant authenticates each before it publishes the
#: bundle; this reader rechecks the schema and reads no other field.
CATALOG_EXTENSION_SCHEMAS = frozenset({"prismaquant.joint_catalog_extension.v1",
                                       "prismaquant.joint_catalog_extension.v2",
                                       "prismaquant.joint_catalog_extension.v3"})
CANDIDATE_OVERLAY_SCHEMAS = frozenset({"prismaquant.t4_adopted_catalog.v1"})
ADOPTION_SCHEMA = "prismaquant.joint_catalog_source_adoption.v1"
RESEAL_PROOF_SCHEMA = "prismaquant.reseal_proof_bundle.v1"
SERVED_POLICY_SCHEMA_V1 = "prismaquant.joint_served_activation_policy.v1"
SERVED_POLICY_SCHEMA_V2 = "prismaquant.joint_served_activation_policy.v2"
#: v1 of the served policy names one rung; its original rung stays exact.
SERVED_POLICY_V1_FORMAT = "TESSERA_E2M1_K2_R896"
SERVED_POLICY_FORMAT_PREFIX = "TESSERA_E2M1_K2_R"
#: The canonical calibration cache a ``tessera.hessian_capture.references.v1``
#: document binds, and the storage source each of its payloads names.
CANONICAL_CAPTURE_SCHEMA = "prismaquant.tessera_calibration_cache.v2"
CANONICAL_CAPTURE_SOURCE = "tessera_campaign_prefix_f32_v1"
CANONICAL_CAPTURE = (CANONICAL_CAPTURE_SCHEMA, CANONICAL_CAPTURE_SOURCE)

#: This file, for ``export_tessera_serving.py --producer-authority``.
AUTHORITY_PATH = Path(__file__).resolve()

_DOCUMENT_SCHEMAS = {"catalog_extension": CATALOG_EXTENSION_SCHEMAS,
                     "candidate_overlay": CANDIDATE_OVERLAY_SCHEMAS}


def served_activation_rates(policy):
    """Read the bound policy's scope; v1 retains its original single rung."""
    if isinstance(policy, dict):
        if (policy.get("schema") == SERVED_POLICY_SCHEMA_V1
                and policy.get("format") == SERVED_POLICY_V1_FORMAT):
            return (896,)
        if policy.get("schema") == SERVED_POLICY_SCHEMA_V2:
            formats = policy.get("formats")
            if (isinstance(formats, list) and formats
                    and all(isinstance(fmt, str) and re.fullmatch(r"TESSERA_E2M1_K2_R[1-9][0-9]*", fmt)
                            for fmt in formats)
                    and formats == sorted(set(formats))):
                return tuple(int(fmt.removeprefix(SERVED_POLICY_FORMAT_PREFIX)) for fmt in formats)
    raise ValueError("rooted cached unit served activation policy schema differs")


class PrismaQuantReuseAuthority:
    """PrismaQuant's ``tessera.cached_unit.ReuseAuthority``."""

    canonical_hessian_capture = CANONICAL_CAPTURE

    def check_document(self, role, document):
        schemas = _DOCUMENT_SCHEMAS[role]
        schema = document.get("schema") if isinstance(document, dict) else None
        if not isinstance(schema, str) or schema not in schemas:
            raise ValueError("rooted cached unit authority schema differs")

    def adoption_proof(self, unit, adoption, identity, original):
        name = unit
        if (not isinstance(adoption, dict) or adoption.get("schema") != ADOPTION_SCHEMA):
            raise ValueError("cached unit source adoption schema differs")
        candidate, reference = adoption["candidate_encoding_identity"], adoption["reference_encoding_identity"]
        if (candidate != identity or reference.get("unit") != name
                or adoption.get("reference_pair", [None])[0] != name
                or reference.get("encoder_source_sha256") != original):
            raise ValueError("cached unit source adoption identities differ")
        for field in ("unit", "source", "calibration", "encoder_fixture_id"):
            if field not in reference or reference[field] != candidate.get(field):
                raise ValueError("cached unit source adoption changed " + field)
        if reference.get("projection") != candidate.get("projection"):
            raise ValueError("cached unit source adoption changed projection")
        return adoption["encoder_source_proof"]

    def proof_authorizes(self, proof, adoption, original):
        candidate = adoption["candidate_encoding_identity"]
        return not (not proof or proof.get("schema") != RESEAL_PROOF_SCHEMA
                    or proof.get("ok") is not True or proof.get("encoder_fixture_id_equal") is not True
                    or proof.get("pins", {}).get("old", {}).get("encoder_source_sha256") != original
                    or proof.get("pins", {}).get("new", {}).get("encoder_source_sha256") != candidate["encoder_source_sha256"]
                    or set((proof.get("fixture_id", {}).get("ids") or {}).values()) != {candidate["encoder_fixture_id"]})

    def served_activations(self, policy, adoptions, units):
        rates = served_activation_rates(policy)
        groups = policy["executed_grouping"]["groups"]
        index = {name: (key, group) for key, group in groups.items() for name in group["members"]}
        expected = {}
        for name in adoptions:
            recipe = units[name]["identity"].get("recipe", {})
            if (recipe.get("grid") == "E2M1x2"
                    and recipe.get("q256") in rates):
                if name not in index:
                    raise ValueError("selected A4 unit absent from served activation policy")
                key, group = index[name]
                expected[name] = {"group": key, "input_global_scale": group["input_global_scale"]}
        return expected


#: The object Tessera's exporter reads from this file (``--producer-authority``).
PRODUCER_AUTHORITY = PrismaQuantReuseAuthority()
