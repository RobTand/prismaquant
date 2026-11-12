"""The installed Tessera lane table, and older lane grammars derived from it.

A helper module, not a test module. These helpers and the ``legacy_v4_contract``
and ``legacy_v5_contract`` fixtures lived in ``tests/conftest.py`` until #3215.
Every test then depended on the modules they import, so a test about an OLDER
Tessera lane grammar now imports what it uses from here, and owns its fixture
instead of asserting that the installed contract is a version it no longer is.
"""
from __future__ import annotations

import copy
import json

import pytest


def _installed_contract() -> dict:
    """The Tessera contract the environment installs, parsed."""
    from importlib.resources import as_file

    from prismaquant import tessera_runtime_contract as contract

    with as_file(contract.contract_path()) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def set_cell_census(payload: dict, cell: dict, rungs: list[int]) -> None:
    """Change a valid fixture census and its v11 derived run-table metadata."""
    from prismaquant.lane_eligibility import _allowable_rung_tables

    cell["rungs_q256"] = list(rungs)
    if "run_tables" in cell:
        row = next(r for r in payload["formats"] if r["family"] == cell["family"])
        tables = _allowable_rung_tables(row, "fixture.formats")
        cell["run_tables"] = [list(t) for t in sorted(
            {tables[r] for r in rungs})] if tables else []



def down_convert_lane_table(payload: dict, schema: str) -> dict:
    """The installed contract, expressed in an OLDER lane grammar.

    Legacy-grammar tests need a legacy table. They used to get one by reading
    the installed contract, which was v4 at the time -- so on the day Tessera
    shipped ``tessera.lane-eligibility.v6`` (its PR #176) every such test began
    asserting that the installed contract was a version it no longer is. That
    is a fixture problem, not a reader problem: a test about a legacy grammar
    must OWN its legacy fixture, exactly as ``test_tessera_contract_v5`` already
    did by rewriting ``runtime`` onto every cell.

    Down-converting rather than hand-writing keeps the fixture honest about
    everything the grammar did not change -- families, rungs, route statuses,
    launches -- so the test still exercises real cells. It is a FIXTURE and
    never an attestation: nothing derived from it is recorded anywhere.

    ``schema`` is ``tessera.lane-eligibility.v9`` (reduce each v10
    ``platforms`` entry back to a bare key set and drop the platforms that
    exist only to say what they do NOT execute), ``...v8`` (drop the v9
    ``smoke.record`` as well), ``...v7`` (drop the v8 ``evidence.artifact`` as well),
    ``...v6`` (drop v7's ``smoke.attribution`` and ``smoke.control`` too),
    ``...v5`` (drop the whole ``evidence`` block and the ``runtime`` version
    fields) or ``...v4`` (drop the per-cell ``runtime`` scope as well).

    Dropping v9's record is not only a key removal: v9 DERIVES
    ``smoke.attribution`` from the record, and v7/v8 derive it from the
    control.  A fixture that removed the record and kept the derived
    attribution would be a table no validator would accept, so the
    attribution is re-derived under the older rule -- through the reader's own
    ``derive_smoke_attribution``, which is that rule's home here.
    """
    from prismaquant.lane_eligibility import (
        EVIDENCE_ATTRIBUTION_UNATTRIBUTED, EVIDENCE_OUTCOME_IDENTICAL,
        EVIDENCE_ATTRIBUTION_SHARED, EVIDENCE_ATTRIBUTION_NOT_SHARED)

    payload = copy.deepcopy(payload)
    lane = payload["lane_eligibility"]
    lane["schema"] = schema
    version = int(schema.rsplit(".v", 1)[1])
    if version <= 9:
        # v10 made a platform entry an OBJECT that states what the platform
        # executes.  Under v9 the value was never read, and a platform with no
        # cell said nothing at all -- so a v9 fixture keeps only the platforms
        # that carry cells, with only the identity key v22 published.
        attested = {cell["platform"] for cell in lane["cells"]}
        lane["platforms"] = {
            key: {k: v for k, v in entry.items()
                  if k in ("compute_capability", "gcn_arch")}
            for key, entry in lane["platforms"].items()
            if key in attested
        }
    for cell in lane["cells"]:
        if version <= 5:
            cell.pop("evidence", None)
        else:
            evidence = cell["evidence"]
            if version <= 8:
                smoke = evidence["smoke"]
                if smoke.pop("record", None) is not None:
                    control = smoke.get("control")
                    smoke["attribution"] = (
                        EVIDENCE_ATTRIBUTION_UNATTRIBUTED if control is None
                        else EVIDENCE_ATTRIBUTION_SHARED
                        if control["outcome"] == EVIDENCE_OUTCOME_IDENTICAL
                        else EVIDENCE_ATTRIBUTION_NOT_SHARED)
            if version <= 7:
                evidence.pop("artifact", None)
            if version <= 6:
                evidence["smoke"].pop("attribution", None)
                evidence["smoke"].pop("control", None)
        if version <= 4:
            cell.pop("runtime", None)
        elif version <= 5:
            runtime = cell.get("runtime", {})
            cell["runtime"] = {"image": runtime["image"],
                               "execution_modes": runtime["execution_modes"]}
    return payload


def lane_cells_on_one_image(payload: dict, image: str | None = None, *,
                            family: str | None = None) -> dict:
    """The installed table's cells measured on ONE runtime image.

    Several fixtures flatten every cell's ``runtime`` onto one fixture image
    (or down-convert below v6, where the image axis does not exist). That was
    safe while no two cells of one (platform, family, structure, regime)
    scope lived on different images. Tessera contract v38 broke it: the stock
    image's dense ``TESSERA_E4M3_K1`` pair (resident|streamed) and the GLM
    image's dense ``_resident`` pair are one scope on two images, and the
    flattened union is a table in which two cells claim the same scope --
    which the reader correctly refuses. A fixture therefore keeps one real
    image's roster and never relabels a union.

    ``image`` defaults to ``versions.default_serve_image`` (dense E4M3 at
    v39). A fixture needing a specific family can instead name ``family``;
    its published cells must select exactly one image, never an arbitrary
    first image. A FIXTURE, never an attestation.
    """
    payload = copy.deepcopy(payload)
    if family is not None:
        assert image is None, 'fixture must select image or family, not both'
        images = {c['runtime']['image'] for c in payload['lane_eligibility']['cells']
                  if c['family'] == family}
        assert len(images) == 1, f'{family}: fixture needs exactly one image, got {images}'
        image = images.pop()
    if image is None:
        image = payload["versions"]["default_serve_image"]
    lane = payload["lane_eligibility"]
    kept = [cell for cell in lane["cells"] if cell["runtime"]["image"] == image]
    assert kept, f"the installed table publishes no cell on {image!r}"
    lane["cells"] = kept
    return payload


def project_lane_cells_onto_structures(payload: dict, structures) -> dict:
    """The installed contract's cell coverage, re-addressed to ``structures``.

    Contract v23 (lane schema v10) made family coverage structure-specific:
    the packaged dense cells publish TESSERA_E2M1_K2 only, while
    TESSERA_E4M3_K1 is published only as routed_moe. A fixture that
    synthesizes one structure by relabelling the OTHER structure's cells --
    how the allocator fixtures were built when the packaged contract carried
    no routed_moe cell at all -- therefore loses every family the source
    structure did not cover, and the admission gate (correctly) refuses the
    rungs those tests select. This helper instead keeps each (platform,
    family, regime) scope's own coverage and re-addresses it onto exactly
    ``structures``:

    * a cell whose structure is requested is kept as published;
    * every published cell is cloned into each requested structure its
      scope does not already cover, so a clone carries its family's own
      admission facts -- rungs, activation contract, route status, plugin
      requirement, launch set -- and only ``structure``, a pure lookup
      discriminator (``cell_matches_serving_context``), is synthesized;
    * cells whose structure is not requested are dropped after cloning, so
      ``project_lane_cells_onto_structures(payload, ("dense",))`` is a
      dense-only table whose dense coverage is still every scope's.

    A clone is added only when neither a kept cell nor an earlier clone in
    the same scope claims an overlapping residency, read off the published
    ``TESSERA_SERVE_MODE`` flag by the same grammar
    ``parse_v4_cell_contract`` reads. The overlap
    key deliberately ignores the runtime image/execution axes because the
    fixtures that call this then flatten every cell's runtime onto one
    fixture image, which would otherwise reintroduce the overlap the reader
    refuses. Like ``down_convert_lane_table`` this is a FIXTURE, never an
    attestation: nothing derived from it is recorded anywhere.
    """
    wanted = list(dict.fromkeys(str(structure) for structure in structures))
    payload = copy.deepcopy(payload)
    lane = payload["lane_eligibility"]
    head = "TESSERA_SERVE_MODE="

    def _residencies(cell: dict) -> frozenset[str]:
        for flag in cell.get("requires_serve_flags", ()):
            if str(flag).startswith(head):
                return frozenset(str(flag)[len(head):].split("|"))
        return frozenset()

    kept = [cell for cell in lane["cells"] if cell.get("structure") in wanted]
    claimed: dict[tuple[str, str, str, str], set[str]] = {}
    for cell in kept:
        key = (cell["platform"], cell["family"], cell["regime"], cell["structure"])
        claimed.setdefault(key, set()).update(_residencies(cell))
    clones = []
    # Clone from EVERY published cell, not only the kept ones: a scope whose
    # only cell lives in a dropped structure still owns that scope's family
    # facts, and the requested structures inherit them from it.
    for cell in lane["cells"]:
        for structure in wanted:
            if structure == cell.get("structure"):
                continue
            key = (cell["platform"], cell["family"], cell["regime"], structure)
            if _residencies(cell) & claimed.get(key, set()):
                continue
            clone = copy.deepcopy(cell)
            clone["id"] = f"{cell['id']}_{structure}_fixture"
            clone["structure"] = structure
            clones.append(clone)
            claimed.setdefault(key, set()).update(_residencies(cell))
    lane["cells"] = kept + clones
    published = [s for s in lane.get("structures", []) if s in wanted]
    lane["structures"] = published + [s for s in wanted if s not in published]
    return payload


@pytest.fixture
def legacy_v4_contract() -> dict:
    """The installed contract expressed as a v4 lane table.

    v4 has no runtime-image axis, so the default image's roster is kept
    first (``lane_cells_on_one_image``): since contract v38 two images
    publish the same dense E4M3 scopes, and without the axis they overlap.
    """
    return down_convert_lane_table(lane_cells_on_one_image(_installed_contract()),
                                   "tessera.lane-eligibility.v4")


@pytest.fixture
def legacy_v5_contract() -> dict:
    """The installed contract expressed as a v5 lane table."""
    return down_convert_lane_table(_installed_contract(),
                                   "tessera.lane-eligibility.v5")
