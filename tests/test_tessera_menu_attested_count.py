"""An explicitly named Tessera rung is not "attested" until the predicate says so (#278)."""
from prismaquant import format_registry as fr
from prismaquant import tessera_menu as menu

ATTESTED = "TESSERA_E4M3_K1_R1024"
UNATTESTED = "TESSERA_E4M3_K1_R896"


def _admits_only_r1024(monkeypatch):
    monkeypatch.setattr(fr, "format_is_producer_eligible",
                        lambda name, **_kw: str(name).endswith("R1024"))


def test_explicit_unattested_rung_is_partitioned_out(monkeypatch):
    _admits_only_r1024(monkeypatch)
    admitted, refused = menu.partition_attested([ATTESTED, UNATTESTED, "NVFP4"])
    assert admitted == [ATTESTED]
    assert refused == [UNATTESTED]


def test_explicit_menu_report_counts_the_predicate_not_the_caller(monkeypatch):
    _admits_only_r1024(monkeypatch)
    kept = [ATTESTED, UNATTESTED]
    admitted, refused = menu.partition_attested(kept)
    widths, line = menu.menu_width_report(kept, admitted, [], refused, menu.MENU_ATTESTED)
    assert widths["attested_rungs"] == 1
    assert widths["explicit_unattested_rungs"] == 1
    assert widths["menu_mode"] == menu.MENU_ATTESTED
    assert line.startswith("[alloc] Tessera menu: 1 of 2 priced rungs are attested by the pinned runtime")
    assert "1 explicitly named rungs are unattested" in line


def test_all_explicit_rungs_unattested_reads_zero_not_all(monkeypatch):
    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name, **_kw: False)
    kept = [ATTESTED, UNATTESTED, "TESSERA_BF16_K1_R896"]
    admitted, refused = menu.partition_attested(kept)
    widths, line = menu.menu_width_report(kept, admitted, [], refused, menu.MENU_ATTESTED)
    assert widths["attested_rungs"] == 0 and refused == kept
    assert line.startswith("[alloc] Tessera menu: 0 of 3 priced rungs are attested")
    assert "sample" not in line


def test_research_mode_is_labelled_admitted_not_attested(monkeypatch):
    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name, **_kw: True)
    kept = [ATTESTED, UNATTESTED]
    admitted, refused = menu.partition_attested(kept)
    widths, line = menu.menu_width_report(kept, admitted, [], refused, menu.MENU_RESEARCH)
    assert widths["attested_rungs"] == 0
    assert widths["research_admitted_rungs"] == 2
    assert "PRISMAQUANT_TESSERA_MENU=research" in line
    assert "attested by the pinned runtime" not in line.split("(not attested")[0]


# --------------------------------------------------------------------------- #
# A refusal that cannot name its own cause (RobTand/prismaquant#572)
# --------------------------------------------------------------------------- #
#
# "0 of 16 priced rungs are attested" was read three ways at once -- the
# contract lists none of them, no serving scope was supplied, or the menu mode
# is too strict -- and answering it took a full investigation. The two the
# contract can tell apart are already structured on ``RouteAdmission``
# (``requires_serving_context``) and in the packaged contract
# (``attested_rungs_q256``); the report just never read them.


class _Admission:
    def __init__(self, requires_serving_context):
        self.requires_serving_context = requires_serving_context


def test_diagnosis_names_a_missing_serving_scope(monkeypatch):
    monkeypatch.setattr(menu, "route_admission",
                        lambda name, **_kw: _Admission(True))
    monkeypatch.setattr(menu, "contract_attested_rung_names",
                        lambda: ("TESSERA_E4M3_K1_R1024",))
    refused = [ATTESTED, UNATTESTED]
    diagnosis = menu.unattested_diagnosis(refused, priced=refused)
    assert diagnosis["serving_scope_supplied"] is False
    assert diagnosis["awaiting_serving_scope"] == sorted(refused)
    assert diagnosis["unattested_by_contract"] == []
    assert diagnosis["contract_attested_rungs"] == ["TESSERA_E4M3_K1_R1024"]
    _, line = menu.menu_width_report(refused, [], [], refused, menu.MENU_ATTESTED,
                                     diagnosis=diagnosis)
    assert "no Tessera serving scope was supplied" in line
    assert "--tessera-platform" in line


def test_diagnosis_separates_a_real_coverage_gap(monkeypatch):
    # A scope WAS supplied, so what comes back is the contract's own verdict,
    # and the useful thing to print beside it is what the runtime does attest.
    monkeypatch.setattr(menu, "route_admission",
                        lambda name, **_kw: _Admission(True))
    monkeypatch.setattr(menu, "contract_attested_rung_names",
                        lambda: ("TESSERA_BF16_K1_R1792", "TESSERA_E4M3_K1_R1024"))

    class _Ctx:
        def key(self):
            return "sm_121/dense"

    diagnosis = menu.unattested_diagnosis([UNATTESTED], priced=[UNATTESTED, ATTESTED],
                                          context_by_unit={"u": _Ctx()})
    assert diagnosis["serving_scope_supplied"] is True
    assert diagnosis["awaiting_serving_scope"] == []
    assert diagnosis["unattested_by_contract"] == [UNATTESTED]
    assert diagnosis["contract_attested_but_unpriced"] == ["TESSERA_BF16_K1_R1792"]
    _, line = menu.menu_width_report([UNATTESTED, ATTESTED], [ATTESTED], [], [UNATTESTED],
                                     menu.MENU_ATTESTED, diagnosis=diagnosis)
    assert "no Tessera serving scope was supplied" not in line
    assert "the pinned contract attests" in line
    assert "TESSERA_BF16_K1_R1792" in line


def test_contract_attested_rung_names_are_read_from_the_packaged_contract(monkeypatch):
    # Principle 14: the names are DERIVED from the table the runtime publishes,
    # never typed here. A family the contract publishes with no attested rung
    # contributes nothing rather than its whole reader range.
    formats = {
        "TESSERA_E2M1_K2": {"name_pattern": "TESSERA_E2M1_K2_R{k}",
                            "attested_rungs_q256": [896],
                            "reader_rate_range_q256": [896, 896]},
        "TESSERA_BF16_K1": {"name_pattern": "TESSERA_BF16_K1_R{k}",
                            "attested_rungs_q256": [],
                            "reader_rate_range_q256": [256, 4096]},
    }
    from prismaquant import tessera_render
    monkeypatch.setattr(tessera_render, "_pinned_serving_table",
                        lambda: ({}, formats))
    assert menu.contract_attested_rung_names() == ("TESSERA_E2M1_K2_R896",)


def test_report_without_a_diagnosis_is_byte_identical(monkeypatch):
    _admits_only_r1024(monkeypatch)
    kept = [ATTESTED, UNATTESTED]
    admitted, refused = menu.partition_attested(kept)
    plain = menu.menu_width_report(kept, admitted, [], refused, menu.MENU_ATTESTED)
    explicit = menu.menu_width_report(kept, admitted, [], refused, menu.MENU_ATTESTED,
                                      diagnosis=None)
    assert plain == explicit
    assert "serving scope" not in plain[1]


def test_the_fatal_refusal_carries_the_cause_too():
    # The menu line prints one line above the SystemExit an operator greps for,
    # so the cause rides on both. Same structured answer, rendered twice.
    diagnosis = {"awaiting_serving_scope": ["TESSERA_E4M3_K1_R1006"],
                 "unattested_by_contract": [],
                 "contract_attested_rungs": ["TESSERA_E4M3_K1_R1024"],
                 "contract_attested_but_unpriced": ["TESSERA_E4M3_K1_R1024"],
                 "serving_scope_supplied": False}
    cause = menu.tessera_refusal_cause(diagnosis)
    assert cause.startswith("; ")
    assert "no Tessera serving scope was supplied" in cause
    assert "--tessera-platform" in cause
    assert "the pinned contract attests TESSERA_E4M3_K1_R1024" in cause
    # No Tessera rungs refused -> nothing appended, so a non-Tessera menu
    # refusal reads exactly as it did.
    assert menu.tessera_refusal_cause(None) == ""
    assert menu.tessera_refusal_cause({}) == ""
