"""Check the frontier publication calls against the recorded results."""

import json

import pytest

from prismaquant import select_validated_frontier as frontier
from prismaquant.digests import DIRECT_UTF8_LAX, JsonProfile
from prismaquant.footprint import assignment_serialization_sha256

LAYER = b'''{
  "model.layers.0.mlp.caf\\u00e9": {
    "act_bits": 16,
    "act_data_type": "float",
    "bits": 16,
    "data_type": "float",
    "group_size": 0
  }
}
'''
EMPTY_ASSIGNMENT = b'''{
  "assignment": {
    "model.layers.0.mlp.caf\\u00e9": "BF16"
  },
  "schema": "prismaquant.validated_frontier_assignment.v1",
  "selected": {
    "artifact_bytes": null,
    "assignment_hash": null,
    "assignment_sha256": "a7b578d46a81859041342570f9e479daf2a76fb1b6f03af30a032b22f7f93331",
    "bpp": 16.0,
    "changed_vs_base": 0,
    "format_counts": {
      "BF16": 1
    },
    "kl": 0.01,
    "kl_repeats": [],
    "kl_std": null,
    "kl_stderr": null,
    "kl_ucb": null,
    "label": "fixture-\\u2603",
    "mse": {},
    "path": "candidate.json",
    "surrogate_basis": null,
    "surrogate_loss": null
  },
  "selection_mode": "best-kl"
}
'''
EMPTY_SUMMARY = (
    b'{\n  "frontier": [\n    {\n      "artifact_bytes": null,\n      "assignm'
    b'ent_hash": null,\n      "assignment_sha256": "a7b578d46a8185904134257'
    b'0f9e479daf2a76fb1b6f03af30a032b22f7f93331",\n      "bpp": 16.0,\n     '
    b' "changed_vs_base": 0,\n      "format_counts": {\n        "BF16": 1\n  '
    b'    },\n      "kl": 0.01,\n      "kl_repeats": [],\n      "kl_std": nul'
    b'l,\n      "kl_stderr": null,\n      "kl_ucb": null,\n      "label": "fi'
    b'xture-\\u2603",\n      "mse": {},\n      "path": "candidate.json",\n    '
    b'  "surrogate_basis": null,\n      "surrogate_loss": null\n    }\n  ],\n '
    b' "kl_noise_floor": 0.0,\n  "kneedle_comparison": {\n    "enabled": fal'
    b'se,\n    "reason": "too_few_frontier_points"\n  },\n  "leave_one_out": '
    b'{\n    "enabled": false,\n    "reason": "mode_not_kneedle"\n  },\n  "met'
    b'ric": "kl",\n  "n_frontier": 1,\n  "n_results": 1,\n  "output_assignmen'
    b't": "assignment.json",\n  "output_layer_config": "layer.json",\n  "pra'
    b'ctical_abs_eps": 0.0,\n  "practical_knee": {\n    "artifact_bytes": nu'
    b'll,\n    "assignment_hash": null,\n    "assignment_sha256": "a7b578d46'
    b'a81859041342570f9e479daf2a76fb1b6f03af30a032b22f7f93331",\n    "best_'
    b'kl": 0.01,\n    "best_kl_label": "fixture-\\u2603",\n    "bpp": 16.0,\n '
    b'   "changed_vs_base": 0,\n    "format_counts": {\n      "BF16": 1\n    '
    b'},\n    "kl": 0.01,\n    "kl_repeats": [],\n    "kl_std": null,\n    "kl'
    b'_stderr": null,\n    "kl_ucb": null,\n    "label": "fixture-\\u2603",\n '
    b'   "mse": {},\n    "path": "candidate.json",\n    "surrogate_basis": n'
    b'ull,\n    "surrogate_loss": null,\n    "tolerance": 5e-05\n  },\n  "prac'
    b'tical_rel_eps": 0.005,\n  "saturation": null,\n  "schema": "prismaquan'
    b't.validated_frontier_selection.v1",\n  "selected": {\n    "artifact_by'
    b'tes": null,\n    "assignment_hash": null,\n    "assignment_sha256": "a'
    b'7b578d46a81859041342570f9e479daf2a76fb1b6f03af30a032b22f7f93331",\n  '
    b'  "bpp": 16.0,\n    "changed_vs_base": 0,\n    "format_counts": {\n    '
    b'  "BF16": 1\n    },\n    "kl": 0.01,\n    "kl_repeats": [],\n    "kl_std'
    b'": null,\n    "kl_stderr": null,\n    "kl_ucb": null,\n    "label": "fi'
    b'xture-\\u2603",\n    "mse": {},\n    "path": "candidate.json",\n    "sur'
    b'rogate_basis": null,\n    "surrogate_loss": null\n  },\n  "selected_art'
    b'ifact_bytes": null,\n  "selection_mode": "best-kl",\n  "surrogate_pair'
    b'ing": {\n    "bases": [],\n    "basis": null,\n    "n_paired": 0,\n    "'
    b'n_without_surrogate": 1,\n    "pooled": false\n  },\n  "surrogate_spear'
    b'man": null,\n  "surrogate_worst_rank_inversion": null,\n  "tail_veto":'
    b' {\n    "column": null,\n    "eta": "auto",\n    "eta_mode": "auto",\n  '
    b'  "eta_resolved": null,\n    "inert_reason": null,\n    "n_vetoed": 0\n'
    b'  },\n  "target_disk_gb": null,\n  "uniform_control": null,\n  "unstabl'
    b'e_policy": "keep-kneedle",\n  "validation_json": "validation.json",\n '
    b' "vetoed_rows": []\n}\n'
)
RICH_ASSIGNMENT = (
    b'{\n  "assignment": {\n    "model.layers.0.mlp.caf\\u00e9": "BF16"\n '
    b' },\n  "schema": "prismaquant.validated_frontier_assignment.v1",\n  "s'
    b'elected": {\n    "artifact_bytes": null,\n    "assignment_hash": null,'
    b'\n    "assignment_sha256": "a7b578d46a81859041342570f9e479daf2a76fb1b'
    b'6f03af30a032b22f7f93331",\n    "bpp": 16.0,\n    "changed_vs_base": 0,'
    b'\n    "format_counts": {\n      "BF16": 1\n    },\n    "kl": 0.01,\n '
    b'   "kl_repeats": [],\n    "kl_std": null,\n    "kl_stderr": null,\n'
    b'    "kl_ucb": null,\n    "label": "fixture-\\u2603",\n    "mse": {\n'
    b'      "infinity": Infinity,\n      "nan": NaN,\n      "negative_infini'
    b'ty": -Infinity,\n      "nested": [\n        null,\n        true,\n  '
    b'      -1,\n        1.5,\n        [],\n        {}\n      ],\n      "un'
    b'icode": "snowman \\u2603 / caf\\u00e9"\n    },\n    "path": "candida'
    b'te.json",\n    "surrogate_basis": null,\n    "surrogate_loss": nul'
    b'l\n  },\n  "selection_mode": "best-kl"\n}\n'
)
RICH_SUMMARY = (
    b'{\n  "frontier": [\n    {\n      "artifact_bytes": null,\n      "assignm'
    b'ent_hash": null,\n      "assignment_sha256": "a7b578d46a81859041342570f9e'
    b'479daf2a76fb1b6f03af30a032b22f7f93331",\n      "bpp": 16.0,\n      "ch'
    b'anged_vs_base": 0,\n      "format_counts": {\n        "BF16": 1\n      '
    b'},\n      "kl": 0.01,\n      "kl_repeats": [],\n      "kl_std": null,\n  '
    b'    "kl_stderr": null,\n      "kl_ucb": null,\n      "label": "fixtur'
    b'e-\\u2603",\n      "mse": {\n        "infinity": Infinity,\n        "nan'
    b'": NaN,\n        "negative_infinity": -Infinity,\n        "nested": [\n'
    b'          null,\n          true,\n          -1,\n          1.5,\n       '
    b'   [],\n          {}\n        ],\n        "unicode": "snowman \\u2603 / '
    b'caf\\u00e9"\n      },\n      "path": "candidate.json",\n      "surrogate'
    b'_basis": null,\n      "surrogate_loss": null\n    }\n  ],\n  "kl_noise_f'
    b'loor": 0.0,\n  "kneedle_comparison": {\n    "enabled": false,\n    "rea'
    b'son": "too_few_frontier_points"\n  },\n  "leave_one_out": {\n    "enabl'
    b'ed": false,\n    "reason": "mode_not_kneedle"\n  },\n  "metric": "kl",\n'
    b'  "n_frontier": 1,\n  "n_results": 1,\n  "output_assignment": "assignm'
    b'ent.json",\n  "output_layer_config": "layer.json",\n  "practical_abs_e'
    b'ps": 0.0,\n  "practical_knee": {\n    "artifact_bytes": null,\n    "ass'
    b'ignment_hash": null,\n    "assignment_sha256": "a7b578d46a81859041342570f'
    b'9e479daf2a76fb1b6f03af30a032b22f7f93331",\n    "best_kl": 0.01,\n    "'
    b'best_kl_label": "fixture-\\u2603",\n    "bpp": 16.0,\n    "changed_vs_b'
    b'ase": 0,\n    "format_counts": {\n      "BF16": 1\n    },\n    "kl": 0.0'
    b'1,\n    "kl_repeats": [],\n    "kl_std": null,\n    "kl_stderr": null,\n'
    b'    "kl_ucb": null,\n    "label": "fixture-\\u2603",\n    "mse": {\n    '
    b'  "infinity": Infinity,\n      "nan": NaN,\n      "negative_infinity":'
    b' -Infinity,\n      "nested": [\n        null,\n        true,\n        -1'
    b',\n        1.5,\n        [],\n        {}\n      ],\n      "unicode": "sno'
    b'wman \\u2603 / caf\\u00e9"\n    },\n    "path": "candidate.json",\n    "s'
    b'urrogate_basis": null,\n    "surrogate_loss": null,\n    "tolerance": '
    b'5e-05\n  },\n  "practical_rel_eps": 0.005,\n  "saturation": null,\n  "sc'
    b'hema": "prismaquant.validated_frontier_selection.v1",\n  "selected": '
    b'{\n    "artifact_bytes": null,\n    "assignment_hash": null,\n    "assi'
    b'gnment_sha256": "a7b578d46a81859041342570f9e479daf2a76fb1b6f03af30a032b2'
    b'2f7f93331",\n    "bpp": 16.0,\n    "changed_vs_base": 0,\n    "format_c'
    b'ounts": {\n      "BF16": 1\n    },\n    "kl": 0.01,\n    "kl_repeats": ['
    b'],\n    "kl_std": null,\n    "kl_stderr": null,\n    "kl_ucb": null,\n  '
    b'  "label": "fixture-\\u2603",\n    "mse": {\n      "infinity": Infinity'
    b',\n      "nan": NaN,\n      "negative_infinity": -Infinity,\n      "nes'
    b'ted": [\n        null,\n        true,\n        -1,\n        1.5,\n       '
    b' [],\n        {}\n      ],\n      "unicode": "snowman \\u2603 / caf\\u00e'
    b'9"\n    },\n    "path": "candidate.json",\n    "surrogate_basis": null,'
    b'\n    "surrogate_loss": null\n  },\n  "selected_artifact_bytes": null,\n'
    b'  "selection_mode": "best-kl",\n  "surrogate_pairing": {\n    "bases":'
    b' [],\n    "basis": null,\n    "n_paired": 0,\n    "n_without_surrogate"'
    b': 1,\n    "pooled": false\n  },\n  "surrogate_spearman": null,\n  "surro'
    b'gate_worst_rank_inversion": null,\n  "tail_veto": {\n    "column": nul'
    b'l,\n    "eta": "auto",\n    "eta_mode": "auto",\n    "eta_resolved": nu'
    b'll,\n    "inert_reason": null,\n    "n_vetoed": 0\n  },\n  "target_disk_'
    b'gb": null,\n  "uniform_control": null,\n  "unstable_policy": "keep-kne'
    b'edle",\n  "validation_json": "validation.json",\n  "vetoed_rows": []\n}'
    b'\n'
)
STDOUT = (
    b'[frontier-select] selected fixture-\xe2\x98\x83 bpp=16.000000 KL=0.01 mode=best-kl\n'
    b'[frontier-select] surrogate-vs-KL fidelity: unavailable (need >=3 measured points carrying'
    b' predicted_dloss_sum)\n[frontier-select] layer_config -> layer.json\n'
    b'[frontier-select] summary -> summary.json\n'
)
RICH = {
    "unicode": "snowman \u2603 / caf\u00e9",
    "nested": [None, True, -1, 1.5, [], {}],
    "nan": float("nan"),
    "infinity": float("inf"),
    "negative_infinity": -float("inf"),
}


@pytest.mark.parametrize(
    "mse, assignment_bytes, summary_bytes, message",
    [
        pytest.param({}, EMPTY_ASSIGNMENT, EMPTY_SUMMARY, None, id="empty"),
        pytest.param(RICH, RICH_ASSIGNMENT, RICH_SUMMARY, None, id="rich"),
        pytest.param({1: "number", "x": "text"}, None, None,
                     "'<' not supported between instances of 'str' and 'int'", id="mixed"),
        pytest.param({"bad": object()}, None, None,
                     "Object of type object is not JSON serializable", id="unsupported"),
    ],
)
def test_frontier_publication_profile_calls(
    tmp_path, monkeypatch, capsysbinary, mse, assignment_bytes, summary_bytes, message,
):
    monkeypatch.chdir(tmp_path)
    assignment = {"model.layers.0.mlp.café": "BF16"}
    digest = assignment_serialization_sha256(assignment)
    assert digest == "a7b578d46a81859041342570f9e479daf2a76fb1b6f03af30a032b22f7f93331"
    inputs = {
        "validation.json": {"results": [{
            "label": "fixture-☃", "path": "candidate.json", "bpp": 16.0,
            "last_token_kl": 0.01, "format_counts": {"BF16": 1},
            "assignment_sha256": digest, "mse": mse,
        }]},
        "candidate.json": {
            "schema": "prismaquant.allocator.pareto_assignment.v1", "assignment": assignment,
        },
    }
    monkeypatch.setattr(frontier, "_load_json", lambda path: inputs[str(path)])
    calls = []
    original = JsonProfile.text

    def recording_text(self, value):
        calls.append((self.name, value))
        return original(self, value)

    monkeypatch.setattr(JsonProfile, "text", recording_text)
    argv = [
        "--validation-json", "validation.json", "--mode", "best-kl", "--tail-veto", "none",
        "--output-layer-config", "layer.json", "--output-assignment", "assignment.json",
        "--output-summary", "summary.json",
    ]
    status = terminal = None
    try:
        status = frontier.main(argv)
    except TypeError as error:
        terminal = (type(error), str(error), error.__cause__)

    captured = capsysbinary.readouterr()
    files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    expected_files = {"layer.json": LAYER}
    if message is None:
        expected_files.update({"assignment.json": assignment_bytes, "summary.json": summary_bytes})
    assert files == expected_files
    assert captured.out == (STDOUT if message is None else b"")
    assert captured.err == b""
    assert status == (0 if message is None else None)
    assert terminal == (None if message is None else (TypeError, message, None))

    expected_assignment = json.loads(EMPTY_ASSIGNMENT)
    expected_assignment["selected"]["mse"] = mse
    expected_values = [json.loads(LAYER), expected_assignment]
    if message is None:
        expected_summary = json.loads(EMPTY_SUMMARY)
        for row in (expected_summary["selected"], expected_summary["practical_knee"],
                    *expected_summary["frontier"]):
            row["mse"] = mse
        expected_values.append(expected_summary)
    assert [call for call in calls if call[0] != DIRECT_UTF8_LAX.name] == [
        ("direct-ascii-indent2-lax", value) for value in expected_values
    ]
