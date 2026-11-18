"""Frontier selection publishes its JSON files through the bound profile.

The literals come from the real unedited selector. A JSON-read seam supplies
the validation rows and the candidate. CPU-only.
"""
from __future__ import annotations

import pytest

from prismaquant import select_validated_frontier as frontier
from prismaquant.digests import JsonProfile
from prismaquant.footprint import assignment_serialization_sha256

PROFILE = "direct-ascii-indent2-lax"
ASSIGNMENT = {"model.layers.0.mlp.caf\u00e9": "BF16"}
CASES = {
    "empty": {},
    "rich": {"unicode": "snowman \u2603 / caf\u00e9",
             "nested": [None, True, -1, 1.5, [], {}],
             "nan": float("nan"), "infinity": float("inf"),
             "negative_infinity": -float("inf")},
    "mixed": {1: "number", "x": "text"},
    "unsupported": {"bad": object()},
}
LAYER_BYTES = (
    b'{\n  "model.layers.0.mlp.caf\\u00e9": {\n    "act_bits": 16,\n'
    b'    "act_data_type": "float",\n    "bits": 16,\n'
    b'    "data_type": "float",\n    "group_size": 0\n  }\n}\n'
)
DIGEST = "a7b578d46a81859041342570f9e479daf2a76fb1b6f03af30a032b22f7f93331"
ASSIGNMENT_BYTES = {
    "empty": r"""{
  "assignment": {
    "model.layers.0.mlp.caf\u00e9": "BF16"
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
    "label": "fixture-\u2603",
    "mse": {},
    "path": "candidate.json",
    "surrogate_basis": null,
    "surrogate_loss": null
  },
  "selection_mode": "best-kl"
}
""".encode("ascii"),
    "rich": r"""{
  "assignment": {
    "model.layers.0.mlp.caf\u00e9": "BF16"
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
    "label": "fixture-\u2603",
    "mse": {
      "infinity": Infinity,
      "nan": NaN,
      "negative_infinity": -Infinity,
      "nested": [
        null,
        true,
        -1,
        1.5,
        [],
        {}
      ],
      "unicode": "snowman \u2603 / caf\u00e9"
    },
    "path": "candidate.json",
    "surrogate_basis": null,
    "surrogate_loss": null
  },
  "selection_mode": "best-kl"
}
""".encode("ascii"),
}
SUMMARY_BYTES = {
    "empty": r"""{
  "frontier": [
    {
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
      "label": "fixture-\u2603",
      "mse": {},
      "path": "candidate.json",
      "surrogate_basis": null,
      "surrogate_loss": null
    }
  ],
  "kl_noise_floor": 0.0,
  "kneedle_comparison": {
    "enabled": false,
    "reason": "too_few_frontier_points"
  },
  "leave_one_out": {
    "enabled": false,
    "reason": "mode_not_kneedle"
  },
  "metric": "kl",
  "n_frontier": 1,
  "n_results": 1,
  "output_assignment": "assignment.json",
  "output_layer_config": "layer.json",
  "practical_abs_eps": 0.0,
  "practical_knee": {
    "artifact_bytes": null,
    "assignment_hash": null,
    "assignment_sha256": "a7b578d46a81859041342570f9e479daf2a76fb1b6f03af30a032b22f7f93331",
    "best_kl": 0.01,
    "best_kl_label": "fixture-\u2603",
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
    "label": "fixture-\u2603",
    "mse": {},
    "path": "candidate.json",
    "surrogate_basis": null,
    "surrogate_loss": null,
    "tolerance": 5e-05
  },
  "practical_rel_eps": 0.005,
  "saturation": null,
  "schema": "prismaquant.validated_frontier_selection.v1",
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
    "label": "fixture-\u2603",
    "mse": {},
    "path": "candidate.json",
    "surrogate_basis": null,
    "surrogate_loss": null
  },
  "selected_artifact_bytes": null,
  "selection_mode": "best-kl",
  "surrogate_pairing": {
    "bases": [],
    "basis": null,
    "n_paired": 0,
    "n_without_surrogate": 1,
    "pooled": false
  },
  "surrogate_spearman": null,
  "surrogate_worst_rank_inversion": null,
  "tail_veto": {
    "column": null,
    "eta": "auto",
    "eta_mode": "auto",
    "eta_resolved": null,
    "inert_reason": null,
    "n_vetoed": 0
  },
  "target_disk_gb": null,
  "uniform_control": null,
  "unstable_policy": "keep-kneedle",
  "validation_json": "validation.json",
  "vetoed_rows": []
}
""".encode("ascii"),
    "rich": r"""{
  "frontier": [
    {
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
      "label": "fixture-\u2603",
      "mse": {
        "infinity": Infinity,
        "nan": NaN,
        "negative_infinity": -Infinity,
        "nested": [
          null,
          true,
          -1,
          1.5,
          [],
          {}
        ],
        "unicode": "snowman \u2603 / caf\u00e9"
      },
      "path": "candidate.json",
      "surrogate_basis": null,
      "surrogate_loss": null
    }
  ],
  "kl_noise_floor": 0.0,
  "kneedle_comparison": {
    "enabled": false,
    "reason": "too_few_frontier_points"
  },
  "leave_one_out": {
    "enabled": false,
    "reason": "mode_not_kneedle"
  },
  "metric": "kl",
  "n_frontier": 1,
  "n_results": 1,
  "output_assignment": "assignment.json",
  "output_layer_config": "layer.json",
  "practical_abs_eps": 0.0,
  "practical_knee": {
    "artifact_bytes": null,
    "assignment_hash": null,
    "assignment_sha256": "a7b578d46a81859041342570f9e479daf2a76fb1b6f03af30a032b22f7f93331",
    "best_kl": 0.01,
    "best_kl_label": "fixture-\u2603",
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
    "label": "fixture-\u2603",
    "mse": {
      "infinity": Infinity,
      "nan": NaN,
      "negative_infinity": -Infinity,
      "nested": [
        null,
        true,
        -1,
        1.5,
        [],
        {}
      ],
      "unicode": "snowman \u2603 / caf\u00e9"
    },
    "path": "candidate.json",
    "surrogate_basis": null,
    "surrogate_loss": null,
    "tolerance": 5e-05
  },
  "practical_rel_eps": 0.005,
  "saturation": null,
  "schema": "prismaquant.validated_frontier_selection.v1",
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
    "label": "fixture-\u2603",
    "mse": {
      "infinity": Infinity,
      "nan": NaN,
      "negative_infinity": -Infinity,
      "nested": [
        null,
        true,
        -1,
        1.5,
        [],
        {}
      ],
      "unicode": "snowman \u2603 / caf\u00e9"
    },
    "path": "candidate.json",
    "surrogate_basis": null,
    "surrogate_loss": null
  },
  "selected_artifact_bytes": null,
  "selection_mode": "best-kl",
  "surrogate_pairing": {
    "bases": [],
    "basis": null,
    "n_paired": 0,
    "n_without_surrogate": 1,
    "pooled": false
  },
  "surrogate_spearman": null,
  "surrogate_worst_rank_inversion": null,
  "tail_veto": {
    "column": null,
    "eta": "auto",
    "eta_mode": "auto",
    "eta_resolved": null,
    "inert_reason": null,
    "n_vetoed": 0
  },
  "target_disk_gb": null,
  "uniform_control": null,
  "unstable_policy": "keep-kneedle",
  "validation_json": "validation.json",
  "vetoed_rows": []
}
""".encode("ascii"),
}
STDOUT = (
    "[frontier-select] selected fixture-\u2603 bpp=16.000000 KL=0.01 mode=best-kl\n"
    "[frontier-select] surrogate-vs-KL fidelity: unavailable (need >=3 measured "
    "points carrying predicted_dloss_sum)\n"
    "[frontier-select] layer_config -> layer.json\n"
    "[frontier-select] summary -> summary.json\n"
).encode("utf-8")
REFUSALS = {
    "mixed": "'<' not supported between instances of 'str' and 'int'",
    "unsupported": "Object of type object is not JSON serializable",
}


def _select(mp, root, value):
    """Run the real selector main on one case value in a fresh directory."""
    root.mkdir()
    mp.chdir(root)
    row = {"label": "fixture-\u2603", "path": "candidate.json", "bpp": 16.0,
           "last_token_kl": 0.01, "format_counts": {"BF16": 1}, "mse": value,
           "assignment_sha256": assignment_serialization_sha256(ASSIGNMENT)}
    inputs = {
        "validation.json": {"results": [row]},
        "candidate.json": {"schema": "prismaquant.allocator.pareto_assignment.v1",
                           "assignment": ASSIGNMENT},
    }
    mp.setattr(frontier, "_load_json", lambda path: inputs[str(path)])
    return frontier.main([
        "--validation-json", "validation.json", "--mode", "best-kl",
        "--tail-veto", "none", "--output-layer-config", "layer.json",
        "--output-assignment", "assignment.json", "--output-summary", "summary.json",
    ])


@pytest.mark.parametrize("case", list(CASES))
def test_frontier_publication_profile_calls(tmp_path, monkeypatch, capsysbinary, case):
    assert assignment_serialization_sha256(ASSIGNMENT) == DIGEST
    calls = []
    original = JsonProfile.text

    def spy(self, value):
        calls.append((self.name, value))
        return original(self, value)

    monkeypatch.setattr(JsonProfile, "text", spy)
    root = tmp_path / case
    refusal = REFUSALS.get(case)
    if refusal is None:
        assert _select(monkeypatch, root, CASES[case]) == 0
    else:
        with pytest.raises(TypeError) as info:
            _select(monkeypatch, root, CASES[case])
        assert str(info.value) == refusal
        assert info.value.__cause__ is None
    out, err = capsysbinary.readouterr()
    assert err == b""
    ours = [value for name, value in calls if name == PROFILE]
    assert len(ours) == (3 if refusal is None else 2)
    assert ours[0] == {"model.layers.0.mlp.caf\u00e9": {
        "act_bits": 16, "act_data_type": "float", "bits": 16,
        "data_type": "float", "group_size": 0}}
    assert ours[1]["schema"] == "prismaquant.validated_frontier_assignment.v1"
    assert (root / "layer.json").read_bytes() == LAYER_BYTES
    if refusal is not None:
        assert out == b""
        assert sorted(p.name for p in root.iterdir()) == ["layer.json"]
        return
    assert ours[2]["schema"] == "prismaquant.validated_frontier_selection.v1"
    assert out == STDOUT
    assert sorted(p.name for p in root.iterdir()) == [
        "assignment.json", "layer.json", "summary.json"]
    assert (root / "assignment.json").read_bytes() == ASSIGNMENT_BYTES[case]
    assert (root / "summary.json").read_bytes() == SUMMARY_BYTES[case]
