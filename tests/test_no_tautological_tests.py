from pathlib import Path
import runpy

tautology = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / "prismaquant/vendored/tautology.py"
))


def test_no_tautological_tests():
    root = Path(__file__).resolve().parents[1]
    findings = tautology["scan"](root)
    if findings:
        raise AssertionError("\n".join(
            f"{finding.path}:{finding.line}: {finding.test}: {finding.rule}"
            for finding in findings
        ))
