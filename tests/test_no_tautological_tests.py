from pathlib import Path

from prismaquant.tautology import scan


def test_no_tautological_tests():
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    if findings:
        raise AssertionError("\n".join(
            f"{finding.path}:{finding.line}: {finding.test}: {finding.rule}"
            for finding in findings
        ))
