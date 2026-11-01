#!/usr/bin/env python3
"""Verify the validation-repeat sampler moved without changing its body."""
import argparse
import ast
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    before = subprocess.check_output(
        ['git','show',f'{args.base}:prismaquant/validate_assignments_kl.py'], cwd=root, text=True)
    old = next(node for node in ast.parse(before).body if isinstance(node,ast.FunctionDef)
               and node.name=='_load_calibration_repeats')
    current = (root/'prismaquant/calibration_data.py').read_text()
    new = next(node for node in ast.parse(current).body if isinstance(node,ast.FunctionDef)
               and node.name=='load_calibration_repeats')
    old.name = new.name
    if ast.dump(old,include_attributes=False) != ast.dump(new,include_attributes=False):
        raise SystemExit('Validation-repeat sampler body changed')
    print('Validation-repeat sampler retains its original body.')


if __name__=='__main__':
    main()
