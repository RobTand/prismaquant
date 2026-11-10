"""Standalone real measurement, outside pytest's rewriting import loader."""
import json
import pytest
from prismaquant.native_results import measure_runtime

if __name__ == '__main__':
    print(json.dumps(measure_runtime(dependencies=('pytest',))))
