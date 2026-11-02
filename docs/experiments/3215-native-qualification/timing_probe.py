"""Temporary diagnostic plugin. Does not change tests, gate policy, or budgets."""
import collections
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import time

REPORTS = []
IMPORTS = []
SAMPLES = collections.Counter()
PHASE = 'collection'
STOP = threading.Event()
MAIN_THREAD = threading.get_ident()
LOAD = importlib.machinery.SourceFileLoader.exec_module


def load(loader, module):
    phase = PHASE
    start = time.perf_counter()
    try:
        return LOAD(loader, module)
    finally:
        IMPORTS.append({'module': module.__name__, 'phase': phase,
                        'seconds': time.perf_counter() - start})


importlib.machinery.SourceFileLoader.exec_module = load


def sample():
    while not STOP.wait(0.005):
        frame = sys._current_frames().get(MAIN_THREAD)
        stack = []
        while frame is not None:
            stack.append(f'{frame.f_code.co_filename}:{frame.f_code.co_firstlineno}:{frame.f_code.co_name}')
            frame = frame.f_back
        SAMPLES[(PHASE, '\n'.join(stack[:16]))] += 1


THREAD = threading.Thread(target=sample, daemon=True)
THREAD.start()


def pytest_runtest_setup(item):
    global PHASE
    PHASE = 'setup:' + item.nodeid


def pytest_runtest_call(item):
    global PHASE
    PHASE = 'call:' + item.nodeid


def pytest_runtest_teardown(item):
    global PHASE
    PHASE = 'teardown:' + item.nodeid


def pytest_runtest_logreport(report):
    REPORTS.append({'nodeid': report.nodeid, 'phase': report.when,
                    'seconds': report.duration, 'outcome': report.outcome})


def pytest_sessionfinish(session, exitstatus):
    STOP.set()
    THREAD.join(timeout=1)
    data = {'exitstatus': int(exitstatus), 'reports': REPORTS,
            'imports': IMPORTS,
            'loaded_edited_modules': sorted(n for n in sys.modules if n in {
                'prismaquant.glm_mtp', 'prismaquant.glm_mtp_loading',
                'prismaquant.glm_mtp_quantum', 'prismaquant.calibration_data',
                'prismaquant.model_dtypes', 'prismaquant.model_walk'}),
            'samples': [{'phase': p, 'stack': s, 'count': n}
                        for (p, s), n in SAMPLES.most_common()]}
    Path(os.environ['PQ_TIMING_REPORT']).write_text(json.dumps(data, indent=2))
