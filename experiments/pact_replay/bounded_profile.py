"""Capture one science item. Release the profiler before all later items."""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import gc
import json
from pathlib import Path
import runpy
import sys
import time

_CAPTURE = None


class _Capture:
    def __init__(self, output, activities):
        self.output, self.activities = output, activities
        self.capture_items = 0
        self.started_unix = self.ended_unix = None
        self.trace = None

    @contextmanager
    def first_item(self):
        import torch
        self.capture_items = 1
        profiler = torch.profiler.profile(activities=self.activities, record_shapes=True,
            profile_memory=False, with_stack=False)
        try:
            self.started_unix = time.time()
            with profiler:
                yield
            self.ended_unix = time.time()
            trace = self.output / "trace.json"
            profiler.export_chrome_trace(str(trace))
            self.trace = str(trace)
        finally:
            # Torch transition actions form an owner cycle through bound methods.
            del profiler
            gc.collect()


def capture_item():
    """Capture the first item only. The cap applies across the complete entry."""
    if _CAPTURE is None or _CAPTURE.capture_items:
        return nullcontext()
    return _CAPTURE.first_item()


def profile_run(output, entry, entry_args, expected_device):
    """Run an unchanged entry with one bounded diagnostic trace and no tables."""
    import torch
    if expected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The GPU profile requires a live CUDA device")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if expected_device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    capture = _Capture(output, activities)
    global _CAPTURE
    previous, previous_argv = _CAPTURE, sys.argv
    _CAPTURE = capture
    sys.argv = [entry, *entry_args]
    started_unix = time.time()
    try:
        runpy.run_path(entry, run_name="__main__")
    finally:
        _CAPTURE, sys.argv = previous, previous_argv
    ended_unix = time.time()
    summary = {"schema": "pact.bounded_rate_profile.v1", "entry": entry,
        "expected_device": expected_device, "started_unix": started_unix,
        "ended_unix": ended_unix, "entry_wall_seconds": ended_unix - started_unix,
        "capture_item_cap": 1, "capture_items": capture.capture_items,
        "capture_started_unix": capture.started_unix, "capture_ended_unix": capture.ended_unix,
        "trace": capture.trace, "record_shapes": True, "profile_memory": False,
        "with_stack": False, "operator_tables": False, "full_run_operator_totals": False,
        "scope": "The trace covers the first science item only. Entry wall time includes its trace export. No full-run operator totals are available."}
    path = output / "profile-summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"profile_summary": str(path), "profiled_entry_completed": True}), flush=True)
    return summary
