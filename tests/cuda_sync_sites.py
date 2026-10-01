"""Count the CUDA host syncs a call makes, and name the line of each.

``torch.cuda.set_sync_debug_mode("warn")`` warns ``SYNC_WARNING`` at every
synchronizing call. The first switch into a debug mode in a process also
warns, once, that the mode is a prototype that "does not yet detect all
synchronizing operations". That notice is not a sync, so only the exact sync
message is counted; matching on "synchroniz" counted it too, and charged it
to whichever call was measured first (PQ #1931).
"""
from __future__ import annotations

import traceback
import warnings

import torch

SYNC_WARNING = "called a synchronizing CUDA operation"


def sync_sites(fn) -> list[str]:
    """Run ``fn`` with CUDA sync warnings on; return the call site of each sync."""
    sites: list[str] = []

    def _record(message, category, filename, lineno, file=None, line=None):
        if SYNC_WARNING not in str(message):
            return
        frames = [f for f in traceback.extract_stack()[:-1]
                  if not f.filename.endswith("warnings.py")]
        sites.append(" <- ".join(
            f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.line}"
            for f in reversed(frames[-3:])))

    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = _record
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return sites
