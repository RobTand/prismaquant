"""A Stage B window's staged-render wait and its unit commit each have a span (PQ #1207).

PQ #1207 attributed 25-30 s of a 45-50 s render window to the unit commit
in ``after_window``, from a py-spy sample of row 42's windows 10 and 11. The
row's own counters could not settle it: the ``window`` span holds the
staged-render wait, the replay and the commit, and ``windows[i].wall_s``
holds only the part between them, so their difference is the wait plus the
commit, about 2 s on rows 38, 39 and 43 and 18-23 s on the two sampled
windows. Each part now has its own child span of the window, so the table
says which one a slow window spent its time in.
"""
from __future__ import annotations

from test_stage_b_tail_1187 import _quantum


def test_each_window_spans_its_wait_and_its_commit(tmp_path, monkeypatch):
    payload, _record, block = _quantum(tmp_path, monkeypatch)
    spans = block["io_spans"]
    windows = [span for span in spans if span["span"] == "window"
               and span["outcome"] == "ok"]
    assert windows, [span["span"] for span in spans]
    units = 0
    for window in windows:
        index = window["window"]
        inside = [span for span in spans if span.get("window") == index
                  and span["parent"] == "window"]
        waits = [span for span in inside if span["span"] == "window-wait"]
        commits = [span for span in inside if span["span"] == "commit"]
        assert len(waits) == 1 and len(commits) == 1, [span["span"] for span in inside]
        (wait,), (commit,) = waits, commits
        assert wait["outcome"] == "ok" and commit["outcome"] == "ok"
        assert window["start_unix"] <= wait["start_unix"] <= wait["end_unix"] \
            <= commit["start_unix"] <= commit["end_unix"] <= window["end_unix"]
        # The commit names the units it made durable in this window.
        assert commit["units"] == len(block["windows"][index]["names"])
        units += commit["units"]
    assert units == len(payload["costs"])
