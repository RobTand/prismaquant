import sys
from types import SimpleNamespace
from unittest.mock import Mock

from prismaquant import digests
from tools import profile_produced_group_lookup as lookup


def test_lookup_report_uses_indent1_profile(monkeypatch, capsysbinary):
    expected = rb'''{
 "groups": 1,
 "per_group": 1,
 "entries": 1,
 "windows": 1,
 "produced_group_for": {
  "wall_s_total": 2.0,
  "wall_s_per_window": 2.0,
  "profile_head": [
   "caf\u00e9"
  ]
 },
 "reference_index_control": {
  "wall_s_total": 1.0,
  "wall_s_per_window": 1.0,
  "profile_head": [
   "caf\u00e9"
  ]
 },
 "ratio_to_dict_control": 2.0,
 "window_payload_bytes": 16777216
}
'''
    monkeypatch.setattr(sys, "argv", ["lookup", "--groups", "1", "--per-group", "1", "--windows", "1"])

    def stats(profiler, *, stream):
        return Mock(**{"sort_stats.return_value.print_stats.side_effect":
                       lambda count: stream.write("café\n")})

    monkeypatch.setattr(lookup, "pstats", SimpleNamespace(Stats=stats))
    calls = []
    real_text = digests.JsonProfile.text

    def observe_text(self, value):
        text = real_text(self, value)
        if self is digests.INSERTION_ASCII_INDENT1_LAX:
            calls.append(self)
            return marker.decode("ascii") + text
        return text

    monkeypatch.setattr(digests.JsonProfile, "text", observe_text)
    for marker in (b"", b" "):
        ticks = iter((1.0, 3.0, 4.0, 5.0))
        monkeypatch.setattr(lookup, "time", SimpleNamespace(perf_counter=ticks.__next__))
        calls.clear()
        lookup.main()
        assert capsysbinary.readouterr().out == marker + expected
        assert calls == [digests.INSERTION_ASCII_INDENT1_LAX], "The report must use the indent-1 profile once."
