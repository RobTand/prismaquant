# Original consumer excerpt; dependencies are injected by compatibility tests.
def _emit(self, now, *, final):
    try:
        elapsed = now - self._started
        interval = now - self._last_clock
        interval_bytes = self.bytes - self._last_bytes
        interval_rate = interval_bytes / interval if interval > 0 else None
        mean_rate = self.bytes / elapsed if elapsed > 0 else None
        rate = interval_rate if interval_rate else mean_rate
        remaining = max(0, self.total_bytes - self.bytes)
        proc = counter_delta(self._read_proc_io(), self._proc_start)
        record = {
            "schema": READ_RATE_SCHEMA,
            "label": self.label,
            "final": bool(final),
            "entries": self.entries,
            "entries_total": self.total_entries,
            "bytes": self.bytes,
            "bytes_total": self.total_bytes,
            "elapsed_s": elapsed,
            "interval_s": interval,
            "interval_mb_s": (None if interval_rate is None
                              else interval_rate / 1e6),
            "mean_mb_s": None if mean_rate is None else mean_rate / 1e6,
            "eta_s": (0.0 if remaining == 0 else
                      remaining / rate if rate else None),
            "proc_io": proc,
        }
        self._log(f"{READ_RATE_MARKER} "
                  + json.dumps(record, sort_keys=True, allow_nan=False))
        self.lines += 1
    except Exception as exc:  # noqa: BLE001 -- instrumentation only
        try:
            self._log(f"{READ_RATE_MARKER} {self.label}: rate line failed: "
                      f"{_error_text(exc)}")
        except Exception:  # noqa: BLE001
            pass
    self._last_clock = now
    self._last_entries = self.entries
    self._last_bytes = self.bytes
