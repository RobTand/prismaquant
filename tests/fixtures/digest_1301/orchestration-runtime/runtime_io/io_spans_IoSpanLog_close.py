# Original consumer excerpt; dependencies are injected by compatibility tests.
def close(self, span: IoSpan, *, error: BaseException | None = None,
          outcome: str | None = None) -> dict | None:
    """Close ``span``, print its line and return its record.

    ``error`` records a workload exception; ``outcome`` overrides the
    outcome name. Closing a span twice is a no-op that returns ``None``.
    """
    if span.closed:
        return None
    span.closed = True
    try:
        self._open.remove(span)
    except ValueError:
        pass
    try:
        record = self._record(span, error=error, outcome=outcome)
    except Exception as exc:  # noqa: BLE001 -- instrumentation only
        self._emit_failure(span.name, exc)
        return None
    self.records.append(record)
    try:
        self._log(f"{IO_SPAN_MARKER} "
                  + json.dumps(record, sort_keys=True, allow_nan=False,
                               default=str))
    except Exception as exc:  # noqa: BLE001 -- instrumentation only
        self._emit_failure(span.name, exc)
    return record
