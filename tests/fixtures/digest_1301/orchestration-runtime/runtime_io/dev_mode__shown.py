# Original consumer excerpt; dependencies are injected by compatibility tests.
def _shown(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= _SHOWN_CHARS:
        return text
    try:
        encoded = json.dumps(value, sort_keys=True, default=repr).encode()
    except (TypeError, ValueError):
        encoded = text.encode()
    digest = bytes_sha256hex(encoded)[:16]
    return f"{text[:_SHOWN_CHARS]}... ({len(text)} chars, sha256 {digest}...)"
