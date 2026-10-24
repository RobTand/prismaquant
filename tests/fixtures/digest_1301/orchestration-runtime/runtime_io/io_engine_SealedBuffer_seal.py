# Original consumer excerpt; dependencies are injected by compatibility tests.
def seal(self) -> str:
    """Hash the bytes read, drop the writable mapping and seal; the SHA-256 hex."""
    digest = hashlib.sha256()
    if self._map is not None:
        view = memoryview(self._map)
        try:
            digest.update(view)
        finally:
            view.release()
        self._map.close()
        self._map = None
    fcntl.fcntl(self._fd, _F_ADD_SEALS, _SEALS)
    return digest.hexdigest()
