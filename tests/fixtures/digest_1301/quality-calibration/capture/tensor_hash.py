def _hash(tensor):
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
