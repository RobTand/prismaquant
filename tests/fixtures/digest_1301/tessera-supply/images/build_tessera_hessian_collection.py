# Original source fragments from 6d8288adc94b3c4b624c143ed372fb6513f32784:tools/build_tessera_hessian_collection.py
# Domain dependencies are injected by the behavioral tests.
from __future__ import annotations
import json
import hashlib
from pathlib import Path

def publish_collection(paths, output: str | Path) -> dict:
    from tessera import hessian_capture as reader
    from tessera.hessian_capture import ReferenceHessianCollection

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Hessian collection output exists: {output}")
    document = collection_document(paths)
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the exact bytes under the producer's reader before publication.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".collection.references.json") as temp:
        temp.write(raw)
        temp.flush()
        with ReferenceHessianCollection(temp.name, canonical_capture=CANONICAL_CAPTURE) as checked:
            if sorted(checked) != document["units"]:
                raise ValueError("Tessera reader disagrees with the authored collection")
    if not publish_new_bytes(output, raw):
        raise FileExistsError(f"Hessian collection output exists: {output}")
    with ReferenceHessianCollection(output, canonical_capture=CANONICAL_CAPTURE) as checked:
        return {"path": str(output.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                "capture_sha256": document["capture_sha256"],
                "reference_binding": checked.binding(),
                "units": len(checked), "reader_file": str(Path(reader.__file__).resolve()),
                "reader_sha256": hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest()}
