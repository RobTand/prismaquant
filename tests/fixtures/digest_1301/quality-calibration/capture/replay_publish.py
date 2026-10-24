        buffer = io.BytesIO()
        torch.save(result, buffer)
        raw = buffer.getvalue()
        target = root / "routing_boundary.pt"
        if not publish_new_bytes(target, raw):
            raise ValueError("routing capture output already exists")
        receipt = {"schema": "prismaquant.glm_routing_replay_receipt.v1", "status": "complete",
                   "spec": spec_binding, "boundary": {"path": str(target), "sha256": hashlib.sha256(raw).hexdigest()},
                   "metadata": result["metadata"],
                   # Absent at PyTorch's default (PQ #1028).
                   **bf16_reduction_stamp()}
        if not publish_new_bytes(root / "receipt.json", (json.dumps(receipt, sort_keys=True) + "\n").encode()):
            raise ValueError("routing capture receipt already exists")
        print(json.dumps({"status": "complete", "boundary": receipt["boundary"]}, sort_keys=True))
