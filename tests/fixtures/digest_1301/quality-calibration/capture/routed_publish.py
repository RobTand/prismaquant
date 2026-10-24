    def publish(layer, result):
        # Exercise the actual intake for every file before it is published.
        transported = {"source": "routed_boundary_capture",
                       "boundary_metadata": result["metadata"], **result["tensors"]}
        routed_boundary_inputs(transported, calibration_receipt=calibration,
            capture_manifest=manifest, device="cpu", source_model_identity=source)
        buffer = io.BytesIO()
        torch.save(result, buffer)
        raw = buffer.getvalue()
        target = root / f"layer-{layer:03d}.pt"
        if not publish_new_bytes(target, raw):
            raise ValueError(f"routed capture output already exists: {target}")
        entries.append({"layer": layer, "unit": result["metadata"]["unit"],
                        "path": str(target), "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw)})
        report("capture", len(entries), unit=result["metadata"]["unit"])
        print(json.dumps({"captured": layer, "count": len(entries),
                          "elapsed_s": time.monotonic() - started}), flush=True)
