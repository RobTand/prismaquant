"""Existing render adapters. Metadata only persists; decoded weights are layer-scoped.

A8S uses the current pick's wire/SOURCE contract. EXL3 uses the complete
accepted G3 v3 manifest, four real tensor ranges and accepted exl3_torch
operator decode; nonrouted EXL3 is source passthrough. A4 uses the actual
canonical export config's fused membership, shared ExportScales and its
complete layer served-scale reduction. No new encoder or residency owner.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

ROLES = {"gate_proj", "up_proj", "down_proj"}

@contextmanager
def source_reader(args):
    """Route checkpoint reads through the accepted source lease owner."""
    from g3_readset import SourceReads
    from prismaquant import layer_streaming, streaming_model
    reader = SourceReads(args.source_root, ["null"], {layer: [] for layer in range(45)})
    previous = (layer_streaming._source_safe_open, layer_streaming._source_json,
                streaming_model.safe_open, streaming_model._source_json)
    class Slice:
        def __init__(self, opened, name):
            self.opened, self.name = opened, name
        def get_shape(self):
            return self.opened.header[self.name]["shape"]
        def get_dtype(self):
            from safetensors.torch import _TYPES
            dtype = self.opened.header[self.name]["dtype"]
            return next(name for name, value in _TYPES.items() if value == dtype)
        def __getitem__(self, key):
            return self.opened.get_tensor(self.name)[key]
    def opened(path, **kwargs):
        value = reader.open(path, **kwargs)
        if not hasattr(value,"get_slice"):
            value.get_slice = lambda name: Slice(value,name)
        return value
    try:
        reader.install()
        layer_streaming._source_safe_open = opened
        streaming_model.safe_open = opened
        streaming_model._source_json = reader.read_json
        yield reader
    finally:
        (layer_streaming._source_safe_open, layer_streaming._source_json,
         streaming_model.safe_open, streaming_model._source_json) = previous


class ExistingRenders:
    def __init__(self, args):
        from g3_residency import read_file
        from mixed_manifest import ExportScales
        self.args = args
        self.a4 = ExportScales(args.a4_root)
        manifest = json.loads(read_file(args.manifest))
        reference = json.loads(read_file(args.reference_manifest))
        self.rows = self._unique(manifest["rows"], "A8S manifest")
        self.references = self._unique(reference["rows"], "EXL3 reference manifest")
        if set(self.rows) != set(self.references):
            raise ValueError("Current A8S and complete EXL3 manifest rosters differ")
        for name, row in self.rows.items():
            if row["role"] not in ROLES or row["kind"] not in {"routed", "shared", "dense"}:
                raise ValueError("Manifest includes non-MLP pricing unit: " + name)
            if row["kind"] == "routed" and "EXL3" not in self.references[name]["formats"]:
                raise ValueError("Missing real EXL3 operator for " + name)
        self.source_index = json.loads(read_file(Path(args.source_root) / "model.safetensors.index.json"))["weight_map"]
        self.source_headers = {}
        self.layer_facts = {}
        self.read_ranges = []
        from menu_contract import extend_rows
        extend_rows(self, args)
        self.t4_units = {}
        if args.t4_inventory:
            document = json.loads(read_file(args.t4_inventory))
            if document.get("schema") != "pact.t4_body_inventory.v4":
                raise ValueError("The actual T4 inventory schema differs")
            self.t4_units = self._unique(document["units"], "T4 inventory")
            mlp_names = {name for name,row in self.rows.items() if row["kind"] in {"routed","shared","dense"}}
            if set(self.t4_units) != mlp_names:
                raise ValueError("The actual T4 inventory and MLP populations differ")
            self.menu_gaps.append({"family":"T4","body":"WINDOW","reason":document["missing_window_scope"]})

    @staticmethod
    def _unique(rows, label):
        result = {row["qname"]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError(label + " duplicates units")
        return result

    def complete_layer(self, layer):
        rows = [r for r in self.rows.values() if r["layer"] == layer and r["kind"] in {"routed", "shared", "dense"}]
        if not rows:
            raise ValueError("Layer has no complete MLP roster")
        return sorted(rows, key=lambda r: r["qname"])

    def static_layer(self, layer):
        if layer not in self.layer_facts:
            specs, evidence = self.a4.layer_specs(self.complete_layer(layer))
            self.layer_facts[layer] = ({s.qname: s for s in specs}, evidence)
        return self.layer_facts[layer]

    def tensor_range(self, root, name, index, headers):
        from g3_readset import shard_header
        shard = index[name]
        if shard not in headers:
            headers[shard] = shard_header(Path(root) / shard)
        return Path(root) / shard, headers[shard][name]

    def read_source(self, row, device):
        import torch
        from g3_lib import read_range
        path, extent = self.tensor_range(self.args.source_root, row.get("checkpoint", row["qname"] + ".weight"),
                                        self.source_index, self.source_headers)
        raw = read_range(str(path), extent["offset"], extent["bytes"])
        if len(raw) != extent["bytes"]:
            raise ValueError("Truncated source weight")
        value = torch.frombuffer(bytearray(raw), dtype=extent["dtype"]).reshape(extent["shape"])
        return value.to(device), {"path": str(path), "offset": extent["offset"], "bytes": extent["bytes"]}

    def options(self, row, *, include_diagnostics=True):
        """Metadata plans only; no fractional render is produced here."""
        if row["kind"] not in {"routed", "shared", "dense"}:
            return [{"name": "A8S", "family": "BF16", "q256": None, "contract": "bf16_unquantized",
                "passthrough": True, "weight_reference_definition": "source_bf16"}] + list(self.extra_options.get(row["qname"], []))
        specs, _ = self.static_layer(row["layer"])
        static = specs[row["qname"]]
        a4_tensor = (row["qname"] + ".wire" if row["kind"] == "routed" else
                     self.a4.module_for_member[row["qname"]] + ".wire_bytes")
        path, extent = self.a4.tensor_location(a4_tensor)
        a4 = {"name": "A4-q896", "family": "T4", "q256": 896,
              "contract": "e2m1_group16_ue4m3_static", "diagnostic_only": True,
              "wire": {"path": str(path), "offset": extent["offset"], "bytes": extent["bytes"],
                       "member": None if row["kind"] == "routed" else row["role"]},
              "scale": asdict(static.scale)}
        if self.t4_units:
            actual = self.t4_units[row["qname"]]
            a4["q256"] = actual["q256"]
            a4["body"] = actual["body"]
            a4["outer_scheme"] = actual["plane"]
            if float(a4["scale"]["effective"]) != float(actual["effective_served_scale"]):
                raise ValueError("The actual served T4 scales differ between current readers")
        pick = row["pick"]["a8s"]
        if pick == "SOURCE":
            a8 = {"name": "A8S", "family": "BF16", "q256": None,
                  "contract": "bf16_unquantized", "passthrough": True}
        else:
            option = row["formats"][pick]
            if option["contract"] not in ("fp8_per_token_dynamic", "bf16_unquantized"):
                raise ValueError("A8S pick has an unexpected served contract")
            a8 = {"name": "A8S", "family": "T8" if option["contract"] == "fp8_per_token_dynamic" else "T16",
                  "q256": int(pick.rsplit("_R", 1)[1]), "contract": option["contract"],
                  "location": option["wire"], "rendered_shape": option["rendered_shape"],
                  "recorded_rendered_sha256": option.get("rendered_sha256")}
        ex = self.references[row["qname"]]
        if row["kind"] == "routed":
            option = ex["formats"]["EXL3"]
            reference = {"name": "EXL3", "family": "EXL3", "q256": None,
                         "contract": "bf16_unquantized", "diagnostic_only": True,
                         "location": option["wire"], "rendered_shape": option["rendered_shape"],
                         "recorded_rendered_sha256": option.get("rendered_sha256")}
        else:
            reference = {"name": "EXL3", "family": "EXL3", "q256": None,
                         "contract": "bf16_unquantized", "passthrough": True, "diagnostic_only": True}
        result = [a8, a4, reference]
        if include_diagnostics:
            for name, option in row["formats"].items():
                if name.startswith("v1::") and name.rsplit("_R", 1)[-1] in {"832", "1088"}:
                    result.append({"name": "allrates-" + name, "family": "T8",
                        "q256": int(name.rsplit("_R", 1)[1]), "contract": option["contract"],
                        "location": option["wire"], "rendered_shape": option["rendered_shape"],
                        "diagnostic_only": True, "quality_chord_check_only": True})
        for option in result:
            option["unit_row"] = {name: row[name] for name in ("qname", "kind", "role", "expert", "layer")}
            option["weight_reference_definition"] = ("legacy_T16_folded_bf16"
                if option["family"] == "T16" else "source_bf16" if option.get("passthrough") else "decoded_bf16")
            if option["family"] == "T16":
                option["canonical_T16_priceable"] = False
        from menu_contract import all_options
        return all_options(self, row, result)

    def read_location(self, option):
        """Read the stored EXL3 tensor ranges."""
        from g3_lib import read_range
        if option["family"] != "EXL3":
            raise ValueError("Use the shared adapter for Tessera units")
        location = option["location"]
        if "ranges" not in location:
            raise ValueError("The EXL3 source lacks its stored tensor ranges")
        root = Path(self.args.exl3_root)
        raw = b"".join(read_range(str(root / shard), offset, size) for shard, offset, size in location["ranges"])
        if len(raw) != location["member_bytes"] or hashlib.sha256(raw).hexdigest() != location["wire_sha256"]:
            raise ValueError("The EXL3 bytes fail their own length or digest check")
        return raw

    def read_weight(self, option, shape):
        """Read one actual weight through its existing byte owner."""
        if option["family"] == "EXL3":
            return self.read_location(option), None
        from mixed_manifest import read_unit_blob
        if "wire" in option and option["name"] != "A4-q896":
            from g3_lib import read_range
            wire = option["wire"]
            raw = read_range(wire["path"], wire["offset"], wire["bytes"])
            if len(raw) != wire["bytes"] or wire.get("sha256") and hashlib.sha256(raw).hexdigest() != wire["sha256"]:
                raise ValueError("The actual sample bytes fail their own length or digest")
            return read_unit_blob(option["unit_row"], expected_shape=tuple(shape), data=raw)
        kwargs = {"export": self.a4} if "wire" in option else {
            "location": option["location"],
            "roots": {"a8": self.args.a8_root, "t8r": self.args.t8r_root},
            "root": self.args.a8_root}
        return read_unit_blob(option["unit_row"], expected_shape=tuple(shape), **kwargs)

    @staticmethod
    def validate_decoded(option, value, source):
        """Keep the BF16 weight reference and reject invalid planes."""
        import torch
        value = value.to(torch.bfloat16)
        if value.shape != source.shape or value.device != source.device:
            raise ValueError("Actual source and decoded weight geometry or device differ")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("The actual decoded weights are not finite")
        expected = option.get("recorded_rendered_sha256")
        if expected:
            from g3_pq_policy.dev_mode import seal_check, NOT_COMPUTED
            seal_check("previously recorded decoded weight", expected, NOT_COMPUTED, where=option["name"])
        return value

    def decode(self, option, source, *, device, prefetched=None):
        """Use the same checks after synchronous or prefetched byte reads."""
        if option["family"] == "T16":
            from canonical_weight import read_canonical_weight
            raw, fact = prefetched if prefetched is not None else self.read_weight(option, source.shape)
            if fact["format"]["q256"] != option["q256"]:
                raise ValueError("The requested canonical rate differs from the actual wire")
            option["body"] = fact["format"]["body"]
            option["outer_scheme"] = fact["format"]["plane"]
            option["wire_facts"] = fact
            return read_canonical_weight(raw, device=device, expected_shape=tuple(source.shape))
        if option.get("passthrough"):
            return self.validate_decoded(option, source, source)
        raw, fact = prefetched if prefetched is not None else self.read_weight(option, source.shape)
        if fact is not None:
            actual = fact["format"]
            if actual["q256"] != option["q256"] or option.get("body", actual["body"]) != actual["body"]:
                raise ValueError("The requested rate or body differs from the actual wire")
            grid = actual["grid"]
            if option["family"] == "T8" and grid != "E4M3" or option["family"] == "T4" and not grid.startswith("E2M1"):
                raise ValueError("The requested family differs from the actual wire grid")
            option["body"] = actual["body"]
            option["outer_scheme"] = actual["plane"]
        if option["family"] == "EXL3":
            from exl3_torch import decode_wire
            out_features, in_features = option["rendered_shape"]
            value = decode_wire(raw, in_features, out_features, device=device)
        else:
            from tessera.unit_artifact import read_unit_artifact
            option["wire_facts"] = fact
            value = read_unit_artifact(raw, device=device)
        return self.validate_decoded(option, value, source)

    @contextmanager
    def retained_decoded_window(self, sources, options_by_name, *, max_resident_bytes, device, guard):
        """Admit the complete render lifetime through ProductionWeightCache."""
        import torch
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import nullcontext
        from canonical_weight import CanonicalWeight
        from prismaquant.production_weight_cache import ProductionWeightCache
        if type(max_resident_bytes) is not int or max_resident_bytes <= 0:
            raise ValueError("A decoded render lifetime needs its positive admitted byte budget")
        planned = 0
        for name, options in options_by_name.items():
            source = sources[name]
            if source.ndim != 2:
                raise ValueError("A retained render source must be an actual Linear plane")
            for option in options:
                if not option.get("passthrough"):
                    planned += 2 * source.numel()
                    if option["family"] == "T16":
                        planned += 4 * source.shape[0]
        if planned > max_resident_bytes:
            raise MemoryError("The complete decoded render lifetime exceeds its admitted byte budget")
        cache = ProductionWeightCache(weights={}, levers={}, metadata={"scope": "actual research renders"})
        weights = {name: {"source": source, "options": []} for name, source in sources.items()}
        resolved = []
        def compact(tensor):
            if not tensor.is_contiguous():
                return tensor.contiguous()
            if tensor.storage_offset() or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size():
                return tensor.clone(memory_format=torch.contiguous_format)
            return tensor
        def jobs():
            for name, options in options_by_name.items():
                for option in options:
                    yield name, sources[name], option
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="existing-render-reader") as reads:
                pending, iterator = [], iter(jobs())
                def enqueue():
                    try:
                        name, source, option = next(iterator)
                    except StopIteration:
                        return False
                    future = None if option.get("passthrough") else reads.submit(self.read_weight, option, source.shape)
                    pending.append((name, source, option, future))
                    return True
                for _ in range(2):
                    enqueue()
                while pending:
                    name, source, option, future = pending.pop(0)
                    guard("decode admitted render")
                    prefetched = future.result() if future is not None else None
                    decoded = self.decode(option, source, device=device, prefetched=prefetched)
                    key = (name, "PACT_" + hashlib.sha256(option["name"].encode()).hexdigest().upper())
                    if option.get("passthrough"):
                        resolved.append((name, option, "source", source))
                    elif isinstance(decoded, CanonicalWeight):
                        value_key, scale_key = (name, key[1] + ":VALUES"), (name, key[1] + ":ROW_SCALES")
                        cache.weights[value_key] = compact(decoded.values)
                        cache.weights[scale_key] = compact(decoded.row_scales)
                        resolved.append((name, option, "canonical", (value_key, scale_key)))
                    else:
                        cache.weights[key] = compact(decoded)
                        resolved.append((name, option, "tensor", key))
                    del decoded, prefetched, future
                    enqueue()
            keys = list(cache.weights)
            context = cache.retained_window(keys, max_resident_bytes=max_resident_bytes, max_workers=1) if keys else nullcontext({"resident_bytes": 0})
            with context as resident:
                for name, option, kind, reference in resolved:
                    if kind == "canonical":
                        decoded = CanonicalWeight(cache.get_resident(*reference[0]), cache.get_resident(*reference[1]))
                    elif kind == "tensor":
                        decoded = cache.get_resident(*reference)
                    else:
                        decoded = reference
                    weights[name]["options"].append((option, decoded))
                evidence = {"owner": "ProductionWeightCache", "planned_render_bytes": planned,
                    "admitted_render_bytes": max_resident_bytes, "actual_resident_bytes": resident["resident_bytes"],
                    "keys": [list(key) for key in keys], "no_lazy_file_load": True,
                    "source_passthrough_owner": "existing streamed source cache", "scope": "Persistent render storage, not transient peak qualification."}
                try:
                    yield weights, evidence
                finally:
                    weights.clear()
                    resolved.clear()
                    decoded = None
        finally:
            weights.clear()
            resolved.clear()
            cache.weights.clear()

    def capture_probe(self, row):
        import io
        import torch
        from g3_residency import read_file
        directory = Path(self.args.capture_root) / "layers" / ("L%03d" % row["layer"])
        meta = json.loads(read_file(directory / "manifest.json"))
        rec = meta["units"][row["qname"]]["heldout"]
        raw = read_file(Path(self.args.capture_root) / rec["file"])
        if len(raw) != rec["bytes"] or hashlib.sha256(raw).hexdigest() != rec["sha256"]:
            raise ValueError("CPU probe capture fails its own length/digest")
        capture = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if capture["name"] != row["qname"] or capture["role"] != "heldout":
            raise ValueError("CPU probe capture unit/role differ")
        x = capture["inputs"][:self.args.cpu_rows]
        if x.ndim != 2 or not len(x):
            raise ValueError("CPU probe has no actual retained input rows")
        return x, {"capture": rec, "represented_rows": int(capture["count"]),
                   "retained_rows": len(capture["inputs"]), "consumed_rows": len(x),
                   "not_full_population": True}
