"""Existing-render adapters and the single authoritative static scale reduction.

No choice of candidate, interpolation, encoding, timer pricing or seal is made.
An unreadable wire, absent actual scale or incomplete declared cohort is refused.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from mixed_injection import UnitSpec, StaticScale, CONTRACT_FAMILY, T4, T8, BF16

# Legacy accepted range/header owners are standalone modules, not a package.
# Make their bundled directory discoverable without overriding a parent-selected
# numerical PrismaQuant root or replacing an already imported reader owner.
_accepted_helpers = str(Path(__file__).resolve().parent / "accepted")
if _accepted_helpers not in sys.path:
    sys.path.append(_accepted_helpers)


def resolve_served_scales(specs, *, expected_cohorts=None, raw_diagnostic=False):
    """Resolve pinned native routed group G from every actual member scalar.

    Pinned nvfp4_moe_route.py:655-670 executes FP32 1/max(1/raw_G),
    separately across all loaded gate/up projections and all loaded down roles.
    Dense/shared fused module G is read directly from its exported module tensor.
    expected_cohorts: {group: set(qname)}; supplying incomplete metadata refuses.
    raw_diagnostic=True is named diagnostic-only, never primary served pricing.
    """
    specs = list(specs)
    groups = {}
    for spec in specs:
        if spec.contract == T4:
            groups.setdefault(spec.scale.group, []).append(spec)
    expected_cohorts = expected_cohorts or {}
    evidence = []
    resolved = {}
    for group, members in groups.items():
        names = tuple(sorted(s.qname for s in members))
        if len(set(names)) != len(names):
            raise ValueError(group + ": duplicate static cohort members")
        if group in expected_cohorts and set(names) != set(expected_cohorts[group]):
            raise ValueError(group + ": static scale cohort is incomplete or contains extras")
        if len({s.kind == "routed" for s in members}) != 1:
            raise ValueError(group + ": static cohort mixes routed and nonrouted modules")
        routed = members[0].kind == "routed"
        raw = torch.tensor([s.scale.raw for s in members], dtype=torch.float32)
        if not bool((torch.isfinite(raw) & (raw > 0)).all()):
            raise ValueError(group + ": invalid actual FP32 scale values")
        if routed:
            effective = float(raw.reciprocal().max().reciprocal())
            reduction = "FP32 reciprocal(max(reciprocal(raw G))); pinned native routed group"
        else:
            if len({s.scale.source for s in members}) != 1 or not bool((raw == raw[0]).all()):
                raise ValueError(group + ": fused module must name its one actual exported G tensor")
            effective = float(raw[0])
            reduction = "actual exported fused module scalar, no cross-module reduction"
        for s in members:
            resolved[s.qname] = replace(s, scale=replace(s.scale,
                effective=s.scale.raw if raw_diagnostic else effective, members=names))
        evidence.append({"group": group, "reduction": reduction, "member_count": len(members),
            "members": names, "raw": {s.qname: asdict(s.scale) for s in members},
            "effective": effective, "raw_diagnostic_only": raw_diagnostic})
    if set(expected_cohorts) - set(groups):
        raise ValueError("Declared static cohorts have no supplied member scales")
    return [resolved.get(s.qname, s) for s in specs], evidence


class ExportScales:
    """Read exact scale tensors through the accepted staged/range reader.

    Resolves dense/shared fused membership from actual config_groups schemes,
    not from a guessed gate_up name. Routed tensors bind each expert and role.
    Scalar reads are retained metadata facts, not a new activation/weight cache.
    """
    def __init__(self, root):
        from g3_residency import read_file
        self.root = Path(root)
        self.index = json.loads(read_file(self.root / "model.safetensors.index.json"))["weight_map"]
        self.config = json.loads(read_file(self.root / "config.json"))
        self.headers, self.header_sizes, self.facts = {}, {}, {}
        self.module_for_member = {}
        self.roles_for_module = {}
        groups = self.config.get("quantization_config", {}).get("config_groups", {})
        for block in groups.values():
            scheme = block.get("scheme", {})
            for target in block.get("targets", []):
                members = scheme.get("roles", [])
                self.roles_for_module[target] = {str(name): int(rows) for name, rows in members}
                if members:
                    prefix = target.rsplit(".", 1)[0]
                    for name, _rows in members:
                        self.module_for_member[prefix + "." + name] = target
                else:
                    self.module_for_member[target] = target
        self.reads = []

    def tensor_location(self, tensor):
        from g3_readset import shard_header
        if tensor not in self.index:
            raise ValueError("Actual exported tensor is absent: " + tensor)
        shard = self.index[tensor]
        if shard not in self.headers:
            self.headers[shard], self.header_sizes[shard] = shard_header(self.root / shard, with_size=True)
        row = self.headers[shard][tensor]
        return self.root / shard, row

    def scalar(self, tensor):
        from g3_lib import read_range
        if tensor not in self.facts:
            path, row = self.tensor_location(tensor)
            if math.prod(row["shape"]) != 1 or row["dtype"] != torch.float32:
                raise ValueError(tensor + ": static G must be one actual FP32 scalar")
            raw = read_range(str(path), row["offset"], row["bytes"])
            value = float(torch.frombuffer(bytearray(raw), dtype=torch.float32)[0])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(tensor + ": invalid actual static G")
            fact = {"path": str(path), "tensor": tensor, "offset": row["offset"],
                    "bytes": row["bytes"], "sha256": hashlib.sha256(raw).hexdigest(), "value": value}
            self.facts[tensor] = fact
            self.reads.append(fact)
        return self.facts[tensor]

    def scale_for(self, row):
        qname = row["qname"]
        if row["kind"] == "routed":
            tensor = qname + ".input_global_scale"
            stage = "w2" if row["role"] == "down_proj" else "w13"
            group = qname.split(".experts.")[0] + ".experts." + stage
        else:
            module = self.module_for_member.get(qname)
            if module is None:
                raise ValueError(qname + ": scale export config does not bind its fused module")
            tensor = module + ".trellis_input_global_scale"
            group = module
        fact = self.scalar(tensor)
        return StaticScale(fact["value"], fact["value"], str(self.root) + "::" + tensor, group)

    def layer_specs(self, rows, *, family="T4"):
        """Complete actual static cohort for parent A4 band stream, unselected yet."""
        rows = list(rows)
        if not rows or len({int(r["layer"]) for r in rows}) != 1:
            raise ValueError("Static cohort must contain exactly one complete layer")
        routed = [r for r in rows if r["kind"] == "routed"]
        if routed:
            count = self.config.get("text_config", self.config).get("n_routed_experts")
            expected_keys = {(e, role) for e in range(count or 0) for role in ("gate_proj", "up_proj", "down_proj")}
            keys = {(r["expert"], r["role"]) for r in routed}
            if not count or keys != expected_keys or len(keys) != len(routed):
                raise ValueError("Static routed cohort omits or duplicates artifact experts/projections")
        for kind in ("shared", "dense"):
            members = [r for r in rows if r["kind"] == kind]
            if members and ({r["role"] for r in members} != {"gate_proj", "up_proj", "down_proj"} or len(members) != 3):
                raise ValueError("Static nonrouted cohort is not the complete fused module roster")
        specs = [UnitSpec(r["qname"], r["kind"], r["role"], r["expert"], family,
                          T4, self.scale_for(r)) for r in rows]
        expected = {}
        for spec in specs:
            expected.setdefault(spec.scale.group, set()).add(spec.qname)
        return resolve_served_scales(specs, expected_cohorts=expected)


def selected_specs(rows, key, scales=None):
    """Adapt materialized G3 rows; each SOURCE/T4/T8/BF16 role remains explicit.

    For a T4 routed member, obtain the effective group scalar from the complete
    scale artifact cohort on this layer, then retain that full raw member scope.
    The candidate may select a subset; this is a diagnostic, not mixed-expert
    kernel admission. Non-T4 rows never acquire static scale metadata.
    """
    complete, evidence = {}, []
    if any(r.get(key + "_contract") == T4 for r in rows):
        if scales is None:
            raise ValueError("Selected static T4 entries require an actual scale export")
        complete_specs, evidence = scales.layer_specs(rows)
        complete = {s.qname: s for s in complete_specs}
    specs = []
    for row in rows:
        contract = row.get(key + "_contract")
        if contract not in CONTRACT_FAMILY:
            raise ValueError(row["qname"] + ": unsupported selected activation contract " + str(contract))
        specs.append(UnitSpec(row["qname"], row["kind"], row["role"], row["expert"],
            CONTRACT_FAMILY[contract], contract, complete[row["qname"]].scale if contract == T4 else None))
    # A fused pair shares one quantization when contracts match. Mixed-role
    # diagnostics split output halves in the packed tap, never quantize twice.
    groups = {}
    for s in specs:
        if s.role in ("gate_proj", "up_proj"):
            groups.setdefault((s.kind, s.expert), []).append(s)
    for group, pair in groups.items():
        if len(pair) != 2:
            raise ValueError("Selected fused gate/up pair is incomplete: " + str(group))
    return specs, evidence


def attach_static_export(manifest, root):
    """Manifest adapter: bind readable existing T4 entries to actual scale refs.

    Operates only on already-present formats and picks. No T4 wire is invented,
    no entry is removed, no candidate is selected, and missing scales refuse.
    """
    reader = ExportScales(root)
    count = 0
    for row in manifest["rows"]:
        for option in row.get("formats", {}).values():
            if option.get("contract") == T4:
                scale = reader.scale_for(row)
                option["static_scale"] = asdict(scale)
                count += 1
    manifest["mixed_static_export"] = str(root)
    return {"static_entries_bound": count, "scales_read": reader.reads,
            "limit": "Actual static metadata only; no format selection or timing admission"}



def read_unit_blob(row, *, expected_shape, export=None, location=None, roots=None, root=None, data=None):
    """ONE role-bound blob/extent API for canonical and manifest wire reads.

    export is an ExportScales reader over the actual export (its wire lookup is
    also valid for nonstatic families). location/roots select an existing G3
    manifest range. data optionally supplies bytes already read by WireReader.
    The returned bare unit blob has exact requested [out,in] geometry verified
    by public parse_unit_metadata, with no weight-plane expansion. Framing uses
    the accepted unwrap_members/fused_header/member_location owners, never a
    foreign-magic suppression. No wire or missing role is synthesized.
    """
    from g3_lib import read_range, member_location, unwrap_members, fused_header
    from tessera.unit_artifact import parse_unit_metadata
    role = row["role"]
    if not isinstance(role, str) or not role:
        raise ValueError("The requested wire role is empty")
    shape = tuple(expected_shape)
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Requested wire geometry must be positive integral [out,in]")
    path, offset, length, tensor = None, 0, None, None
    config_roles = {}
    declared_member = None
    if export is not None:
        if location is not None:
            raise ValueError("Choose canonical export or manifest location, not both")
        if row["kind"] == "routed":
            tensor = row["qname"] + ".wire"
        else:
            module = export.module_for_member.get(row["qname"])
            if module is None:
                raise ValueError(row["qname"] + ": actual export config lacks its wire module")
            config_roles = export.roles_for_module[module]
            if config_roles.get(role) != shape[0]:
                raise ValueError(row["qname"] + ": requested rows differ from actual configured role")
            tensor = module + ".wire_bytes"
        path, extent = export.tensor_location(tensor)
        if extent["dtype"] != torch.uint8:
            raise ValueError(tensor + ": actual wire tensor is not byte storage")
        offset, length = extent["offset"], extent["bytes"]
    elif location is not None:
        if "ranges" in location:
            raise ValueError("EXL3 range framing belongs to its existing decoder, not a Tessera unit")
        declared_member = location.get("member")
        if declared_member is not None and declared_member not in (role, row["qname"]):
            raise ValueError("Manifest member does not bind the requested role")
        root_key = location.get("root")
        actual_root = roots.get(root_key) if roots is not None and root_key is not None else root
        if actual_root is None:
            raise ValueError("Manifest wire must name its actual supplied root")
        path = Path(actual_root) / location["shard"]
        offset, length = member_location(location)
        tensor = location.get("tensor")
    elif data is None:
        raise ValueError("Role-bound unit requires actual wire bytes or an actual location")
    if data is None:
        data = read_range(str(path), offset, length)
    else:
        data = bytes(data)
        if length is not None and len(data) != length:
            raise ValueError("Supplied wire bytes differ from their declared extent")
    length = len(data)
    members = unwrap_members(data)
    header = fused_header(data)
    relative_offset = 0
    member_name = None
    if header is None:
        unit_blob = members[None]
    else:
        header_bytes, heads = header
        wanted = declared_member or role
        if wanted in members:
            member_name = wanted
        elif row["qname"] in members:
            member_name = row["qname"]
        elif len(members) == 1:
            (only,) = members
            canonical_role_binding = tensor == row["qname"] + ".wire"
            singleton_config_binding = set(config_roles) == {role}
            if only in {"gate_proj", "up_proj", "down_proj"} or not (canonical_role_binding or singleton_config_binding):
                raise ValueError("Singleton fused wire does not unambiguously bind the requested role")
            member_name = only
        else:
            raise ValueError("Multi-member fused wire lacks the exact requested role/member")
        cursor = header_bytes
        for name, declared_rows, size in heads:
            if name == member_name:
                if declared_rows != shape[0]:
                    raise ValueError("Fused member rows differ from requested Linear geometry")
                relative_offset = cursor
                break
            cursor += size
        unit_blob = members[member_name]
    metadata = parse_unit_metadata(unit_blob, device="cpu")
    actual_shape = metadata.rows, metadata.columns
    if actual_shape != shape:
        raise ValueError("Actual unit geometry differs from requested role: " + str(actual_shape) + " versus " + str(shape))
    fact = {"qname": row["qname"], "role": role, "tensor": tensor,
        "container_path": str(path) if path is not None else None,
        "container_offset": offset, "container_bytes": length,
        "container_sha256": hashlib.sha256(data).hexdigest(),
        "fused_members": [name for name in members if name is not None],
        "member": member_name, "member_relative_offset": relative_offset,
        "unit_offset": offset + relative_offset, "unit_bytes": len(unit_blob),
        "unit_sha256": hashlib.sha256(unit_blob).hexdigest(), "shape": list(actual_shape),
        "configured_roles": config_roles, "format": metadata.role_facts()}
    return unit_blob, fact
