#!/usr/bin/env python3
"""Build the CPU-only COLD-EXPERT ATLAS for a sharded MoE checkpoint.

This is an offline screening tool. It deliberately reads only explicitly
requested tensors from safetensors shards and never creates CUDA tensors.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open


DEFAULT_MODEL = Path("/home/rob/dq-runs/dsv4-flash-0731/source")
DEFAULT_COST_TABLE = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/artifacts/cost_full.pkl"
)
DEFAULT_OUTPUT_DIR = Path("atlas-out")
EXPERT_QNAME_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)$"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted local experiment artifact


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _first_numeric(mapping: Any, key: str) -> float | int | None:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    for child in mapping.values():
        if isinstance(child, dict):
            value = child.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
    return None


def _cost_rows(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return qname-keyed cost rows and any embedded probe stats."""
    if isinstance(payload, dict) and isinstance(payload.get("costs"), dict):
        stats = payload.get("stats", {})
        return payload["costs"], stats if isinstance(stats, dict) else {}
    if isinstance(payload, dict) and all(isinstance(key, str) for key in payload):
        return payload, {}
    if isinstance(payload, list):
        rows: dict[str, Any] = {}
        for row in payload:
            if not isinstance(row, dict) or not isinstance(row.get("qname"), str):
                raise ValueError("list cost tables require dict rows carrying a string qname")
            rows[row["qname"]] = row
        return rows, {}
    raise ValueError("unsupported cost-table payload; expected {'costs': ...}, dict, or rows")


def derive_cold_experts(
    cost_table_path: str | Path,
    *,
    probe_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Derive cold expert groups from a cost pickle and h_trace provenance.

    Portable cost tables may embed ``stats`` or put ``h_trace`` in each cost
    row. The production v2 artifact keeps those stats in its adjacent
    ``probe.pkl``; that file is joined by qname when needed.
    """
    cost_path = Path(cost_table_path)
    costs, embedded_stats = _cost_rows(_read_pickle(cost_path))
    stats = embedded_stats
    needs_external_stats = any(
        EXPERT_QNAME_RE.match(qname)
        and _first_numeric(row, "h_trace") is None
        and _first_numeric(embedded_stats.get(qname, {}), "h_trace") is None
        for qname, row in costs.items()
    )
    resolved_probe: Path | None = None
    if needs_external_stats:
        resolved_probe = Path(probe_path) if probe_path else cost_path.parent / "probe.pkl"
        if not resolved_probe.exists():
            raise ValueError(
                "cost rows do not carry h_trace and no probe provenance exists at "
                f"{resolved_probe}"
            )
        probe_payload = _read_pickle(resolved_probe)
        if not isinstance(probe_payload, dict) or not isinstance(
            probe_payload.get("stats"), dict
        ):
            raise ValueError(f"probe provenance at {resolved_probe} has no stats mapping")
        stats = probe_payload["stats"]

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for qname, cost_row in costs.items():
        match = EXPERT_QNAME_RE.match(qname)
        if match is None:
            continue
        stat_row = stats.get(qname, {}) if isinstance(stats, dict) else {}
        h_trace = _first_numeric(cost_row, "h_trace")
        if h_trace is None:
            h_trace = _first_numeric(stat_row, "h_trace")
        if h_trace is None:
            raise ValueError(f"missing h_trace provenance for {qname}")
        n_rows = _first_numeric(cost_row, "n_activation_rows")
        if n_rows is None:
            n_rows = _first_numeric(stat_row, "n_activation_rows")
        if n_rows is None:
            n_rows = _first_numeric(stat_row, "n_tokens_seen")
        if n_rows is None and float(h_trace) == 0.0:
            n_rows = 0
        grouped[(int(match["layer"]), int(match["expert"]))].append(
            {
                "qname": qname,
                "projection": match["projection"],
                "h_trace": float(h_trace),
                "n_activation_rows": int(n_rows) if n_rows is not None else None,
            }
        )

    cold: list[dict[str, Any]] = []
    projection_order = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
    for (layer, expert_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: projection_order.get(row["projection"], 99))
        if rows and all(row["h_trace"] == 0.0 for row in rows):
            row_counts = [row["n_activation_rows"] for row in rows]
            known_counts = [count for count in row_counts if count is not None]
            cold.append(
                {
                    "layer": layer,
                    "expert_id": expert_id,
                    "qnames": [row["qname"] for row in rows],
                    "h_trace": 0.0,
                    "n_activation_rows": max(known_counts) if known_counts else None,
                    "constituents": rows,
                }
            )
    return cold


def invert_tid2eid(tid2eid: torch.Tensor, num_experts: int | None = None) -> dict[int, list[int]]:
    """Invert a token-id -> expert-id membership map, retaining empty experts."""
    if tid2eid.device.type != "cpu":
        raise ValueError("tid2eid must be CPU-resident")
    if tid2eid.ndim not in (1, 2):
        raise ValueError(f"tid2eid must be 1-D or 2-D, got shape {tuple(tid2eid.shape)}")
    values = tid2eid.to(dtype=torch.int64).tolist()
    memberships = [[value] for value in values] if tid2eid.ndim == 1 else values
    inferred = max((value for row in memberships for value in row), default=-1) + 1
    count = inferred if num_experts is None else num_experts
    inverted = {expert_id: [] for expert_id in range(count)}
    for token_id, expert_ids in enumerate(memberships):
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError(f"token {token_id} has duplicate expert memberships")
        for expert_id in expert_ids:
            if expert_id not in inverted:
                raise ValueError(
                    f"token {token_id} maps to expert {expert_id}, outside [0, {count})"
                )
            inverted[expert_id].append(token_id)
    return inverted


def router_lens_topk(
    gate_rows: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    top_k: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact top-k gate-row/embedding affinities using CPU float32 matmul."""
    if gate_rows.device.type != "cpu" or embeddings.device.type != "cpu":
        raise ValueError("router lens is strictly CPU-only")
    if gate_rows.ndim != 2 or embeddings.ndim != 2:
        raise ValueError("gate_rows and embeddings must both be matrices")
    if gate_rows.shape[1] != embeddings.shape[1]:
        raise ValueError("gate rows and embeddings have different hidden widths")
    if not 0 < top_k <= embeddings.shape[0]:
        raise ValueError("top_k must be in [1, vocab_size]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    values: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    embedding_f32 = embeddings.to(dtype=torch.float32)
    embedding_t = embedding_f32.transpose(0, 1)
    with torch.inference_mode():
        for start in range(0, gate_rows.shape[0], batch_size):
            scores = gate_rows[start : start + batch_size].to(torch.float32) @ embedding_t
            batch_values, batch_indices = torch.topk(
                scores, k=top_k, dim=1, largest=True, sorted=True
            )
            values.append(batch_values)
            indices.append(batch_indices)
    return torch.cat(values, dim=0), torch.cat(indices, dim=0)


class CheckpointReader:
    """Index-driven reader that loads one named CPU tensor at a time."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = payload["weight_map"]

    def tensor(self, name: str) -> torch.Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"tensor {name!r} is absent from the checkpoint index") from exc
        with safe_open(
            str(self.model_dir / shard), framework="pt", device="cpu"
        ) as handle:
            tensor = handle.get_tensor(name)
        if tensor.device.type != "cpu":
            raise RuntimeError(f"CPU-only invariant violated while reading {name}")
        return tensor

    def gate_name(self, layer: int) -> str:
        pattern = re.compile(
            rf"(?:^|\.)layers\.{layer}\..*(?:ffn|mlp)\.gate\.weight$"
        )
        matches = [
            name
            for name in self.weight_map
            if pattern.search(name) and "attn" not in name and not name.startswith("mtp.")
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one learned gate for layer {layer}, found {matches}")
        return matches[0]

    def tid2eid_name(self, layer: int) -> str:
        suffix = f"layers.{layer}.ffn.gate.tid2eid"
        matches = [name for name in self.weight_map if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one tid2eid for layer {layer}, found {matches}")
        return matches[0]

    def embedding_name(self) -> str:
        preferred_suffixes = ("embed_tokens.weight", "embed.weight")
        for suffix in preferred_suffixes:
            matches = [name for name in self.weight_map if name.endswith(suffix)]
            if len(matches) == 1:
                return matches[0]
        raise ValueError("could not identify a unique token embedding tensor in the index")


def _load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer, PreTrainedConfig

    tokenizer_payload = json.loads(
        (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    tokenizer_config = PreTrainedConfig()
    tokenizer_config.tokenizer_class = tokenizer_payload.get(
        "tokenizer_class", "PreTrainedTokenizerFast"
    )
    return AutoTokenizer.from_pretrained(
        model_dir,
        config=tokenizer_config,
        local_files_only=True,
        trust_remote_code=True,
    )


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def run_inventory(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    cold = derive_cold_experts(args.cost_table, probe_path=args.probe)
    counts = Counter(item["layer"] for item in cold)
    default_probe = Path(args.cost_table).parent / "probe.pkl"
    resolved_probe = Path(args.probe) if args.probe else default_probe
    payload = {
        "schema": "prismaquant.cold_expert_atlas.inventory.v1",
        "generated_at": _utc_now(),
        "cost_table": str(Path(args.cost_table).resolve()),
        "probe": str(resolved_probe.resolve()) if resolved_probe.exists() else None,
        "definition": "all constituent expert rows have h_trace == 0",
        "cold_expert_count": len(cold),
        "cold_counts_by_layer": {str(layer): counts[layer] for layer in sorted(counts)},
        "experts": cold,
    }
    path = Path(args.output_dir) / "cold_experts.json"
    _json_dump(path, payload)
    elapsed = time.perf_counter() - started
    print(f"inventory: wrote {len(cold)} cold experts to {path} in {elapsed:.2f}s")
    return path


def run_hash_invert(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    model_dir = Path(args.model)
    reader = CheckpointReader(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    num_experts = int(text_config.get("n_routed_experts", 256))
    tokenizer = _load_tokenizer(model_dir)
    layers = []
    for layer in range(3):
        tensor_name = reader.tid2eid_name(layer)
        inverted = invert_tid2eid(reader.tensor(tensor_name), num_experts=num_experts)
        frequency = Counter(len(token_ids) for token_ids in inverted.values())
        experts = []
        for expert_id, token_ids in inverted.items():
            examples = token_ids[: args.max_examples]
            experts.append(
                {
                    "expert_id": expert_id,
                    "token_count": len(token_ids),
                    "example_token_ids": examples,
                    "example_tokens": [_decode_token(tokenizer, tid) for tid in examples],
                }
            )
        layers.append(
            {
                "layer": layer,
                "gate_tensor": tensor_name,
                "tokens_per_expert_histogram": {
                    str(token_count): expert_frequency
                    for token_count, expert_frequency in sorted(frequency.items())
                },
                "experts": experts,
            }
        )
        print(f"hash-invert: layer {layer} inverted ({sum(map(len, inverted.values()))} tokens)")
    payload = {
        "schema": "prismaquant.cold_expert_atlas.hash_inversion.v1",
        "generated_at": _utc_now(),
        "model": str(model_dir.resolve()),
        "example_cap_per_expert": args.max_examples,
        "note": "Calibration token-id coverage/ranges are intentionally future work.",
        "layers": layers,
    }
    path = Path(args.output_dir) / "hash_layer_token_map.json"
    _json_dump(path, payload)
    print(f"hash-invert: wrote {path} in {time.perf_counter() - started:.2f}s")
    return path


def _norm_summary(norms: torch.Tensor, row_norm: float) -> dict[str, float]:
    norms_f32 = norms.to(torch.float32)
    return {
        "row_norm": row_norm,
        "layer_mean": float(norms_f32.mean()),
        "layer_std": float(norms_f32.std(unbiased=False)),
        "layer_min": float(norms_f32.min()),
        "layer_max": float(norms_f32.max()),
        "percentile": float((norms_f32 <= row_norm).to(torch.float32).mean() * 100.0),
    }


def run_router_lens(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    inventory_path = output_dir / "cold_experts.json"
    if not inventory_path.exists():
        raise FileNotFoundError(f"run inventory first: {inventory_path} does not exist")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    cold_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in inventory["experts"]:
        if int(item["layer"]) >= 3:
            cold_by_layer[int(item["layer"])].append(item)

    model_dir = Path(args.model)
    reader = CheckpointReader(model_dir)
    embedding_name = reader.embedding_name()
    print(f"router-lens: loading CPU embedding tensor {embedding_name}")
    embeddings = reader.tensor(embedding_name).to(torch.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embedding tensor must be 2-D, got {tuple(embeddings.shape)}")
    tokenizer = _load_tokenizer(model_dir)
    results: list[dict[str, Any]] = []
    layer_timings: dict[str, float] = {}

    for layer in sorted(cold_by_layer):
        layer_started = time.perf_counter()
        cold = sorted(cold_by_layer[layer], key=lambda item: int(item["expert_id"]))
        expert_ids = [int(item["expert_id"]) for item in cold]
        gate_name = reader.gate_name(layer)
        gate = reader.tensor(gate_name)
        if gate.ndim != 2 or gate.shape[1] != embeddings.shape[1]:
            raise ValueError(
                f"gate {gate_name} shape {tuple(gate.shape)} is incompatible with "
                f"embedding shape {tuple(embeddings.shape)}"
            )
        if max(expert_ids) >= gate.shape[0]:
            raise ValueError(f"cold expert id exceeds row count for {gate_name}")
        norms = torch.linalg.vector_norm(gate.to(torch.float32), dim=1)
        selected = gate[expert_ids]
        top_values, top_indices = router_lens_topk(
            selected,
            embeddings,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        for row_index, cold_item in enumerate(cold):
            token_ids = top_indices[row_index].tolist()
            scores = top_values[row_index].tolist()
            expert_id = int(cold_item["expert_id"])
            row_norm = float(norms[expert_id])
            results.append(
                {
                    "layer": layer,
                    "expert_id": expert_id,
                    "qnames": cold_item["qnames"],
                    "gate_tensor": gate_name,
                    "gate_row_norm": _norm_summary(norms, row_norm),
                    "top_tokens": [
                        {
                            "token_id": int(token_id),
                            "token": _decode_token(tokenizer, int(token_id)),
                            "score": float(score),
                        }
                        for token_id, score in zip(token_ids, scores, strict=True)
                    ],
                }
            )
        elapsed = time.perf_counter() - layer_started
        layer_timings[str(layer)] = elapsed
        print(
            f"router-lens: layer {layer:02d}, {len(cold):3d} cold experts, "
            f"{elapsed:.2f}s"
        )

    duration = time.perf_counter() - started
    payload = {
        "schema": "prismaquant.cold_expert_atlas.router_lens.v1",
        "generated_at": _utc_now(),
        "model": str(model_dir.resolve()),
        "embedding_tensor": embedding_name,
        "calculation": "float32 CPU gate_row @ embed_tokens.T over the full vocabulary",
        "top_k": args.top_k,
        "expert_batch_size": args.batch_size,
        "cold_expert_count": len(results),
        "layer_wall_seconds": layer_timings,
        "wall_seconds": duration,
        "experts": results,
    }
    path = output_dir / "router_lens.json"
    _json_dump(path, payload)
    print(f"router-lens: wrote {len(results)} experts to {path} in {duration:.2f}s")
    return path


def _script_class(text: str) -> str:
    classes = []
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        for script in (
            "LATIN",
            "CJK",
            "HIRAGANA",
            "KATAKANA",
            "HANGUL",
            "CYRILLIC",
            "ARABIC",
            "HEBREW",
            "DEVANAGARI",
            "THAI",
            "GREEK",
        ):
            if script in name:
                classes.append(script)
                break
    return Counter(classes).most_common(1)[0][0] if classes else "OTHER"


def coherence_score(tokens: Iterable[str]) -> tuple[float, str]:
    """Score simple shared-script or shared-prefix coherence."""
    cleaned = [token.strip().casefold() for token in tokens if token.strip()]
    if not cleaned:
        return 0.0, "no visible tokens"
    scripts = Counter(_script_class(token) for token in cleaned)
    script, script_count = scripts.most_common(1)[0]
    prefixes = Counter(token[:3] for token in cleaned if len(token) >= 3)
    prefix, prefix_count = prefixes.most_common(1)[0] if prefixes else ("", 0)
    script_fraction = script_count / len(cleaned) if script != "OTHER" else 0.0
    prefix_fraction = prefix_count / len(cleaned)
    if prefix_fraction > script_fraction:
        return prefix_fraction, f"shared prefix {prefix!r}"
    return script_fraction, f"shared {script} script"


def _md_token(token: str) -> str:
    visible = token.replace("\n", "\\n").replace("\r", "\\r").replace("`", "\\`")
    return visible if visible else "<empty>"


def run_report(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    inventory = json.loads((output_dir / "cold_experts.json").read_text(encoding="utf-8"))
    lens = json.loads((output_dir / "router_lens.json").read_text(encoding="utf-8"))
    counts = Counter(int(item["layer"]) for item in inventory["experts"])
    ranked = []
    for expert in lens["experts"]:
        tokens = [item["token"] for item in expert["top_tokens"]]
        score, reason = coherence_score(tokens)
        visible = sum(bool(token.strip()) for token in tokens) / max(1, len(tokens))
        ranked.append((score + 0.05 * visible, reason, expert))
    ranked.sort(key=lambda item: (-item[0], item[2]["layer"], item[2]["expert_id"]))

    lines = [
        "# COLD-EXPERT ATLAS",
        "",
        "CPU-only screening of experts with zero recorded Fisher trace by projecting each learned router row "
        "back onto the checkpoint token-embedding vocabulary.",
        "",
        f"Cold definition: `{inventory['definition']}`. Total: **{inventory['cold_expert_count']:,}**.",
        "",
        "## Cold experts by layer",
        "",
        "| Layer | Cold experts |",
        "|---:|---:|",
    ]
    lines.extend(f"| {layer} | {counts[layer]} |" for layer in range(43))
    lines.extend(
        [
            "",
            "Layers 0–2 are hash-routed and have no cold experts. Their token-to-expert "
            "inversion and distribution check are in `hash_layer_token_map.json`.",
            "",
            "## Interpretable screening examples",
            "",
            "These twelve are selected mechanically by the strongest shared Unicode-script "
            "or three-character-prefix concentration among each expert's top-40 tokens.",
            "",
        ]
    )
    for _, reason, expert in ranked[:12]:
        norm = expert["gate_row_norm"]
        tokens = expert["top_tokens"][:12]
        token_text = ", ".join(f"`{_md_token(item['token'])}`" for item in tokens)
        lines.extend(
            [
                f"### Layer {expert['layer']}, expert {expert['expert_id']}",
                "",
                f"Heuristic: {reason}. Gate-row norm {norm['row_norm']:.4f} "
                f"(percentile {norm['percentile']:.1f} within the layer).",
                "",
                f"Top tokens: {token_text}",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "- This is a token-level lens. It can miss experts specialized for phrases, "
            "syntax, long-range dependencies, or other contextual patterns.",
            "",
            "- Hidden-state geometry at MoE layer L is not embedding-space geometry. The "
            "projection omits every intervening attention/MLP transformation and router bias.",
            "",
            "- A coherent top-token list is a hypothesis about specialization, not evidence "
            "that those tokens would route to the expert in a real forward pass.",
            "",
            "- A zero Fisher trace does not prove that no token reached the expert. "
            "This atlas is a screening tool for the future truncated-forward miner, "
            "not proof that an expert is dead or semantically specialized.",
            "",
        ]
    )
    path = output_dir / "ATLAS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: wrote {path} in {time.perf_counter() - started:.2f}s")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="derive the cold-expert inventory")
    inventory.add_argument("--cost-table", type=Path, default=DEFAULT_COST_TABLE)
    inventory.add_argument("--probe", type=Path, default=None)
    inventory.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    inventory.set_defaults(func=run_inventory)

    hash_invert = subparsers.add_parser("hash-invert", help="invert hash-router token maps")
    hash_invert.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    hash_invert.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    hash_invert.add_argument("--max-examples", type=int, default=200)
    hash_invert.set_defaults(func=run_hash_invert)

    router = subparsers.add_parser("router-lens", help="project cold gate rows onto tokens")
    router.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    router.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    router.add_argument("--top-k", type=int, default=40)
    router.add_argument("--batch-size", type=int, default=8)
    router.add_argument(
        "--threads", type=int, default=0, help="torch CPU threads (0 keeps environment default)"
    )
    router.set_defaults(func=run_router_lens)

    report = subparsers.add_parser("report", help="render ATLAS.md from JSON artifacts")
    report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    report.set_defaults(func=run_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "threads", 0):
        torch.set_num_threads(args.threads)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
