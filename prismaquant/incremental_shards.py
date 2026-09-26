"""Shared helpers for incremental probe/cost shard pickles."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any


def build_layer_shard_regexes(num_hidden_layers: int,
                              layers_per_shard: int,
                              layer_prefix: str = "model.layers") -> list[str]:
    """One include regex per shard of ``layers_per_shard`` decoder layers.

    A one-layer shard matches ``<prefix>.<i>.``; a wider shard matches any of
    its layer indices. The streaming probe and every model profile's shard
    plan build their shards here (#1394).
    """
    regexes: list[str] = []
    for start in range(0, num_hidden_layers, layers_per_shard):
        end = min(start + layers_per_shard, num_hidden_layers)
        if end - start == 1:
            body = rf"{re.escape(layer_prefix)}\.{start}\."
        else:
            idxs = "|".join(str(i) for i in range(start, end))
            body = rf"{re.escape(layer_prefix)}\.(?:{idxs})\."
        regexes.append(body)
    return regexes


def read_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def annotate_incremental_shard(path: Path, extra_meta: dict[str, Any]) -> None:
    data = read_pickle(path)
    meta = dict(data.get("meta", {}))
    inc = dict(meta.get("incremental_shard", {}))
    inc.update(extra_meta)
    meta["incremental_shard"] = inc
    data["meta"] = meta
    with open(path, "wb") as f:
        pickle.dump(data, f)
