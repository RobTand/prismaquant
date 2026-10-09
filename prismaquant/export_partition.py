"""Whole-layer export work under the producer's existing ownership rule."""
from __future__ import annotations


def whole_layer_partitions(tensors, producer) -> list[dict]:
    """Use the largest modulo domain with no empty owner.

    The producer owns names and ownership. Physical source shards do not
    define independent work: fused members and routed stacks stay together.
    """
    layers = {int(match.group(1)) for name in tensors
              if (match := producer.BODY_LAYER.match(name))}
    if not layers:
        raise ValueError("source has no whole-layer export work")
    for count in range(len(layers), 0, -1):
        owned = [[] for _ in range(count)]
        for name in sorted(tensors):
            owner = producer.partition_owner(name, count)
            if type(owner) is not int or not 0 <= owner < count:
                raise ValueError(f"producer returned an invalid partition owner for {name}")
            owned[owner].append(name)
        if all(owned):
            return [{"index": index, "source_tensors": names,
                     "source_shards": sorted({tensors[name] for name in names})}
                    for index, names in enumerate(owned)]
    raise ValueError("producer cannot partition this source")
