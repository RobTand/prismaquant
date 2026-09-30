"""Lightweight owners for explicitly shared qname grammars.

LAYER_QNAME is the legacy roster grammar, not a general model-name parser.
It is greedy, permits Unicode decimal digits and requires a dot before layers.
Caller-specific exact-string checks and refusals stay with each consumer.
"""
import re


LAYER_QNAME: re.Pattern[str] = re.compile(r"^.*\.layers\.(\d+)(?:\.|$)")
