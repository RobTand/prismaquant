"""Lightweight owners for explicitly shared qname grammars.

LAYER_QNAME is the legacy roster grammar, not a general model-name parser.
It is greedy, permits Unicode decimal digits and requires a dot before layers.
DOTTED_LAYER_QNAME searches the first complete dotted layer component.
Caller-specific exact-string checks and refusals stay with each consumer.
"""
import re


LAYER_QNAME: re.Pattern[str] = re.compile(r"^.*\.layers\.(\d+)(?:\.|$)")

# Distinct from the greedy roster grammar above: both dots are required,
# and search() returns the first component even in a repeated layer path.
DOTTED_LAYER_QNAME: re.Pattern[str] = re.compile(r"\.layers\.(\d+)\.")
