"""PACT frontier and price adapter over the model profile.

Move the D43 frontier and the measured-price adapter into prismaquant
without moving any baseline. Read layers, bands, vocab, prefix ids,
hidden layout and TP rules from the model profile. Keep the running
GLM measurement paths bit-identical until those runs finish (D50 item 3).

Boundary: D50 item 4 runs under testcost (parent
testcost-d50-next-model-quality-20261007). Leave live mixed G3 and PACT
workspaces unchanged. PACT and profile ownership stays intact. Route
shared profile needs through the existing owner. Never build a parallel
model map. This module therefore adds no profile class and no registry.
It reads only the existing owner API and plain config dicts.

Sources (read-only, never edited here):
- harness /home/rob/tmp/eng-pact-pricing-20261006
- frontier /mnt/shared/tessera-measurements/pact-frontier-d43-20261006/frontier.py
- adapter checkout /home/rob/tmp/eng-pact-energy-20261007
"""

from __future__ import annotations

# GLM-5.3 Flash legacy values. These are the exact constants the running
# measurement uses today. Keep them unchanged while those runs finish.
GLM_LEGACY = {
    "num_layers": 45,
    "dense_layers": (0, 3),
    "band_width": 6,
    "prefix_ids": (154822, 154824),
    # Measured-consumer contract (issue 2427): the running measurement
    # excludes the local prefix rows; each input carries the prefix.
    # The identity guard rejects any cohort that lacks or changes these.
    "local_prefix_rows": "excluded",
    "input_contract": "prefixed_514",
    "vocab_size": 154880,
    "scored_positions": 511,
    "sample_range": (384, 448),
    "raw_tokens_per_sequence": 512,
    "global_original_tokens": 32768,
    "tp_splits_down_proj": 2,
    "tp_splits_other": 1,
    "hidden_streams": 4,
    "hidden_size": 4096,
}

# Manifest keys the D43 frontier script reads. The overlay below keeps
# every one of them; it substitutes only profile-derived scalar fields.
FRONTIER_MANIFEST_KEYS = (
    "d41_index",
    "price_tables",
    "anchors",
    "heldout_preregistration",
    "stack_scope_source",
    "thresholds",
    "predictor",
)


def _config_value(config, *names, default=None):
    """Return the first present config value. Walk nested text_config."""
    if not isinstance(config, dict):
        return default
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    text = config.get("text_config")
    if isinstance(text, dict):
        for name in names:
            if name in text and text[name] is not None:
                return text[name]
    return default


def num_layers_from_profile(profile=None, config=None):
    """Return the decoder layer count. Fall back to the GLM legacy value."""
    value = _config_value(config, "num_hidden_layers")
    if type(value) is int and value > 0:
        return value
    layers = getattr(profile, "_declared_config", None)
    if isinstance(layers, dict):
        value = _config_value(layers, "num_hidden_layers")
        if type(value) is int and value > 0:
            return value
    return GLM_LEGACY["num_layers"]


def pact_bands(num_layers=None, profile=None, config=None):
    """Return the PACT band list. Keep the GLM shape for 45 layers."""
    count = num_layers if type(num_layers) is int and num_layers > 0 else num_layers_from_profile(profile, config)
    dense = GLM_LEGACY["dense_layers"]
    width = GLM_LEGACY["band_width"]
    bands = [dense]
    start = dense[1]
    while start < count:
        bands.append((start, min(start + width, count)))
        start += width
    return tuple(bands)


def pact_cohort_from_profile(profile=None, config=None):
    """Return the pricing cohort dict. Resolve GLM values bit-identically."""
    prefix = _config_value(config, "prefix_ids", "serving_prefix_ids", default=None)
    if prefix is None:
        prefix = GLM_LEGACY["prefix_ids"]
    vocab = _config_value(config, "vocab_size", default=GLM_LEGACY["vocab_size"])
    scored = _config_value(config, "scored_positions_per_sequence", default=GLM_LEGACY["scored_positions"])
    raw_tokens = _config_value(
        config, "raw_tokens_per_sequence", default=GLM_LEGACY["raw_tokens_per_sequence"])
    cohort = {
        "sample_range": list(GLM_LEGACY["sample_range"]),
        "raw_tokens_per_sequence": int(raw_tokens),
        "prefix_ids": list(prefix),
        "local_prefix_rows": GLM_LEGACY["local_prefix_rows"],
        "input_contract": GLM_LEGACY["input_contract"],
        "global_original_tokens": GLM_LEGACY["global_original_tokens"],
        "scored_positions_per_sequence": int(scored),
        "vocab_size": int(vocab),
    }
    return cohort


def tp_splits_for_role(role, profile=None):
    """Return the TP split count for a unit role. Keep the TP2 rule."""
    del profile
    if role == "down_proj":
        return GLM_LEGACY["tp_splits_down_proj"]
    return GLM_LEGACY["tp_splits_other"]


def glm_paths_identical(cohort):
    """Check the cohort against the running GLM measurement values."""
    legacy = pact_cohort_from_profile()
    return all(cohort.get(key) == legacy[key] for key in legacy)


def frontier_manifest_overlay(manifest, profile=None, config=None):
    """Overlay profile-derived scalars on a D43 manifest. Keep all paths."""
    if not isinstance(manifest, dict):
        raise ValueError("frontier manifest is not a JSON object")
    missing = [key for key in FRONTIER_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError("frontier manifest lacks keys: " + ",".join(missing))
    overlay = dict(manifest)
    overlay["model_profile"] = getattr(profile, "name", "unknown")
    overlay["num_layers"] = num_layers_from_profile(profile, config)
    overlay["bands"] = [list(band) for band in pact_bands(profile=profile, config=config)]
    overlay["cohort"] = pact_cohort_from_profile(profile, config)
    return overlay


# Shared profile need, routed to the existing owner (campaign):
# pact prefix ids, vocab, scored positions and TP rules belong in the
# model profile / structure spec. This adapter reads plain config dicts
# until the owner adds them. Do not add profile methods here.
PROFILE_OWNER_REQUEST = (
    "campaign: expose pact cohort (prefix ids, vocab, scored positions), "
    "layer count, hidden layout and TP rules on ModelProfile/structure spec"
)
