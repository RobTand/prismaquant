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
# The shared contract now owns them: Glm5NextProfile states the cohort
# and layer count, and specs/glm5_next.json states the scope. This
# table stays as the frozen equivalence oracle only.
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
    """First present alias value. Top level beats nested text_config."""
    from .model_profiles.base import _read_config_alias

    value = _read_config_alias(config, *names)
    return default if value is None else value


def _require_profile(profile):
    """Return the profile, or refuse a call with no declared scope."""
    if profile is None or getattr(profile, "pact_scope_declared", None) is None:
        raise ValueError(
            "PACT scope needs a declared model profile; "
            "explicit non-GLM resolution states no GLM fallback"
        )
    if not profile.pact_scope_declared():
        raise ValueError(
            f"profile {getattr(profile, 'name', 'unknown')} declares no PACT scope"
        )
    return profile


def num_layers_from_profile(profile=None, config=None):
    """Return the decoder layer count from the profile contract."""
    _require_profile(profile)
    count = profile.pact_layer_count(config)
    if count is not None:
        return count
    raise ValueError(
        "PACT layer count needs num_hidden_layers in the explicit "
        "or declared config; no fallback supplies it"
    )


def pact_bands(num_layers=None, profile=None, config=None):
    """Return the PACT band list from the profile scope."""
    if type(num_layers) is int:
        if num_layers <= 0:
            raise ValueError("PACT bands need a positive layer count")
        count = num_layers
    else:
        _require_profile(profile)
        count = num_layers_from_profile(profile, config)
    owner = _require_profile(profile)
    dense_end = owner.pact_dense_layer_end()
    width = owner.pact_band_width()
    if dense_end is None or width is None:
        raise ValueError(
            f"profile {getattr(owner, 'name', 'unknown')} declares no PACT band scope"
        )
    if type(count) is not int or count <= 0:
        raise ValueError("PACT bands need a positive layer count")
    if dense_end < 0 or dense_end > count:
        raise ValueError(
            f"PACT dense_layer_end {dense_end} is outside 0..{count}"
        )
    bands = [(0, dense_end)] if dense_end > 0 else []
    if dense_end == 0:
        bands = [(0, min(width, count))]
        start = width
    else:
        bands = [(0, dense_end)]
        start = dense_end
    while start < count:
        bands.append((start, min(start + width, count)))
        start += width
    return tuple(bands)


def pact_cohort_from_profile(profile=None, config=None):
    """Return the pricing cohort dict from the profile contract."""
    owner = _require_profile(profile)
    cohort_fn = getattr(owner, "pact_cohort_values", None)
    if not callable(cohort_fn):
        raise ValueError(
            f"profile {getattr(owner, 'name', 'unknown')} declares no PACT cohort"
        )
    values = cohort_fn(config)
    for key in (
        "sample_range",
        "raw_tokens_per_sequence",
        "prefix_ids",
        "local_prefix_rows",
        "input_contract",
        "global_original_tokens",
        "scored_positions_per_sequence",
        "vocab_size",
    ):
        if key not in values:
            raise ValueError(
                f"PACT cohort from {getattr(owner, 'name', 'unknown')} lacks {key}"
            )
    return {
        "sample_range": list(values["sample_range"]),
        "raw_tokens_per_sequence": int(values["raw_tokens_per_sequence"]),
        "prefix_ids": list(values["prefix_ids"]),
        "local_prefix_rows": values["local_prefix_rows"],
        "input_contract": values["input_contract"],
        "global_original_tokens": int(values["global_original_tokens"]),
        "scored_positions_per_sequence": int(
            values["scored_positions_per_sequence"]
        ),
        "vocab_size": int(values["vocab_size"]),
    }


def tp_splits_for_role(role, profile=None):
    """Return the TP split count from the profile scope."""
    owner = _require_profile(profile)
    count = owner.pact_tp_splits_for_role(role)
    if count is not None:
        return count
    if role == "down_proj":
        raise ValueError(
            f"profile {getattr(owner, 'name', 'unknown')} declares no TP split "
            f"for {role}"
        )
    return 1


def _legacy_cohort():
    """Frozen GLM measurement cohort. Never reads a profile."""
    return {
        "sample_range": list(GLM_LEGACY["sample_range"]),
        "raw_tokens_per_sequence": GLM_LEGACY["raw_tokens_per_sequence"],
        "prefix_ids": list(GLM_LEGACY["prefix_ids"]),
        "local_prefix_rows": GLM_LEGACY["local_prefix_rows"],
        "input_contract": GLM_LEGACY["input_contract"],
        "global_original_tokens": GLM_LEGACY["global_original_tokens"],
        "scored_positions_per_sequence": GLM_LEGACY["scored_positions"],
        "vocab_size": GLM_LEGACY["vocab_size"],
    }


def glm_paths_identical(cohort):
    """Check the cohort against the running GLM measurement values."""
    legacy = _legacy_cohort()
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


def pact_hidden_layout(profile=None, config=None):
    """Return hidden streams and width from the profile contract."""
    owner = _require_profile(profile)
    streams = owner.pact_hidden_streams()
    width = owner.pact_hidden_size(config)
    if streams is None:
        raise ValueError(
            f"profile {getattr(owner, 'name', 'unknown')} declares no hidden layout"
        )
    if width is None:
        raise ValueError(
            "PACT hidden width needs hidden_size in the explicit "
            "or declared config; no fallback supplies it"
        )
    return {"hidden_streams": streams, "hidden_size": width}
