"""Files whose bytes identify each gold measurement producer."""

# Six lane-owned members (the assignment/environment registries, the two
# runtime pins, the serving-pin reader and the DSv4 contract) left this
# closure when the Gridbook lane was retired 2026-09-02; the files now live
# under archive/gridbook_lane_2026-09-02/ and no gold tool imports them.
# The Tessera serving pin JSON joined it on 2026-09-03 instead: the serve
# fingerprint reads its extension rows from that file at runtime (below), so
# the transported snapshot is digest-covered for the very bytes the
# container-side tool decides on. It is the one non-`.py` member;
# `gold_producer_identity` hashes bytes, not language.
_GOLD_PRODUCER_COMMON_FILES = (
    "prismaquant/tessera_runtime/tessera_serving_runtime_pin.json",
    "tools/dsv4_wikitext_inputs.py",
    "tools/prepare_dsv4_wikitext_inputs.py",
    "tools/prismaquant_source_bootstrap.py",
    "tools/serve_fingerprint.py",
    "prismaquant/gold_producer_files.py",
    "tools/spec_decode_guard.py",
)
_GOLD_PRODUCER_TOOL_FILES = {
    "build_streamed_full_kl_teacher": (
        "tools/build_streamed_full_kl_teacher.py",
        "prismaquant/container_runtime_identity.py",
        "tools/full_kl_teacher_payload.py",
    ),
    # `gold_measurement_fidelity.py` binds the bytes that decide which
    # estimator a result says it is.  A number labelled full-vocabulary that
    # was top-K, or the reverse, is a worse confound than a wrong version
    # string, so the label's source travels in the producer identity.
    "measure_vllm_full_kl": (
        "tools/gold_engine_options.py",
        "tools/gold_measurement_fidelity.py",
        "tools/full_kl_teacher_payload.py",
        "tools/measure_vllm_full_kl.py",
    ),
    "measure_vllm_wikitext_ppl": (
        "tools/gold_engine_options.py",
        "tools/gold_measurement_fidelity.py",
        "tools/full_kl_teacher_payload.py",
        "tools/measure_vllm_wikitext_ppl.py",
    ),
    # The served-lane gold tool.  It owns no `LLM`, so it needs neither the
    # teacher payload nor the DSv4 in-process contract -- the common files
    # above already bind the serve fingerprint and spec-decode guard it does
    # use.
    "measure_served_gold": (
        "tools/measure_served_gold.py",
    ),
}

