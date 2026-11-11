"""Tests that cannot run hermetically, or need hardware, each with its reason.

The remote platform turns networking off and runs each test in a container
that has a few CPUs of a larger host, as an unprivileged user, on a runfiles
tree with every input read-only and no .git. A test listed here gets back
what its tags name. Keep this table short and the reasons true; the PR that
adds an entry says why the test can't use a declared input instead.
"""

NONHERMETIC = {
    "//tests:test_band_serial_batched_regime": struct(
        tags = ["gpu", "hosted-arm64"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU. Stage B spill requires local ext4/xfs/btrfs; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_band_serial_scratch_plane": struct(
        tags = ["hosted-arm64"],
        reason = "Cotangent scratch requires local disk; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_campaign_recovery_scratch_seam": struct(
        tags = ["hosted-arm64"],
        reason = "Cotangent scratch requires local disk; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_canonical_consumer_join": struct(
        tags = ["requires-network"],
        reason = "it fetches Tessera's rung_allowability.py from GitHub at a declared commit (cannot fetch declared dependency); hosted shards have a network.",
    ),
    "//tests:test_canonical_quality_quantity": struct(
        tags = ["requires-network"],
        reason = "it fetches Tessera's rung_allowability.py from GitHub at a declared commit (cannot fetch declared dependency); hosted shards have a network.",
    ),
    "//tests:test_capture_layer_chain": struct(
        tags = ["hosted-arm64"],
        reason = "CaptureWriter free-space check sees 0 bytes on arm64 tmpfs; hosted arm64 has a real disk.",
    ),
    "//tests:test_capture_layer_chain_glm": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_capture_single_pass_source_1896": struct(
        tags = ["gpu", "hosted-arm64"],
        reason = "1 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU. CaptureWriter free-space check sees 0 bytes on arm64 tmpfs; hosted arm64 has a real disk.",
    ),
    "//tests:test_chain_roll_overlap": struct(
        tags = ["gpu"],
        reason = "3 of its test functions (6 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_checkpoint_staged_readset_1366": struct(
        tags = ["hosted-arm64"],
        reason = "Stage B spill requires local ext4/xfs/btrfs; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_derive_per_expert_routing_1931": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_dispatch_capture_chain": struct(
        tags = ["hosted-arm64"],
        reason = "CaptureWriter free-space check sees 0 bytes on arm64 tmpfs; hosted arm64 has a real disk.",
    ),
    "//tests:test_export_inline_expert_gptq": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_fleet_acceptance_level1": struct(
        tags = ["runner"],
        reason = "Its helper CLI qualifies the real checkout through fleet_acceptance_runner and record_snapshots, which require git; remote runfiles have no .git. Existing absent-PB-bundle skips remain. The amd64 runner covers it; arm64 action coverage is excluded.",
    ),
    "//tests:test_fp8_batched_midpoint_parity_audit": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (4 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_fp8_dynamic_native_parity": struct(
        tags = ["gpu"],
        reason = "2 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_glm_attention_census": struct(
        tags = ["hosted-arm64"],
        reason = "CaptureWriter free-space check sees 0 bytes on arm64 tmpfs; hosted arm64 has a real disk.",
    ),
    "//tests:test_glm_full_capture_profile": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_glm_mtp_priced_wires_1413": struct(
        tags = ["requires-network"],
        reason = "it fetches Tessera's rung_allowability.py from GitHub at a declared commit (cannot fetch declared dependency); hosted shards have a network.",
    ),
    "//tests:test_glm_mtp_selection": struct(
        tags = ["requires-network"],
        reason = "it fetches Tessera's rung_allowability.py from GitHub at a declared commit (cannot fetch declared dependency); hosted shards have a network.",
    ),
    "//tests:test_glm_routing_capture": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_glm_selected_anchor_profile": struct(
        tags = ["gpu"],
        reason = "2 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_glm_tr3_dual_teacher": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_glm_tr3_full_vocab": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_graph_impact": struct(
        tags = ["runner"],
        reason = "The live per-PR graph check reads base/head source snapshots from Git history; remote runfiles have no .git.",
    ),
    "//tests:test_indomain_split_capture": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_joint_aura_streamed": struct(
        tags = ["runner"],
        reason = "it resolves the source identity with git (rev-parse or ls-files) in the checkout; remote runfiles have no .git.",
    ),
    "//tests:test_joint_operator_statistics": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_joint_projection_reduce": struct(
        tags = ["gpu"],
        reason = "6 of its test functions (21 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_kda_capture_kernel": struct(
        tags = ["gpu"],
        reason = "10 of its test functions (10 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_kda_kernel_default_1252": struct(
        tags = ["gpu"],
        reason = "4 of its test functions (4 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_kda_kernel_mode": struct(
        tags = ["gpu"],
        reason = "5 of its test functions (7 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_layer_major_boundary_capture": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_memory_management": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_model_profile_conformance": struct(
        tags = ["gpu"],
        reason = "3 of its test functions (36 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_mtp_formats_menu_1692": struct(
        tags = ["requires-network"],
        reason = "it fetches Tessera's rung_allowability.py from GitHub at a declared commit (cannot fetch declared dependency); hosted shards have a network.",
    ),
    "//tests:test_native_render_fused_joint_global": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_native_retained_cost_source_admission_audit": struct(
        tags = ["runner"],
        reason = "it resolves the source identity with git (rev-parse or ls-files) in the checkout; remote runfiles have no .git.",
    ),
    "//tests:test_no_secret_on_command_line": struct(
        tags = ["runner"],
        reason = "It scans tracked files with git ls-files in the actual ROOT checkout; remote runfiles have no .git. The amd64 runner covers it; arm64 action coverage is excluded.",
    ),
    "//tests:test_numerics_pairs_1394": struct(
        tags = ["gpu"],
        reason = "2 of its test functions (4 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_nvfp4_a16_render_equivalence": struct(
        tags = ["gpu"],
        reason = "3 of its test functions (3 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_nvfp4_codebook_cache": struct(
        tags = ["gpu"],
        reason = "2 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_nvfp4_served_codebook_residency": struct(
        tags = ["gpu"],
        reason = "4 of its test functions (6 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_nvfp4_served_dequant_kernel": struct(
        tags = ["gpu"],
        reason = "11 of its test functions (49 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_nvfp4_served_qdq_no_sync": struct(
        tags = ["gpu"],
        reason = "2 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_original_diagnostic_context_2149": struct(
        tags = ["hosted"],
        reason = "Bazel canonical repo paths contain '+', which the modeling-source schema (schemas.py:592) correctly rejects. Hosted setup-python retains amd64 coverage; arm64 action coverage is deferred for this same layout constraint.",
    ),
    "//tests:test_original_source_admission": struct(
        tags = ["hosted"],
        reason = "Bazel canonical repo paths contain '+', which the modeling-source schema (schemas.py:592) correctly rejects. Hosted setup-python retains amd64 coverage; arm64 action coverage is deferred for this same layout constraint.",
    ),
    "//tests:test_original_source_copy_completion_cuda": struct(
        tags = ["gpu"],
        reason = "2 of its test functions (64 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_packed_grouped_qdq_audit": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_parallel_capture_prefetch": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_pilot_verified_result_1293": struct(
        tags = ["hosted"],
        reason = "needs os.pidfd_open/P_PIDFD/signal.pidfd_send_signal, absent from python-build-standalone 3.12 (rules_python 2.4.6) on amd64; arm64 standalone has pidfd and is excluded only for table simplicity. Hosted setup-python retains amd64 coverage.",
    ),
    "//tests:test_production_cache_source_model_binding": struct(
        tags = ["runner"],
        reason = "it resolves the source identity with git (rev-parse or ls-files) in the checkout; remote runfiles have no .git.",
    ),
    "//tests:test_projected_preparation_2039": struct(
        tags = ["gpu", "hosted"],
        reason = "it builds a torch C++ extension, which needs ninja; the hosted image has it, the executor and runner images don't, and .[test] doesn't install it; 3 of its test functions (6 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_projected_unit_check_1935": struct(
        tags = ["gpu"],
        reason = "2 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_resident_source_decode_audit": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_row_chunked_qdq": struct(
        tags = ["gpu"],
        reason = "4 of its test functions (12 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_rung_allowability": struct(
        tags = ["requires-network"],
        reason = "it fetches Tessera's rung_allowability.py from GitHub at a declared commit (cannot fetch declared dependency); hosted shards have a network.",
    ),
    "//tests:test_rung_allowability_versions": struct(
        tags = ["requires-network"],
        reason = "it fetches Tessera's rung_allowability.py from GitHub at a declared commit (cannot fetch declared dependency); hosted shards have a network.",
    ),
    "//tests:test_selected_calibration_capture": struct(
        tags = ["hosted-arm64"],
        reason = "CaptureWriter free-space check sees 0 bytes on arm64 tmpfs; hosted arm64 has a real disk.",
    ),
    "//tests:test_spaced_lax_tools_routing_1301": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_spill_readplan_metadata_2115": struct(
        tags = ["hosted-arm64"],
        reason = "Stage B spill requires local ext4/xfs/btrfs; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_stage_a_plane_window_1121": struct(
        tags = ["hosted-arm64"],
        reason = "The produced-output two-plane window reads statvfs free space; arm64 tmpfs reports 0 bytes. The forced-zero probe reproduced ProducedWindowRefused at its real bind.",
    ),
    "//tests:test_stage_b_pass_profile": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_stage_b_streamed_handoff": struct(
        tags = ["hosted-arm64"],
        reason = "Cotangent scratch requires local disk; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_stage_b_streamed_incoming_1143": struct(
        tags = ["hosted-arm64"],
        reason = "Its staging tests construct ExactCotangentScratch on tmp_path; that real disk guard refuses arm64 tmpfs. Static audit; fresh runtime validation pending.",
    ),
    "//tests:test_stage_b_workspace_profile": struct(
        tags = ["gpu"],
        reason = "2 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_stageb_checkpoint_incoming_1802": struct(
        tags = ["hosted-arm64"],
        reason = "Stage B spill requires local ext4/xfs/btrfs; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_stageb_cotangent_scratch": struct(
        tags = ["hosted-arm64"],
        reason = "Cotangent scratch requires local disk; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_stageb_host_staging_1246": struct(
        tags = ["gpu", "hosted-arm64"],
        reason = "1 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU. Cotangent scratch requires local disk; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_stageb_one_pass_spill": struct(
        tags = ["gpu", "hosted-arm64"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU. Stage B spill requires local ext4/xfs/btrfs; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_stageb_scatter_reads_1794": struct(
        tags = ["hosted-arm64"],
        reason = "Stage B spill requires local ext4/xfs/btrfs; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_stageb_spill_input_digest": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_stageb_spill_integrity_1369": struct(
        tags = ["hosted-arm64"],
        reason = "Stage B spill requires local ext4/xfs/btrfs; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_stageb_spill_records_1086": struct(
        tags = ["hosted-arm64"],
        reason = "Stage B spill requires local ext4/xfs/btrfs; arm64 executors use tmpfs with 0 free bytes, so hosted arm64 runs this file.",
    ),
    "//tests:test_streamed_cost_checkpoints": struct(
        tags = ["runner"],
        reason = "it resolves the source identity with git (rev-parse or ls-files) in the checkout; remote runfiles have no .git.",
    ),
    "//tests:test_streaming_production_cache": struct(
        tags = ["gpu"],
        reason = "3 of its test functions (3 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_streaming_static_activations": struct(
        tags = ["gpu", "runner"],
        reason = "it resolves the source identity with git (rev-parse or ls-files) in the checkout; remote runfiles have no .git; 1 of its test functions (2 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_strict_reader_tier_enforcement": struct(
        tags = ["hosted-arm64"],
        reason = "The local_scratch=True checkpoint test constructs ExactCotangentScratch on tmp_path; its disk guard refuses arm64 tmpfs. Static audit; runtime validation pending.",
    ),
    "//tests:test_tensor_digest_owner_recipes_2638": struct(
        tags = ["gpu"],
        reason = "3 of its test functions (3 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_tensor_digests_1384": struct(
        tags = ["gpu"],
        reason = "5 of its test functions (6 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_tessera_calibration_cache": struct(
        tags = ["hosted-arm64"],
        reason = "CaptureWriter free-space check sees 0 bytes on arm64 tmpfs; hosted arm64 has a real disk.",
    ),
    "//tests:test_tessera_campaign": struct(
        tags = ["gpu"],
        reason = "5 of its test functions (5 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_tessera_campaign_batch": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_tessera_joint_aura": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (3 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_tessera_partial_superblock": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (3 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_tessera_selected_source": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_tessera_serve_fingerprint": struct(
        tags = ["runner"],
        reason = "it resolves the source identity with git (rev-parse or ls-files) in the checkout; remote runfiles have no .git.",
    ),
    "//tests:test_validation_harness": struct(
        tags = ["gpu"],
        reason = "1 of its test functions (1 nodes) skip on CPU for CUDA, Triton or vLLM (notes/inventory control run); the rest run on CPU.",
    ),
    "//tests:test_verified_capture_load": struct(
        tags = ["hosted-arm64"],
        reason = "CaptureWriter free-space check sees 0 bytes on arm64 tmpfs; hosted arm64 has a real disk.",
    ),
}

def nonhermetic(name):
    """Returns (tags, exec_properties, env) for the named test in this package.

    gpu: some of its tests need CUDA, Triton or vLLM and skip without them. The
    tag marks it for the GPU lane (tools/bazel/gpu, a follow-up); on the CPU
    executors it still runs, and those tests skip as they do on hosted shards.
    requires-network: the remote executor gets a network.
    runner: the test runs on the Remote Bazel runner itself, a VM like a hosted
    gate runner (all its CPUs, sudo), with pytest in the runner's git checkout.
    Its result is never cached, because it reads that checkout.
    hosted-arm64: the arm64 remote population filters this tag; the hosted
    arm64 post-merge lane runs the file. It still runs on amd64 executors.
    hosted: Bazel never runs the test; the hosted lane runs it with
    tools/ci/gate_run.py whenever Bazel's selection impacts its target.
    """
    entry = NONHERMETIC.get("//%s:%s" % (native.package_name(), name))
    if not entry:
        return [], {}, {}
    tags, properties, env = list(entry.tags), {}, {}
    if "requires-network" in entry.tags:
        properties["dockerNetwork"] = "bridge"
    if "runner" in entry.tags:
        tags += ["no-remote-exec", "no-sandbox", "no-cache"]
        env["PQ_BAZEL_LANE"] = "runner"
    if "hosted" in entry.tags:
        tags += ["manual", "gate-lane-hosted"]
    return tags, properties, env
