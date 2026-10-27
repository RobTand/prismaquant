# EXL3 grouped-down diagnostic

This opt-in driver observes one live grouped-down call at the first routed MoE
boundary. Its output is diagnostic evidence. It cannot qualify an artifact or
produce a gold record. Historical full25 remains measured but incomparable
under [#468](https://github.com/RobTand/prismaquant/issues/468).

## Run one sealed window

Use the existing baseline image and paired launcher after the exact source,
inputs, runtime, host readiness, and scientific run are approved. This command
describes the container entry point; it grants no launch authority.

```sh
python -m experiments.diagnose_exl3_grouped_down_replay \
  --model "$MODEL" --candidate-digest-cache "$CANDIDATE_DIGEST_CACHE" \
  --panel "$PANEL" --arrays-root "$ARRAYS_ROOT" \
  --teacher "$TEACHER" --teacher-sha256 "$TEACHER_SHA256" \
  --serve-image "$SERVE_IMAGE" --quantization exl3 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype auto --expected-kv-cache-dtype fp8_ds_mla \
  --logits-layout vllm_v2_chunk1024 --output "$DIAGNOSTIC_OUTPUT"
```

Supply the original distributed engine arguments through the existing native
options. The CLI selects only panel window zero, issues one outstanding prompt
with seed 42, and uses the scorer's authenticated loaders, prompt capture,
rank-completeness checks, and prompt-logprob alignment. It rechecks identities
and configurations after the request. It does not invoke the panel scorer.

The default boundary is `model.layers.3.mlp`, the required input row count is
2048, and the replay count is eight. `--module` and `--required-rows` are explicit
diagnostic selectors. `--repeats` permits 2 through 16. Other shapes remain
recorded as non-firing evidence; the driver does not change serving allocation.
`--serve-image` must name an immutable digest. Qualification and gold-output
arguments are refused. Use a fresh output path for each run.

## Interpret the record

Probe schema `/3` snapshots all ten arguments before the real call. Nine are
read-only inputs. The output is an accumulator that can already contain thin
expert contributions. Each replay clones its pre-call bytes. The initial hash
of every replay destination must equal the original pre-call destination hash.
The served post-call output must remain unchanged throughout observation.
Schema `/2` zeroed replay destinations and observed arguments after the original
call. The driver refuses those records.

The report binds complete candidate checkpoint metadata, the native producer
and diagnostic source files, the image digest, engine options, both TP rank
rosters, loaded overlay and extension identities, route source hashes and
counter deltas, argument hashes, output preservation, finiteness, and FP32/BF16
comparisons. It reuses the checkpoint and native configuration validators.
Missing or malformed observations produce invalid strict JSON. Valid evidence
publishes only after this invocation's native capture cleanup succeeds.
Pre-existing captures remain owned by their original installer.

An evidence-valid report may contain differing outputs. Identical bounded
replays are not proof of determinism, and synchronized observation can mask the
original behavior. Pointer hashes cover addresses, not pointed-to weight bytes.
The diagnostic does not test gather, gate/up, or establish an atomic-add cause.
Fresh uninstrumented within-engine and cross-boot checks are required after a
supported numerical repair. CPU tests establish orchestration and report
validation only. They establish no native repeatability or served quality.

The `/3` probe retains four output-shaped FP32 buffers plus temporary BF16
comparison tensors. This is static buffer accounting, not a measured residency
or performance result. Ordinary scoring and its qualification predicates stay
in `measure_glm_tr3_vllm.py`.
