# Inspect export structure without loading weights

Use the offline diagnostic after the exporter has stopped writing. Run it as a
separate PrismaBuild action so a diagnostic failure does not require repeating
expensive encoding. The module uses only stdlib and the existing shipcard header
validator, but the `prismaquant` package initializer requires a PQ environment.

For example, using an existing worker-side PQ interpreter:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /path/to/prismaquant --cpus 1 --demand mem_gb=2 --priority -10 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --timeout-s 300 -- \
  /path/to/pq-venv/bin/python -m prismaquant.export_structure \
  /path/to/exported --expect-shards 120 --index-size-basis file
```

Supply a reachable artifact path and the interpreter available on eligible
workers. The shard count is an example, not a model default. The tool enumerates
only the artifact root, reads the index and shard headers, and stats shard files;
it does not load, mmap, hash or decode tensor payloads. No GPU is needed.

## Checks and output

- Flat shard filenames in `model.safetensors.index.json` exactly match all
  `*.safetensors` files in the root. Without an index, only `model.safetensors`
  is accepted. Subdirectories, symlink shards/indexes and nonregular files are
  refused. Other sidecar files are not counted or inspected.
- Every header tensor occurs once and in its indexed shard; every indexed
  tensor is present. Duplicate JSON keys are rejected, not silently overwritten.
- Shared `shipcard.safetensors_header_spans` validation checks dtype, shape,
  offsets, contiguous extents and exact payload length. This is stricter than
  checking only the maximum offset.
- Header/index metadata reads are bounded to the existing 100 MiB header limit.
  Shard reads are unbuffered to avoid a Python read buffer touching payload bytes.
- `--expect-shards N` optionally checks the expected count.
- `--index-size-basis data` compares `metadata.total_size` to tensor payload
  bytes (the HF convention). `file` compares to physical shard bytes including
  headers (the current Tessera export convention). Choose the producer's known
  convention. **Omitting the option skips this comparison**, reported as
  `"index_size_basis": null`; the tool never guesses or rewrites the index.

Exit 0 prints a JSON diagnostic with `status=structure_ok`,
`scope=headers_and_index_only`, shard/tensor counts, `shard_file_bytes`,
`header_bytes` (including length prefixes), `tensor_data_bytes` and the chosen
size basis. Exit 1 prints `structure_error` and its reason. Argument syntax
errors use argparse's exit 2.

## What this does not establish

This is not a content attestation or a release gate. Same-size payload corruption
is intentionally invisible. Stat comparisons detect ordinary concurrent changes,
not hostile mutation; use an inactive export. It does not validate model coverage,
producer/Hessian identity, cached-unit schemas, export-log completion, total
artifact footprint/budget, runtime compatibility, quality or publication status.
A successful result fills no shipcard slots and changes no pipeline defaults.

Continue with existing `prismaquant.artifact_completeness`, authenticated
content/producer checks, native/route/quality qualification, and
`prismaquant.shipcard_cli show` / `verify`. Publish only through the existing
`tools/publish_artifact.py` refusal and frozen-snapshot path. None of those gates
can be replaced by `structure_ok`.
