# Fisher statistics ownership

`probe_statistics` owns the dense and packed marginal field sets, host marginal
reduction, scalar finalization, per-weight detail blobs, legacy normalization
and per-row detail denominator expectations. The probe, incremental probe,
cost measurements and allocator use these owners directly. There are no
compatibility exports at the old owners.

The alternative was a new accumulator class for every hook and writer. That
would require changing the device-resident batched reductions and their
transfer ordering. The existing raw-row wire and reductions already encode
the required math. Moving the shared host operations first preserves them
and gives later readers one denominator owner.

`tools/refactor_probe_statistics.py --base 8c4758c725 --write` performs the move
and caller migration. Without `--write`, it compares the definitions' syntax
trees against that source, allowing only the marginal constant renames.
`--rebuild` explicitly reconstructs the source modules; routine reruns retain
later changes. The tests still exercise scalar/detail agreement, routed and
global counts, visual pass stamps, grouped operands and marginal maxima.

The baseline is PB
`e6766b2dfec13cdb0d4c7d3f2b637d3d05d1e46007c566113995e8feaf065ff9`:
380 passed, 11 subtests, no skips. Final PB
`52ad5c7fbaef3bfed7ea27965baa40c7128b9391d9cb9445f99c82700f8e9743`
proved all moved definition bodies unchanged, compiled all eight touched
modules and passed 145 cases plus 11 subtests, with no skips. Its exit was
zero; its result CAS digest is
`55f8cf4556b37a1c459c1c15d430226e684945f7eca6e3db2d637a9a3ed95634`.
The first run passed 144 cases but caught the stale mount-helper duplication
inventory. Removing that retired entry restored the guard; no allowance grew.
This ownership change alters no calibration, tensor arithmetic,
format, price policy or GPU execution order.
