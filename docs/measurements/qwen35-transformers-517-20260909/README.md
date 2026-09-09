# Streamed Qwen rotary compatibility — 2026-09-09

Transformers 5.17.0 moved text-position expansion out of Qwen3.5's rotary
module and into its model forward. The streamed driver bypasses that forward,
so passing `[batch, tokens]` directly to rotary raises `IndexError`.
The shared `_compute_position_embeddings` helper now asks the model profile
for rotary-only positions. Qwen supplies a three-axis view of text positions;
explicit three-axis inputs are preserved. Mask and layer positions retain their
original shape. Other profiles pass positions through.

The real Qwen integration tests now exercise the shared streaming helper.
Both padded and unpadded forwards compare its rotary output exactly against
the native module with explicit three-axis positions. Explicit unequal axes
are preserved, malformed axes refuse, and the original negative dense-mask
test again reaches the attention layer. Existing mask, generic/multi-rope,
streaming and architecture tests are included.

PB reproduced the original three failures and then the same failures through
the shared helper, before the repair. The repaired suite passed **70 tests
without skips on each of Transformers 5.16.1 and 5.17.0**, plus compilation
of the three changed modules. Runs used the existing scoped Python 3.12.11
CPU environment on DL380; 5.17.0 was installed as a separate pinned overlay.
Each validation action reserved four CPUs and 6 GiB, bounded native threads
to one, and used priority -10. Actual terminal cleanup, CAS receipt/payload,
and source were verified; `cpu-audit.json` contains action keys and hashes.
The only later source difference was the architecture stamp's branch name.

The original hosted CI run was
https://github.com/RobTand/prismaquant/actions/runs/34397560209
(three failures, 7538 passed, 207 skipped, three xfailed, 192 subtests passed).
This is a shape-compatibility correction, not a GPU performance or model-quality
measurement. Tracked by #477.
