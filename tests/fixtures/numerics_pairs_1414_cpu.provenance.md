# CPU numerics-pair golden provenance (PQ #1414)

`numerics_pairs_1414_cpu.json` contains 36 expected calls from the production
functions at Git commit `d6a490cd7561a5a6a2c41ed270563f0f33c023ec`, the
parent of the first #1394 consolidation. The updated test module was placed in
that detached checkout; no production function was copied or regenerated from
the consolidated implementation. PrismaBuild recorded the CPU cases and an
averaging-sensitivity fixture guard with `PQ_GOLDEN_RECORD` on ARM
(`a04505036449`, Python 3.12.3, Torch 2.11.0+cu130, CPU execution) and x86
(`dd0a58d9c40a`, Python 3.12.3, Torch 2.14.0+cpu). Both recordings had 7
passes and produced identical table bytes:
SHA-256 `296f548c31b883ff74de4650cd791750193149ca22e94ab4dcdd74605dbfbf13`.

The historical `numerics_pairs_1394.json` remains byte-for-byte unchanged
(SHA-256 `5dab5cc3a1fd9992d686f85356d3083fe99c274f29ec4aa9edddcc1a8b7a0fa9`).
The new table is selected only for CPU MX, imatrix and marginal cases. The
empty activation file's imatrix remains an all-NaN tensor with its dtype and
shape checked; CPU reduction kernels produced different NaN payload bits on
ARM and x86. Every nonempty result retains an exact raw-byte digest. CUDA
still uses the original inputs and historical table.
The imatrix rows vary in magnitude; the fixture guard proves a first-row
shortcut or omission of the first row changes the intended squared mean.

The unmodified main test failed at its first MX CPU golden on x86
(`580edea41346`, Python 3.12.3, Torch 2.14.0+cpu). The corrected full module
passed there (`fecad34b7d5d`: 11 passed, 4 CUDA skips) and on ARM with Torch
2.11 CPU execution (`a9d1bc1e8814`: 11 passed, 4 CUDA skips). Both runs also
compiled the touched test module. No GPU test was run for this CPU-only repair.
