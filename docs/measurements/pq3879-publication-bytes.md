# Tessera campaign publication bytes for #3879

Row 131 covers three selected-source outputs in `prismaquant/tessera_campaign.py`.
Main at `b782bfa9c` (this branch's base) still uses `json.dumps` at lines 4331, 4571, and 4576.

> The same conditions as #2650 apply: route only to DIRECT_ASCII_INDENT2_LAX, DIRECT_ASCII_SPACED_LAX or DIRECT_ASCII_LAX; change no `digests.py` code; every published byte string and derived digest is byte-equal before and after (recorded literals from the unedited code, STOP on any difference); no gold-producer closure file (`tools/serve_fingerprint.py:89-126`).

| Row 131 output | Main verdict | Main test evidence |
| --- | --- | --- |
| Load-all execution record | PARTLY | `tests/test_tessera_row_stream.py::test_the_stream_head_writes_the_load_all_heads_bytes:196` runs both heads and excludes the execution record from its byte comparison. |
| Selected-anchor refusal file | PARTLY | `tests/test_campaign_demand_derivation.py::test_the_row_refusal_detail_carries_every_term_of_its_predicate:48` checks the input values for the file. |
| Selected-anchor refusal message | PARTLY | The same test checks the input values for the message. |
| Derived digests for these outputs | MOOT | The calls at lines 4331 and 4571 write the two files directly. Line 4576 builds the raised message directly. |

`tests/test_tessera_row_stream.py::test_selected_load_all_writes_exact_execution_and_refusal_bytes:239`
compares the three outputs with literals transcribed from the unedited encoders.
The test uses the existing CPU capture fixture and supplies the CUDA admission
inputs. It checks the actual file bytes and exception from the CLI.

Run the new node in Flux with one CPU, 4 GiB of RAM, and no GPU. Set
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. Flux CPU job `f3dZY1HMhxB` at `187c3f004` (gate venv): 24 passed, exit 0; `tools/ci/static_checks.sh --merge-main` exit 0. The mutants `indent=2` -> `indent=4` at line 4334 and `sort_keys=True` -> `sort_keys=False` at line 4576 each fail the new test.
The three profile changes remain pending. Stop if any output differs from
its literal after the profile changes. This slice changes only a test and this note.
