# Frontier destination metadata ownership — 2026-09-08

A new validated assignment could inherit the overwritten allocator recipe's
Tessera wire/scale/population and serving/runtime claims. The selector now
requires complete canonical assignment equality before carrying those fields;
a changed assignment refuses before writing the layer config, assignment or
summary. Equivalent format string/AutoRound dictionary spellings still match.
The existing selected-payload CB and whole-artifact budget ownership is retained.

This is a fail-closed repair. It does not add selected Tessera metadata to the
allocator Pareto producer or the validator's resolved payload. Until that
handoff is implemented, each Tessera recipe must use the allocator's own final
metadata and selected-wire gates. No performance or serving claim is made.

PrismaBuild CPU qualification on dl380g10, Python 3.12 `pq-cpu312`, native
OMP/MKL/OpenBLAS threads each 1, priority -10, no GPU:

- Red: `2b7762dec63dea6faf3319a2c64c5afc53ae89c8e2191503e2031a37d15b4b85`,
  `pytest -q tests/test_select_validated_frontier.py -k destination_claims`:
  **7 failed, 1 passed**, actual exit 1; each failure reproduced the stale
  destination write. CPU 1 / 4 GiB.
- Green: `a45d7fc4dcfa525cc4132ccfa658f65f75511ab26720850c5cb725d0e2fcb171`,
  `pytest -n 4 -q tests/test_select_validated_frontier.py tests/test_layer_config.py
  tests/test_architecture_doc.py tests/test_docs_staleness.py`:
  **87 passed**, 56 existing Torch deprecation warnings, no skips, exit 0;
  59.64 seconds. CPU 4 / 8 GiB, observed cgroup CPU 219.69 seconds and peak
  memory 2,759,557,120 bytes. Source bundle matches the committed touched files.
- Compile: `30a9a302aab94749c275bb081d3d27cc3ba38e772ae823c647aeb86974b4b398`,
  `/usr/bin/python3 -m py_compile` selector, selector test and metadata sizing
  helper, exit 0. CPU 1 / 1 GiB.

Terminal records, actual log digests, source bundle comparisons and native CAS
receipt/result verification are in `pb-audit.json`. Failed red has no success
CAS receipt. Full sanitized audit files and log locators remain under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/six-variant-recipe-intake-01/`.
