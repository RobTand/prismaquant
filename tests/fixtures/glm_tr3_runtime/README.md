# Actual native TR3 runtime observations

These are byte-identical pre-scoring observations from two EXL3 vLLM boots
on the same two GB10 hosts with the same source, candidate, teacher and
configuration. They are diagnostic observations, not successful KL results.

- `qualified-runtime.json`: native attempt 07, 97,879 bytes,
  SHA256 `1bdb1b41125ce6ef373a4c87200ff6cbf2a440c1e5c5e8d04eac841228ece605`.
  Its binding equals the subsequent successful one-window qualification.
- `restarted-runtime.json`: full-panel attempt 08, 97,879 bytes,
  SHA256 `54f3c5a3e3b98d8af8e54a1fc7e2499ce8e5dd74beeeb9dc49f267b5f69e55e6`.
  This attempt exited 1 before scoring because the previous exact comparison
  treated the new cache allocation capacity as a changed runtime contract.

There are exactly 66 differences: allocated KV shape dimension 0 changes
12441 to 12532, 957 to 964, or 49764 to 50128, each 22 times. Every other field
is identical. The three recorded native backends' `get_kv_cache_shape`
functions declare `num_blocks` first. All remaining dimensions, source
hashes, dtype, device and module identity remain contract fields.

Original observations are retained under
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/` in
`tr3-exl3-hook-02` and `tr3-exl3-full-01`, respectively.
