# Capture tool stdlib startup correction

Both tools reuse `tools/pq_profile_source.py::profile_source_owner("digests")`
via repository-relative file-spec loading, matching corpus/vLLM. The owner
route remains `prismaquant.digests.bytes_sha256hex`; no shared files changed.
Existing serializer and tensor-byte behavior is retained.

`test_capture.py` now runs both real file-spec imports and the MTP script's
`--help` path in fresh `python -I -S` children from an unrelated temporary
working directory. A meta-path guard rejects even attempted imports of
`prismaquant`, `torch`, or `tools`. Children verify unchanged `sys.path`, owner
function identity, and digest output (including the EXL3 `_hash` call with a
controlled byte supplier). These are startup/CPU checks, not GPU qualification.

## Observed checks

- `python -m pytest -q tests/test_digest_1301/quality-calibration/test_capture.py --basetemp="$TMPDIR/pytest"`: **10 passed**.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1261283c402670d7f8f0f7268ce9fe2cfc4ea99c37949dd2f8568390e0e86c85","size":269}},"operation":"60dedf2bcd149a6b15828d9adc224a3bd25ba0eee9201932571263621fa9428c","receipt":"5ef0ac6543d42f116f88b6729125306b4662391389c79d46e95d8ecbef66a7a2"}]

- `python -m pytest -q tests/test_exl3_grouped_down_replay.py tests/test_glm_mtp_capture.py tests/test_mtp_projection_initial_publication_audit.py --basetemp="$TMPDIR/pytest"`: **62 passed, 10 skipped**, 14 torch deprecation warnings. Skips are not passing evidence; no GPU result is claimed.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"988f5f3fb0ecfeeff108afa1a118fa4fe1c62c2294949ca292a6d96eee6b443a","size":833}},"operation":"227fc81340a9cc573c0f97eb587d4bd6d96326445c8c67026713838f3336219d","receipt":"a66b551d256624601168ff0b21929f151bfe88e1a2c5fa4210766d4262f86c4d"}]

- `git diff --check`: exit 0.
