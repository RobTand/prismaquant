# MTP source parameter binding: CPU regression

Closes #2081; Refs #1271. Strict positive integer storage metadata (PR #2069)
still admitted a parameter count unrelated to its priced source tensor.
The MTP selector used that count for BF16 passthrough bytes and the group
menu's bits-per-parameter denominator. The already validated source operator
had the shape needed to check the count, but `_mtp_probe` did not compare it.

The fix extends that shared identity owner: admit each unit's exact positive
integer count with `schemas.Contract.integer`, then compare it against the
product of every priced rung's validated `source_weight.shape`. Selection
and `validate_mtp_selection_record` both call this owner before eligibility
filtering. A malformed second rung therefore cannot hide behind ineligibility.
The existing producer writes `module.weight.numel()` and needs no change.
The same commit updates and re-stamps `docs/ARCHITECTURE.md`.

## Causal evidence

Baseline is the frozen PR #2069 head
`3ec8a96ab3fb2d2aba01ceb80724166f832875a1`. PB RED action
`364c6db752ee4e677fc1653807818fc2f223718b91c8bd89424700098ae9cc47`
ran five complete synthetic joint-AURA cases: five failures, all DID NOT
RAISE, zero skips, no collection errors, action exit 1. The source geometry
remained `[64,128]` for nine units, but counts of one were accepted under a
100-byte budget. Those metadata counts imply an 18-byte BF16 floor; the
validated geometry implies 147,456 bytes. Counts of 8191 or 8193 for a routed
or shared unit were also accepted instead of 8192. RED retains its failure
and has no success receipt.

GREEN PB fanout ran 12 shards: 127 collected, executed and passed, zero skips,
no missing collection, all shard exits 0. The six new cases include those
same five regression bodies plus a correctly hashed, ineligible second-rung
operator whose source shape differs. The remaining cases cover exact storage
counts, canonical/GLM selection, declared menus, priced-wire intake, card
rebasing, cost SHA/output binding, architecture/staleness and duplication.
Independent CAS readback parsed 12 actual `pbtest-outcomes` markers and
reconciled 127 collected IDs and 127 reports, zero skipped.

The new-case action is
`a91ca4d37a79ac4a087f6a26b0a92184ae658c4086aeb31ac232859bb730d256`;
result `d2a6f4c3b686260dbaa5aa8580a32171fbf8dfdf13bcf882d55557efda035d10`,
3,736 bytes; receipt
`582e23f6cc14b2849ab343bdcb1480ad4e193d5fe10296a14f4d74e8dc7aa9cd`.
Compile action
`a7a745f136907aa7b15abeb15494739799e66146569a90a27f9623e6f868852a`
compiled the touched module and new test file, exit 0. Its result is
`25359f6ba71a9f403f567b073a5775bb4288676c237cd0ea8c08c72b19ac97cf`,
1,103 bytes; receipt
`bb2c23d0d871dda01b66311c93f1f85778ca7ce9807c35369b509ce694955452`.

All validation was CPU-only on dl380g10, priority -10, one reserved CPU and
native threads one per action, preserving PB-assigned affinity. Test actions
reserved 4 GiB, GREEN shards had hard 300-second limits; RED had 180 seconds.
Compilation reserved 1 GiB with a hard 60-second limit. Interpreter:
`/home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python`, Python 3.14.4,
Torch 2.11.0+cpu, Transformers 5.16.1, pytest 9.1.1. Published pin preflight
verified PB `95a59051d48cda82eea7927f31870c6c862d7174` and Tessera
`b40c93cb73745097e57a1ba4cf5b9eee166c759a`, including RECORD/import ownership.

| Compiled/contract source | SHA-256 |
| --- | --- |
| `prismaquant/glm_mtp_selection.py` | `ec1db5d04beb70510c08fc6708cac31ba6179519f0b409333737dae1c535b78d` |
| `tests/test_mtp_source_param_count_1271.py` | `9ecac8f1fcd58857c3427ee0bd7825a9b593979450f5a4c0603df82e8cb2721b` |
| Same-commit `docs/ARCHITECTURE.md` | `d19c8e8936c7ddc11d62ece71ac517f6c6169bc6c2671e8bfaa2f51db6aaac81` |

The compile bundle is
`630c51ce12e9d58a1728eb12255a4f3c8ff5e527267152473cdee5b7f67314bb`,
snapshot `ea19d08354599e144d10135906a664b97fb484f1` over the frozen baseline.
Independent bare-Git readback reproduced all three current digests. All 13
GREEN test/compile terminal records and CAS receipts were read back with a
complete, non-timeout query; their actual result lengths and SHA-256 matched.
The report and compiled files remain in
`/mnt/shared/tessera-measurements/pq-serving-instrumentation-20261002/`,
including `mtp-source-count-regressions.json` and `mtp-source-count-red.xml`.
The lane retains `mtp-source-count-receipts.json`,
`mtp-source-count-cas-readback.json` and the bare source repository.

## Execution and scope limits

The two initially queued RED/prose actions waited on a fleet tmpfs census
defect. The coordinator rolled the published runtime back from generation
829a1 to qualified 5b6b97 using the normal publisher; the same sealed requests
then executed. Their submitters had already exited 75 after 600 seconds of
queue waiting, which is not their later action outcome. Neither request was
resubmitted, withdrawn, repinned or run outside PB. The 829a canary did not
qualify dl380 tmpfs and is not used as such here. GREEN was submitted through
the qualified published 5b6b97 generation. CPU resource profiles include
Netdata but report a missing pqteld CSV; this is no performance/energy claim.

A separate prose commit corrects the canonical MTP selector's docstring from
the retired live Gridbook script to its current GLM selector/allocator
integration. Original commit `a0f07996874d3980fdb1448019a5e0343f99f8a0`
has CPU compile action
`5211723481e9d843653f1e417097239725b4d402b1011eb77a5bbe21ee68f68a`,
exit 0, actual compiled file retained; its 50-byte CAS result
`3014d69aa311c21805b789758aaa1b02eb4e658f065e79e3a7bd3ec31cdaa8ef`
and receipt `fe67b2cfa4eaef644908da7bef76c2e9f72584bfc3223480028c83b1db8f63a2`
were independently read back. No behavior test was added for that prose edit.

These are CPU admission regressions, not original-model, encoded-byte,
numerical-quality, serving-throughput or energy qualification. Source count
binding changes no objective currency, priced wire, menu, runtime pin, kernel,
serving gate or default. The #1271 original/current MTP prices and matched
served acceptance against BF16 MTP and the rival remain open.
