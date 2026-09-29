# Tessera serving-runtime pin

`tessera_serving_runtime_pin.json` is PrismaQuant's whole producer-side
boundary to Tessera's vLLM serving plugin, read by
`prismaquant/tessera_serving_runtime_pin.py`. PrismaQuant never vendors or
imports the serving half of that runtime; compatibility crosses the repository
boundary through this pin and through the contract Tessera packages
(`tessera/serving/runtime_contract.json`, read via `importlib.resources`).

**This pin is an exact commit plus the contract's digest — not a release
tag.** Until 2026-09-04 `commit` and `version` were the sentinels
`PENDING_TESSERA_RELEASE_COMMIT` / `PENDING_TESSERA_RELEASE_VERSION` and
admission was fail-closed because no Tessera release tag existed. Rob retired
that: *"can we just pin prismaquant to latest version of tessera? then we won't
have to keep cutting releases."* "Latest" is read here as **an exact commit
plus the packaged contract's raw SHA-256**, never as a floating ref — `main`,
"the installed source tree" and "whatever imports" are precisely the failure
principle 14 exists to prevent.

Of the two, the **digest is the enforced half**. PrismaQuant cannot verify a
sibling checkout's git history from inside its own process, but it can hash the
contract bytes it is about to read, and those bytes are the only thing about
the runtime a gate here consumes. So `require_pinned_tessera_runtime()` refuses
whenever the installed `tessera/serving/runtime_contract.json` does not hash to
`contract_sha256` — which keeps the stray-checkout property the PENDING
sentinels used to provide: a Tessera source tree on `PYTHONPATH` that is not
the pinned one is refused exactly as before. The `commit` is recorded identity:
it says which reviewed tree those bytes came from, and `git` settles any
question the digest raises. The two are bound at review time by one pair of
commands, so they cannot become two independent assertions about one runtime:

```bash
TS=/home/rob/tmp/tessera-pin-probe-$$ && mkdir -p "$TS" && git -C "$TS" init -q
git -C "$TS" fetch -q https://github.com/RobTand/tessera master
SHA=$(git -C "$TS" rev-parse FETCH_HEAD)
git -C "$TS" cat-file -p "$SHA:src/tessera/serving/runtime_contract.json" | sha256sum
```

The commands name the canonical remote rather than somebody's checkout,
because a digest bound from a working tree records what that tree happened to
contain, which nobody else can re-derive.

The current pin is Tessera `83460680ed84e33c82eb62b31345381cc151aa58`,
master's merge of #725 on 2026-09-29 (tessera#724, PQ #1719). It is the
Tessera the GLM-5.3 T8R release serve runs: the routed window intake repacks
each unit in place into the loader's scratch instead of making three to five
fresh large-pool allocations per projection, which under vLLM's
`max_split_size_mb=20` killed the T8R TP2 load on host memory. The package
diff is `compact_prep.py` and `kernel_wire.py`. The contract (v45,
`0869f326…`), `export.py` and `grammar.py` are byte-identical to `a21d74d8`,
so the reviewed answer does not move, the legal inventory renames the
`reader-pin-a21d74d8` byte-state `reader-pin-83460680`, and no rate count
moves. The pin stays schema v2.

The previous pin was Tessera `a21d74d89bd4eca0493a2f913c71b28b03a39d8b`,
master's merge of #701 on 2026-09-29 (contract v45, tessera#694, PQ #1702).
It crosses the 15 master merges since
`a5f3b232cb` (#656, #699, #700, #705, #707, #709, #711, #713, #715-#718, #720,
#722, #723), all at contract v44, and #701's v45: the two fused routed lanes
read `column_rates` [1..8] (was [4]) and publish the structure-scoped
`column_rates_routed_moe` [1..6], the rates their routed-expert launch reaches.
Every rung the four window routed cells list plans rates inside [1..6], so a
mixed-rate routed stack (q832 to q1088) now records the fused pair beside the
compact one instead of the compact pair alone. No cell id, rung, flag or
activation contract moves, the pin JSON's extension rows are unchanged, and
the reviewed answer moves only in the two lanes' `requires`. `export.py` moves
inside `ActivationSource` only (#709's seal-header fold, `427a8f97…` ->
`2127e82b…`) and `grammar.py` does not, so the legal inventory adds the
equivalent `reader-pin-a21d74d8` byte-state and no rate count moves. The SHA-256
moves to `0869f326…`. The pin stays schema v2.

The previous pin was Tessera `a5f3b232cb3c424b537a06713c728c86153d55fb`,
master's merge of #691 on 2026-09-28 (contract v44, tessera#687, on top of
#693's v43; PQ #1616). v43 names the fused window kernel's dense identity as a
second launch in the six dense window cells; v44 moves the supported exporter
into the installed package and moves no admission answer. `export.py` moved by
one docstring line (`e54f3f1b…` -> `427a8f97…`), so the legal inventory added
the equivalent `reader-pin-a5f3b232` byte-state. The SHA-256 was `47b01355…`.
This README was not updated for that pin; PQ #1702 records it here.

The pin before that was Tessera `38e960127478b651e42c14d52acf2274b54bca38`, master
on 2026-09-28 after #678 (contract v41), #685 (contract v42, tessera#640) and
#686 (PQ #1274). v41 lets a cell's `runtime` stamp `tessera_commit` and
`serving_source_sha256`; no packaged cell does. v42 adds the fused routed
window MoE lane: two lane-bearing native extensions,
`tessera_routed_fused_e4m3` (route `TESSERA_FP8`) and
`tessera_routed_fused_value` (route `TESSERA_BF16`), whose `when_unavailable`
substitutes the compact adapter's decoder, and the four window routed cells
name the fused pair beside the compact pair. The pin JSON's
`serving_native_extensions` gains both rows, and the reviewed answer moves in
exactly those two places. `export.py` and `grammar.py` are byte-identical to
`db5b6e23`, so the legal inventory renames that byte-state
`reader-pin-38e96012` and no rate count moves. The SHA-256 moves to
`4aeba5dc…`.

v42 is the first pinned table in which a cell names a lane launch beside a
lane-free one, and PrismaQuant's lane gate refused a whole cell whenever its
lane refused the plan. That would have dropped routed E4M3 at every q256 rung
but 1024, where the fused lane cannot read the mixed-rate plan and the
dispatch keeps the compact adapter. `lane_eligibility.cell_rung_launches` now
drops only the launches through a refusing lane, as Tessera's own
`executes` derivation does, and a route records the launches its rung makes.

The pin stays schema v2. A v3 pin admits only cells censused with the code
fields, and none is. For the record, `serving_source_sha256()` of the
`38e96012` tree is
`445da73b1960f2de20fe248184f49f50c1b1d2f0809a6ab47d2057f43982244d` on all five
of its interpreters.

The previous pin was Tessera `db5b6e23a06869e87d778cb223c1ac5a5154aec5`, master
on 2026-09-28: the merge of Tessera #675 (tessera#599 step 2, PQ #1537), whose
tree equals the PR head `43da1c39` that this change was first tested at. Tessera
no longer names PrismaQuant's record schemas: a rooted cached-unit bundle
(`tessera.cached_units.v2`) takes a `ReuseAuthority` from its caller, and a
Hessian reference takes the canonical `(schema, source)` pair from its caller.
Both refuse by name without one. PrismaQuant's authority is
`prismaquant/tessera_reuse_authority.py`, stdlib-only, with the checks moved
verbatim: `read_cached_unit_bundle`, `open_hessian_reference` and
`tessera_hessian` pass it in process. Tessera's export drivers take its path
as `--producer-authority`, and contract **v40** publishes which drivers do
(`producer_interface.reuse_authority.drivers`: the serving exporter,
`export_glm53_tessera.py`, `glm_routed_owner_inputs.py`,
`bf16_reach_roster.py` and `tools/glm_cpu_cached_pack_probe.py`). PrismaQuant
reads that block and never asserts it: `run-pipeline.sh` and
`dispatch_tessera_campaign.py submit-export` add the option only when the
checkout the export runs from publishes it
(`tessera_export_lane.producer_authority_argv`), so an older pin such as the
live campaign's `a3e83875` gets exactly the argv it got before. v40 is
additive: the contract SHA-256 moves to `d6768313…` and the reviewed admission
answer does not. `grammar.py` is unchanged. `export.py` moves (`from_capture`
gains `canonical_capture`) to `e54f3f1b…` with the wire functions
byte-identical, so the legal inventory adds the equivalent
`reader-pin-db5b6e23` byte-state and no rate count moves.

The previous pin was Tessera `20bf53464f9113f3115f454f8fa80453e71c0308`, master
on 2026-09-27 after #669 (tessera#668, PQ #1527). At a mixed-rate window rung,
the batched LDLQ encode runs a window span's rate calls on per-rate CUDA
streams, bit-exact, which cuts the G2 w01b encode by 18.5-22.4 % a batch. Only
`encode.py` and docs move: the contract, admission answer, `grammar.py` and
`export.py` are byte-identical to `a3e83875d`, so the legal inventory keeps the
`reader-pin-f94929de` byte-state and no rate count moves.

The previous pin was Tessera `a3e83875d20f54c685307da13a34992d70256f02`, master
on 2026-09-27 after #671 (tessera#670, PQ #1524). The rooted cached-unit
reader accepts `prismaquant.joint_catalog_extension.v3`, the schema of the
real Stage B catalog extension. The contract, admission answer, `grammar.py`
and `export.py` are byte-identical to `f94929def`, so the legal inventory
keeps the `reader-pin-f94929de` byte-state and no rate count moves.

The previous pin was Tessera `f94929defd9fa00b8726160a2cd436f02733b6dc`, master
on 2026-09-27 after #663 (tessera#662, PQ #1502). The contract, admission
answer and `grammar.py` are unchanged from `4c4ff1c2e`. `export.py` gains
`served_recipe(grid, q256, structure)`, and the cached-unit receipts stamp it
per structure. The legal inventory records the new `export.py` bytes as the
additive byte-state `reader-pin-f94929de`; no rate count moves.

The previous pin was Tessera `4c4ff1c2eb68d4ffc3f8e0d3fcf9e019db5ab253`, master
on 2026-09-27 after #646 (explicit cached-reader proof mode and v2 multi-rung
activation policies, PQ #1456). Contract **v39**, lane schema **v10**, SHA-256
`f2f909486841c6e21ef6825fdc67ea5f57cf8ff8ffc241ccd89a521ea781c0bb`.
Relative to the previous `09d6559d7…`/v38 pin, the reviewed answer adopts
#641's withdrawn, minted and widened scopes: `f8dbe1a0…` carries every
GLM-image cell while the vanilla image retains dense E4M3 q1024.
`TESSERA_DEV_PIN_ANSWER`, `tests/test_tessera_pin_v38_scope.py` and
`docs/ARCHITECTURE.md` record that review. `grammar.py` and `export.py` are
byte-identical to the previous pin; the legal inventory keeps its unique
`reader-pin-09d6559d` byte-state and unchanged wire counts. Install that revision
and point `TESSERA_REPO` at its complete checkout; the producer scripts live in
`experiments/` and are not wheel entry points.
Provision the pin venv from a git URL so the install records the commit
(`git+https://github.com/RobTand/tessera.git@<pin>`). The PrismaBuild test
interpreter for the current pin is
`/home/rob/venvs/pq-pb059953bc-tessera-83460680`. It carries both pins this
tree tests against: Tessera `83460680` and PrismaBuild `059953bc`
(`staged_lease.PB_READER_LEASE_PIN_COMMIT`, PQ #1541). It is a copy of
`pq-pb059953bc-tessera-a21d74d8` with only Tessera reinstalled, non-editable,
provisioned by host-pinned PB build actions on 2026-09-29: dl380g10
`05240c885661` and sparklina `a2f3caa3007c`. The sparky action
`3a14fa1bb30c` is queued, so sparky has no `83460680` interpreter yet. Each
Spark action also builds the `-tf516` sibling.
The `a21d74d8` interpreters (PQ #1702) are copies of the `a5f3b232` ones
built the same way: dl380g10 `177cb8e57a27`, sparky `53dff8124e53` and
sparklina `8323b283eb2d`.
The `a5f3b232` interpreters (PQ #1616) are copies of the `38e96012` ones
built the same way. The `38e96012` interpreter is a copy of
`pq-pb059953bc-tessera-db5b6e23` with only Tessera reinstalled, non-editable,
provisioned by host-pinned PB build actions on 2026-09-28: dl380g10
`c94b89b4e2ac`, sparky `d36bf663315b`, sparklina `13d94ab22f4d`. Each Spark
also has a `-tf516` sibling, copied from `pq-pb059953bc-tessera-db5b6e23-tf516`
with only Tessera reinstalled (sparky `07013de7488f`, sparklina
`e8cb68eab74e`). Each of those builds also ran the pbtest dependency-pin
preflight (`pbtest_pins.preflight`) against this tree's resolver. The
`db5b6e23` interpreters (dl380g10 `18ee90816fc6`, sparky `7d203c3cdacb`,
sparklina `27b513d61205`; `-tf516` sparky `cb5f94eae9b2`, sparklina
`ddd47cb807ed`) came from `20bf5346` and `43da1c39-tf516`. Each build asserts both commits in `direct_url.json`
(distributions `tessera-quant` and `prismabuild`, neither editable), the
contract, `export.py` and `grammar.py` digests, that `tessera.cached_unit`
defines `ReuseAuthority` and no PQ schema roster, that
`tessera.hessian_capture` names no canonical capture, that
`tessera.producer_authority` spells the option, and that
`prismabuild.client.SDK_VERSION` is 1. The `43da1c39` interpreters (#675's
head, never a merged pin), the `cadc200c` ones, the
`pq-pb059953bc-tessera-20bf5346` ones (PQ #1541's test interpreter), and the
`20bf5346` interpreters (dl380g10 `68bdb80acd8d`, sparky `6940c8f1c592`,
sparklina `062be8b71f67`) and the `a3e83875` ones stay in place. Old interpreters remain untouched for running
sessions. Announce the new `--python` paths before merging a pin move; never
update a live venv in place. On each box, run as a host-pinned PB action
(`--tag <host>`); the GB10 boxes repeat it for the `-tf516` sibling, and
the copy repoints any `base-shadow` `.pth` file from the source venv to the new
one:

```bash
SRC=/home/rob/venvs/pq-pb059953bc-tessera-a21d74d8  # + "-tf516" for the sibling
V=/home/rob/venvs/pq-pb059953bc-tessera-83460680    # + "-tf516" for the sibling
test ! -e "$V" || exit 1
cp -a "$SRC" "$V"
for pth in "$V"/lib/python*/site-packages/*base-shadow*.pth; do
  [ -e "$pth" ] || continue
  grep -qF "$SRC" "$pth" && sed -i "s#$SRC#$V#" "$pth"
done
"$V/bin/python" -m pip install --no-deps --no-build-isolation --force-reinstall \
  'git+https://github.com/RobTand/tessera.git@83460680ed84e33c82eb62b31345381cc151aa58'
```

dl380g10's x86 interpreter descends from `pq881-pb461728e4` (Python 3.14,
CPU torch, no `base-shadow` and no `-tf516` sibling). Each build action's
request in CAS (`cas/requests/<key[:2]>/<key>.json`) carries the exact script
it ran, with its assertions.

**The GLM test modules need transformers 5.16 (PQ #1090).** On the Sparks,
the base interpreter takes transformers 5.6.0 from `pq-cu130`, and
`tests/test_glm5_next_streamed_forward_parity.py` skips at collection below
5.16 (`importorskip("transformers.models.glm5_next")`). Five modules import
from it and skip with it: `test_glm_campaign_streaming`,
`test_tessera_stack_group_cli`, `test_selected_snapshot_scope`,
`test_collector_source_release` and `test_streamed_capture_admission`. Until
2026-09-23 none of the six had run in a PrismaBuild test run.

The sibling interpreter `/home/rob/venvs/pq-pb059953bc-tessera-83460680-tf516`
is the interpreter above with transformers 5.16.1, tokenizers 0.23.1 and
safetensors 0.8.0, the versions in the campaign's `prismaquant-tf516` venv.
Tessera, PrismaBuild, torch and every other package are the same, so
`pbtest` checks the same exact pin. Each pin move copies the previous pin's
`-tf516` sibling in the Spark build actions above; the `83460680` siblings
descend from `a21d74d8-tf516`, the `a21d74d8` ones from `a5f3b232-tf516`,
the `a5f3b232` ones from `38e96012-tf516`, the
`38e96012` ones from `db5b6e23-tf516`, the `db5b6e23` ones from `43da1c39-tf516` (with PrismaBuild reinstalled at `059953bc`), the
`43da1c39` ones from `cadc200c-tf516`, the `cadc200c` ones from `20bf5346-tf516`, the `20bf5346` ones from `a3e83875-tf516`, the `a3e83875` ones from `f94929de-tf516`, and the `4c4ff1c2` ones from `09d6559d-tf516`. The original
transformers installations came from actions `fa510fbf5a02` (sparky) and
`29ec7048687b` (sparklina, 2026-09-25). dl380g10's interpreter of the base name is a
different base (`pq881`, Python 3.14, CPU torch), so pin a run that uses it
with `--tag gb10`. It is built the same way as its base, so a Tessera pin
move re-provisions it too:

```bash
# Historical bootstrap of a tf516 sibling; a pin move copies that sibling.
SRC=/home/rob/venvs/pq-pb461728e4-tessera-4c4ff1c2
V=/home/rob/venvs/pq-pb461728e4-tessera-4c4ff1c2-tf516
test ! -e "$V" || exit 1
cp -a "$SRC" "$V"
echo "$V/base-shadow" > "$V/lib/python3.12/site-packages/pq846-base-shadow.pth"
# Unlink the base's copies from the new base-shadow first, so pip never
# uninstalls through a link into pq-cu130.
for n in transformers transformers-5.6.0.dist-info tokenizers \
         tokenizers-0.22.2.dist-info safetensors safetensors-0.7.0.dist-info; do
  test -L "$V/base-shadow/$n" && rm "$V/base-shadow/$n"
done
"$V/bin/python" -m pip install --no-deps --no-cache-dir \
  'transformers==5.16.1' 'tokenizers==0.23.1' 'safetensors==0.8.0'
```

A PR that touches those six modules, or what they test, runs them there:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py \
  --checkout <this worktree> --tag gb10 --priority -10 \
  --python /home/rob/venvs/pq-pb059953bc-tessera-83460680-tf516/bin/python \
  --threads-per-shard 1 --workers-per-shard 2 --cpus-per-shard 2 --mem-gb 8 \
  --timeout-s 1800 \
  tests/test_glm5_next_streamed_forward_parity.py \
  tests/test_glm_campaign_streaming.py tests/test_tessera_stack_group_cli.py \
  tests/test_selected_snapshot_scope.py tests/test_collector_source_release.py \
  tests/test_streamed_capture_admission.py
```

Every session prints a `pq-test-environment:` line with the interpreter and
its torch, transformers, tessera-quant and prismabuild versions
(`tests/conftest.py`), so each shard's receipt names the transformers it ran
under.

The whole suite passes on this interpreter too. At `26683550bdbf`, 40 shards
on sparky collected and ran 12,040 tests: 11,821 passed, 216 skipped and 3
xfailed, and no skip names transformers (PrismaBuild keys in PQ #1090's pull
request). So no test needs transformers 5.6.0. Whether this interpreter
replaces its base as the PrismaQuant test interpreter is a separate decision;
until then the base keeps running everything else.

The previous interpreter, `pq-pb461728e4-tessera-acf9eafa`, stays on sparky
and sparklina for work sealed against that pin.

**v32 -> v34 (2026-09-22).** The answer diff is five additions and no
removals. v33 (#568) publishes the `sm_121` fp4 activation-quantiser table as a
list of per-image attestations: the stock-image row is byte-identical to v32's,
and a second row for the `glm53-nope-sm121` serving image carries the same
rounding vectors. v34 (#579) attests the fused window GEMM
(`tessera::window_gemm_dense`) and mints four dense `sm_121` cells:
`TESSERA_BF16_K1` q1792 and `TESSERA_E4M3_K1` q1024, batch and decode, graded
`route_only` with `smoke.status: not_recorded`. The status-only evidence gate
does not refuse that grade, so this pin admits those two dense rungs on
`sm_121` as `backed_with_serve_flag` where the v32 pin answered
`unattested`/`no_cell`. `export.py` and `grammar.py` are byte-identical to the
v32 pin, so the legal domain is a re-transcription.

The history below describes the v32 pin (`cc739a55`, 2026-09-19).

It moves the contract from **v29** to **v32**, and the lane-eligibility schema
stays at **v10**: every bump in between is additive for a v10 reader. Four
admission facts move, and `TESSERA_DEV_PIN_ANSWER`'s diff is their review:

- **withdrawn (master-side, v30/v31):** all eight non-`E2M1_K2` dense cells
  -- the four `TESSERA_BF16_K1` rows on `sm_121` and `gfx1201` (which carried
  `recorded` evidence and were the only cells on the AMD platform) and the
  four `TESSERA_E4M3_K1` dense rows, resident and streamed (Tessera #538 and
  the A4 whole-weight retirement). Accepting this pin therefore admits
  STRICTLY LESS than v29 did on dense routes: `TESSERA_BF16_K1_R1792`
  answers `unattested`/`no_cell` everywhere again. The routed-MoE
  `recorded` pair (`TESSERA_E4M3_K1`, q1024) is byte-identical, so the
  status-only evidence gate admits the same scope it did at v29.
- **v32 (#560):** the routed `TESSERA_E2M1_K2` reader domain widens from the
  single rung `[896, 896]` to the full trellis domain `[128, 896]` step 128,
  and the two routed-MoE cells widen with it. The receipt is the seven-rung
  green load (PB `a186d7bc6f1f…`). The cells stay `route_only` /
  `not_recorded`, so that widening is Rob's call under principle 9 (#198),
  flagged in the answer diff rather than decided by it.
- **D2/D2b (#562):** the unreceipted DENSE half of #560's widen is reverted
  (the dense `E2M1_K2` cells keep rung 896 only), and served KL receipts are
  scoped to the rung they measured -- the `q256` key this reader now parses
  and projects as `@q<rung>` in the answer's kl token.
- **A4 retirement:** the span-2 CUDA decoder is gone, so
  `native_extensions` (and this pin's `serving_native_extensions`
  transcription) drops the `tessera_nvfp4_` row and keeps the one
  window-GEMV row.

Nothing else moved: lane schema v10, the TP ceiling stays 2 on the same
receipt, the quantiser table is byte-identical, and `TESSERA_E4M3_K1`'s
family row is unchanged.

**Gated landing (PrismaQuant #760), executed.** The pin is repointed to
Tessera master's merge of #563 -- the head at which the #562/#563 union is
complete -- and the digest is the measured hash at that commit:

```
union contract sha256
3cb67d98b325abdfc1c11c16b6e2edb3dff915ba673dd941f6b0ed41a9c4df34
```

The union digest predicted while the gate was written (`712a15e4…`) went
stale: #563's rework resolved two master conflicts on its branch, so the
landed union bytes differ from the prediction. Verified by the `git
cat-file` command below. Tessera master has since advanced to contract v33
(#568); this pin deliberately does not chase it.

Re-check the exact commit:

```bash
git -C "$TS" cat-file -p 83460680ed84e33c82eb62b31345381cc151aa58:src/tessera/serving/runtime_contract.json | sha256sum
```

No tag names this commit, so `version_is_release` remains `false`.

History worth keeping: this paragraph named `ba582d47` (Tessera #356,
2026-09-05, the `--priced-inputs` producer API) while the JSON beside it had
already moved to `387eda36` (Tessera #441) — regenerated prose that was not
regenerated. The commands below are the procedure; running them is what keeps
this section true.

**`version_is_release` is advisory.** Still required, still parsed, still
recorded, and still unable to be `true` over a PENDING commit — so it keeps
saying something true for an actual release. It gates nothing: a gate that
demanded it would re-impose the tag Rob just removed, and the immutability it
stood in for is now carried directly by the digest.

**Moving the pin is one reviewed commit** that resolves, together: `commit`,
`version` and `contract_sha256` in this file, and
`TESSERA_SERVING_RUNTIME_PINNED_VERSION` /
`TESSERA_SERVING_RUNTIME_PINNED_COMMIT` /
`TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256` in
`prismaquant/tessera_serving_runtime_pin.py`. The reader requires the file to
equal the constants, so neither half admits anything alone.

**A consequence, by design.** A developer checkout of Tessera that has moved
past the pin makes PrismaQuant's Tessera tests red: the installed contract is
not the pinned contract, and fail-closed is the whole point. The fix is
environmental — install Tessera at the pinned commit — never a check that reads
whatever is installed.

**`serving_native_extensions` names what the plugin LOADS, not what it
executes — and it is DERIVED, not asserted.** Tessera's serving plugin
JIT-builds a CUDA decoder and loads it as `tessera_nvfp4_<identity>.so`
(`tessera/serving/ext.py`), and §7.4's reproducibility contract keys KL
comparability on whether a lane's `.so` was resident in the serving process:
two KLs are comparable only across serves whose native-extension residency
matches. Since Tessera contract v7 the runtime publishes that itself, in
`native_extensions`, as four values a consumer can act on — the
`module_name_prefix` the JIT load path itself passes to `cpp_extension.load`,
the `filename_glob` that produces (there is no exact basename: the module name
carries a build-identity hash), `match`, the name of the RULE a gate
applies (`basename_fnmatch`), and `when_unavailable`, what a serve does when
the library is absent (per residency mode: the substitute decoder it keeps
running on, or that there is no serve at all).  The fourth is what lets a
manifest say "the Tessera decoder was expected and is missing" instead of
recording the same absent-basename list as a stack with no Tessera in it
(PrismaQuant #142).

The chain is **contract → pin → fingerprint, with a refusal at each link**:

* `tessera_runtime_contract.require_pin_native_extensions_match_contract`
  refuses a pin whose rows are not the pinned contract's table, in both
  directions — a library the contract publishes and the pin omits makes the
  fingerprint go quietly short, and a library the pin invents is a claim about
  a runtime that does not load it. `contract_answer` carries the table, so a
  Tessera commit that renames the library re-stales the dev pin with a
  field-level diff instead of widening silently.
* `tools/serve_fingerprint.py` is stdlib-only by construction — it runs
  *inside* the serving container from a bootstrapped snapshot of five tool
  files and no package data — so it cannot read either file at runtime and
  carries the same rows as a constant;
  `tests/test_tessera_serve_fingerprint.py` refuses any disagreement, and also
  refuses a tool whose PREDICATE stops being the rule the contract names.
* the member is required, so the JSON and `tessera_serving_runtime_pin.py`'s
  member set move in one commit, the same rule the release constants carry.

History worth keeping: until 2026-09-03 this field was a hand-written
`serving_extension_basenames` with nothing here able to refuse it on drift
(principle 14 read backwards), and it was already wrong by one character —
`"tessera_nvfp4"` where the load path's constant is `"tessera_nvfp4_"`. The
fingerprint also matched it with a bare substring search over the whole mapped
path, which answers yes for
`/root/.cache/torch_extensions/tessera_nvfp4_9f2c/unrelated.so` and is not the
runtime's predicate. RobTand/tessera#28 published the table; PrismaQuant #133
consumed it.

**`repository` is the reviewed identity of the runtime**, not a reachability
claim: nothing in this repository fetches it, and the pin is satisfied by the
contract bytes an installed Tessera packages, never by an origin.

**No wheel digest.** Gridbook's serving pin binds an exact reviewed wheel
SHA-256 because Gridbook is installed into a serving container from a published
archive. Tessera's plugin is installed from a source checkout
(`pip install --no-deps --no-build-isolation -e <tessera>`) and publishes no
wheel; asserting a digest for an archive that does not exist would be exactly
the hand-asserted claim principle 14 refuses. What it DOES bind, since
2026-09-04, is `contract_sha256` — the digest of the one Tessera artifact that
both exists and is read by a gate on this side. When Tessera publishes wheels, a
`wheel_sha256` member joins it here and in the reader in one reviewed commit.

---

## Pin schema v3: the serving code identity (#1561)

The reader accepts a second schema, `prismaquant.tessera_serving_runtime_pin.v3`,
which splits `commit` into three members. The tracked pin stays v2 until a
reviewed bump activates v3.

| Member | What it names | Who reads it |
|---|---|---|
| `producer_commit` | The Tessera the producer venv carries. It equals `TESSERA_DEV_PIN_COMMIT`, and it is the only member that names a venv. | `require_producer_repo_is_pinned`, the venv name |
| `serving_commit` | The Tessera the serve installs. `pin.commit` reads it. | People, and the serve recipe |
| `serving_source_sha256` | `tessera.serving.source_identity.serving_source_sha256()` of `serving_commit`: every source file in the package, algorithm `tessera.package_source.v1` (Tessera #678, contract v41). | The cell matcher and the `route.trace` gate |
| `contract_sha256` | Unchanged: the contract bytes every export and admission gate reads. | Every gate |

A v3 pin admits no PENDING sentinel. The module constants split to match:
`TESSERA_SERVING_RUNTIME_PINNED_COMMIT` is the serving commit, and
`TESSERA_SERVING_RUNTIME_PINNED_PRODUCER_COMMIT` and
`TESSERA_SERVING_RUNTIME_PINNED_SERVING_SOURCE_SHA256` join it. The pin file
and the constants are still one reviewed change.

Under a v3 pin, a lane cell matches only if its `runtime.serving_source_sha256`
equals the pinned digest. A cell that names no code does not match. Every rank
of a traced serve must stamp the pinned digest in its trace header: a
different digest is refused, and an absent one is not verified. A v2 pin names
no digest, so neither check runs.

**Activating v3 needs a cell re-census.** No cell in contract v41 names its
code, so a v3 pin admits nothing until the cells are censused with
`tessera_commit` and `serving_source_sha256` stamped at `serving_commit`. The
same bump adds the code column to `contract_answer`, so the development pin's
answer is re-reviewed then. A code-only bump needs no contract re-take only
when the package's source files are unchanged; every pin bump so far changed
them.

---

## Moving the pin

Verified against `RobTand/tessera` master on 2026-09-29, at its merge of #725:

```
commit           83460680ed84e33c82eb62b31345381cc151aa58
contract_sha256  0869f326543374dbd26b75e1d736befed378280d9a5724c4f170bf398aefdbaa
versions.tessera 0.1.0
contract_version 45
lane schema      tessera.lane-eligibility.v10
```

Five values, two files, one commit — plus the same commit and digest in the
three places that copy them on purpose, each of which refuses on its own if it
is left behind: the development pin (`TESSERA_DEV_PIN_COMMIT` /
`TESSERA_DEV_PIN_CONTRACT_SHA256` in `prismaquant/tessera_runtime_contract.py`,
which `tools/resolve_tessera_dev_pin.py` reads as a literal, so it cannot be
derived), and `FROZEN_PINS` in `prismaquant/tessera_legal_domain.py`, whose
`pin_drift` is what forces the legal-domain re-audit on a pin move. The third
script below edits all of them; when the contract bytes do not move, that
script and this block are the whole code change. Resolve the new commit, digest and version
first — from one `git` object fetched from the canonical remote, so they name
the same tree and no local checkout is trusted:

```bash
TS=/home/rob/tmp/tessera-pin-probe-$$ && mkdir -p "$TS" && git -C "$TS" init -q
git -C "$TS" fetch -q https://github.com/RobTand/tessera master
SHA=$(git -C "$TS" rev-parse FETCH_HEAD)
BLOB="$SHA:src/tessera/serving/runtime_contract.json"
DIGEST=$(git -C "$TS" cat-file -p "$BLOB" | sha256sum | cut -d' ' -f1)
VER=$(git -C "$TS" cat-file -p "$BLOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['versions']['tessera'])")
echo "$SHA $DIGEST $VER"
```

Then edit both halves in one commit — the JSON:

```bash
python3 - "$SHA" "$DIGEST" "$VER" <<'EDIT_JSON'
import json, sys, pathlib
sha, digest, ver = sys.argv[1:4]
p = pathlib.Path("prismaquant/tessera_runtime/tessera_serving_runtime_pin.json")
d = json.loads(p.read_text())
d["commit"], d["contract_sha256"], d["version"] = sha, digest, ver
p.write_text(json.dumps(d, indent=2) + "\n")
EDIT_JSON
```

...and the three reader constants, which the reader requires to EQUAL the pin:

```bash
python3 - "$SHA" "$DIGEST" "$VER" <<'EDIT_CONSTS'
import re, sys, pathlib
sha, digest, ver = sys.argv[1:4]
p = pathlib.Path("prismaquant/tessera_serving_runtime_pin.py")
s = p.read_text()
s = re.sub(r'TESSERA_SERVING_RUNTIME_PINNED_COMMIT = \(\n    "[^"]*"\n\)',
           'TESSERA_SERVING_RUNTIME_PINNED_COMMIT = (\n    "%s"\n)' % sha, s, count=1)
s = re.sub(r'TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256 = \(\n    "[^"]*"\n\)',
           'TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256 = (\n    "%s"\n)' % digest, s, count=1)
s = re.sub(r'TESSERA_SERVING_RUNTIME_PINNED_VERSION = "[^"]*"',
           'TESSERA_SERVING_RUNTIME_PINNED_VERSION = "%s"' % ver, s, count=1)
p.write_text(s)
EDIT_CONSTS
```

...and the development pin plus the legal domain's frozen copy, which name the
same object:

```bash
python3 - "$SHA" "$DIGEST" "$VER" <<'EDIT_COPIES'
import re, sys, pathlib
sha, digest, ver = sys.argv[1:4]
p = pathlib.Path("prismaquant/tessera_runtime_contract.py")
s = p.read_text()
s, a = re.subn(r'TESSERA_DEV_PIN_COMMIT = "[0-9a-f]{40}"',
               'TESSERA_DEV_PIN_COMMIT = "%s"' % sha, s, count=1)
s, b = re.subn(r'TESSERA_DEV_PIN_CONTRACT_SHA256 = \(\n    "[0-9a-f]{64}"\n\)',
               'TESSERA_DEV_PIN_CONTRACT_SHA256 = (\n    "%s"\n)' % digest, s, count=1)
assert (a, b) == (1, 1)
p.write_text(s)
p = pathlib.Path("prismaquant/tessera_legal_domain.py")
s = p.read_text()
head, sep, tail = s.partition("FROZEN_PINS = DomainPins(")
body, close, rest = tail.partition("\n)\n")
body = re.sub(r'"[0-9a-f]{40}"', '"%s"' % sha, body)
body = re.sub(r'"[0-9a-f]{64}"', '"%s"' % digest, body)
body = re.sub(r'serving_runtime_pinned_version="[^"]*"',
              'serving_runtime_pinned_version="%s"' % ver, body)
p.write_text(head + sep + body + close + rest)
EDIT_COPIES
```

A contract whose bytes moved also moves `TESSERA_DEV_PIN_ANSWER`: regenerate
it with `contract_answer` on the new bytes (see below) and review the diff.
Hash `src/tessera/grammar.py` and `src/tessera/export.py` at the new commit
before touching `tessera_legal_domain.py`'s prose; new bytes there are a
re-measurement, not a re-transcription.

Every PrismaBuild lane checks the INSTALLED Tessera against
`TESSERA_DEV_PIN_COMMIT` before pytest runs (`tools/resolve_tessera_dev_pin.py`
inside `pbtest`), so a pin move also needs one new sibling interpreter per
lane, named `pq-<lane>-tessera-<short commit>`, installed from a git commit and
never re-pinned in place: bundle the commit into `/mnt/shared/tessera-pins/`,
then on each box clone the bundle and `pip install --no-deps
--no-build-isolation git+file://<clone>@<commit>`. For the `-tessera-4c384e60`
interpreters that was one bundle plus three builds (dl380g10, sparky,
sparklina), 72-211 s each through PrismaBuild.

Use a fleet interpreter provisioned at the reviewed pin, then route CPU
verification through PrismaBuild:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/prismaquant --tag dl380g10 --cpus 4 --demand mem_gb=8 \
  --priority -10 --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 -- \
  /home/rob/venvs/pq-cpu312-tessera-4c384e60/bin/python -m pytest -q -n 4 \
  tests/test_tessera_serving_pin.py tests/test_tessera_lane_v6.py \
  tests/test_tessera_lane_admission.py tests/test_tessera_export_lane.py
```

**A moved contract is a re-review, not a bump.** The development pin
(`TESSERA_DEV_PIN_COMMIT` / `TESSERA_DEV_PIN_CONTRACT_SHA256` /
`TESSERA_DEV_PIN_ANSWER` in `prismaquant/tessera_runtime_contract.py`) names
the SAME Tessera object, and `tests/test_tessera_serving_pin.py` refuses a
drift between the two. So moving the serving pin means regenerating the
dev-pin answer in the same commit, and the git diff of that literal IS the
review: it shows every value an admission gate decides on, including each
cell's `evidence` block. That is the mechanism which keeps promoting a
`routed_moe` cell a human decision under principle 9, rather than a
consequence of Tessera merging a PR.

**Tests whose content is "the pin refuses everything" are spent.** The
2026-09-04 commit deleted `tests/test_tessera_release_pin_flip.py` (its whole
subject was a tag that no longer gates anything) and inverted the four
PENDING-asserting tests in `test_tessera_lane_admission.py`,
`test_tessera_export_lane.py`, `test_tessera_formats.py` and
`test_tessera_contract_v4.py`. What replaced them asserts the property the tag
stood in for: an exact commit, the installed contract's digest, and a refusal
for any other Tessera.
