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

The current pin is Tessera `e4a3a7d4516de61785e3f8738ccb64b86cf9e8cf`, the
head of Tessera #562 (D2/D2b, stacked on #560's
`flash/506-routed-full-domain-rates-20260918`), re-pinned 2026-09-19 for the
coordinated post-#560-lineage bump (PrismaQuant #760). Install that revision
and point `TESSERA_REPO` at its complete checkout; the producer scripts live in
`experiments/` and are not wheel entry points.

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

**Gated landing (PrismaQuant #760).** This pin is the prerequisite half of a
coordinated two-repo change. #563 (docs fix-forward, pin-coupled) edits only
the contract changelog, so once #562 AND #563 have both landed, the contract
at Tessera master's head is the three-way union of their edits --

```
union contract sha256
712a15e4cb6015e28cb206e82aba542c8af78d8e0e4efda34f78841226e8c61c
```

-- which is NOT this pin's digest. Before the PrismaQuant side merges,
repoint `commit` and `contract_sha256` (both files, one commit) to that
final head and re-verify; the admission answer is already the union's,
because #563 projects nothing the answer reads.

Re-check the exact commit:

```bash
git -C "$TS" cat-file -p e4a3a7d4516de61785e3f8738ccb64b86cf9e8cf:src/tessera/serving/runtime_contract.json | sha256sum
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

## Moving the pin

Verified against `RobTand/tessera` (branch head of #562, stacked on #560's
lineage) on 2026-09-19:

```
commit           e4a3a7d4516de61785e3f8738ccb64b86cf9e8cf
contract_sha256  14acb1f78b2da32272f077e4ae69cb3486846b03912ffa825b677568b84d1780
versions.tessera 0.1.0
contract_version 32
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
