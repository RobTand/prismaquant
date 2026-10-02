# Optional namespace/profile CPU composition (Refs #1750)

The isolated dependency branch preserves namespace source `12eeca815` and
profiler source `ec616e56`. Original qualified sources remain attributable;
the dependency-only PR2017 merge contains no composition feature.

Source-owner refactor precedes behavior. Existing canonical profiler command
construction and local output policy are shared; child, artifact and sampler
source is unchanged. Causal PB action
`cba4b483b5534c4d62d1b477767fbe2905c4429b74f00b5b254d04c444237957`
on source `d8b181598643be7ca9d48c8fc6f9db42d946d15e` observed one failure
(`namespace requires a direct campaign command`) and 24 passing controls,
including rejection of an instrumented command appended after binding. No skips
or success receipt. This is genuine preparation incompatibility, not a GPU or
native-profiler measurement.

The correction binds full actual wrapper argv before hashing/publication and
uses the existing namespace destination owner for observation and host-local
metadata ownership. Profiler input SHA remains a typed declaration. Actual
input byte and PB staging qualification are independent; no read-set capability
is fabricated by metadata preparation. Host-local metadata remains under its
existing policy root and is host-only in use, with explicit identity coverage
required by the namespace owner. Direct/unbound semantics, output identity,
original workload status and workload-attributed sample gates remain intact.

The final affected CPU family, exact command/source/logs/terminal/claim/CAS,
whole-tree snapshot proof and observed outcomes are recorded externally at
`/home/rob/tmp/codex-campaign-takeover-20261002/pq1750-namespace-profile/`.
No final pass is inferred from submission. No Docker, GPU row, measurement
window, serving change, throughput improvement or complete #1750 acceptance
is claimed. Astra owns acceptance, the central batch and merge.


## Append: metadata mount shadow correction

Initial affected-family action `66664112cf124ab0977e4a3ed27e368bbcf210fe051763cb8c4379dfb3e23216`
on `be4a41f0` observed 257 passes and three failed coverage controls. A broad
identity mount legitimately covered the missing dedicated local mount in that
fixture; its metadata root was moved outside that broad mount, and a positive
broad-coverage control was retained. The readonly/remapped failures exposed a
real new-destination gap: the old shadow refusal checked only the cost row.
The existing owner now applies that same refusal to every owned destination,
including host-local metadata. No new mount policy or baseline growth. The
failed action has no successful CAS result and is not called GREEN.
