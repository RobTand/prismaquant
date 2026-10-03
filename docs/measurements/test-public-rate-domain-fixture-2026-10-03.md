# Public rate-domain fixture reuse (Refs #1929)

## Measured duplicate work and change

The retained FAILED current-cohort c080 node clocks identify package-C payload
compatibility (53.52s), dataclass payload reconstruction (17.26s), BF16/E4M3
provider properties (17.95s/9.43s), previous legal neighbour (17.83s), and list
refusal (8.38s) as material repeated derivation work. Those clocks selected
this concern; they are not matched timing or qualified full-suite evidence.
The already shared inventory and duplication ratchet are explicitly excluded.

Only tests/test_tessera_legal_domain.py changes. The six existing consumers
use a lazy module-scoped fixture that invokes BOTH real public entry points,
legal_rate_domain and rate_domain_payload, independently once per requested
family. The payload is not constructed by echoing the stored provider answer.
RateDomain is the existing frozen/slots dataclass containing a string and
strict tuple members; it is safe to share. Every getter returns a new payload
dict, so replacing its rates with a list cannot mutate another case or the
stored baseline. The existing rejection test runs before later valid payload
and package-C reconstructions; all still pass through real constructors.

All 1,793 E4M3 and 3,841 BF16 rates, ordering/uniqueness, tuple refusal,
transition membership, legal predecessor/witness and mandatory-rate assertions
remain. No assertion or node is added, removed or renamed. Malformed-domain,
resolver transition, source-state/pin, custom-shape and inventory controls are
unchanged. No production code, source gate, default, pin, format, algorithm,
mock or fallback is introduced.

## Frozen source and exact targeted population

Base:736fd56aa6853c3ee06655eda6015b18bef763bd.
Test commit:044c55ecc7579971d7dc59ee2adf58bba9bcbded.
Before test SHA-256:8e3a5c5f1585334c5a8fdc7166be7f09737b8e0330b6cb5c4a65f9c292bdd822.
After test SHA-256:78c707af8947dd7ff333194fa0bbab02cb7a513a54284019236bc04e839f62dd.

Selected nodes are the two test_provider_returns_the_full_sorted_unique_domain
family cases and test_rate_domain_refuses_a_payload_whose_arrays_arrived_as_lists,
test_boundary_witnesses_take_the_previous_legal_neighbour,
test_rate_domain_payload_constructs_the_dataclass, and test_payload_satisfies_package_c.
Both arms reconcile exactly these six nodes:6 passed,0 skipped,57 explicitly
deselected,14 existing Torch JIT warnings, no missing/duplicate-collected,
never-ran, missing-file or outcome problems. This does not requalify the other57
nodes or a whole suite. No CUDA skip is counted as coverage.

Published pbtest.py uses an OR -k selector, one PB-owned file shard, explicit
--tag dl380g10, one worker, CPU2/mem4GiB/native threads1,900s deadline,
priority-10, GPU visibility disabled. PB owns partition/placement. Both install
preflights bind Python3.14.4, Torch2.11.0+cpu, Transformers5.16.1,
PB SDK95a59051 and Tessera b40c93cb. Both preferred cpusets are[0,1].

| Observed measurement | Before | After |
|---|---:|---:|
| Pytest wall |126.70s|52.37s|
| PB contained wall |133.706s|59.645s|
| Cgroup CPU |158.198s|65.845s|
| Cgroup peak memory |458887168 bytes|455581696 bytes|
| Observed write bytes |2183640|2142680|
| Observed read bytes |8192|40960|
| legal_rate_domain inclusive sampled time |121.31s|45.65s|
| rate_domain_payload inclusive sampled time |49.48s|23.44s|
| legal_rates inclusive sampled time |60.26s|21.55s|
| resolver_transitions inclusive sampled time |60.87s|24.00s|
| py-spy100Hz samples |12929|5394|

Observed selected-case wall decreases74.33s (58.67%) and CPU decreases92.353s.
These are single sequential profiled observations on a non-isolated box with
lower DL background load after; they are not a guaranteed isolated speedup,
whole-suite critical-path reduction, quality/numerical or GPU qualification.
Actual public derivation remains in the after profile; expensive work moves
into the lazy fixture, so a fast consumer call is not a no-op. Inclusive
sampled times overlap and must not be added. All action/profiler returns are0.

Before action:d2ed4ced779d60b774df3e906c6b257268f03d7428a79300a9b38b4abae103b2.
Receipt:ff3fa6c82f7874d8fb2f6e5ede4736428ca1a27054e3250d7c739cb77550a7c5.
Snapshot:0e3e8b897f407ced1bc19df3f777b81909ecf373; parent exact base.
Input:ce3c8db64a89f2616d870e2ced93632c9332b19fb8271758a46795c398b35117.
Profile:75f97e3f2b57b258c5fdfe01c305665e9a63e3353d502242d5eb86387186e36a.
After action:cf5e286bd50c185ba70cb7c4954c96964f83d892f47494e15b3fe903fa17623f.
Receipt:30bf1ce5654335f884dd19f587d1965870dd574e17304650caecc8fb492a52a6.
Snapshot:0f0113098168f10e4774e6c638ed63a601012447; parent exact test commit.
Input:e9ec45573428fec8b663925d72efc534330b90117d4a00b8a10db6913240e998.
Profile:698a5fbca7dfa27a8b290e53d440db806ad5b83c58959b8ea2be914c48ea4df3.

## Raw host telemetry versus action scopes

The means below cover the FULL retained Netdata response buckets, not claim-only
windows. Queries requested1791059919..1791060054 and1791060354..1791060414.
Before response view20:38:39Z..20:40:54Z (2026-10-03), data20:38:40Z..20:40:54Z,
68 buckets. After response view20:45:55Z..20:46:54Z, data20:45:56Z..20:46:54Z,
30 buckets. Actual PB scopes are1791059919.950875..1791060053.657124 and
1791060354.004930..1791060413.649502. These are distinct intervals; response
bucket alignment does not silently relabel the hardware measurements.

| Host | Before captured busy | After captured busy |
|---|---:|---:|
| dl380g10 |20.559%|13.891%|
| sparky |1.759%|1.772%|
| sparklina |1.471%|1.389%|

Busy excludes idle/iowait. All six responses have zero flagged/empty buckets.
PB's own DL averages20.639%/13.877%, PSI-some avg10 maxima3.90/1.52 and explicit
missing-pqteld diagnostics remain recorded. No energy or monetary claim is made.
The authoritative raw source/action/receipt/outcome/profile/resource/Netdata
packet is /home/rob/tmp/pq1929-domain-evidence.json. Existing prior packets and
frozen integration cohorts remain untouched; normal designated-parent,
independent reviewer and integration gates remain. PQ1929 is still OPEN.
