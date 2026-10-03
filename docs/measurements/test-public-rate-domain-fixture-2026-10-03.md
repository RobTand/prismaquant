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


## Successor: default resolver transition result reuse (Refs #1929)

This separate transition-only successor starts at frozen accepted PR2189 head
06284118e526345608c2a8a7bf5b45108c9a4bc6; that packet is unchanged. The two
read-only resolver convention tests formerly called the same real default BF16
resolver twice. A module-scoped lru_cache fixture now shares its immutable tuple
once per family. Each consumer still constructs its own mutable set. The real
resolver is invoked with rates omitted, so its default legal-rate derivation
and every real schedule_signature still execute; no public-domain cached answer
or mock is substituted. E4M3 is independently derived. Width-boundary, uniform
versus mixed schedule, exact expected transition sets, first-domain exclusion
and every original assertion/node remain. No production, pin, shape or source
gate changes. The earlier six provider/payload consumers and fixture are untouched.

Test commit:a51b9af0cb84bb2d682256eec28f3a5a9d8c69c3.
Test file SHA-256:da246fda9976e653ede6a4c5a229096a1fa6c99c25dfefe696c5060ee7fd2a45.
Both arms select precisely test_a_transition_names_the_first_rate_of_the_new_regime
and E4M3/BF16 cases of test_every_256_multiple_above_the_endpoint_is_a_resolver_transition.
Same3 nodes:3passed,0skipped,60deselected,14 existing Torch warnings and zero
missing/duplicate-collected/never-ran/outcome issues. This does not requalify
PR2189's six nodes, all63 module nodes or the whole CPU suite.

Published pbtest.py uses explicitDL, one PB-owned file shard/worker,
CPU2/mem4GiB/native1,600s deadline, priority-10 and an exact two-name OR -k
selector. The earlier pinned interpreter/package preflight is unchanged and
both preferred cpusets are[0,1]. All action/profiler returns are0.

| New3-case observed measurement | Before | After |
|---|---:|---:|
| Pytest wall |44.13s|28.68s|
| PB contained wall |51.750s|35.631s|
| Cgroup CPU |56.502s|37.864s|
| Cgroup peak memory |456519680 bytes|454959104 bytes|
| Observed writes |2191832 bytes|2122200 bytes|
| Observed reads |40960 bytes|0 bytes|
| resolver_transitions inclusive sampled time |37.90s|22.93s|
| legal_rates inclusive sampled time |18.43s|11.32s|
| schedule_signature inclusive sampled time |19.28s|11.49s|
| py-spy100Hz samples |4573|3041|

Observed selected wall decreases15.45s (35.01%) and CPU decreases18.639s.
This is one sequential profiled pair, not an isolated guaranteed speedup or
whole-suite saving. After host load is higher; both arms use the same preferred
cores. Inclusive samples overlap and must not be added. The real default-path
legal-rate and signature work remains in the after profile.

Before action:0fe264814d1de9ad1eea7be3c343dcd4581374b71a5a96dad745b2ea4ce156bd.
Receipt:edc53ba970a2c78d70eb6122a970cbe6ad1a092961b51345d681ccf0508e0208.
Snapshot:063bcf5aaed35a24d06400d8686b5db8a8dd4b3b; parent exact accepted base.
Input:116b1895455246bacaacfb6d725b71ec3673e23a33d57f0d6b70ecd76ea7e05e.
Profile:4702060bf68cf998d4019c10f75f485b46087b120733b438082d1fa4271bbe35.
After action:06c840dc6cab0acba378f343d843b03000499b552896f7f2ae3e502cae7ae4ff.
Receipt:2a16290ba766a45fc1d1b4ef122aa65f6a2f8dbda6d8714be826bf85c671afef.
Snapshot:8ba137e323daca30d16720002317f19f0de3ca25; parent exact test commit.
Input:89525d2dc24174d6279ef08ec3e10865b8542707d77c91a3520cf06b5d34c28e.
Profile:35d633d27341f5dfadd451eca34052d59771f3044922b8640ea921ea1d450901.
Both profile CAS blobs are successful py-spy0.4.2/100Hz outputs. Read-only SSH
extraction reported an agent-signing communication warning but then completed
successfully with actual profile counts; it did not repeat or cancel a workload.

Captured Netdata means below cover the full returned response buckets, not
claim-only scopes. Queries requested1791063032..1791063085 and1791063229..1791063266.
Before response view21:30:32Z..21:31:25Z (2026-10-03), data21:30:33Z..21:31:25Z,
27buckets. After view21:33:47Z..21:34:26Z, data21:33:48Z..21:34:26Z,20buckets,
about3s of response view before the PB claim. PB scopes are
1791063032.821433..1791063084.571155 and1791063229.940526..1791063265.571539.

| Host | Before full-response busy | After full-response busy |
|---|---:|---:|
| dl380g10 |12.987%|14.456%|
| sparky |1.858%|2.099%|
| sparklina |1.462%|1.389%|

Busy excludes idle/iowait; all six retained responses have zero flagged/empty
buckets. PB's own DL means12.878%/14.673% and missing-pqteld diagnostics are
retained. Raw source/receipt/outcome/scope/profile/Netdata packet:
/home/rob/tmp/pq1929-transition-evidence.json. No blocking wait/poll loop,
measurement replay, GPU/numerical/energy/$ or full-suite qualification occurs.
Designated-parent review, independent review and existing integration gates
remain mandatory; P1 parent1929 stays OPEN.


## Successor: remaining default source-report caller (Refs #1929)

This separately measured concern starts at accepted PR2196 head
b4f3b71d691caac33e304fc4c9f1028c26b872ff. Only one test caller changes:
test_the_report_names_the_source_state_the_numbers_came_from uses the existing,
unchanged _inventory() helper instead of a second default build_inventory().
Both old calls use precisely the same no-argument source/pin/options contract.
The real default derivation still executes for the per-rate table-width donor.
All source-state, actual imported export digest, report ordering and every
per-rate width assertion remain. No inventory helper/algorithm, prior public
provider/transition fixture, pin or grammar guard changes. The moved-pin test
still independently builds its own E4/dense/empty-extra-ledger inventory and
requires PIN DRIFT before the counts; it is not supplied a cached answer.
The public CLI and distinct process/deadline, H-bearing and resume controls
remain independent. No scientific coverage or node is removed.

Test commit:08a548be9cae1e7cb1d26da4d96489deb26cf3a9.
Test-file SHA-256:9c6b173511cdc5c91e108741323c63d2e9c2d04fd49060711b23c0f7503fc802.
Both arms select exactly test_every_candidate_carries_its_own_table_width,
test_the_inventory_carries_its_drift_report, and the changed report node.
Both collect and run the same3 IDs:3passed,0skipped,60deselected,14 existing
Torch warnings; capture reconciliation has no missing files, never-ran,
not-collected, duplicate-collection, extra-phase or outcome problems.
These are a NEW3-node population, not replays or requalification of the earlier
six provider/payload nodes or the three resolver-transition nodes.

Published pbtest.py owns partitioning: one file shard, one pytest worker,
explicit dl380g10, CPU2/mem4GiB/native1,600s deadline,priority-10, sample profile.
The fixed target interpreter is pq-pb95a59051-tessera-b40c93cb/bin/python;
both real preflights match PB95a59051 and verify44 files. Both use preferred
cores[0,1]. All actions and py-spy0.4.2/100Hz backend returns are0.

| New report population observation | Before | After |
|---|---:|---:|
| Pytest wall |58.16s|36.61s|
| PB contained wall |65.461s|43.818s|
| Cgroup CPU |72.956s|47.789s|
| Cgroup peak memory |458530816 bytes|456978432 bytes|
| Process observed writes |2462168 bytes|2687448 bytes|
| Process observed reads |24576 bytes|49152 bytes|
| build_inventory inclusive sampled time |52.97s|29.79s|
| legal_rates inclusive sampled time |24.96s|14.05s|
| resolver_transitions inclusive sampled time |27.24s|15.01s|
| Unchanged default donor inclusive sampled time |23.82s|22.79s|
| Independent moved-pin control inclusive sampled time |7.20s|7.00s|
| Changed report node inclusive sampled time |21.95s|0 sampled seconds|
| py-spy samples |6071|3736|

Observed selected wall decreases21.55s (37.05%) and CPU decreases25.167s.
Zero sampled report time is not a claim of zero execution: its original
assertions and real format_report still run. The unchanged real donor and
independent drift derivation remain in the after profile. Inclusive samples
overlap and must not be added. This is one sequential profiled pair with lower
after DL host load; no guaranteed or whole-suite saving is asserted.

Before action:ec93a3ee58b2a353280c5256d2c1e71f41c6630bd5c08d64f3c6e39f176aeda1.
Receipt:df17b03d9e714fbe7144a9777b46e41a2326f54ae125d60c425febc000cd8d6a.
Snapshot:9c682a9aa648a04ac2bd592dccc18abfcbb6b5f1; parent exact accepted base.
Input:bc535fadbc80ab8597482651c32a2170d5e6d90b753a174bd6b659f5ac3cac69.
Profile:67f87d3ed49e101c66500108da68f3e25d157720f24ba09cdf6b3985fa10029d.
After action:d946a7d356e8a158e5fae30d6ce38e4aff2471130a29c0bb7cf0c42f05cb7a73.
Receipt:0ad0669fffced5c4a3e93ed35907e60610ba46f46467bdef5ba7688c7966fb68.
Snapshot:cea9ab68036236fb7c7e32a655ac94a9684555d1; parent exact test commit.
Input:16daafd977e13276cefeb511e09a6c1c8ee4ade54c21fd8ef4dfbb25f8d079df.
Profile:2fbc582bc0537274df55de250b1f48d18213ee037d96a5da20f3bdb14ec1f090.
Both extracted profile bytes hash to the actual CAS descriptors.

Netdata requested1791065631..1791065698 and1791066104..1791066149. Before
returned view AND actual data span22:13:51Z..22:14:58Z (2026-10-03),68one-second
buckets; after view/data22:21:44Z..22:22:29Z,46buckets. These rounded windows
slightly extend the exact PB scopes1791065631.9567518..1791065697.4175663 and
1791066104.212592..1791066148.0301013. Full-response means below are not
claim-only means; all six responses have zero empty/flagged buckets.

| Host | Before full-response busy | After full-response busy |
|---|---:|---:|
| dl380g10 |13.849%|12.288%|
| sparky |1.984%|4.217%|
| sparklina |1.411%|1.449%|

Busy excludes idle/iowait. PB's own DL scope means13.658%/12.724% and its
missing-pqteld diagnostics are retained separately. Raw source/receipt/node
capture/profile/CPUwallpeakIO/request/view/data packet:
/home/rob/tmp/pq1929-report-evidence.json.

Initial service-option and unsupported -q CLI refusals happened BEFORE
publication, so no admitted baseline was replayed; the refusal log remains.
An empty-dimension telemetry request was rejected locally; exact discovered
dimensions then produced the retained successful captures. Read-only profile
SSH extraction reported an agent-signing warning but completed successfully.
No wait/poll loop, GPU/numerical/energy/$ qualification or frozen-cohort change
occurs. Designated-parent, independent review and normal integration gates
remain; P1 parent1929 stays OPEN.

