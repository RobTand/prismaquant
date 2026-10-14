# Refactor integration checks

Base: `138069b7993`. Branch: `sol/refactor-simplify-20261014`.

The first composed check, PB
`2b4e6d708fc24bed2441ce093a4944e4dc4207cfef00b0501504aa3516872d36`,
passed 590 cases and failed one duplication guard, with no skips. The ledger
had copied the producer's digest validator while leaving the producer copy.
The ledger now owns `require_quantum_digest`; the producer uses it. The exact
string rule, lowercase digest predicate and refusal text remain the same.
The shared digest owner supplies the text predicate.

PB `1d77dee590e483c34e86ae5ca588bcd4231b6e452508e374899ea4b1ceeab24e`
passed all 174 focused producer, ledger, dispatcher, duplication and migration
checks with no skips. Its exit was zero and its result CAS digest is
`2fbdfbdbe8254a61bb8f1756bea6a23979a31bf932639731c89e107fcab241a6`.

The prior submission `f8d059b780094051e275920246d53dbb607520190a3db194e97dc663334becd8`
ran no tests because the command named a nonexistent test file. It failed;
the corrected composed command above records the actual coverage.
