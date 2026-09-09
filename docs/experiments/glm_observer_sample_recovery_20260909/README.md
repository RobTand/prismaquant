# Continue observation after a failed Netdata sample

The live selected expert action `73af05a17fa4` lost Netdata monitoring at
01:39:43 UTC on September 9 after one response omitted its required power
chart. The outer exception handler ended the thread, losing subsequent
samples from both hosts while encoding continued. The original evidence is
retained under the first-proof preparation row 0076; recovered server history
is separately recorded and never substituted into that original stream.

Catch individual sampling failures, retain bounded per-host failure counts and
timestamps, and continue other hosts and later samples. Writer failures still
end monitoring; any missing sample still fails the existing evidence gate.

PB regression `aa80a577b9d0733dd3956481b3bbc482537adc896c037fba040f600cc708cea9`
failed because only the first host was attempted. PB
`ac140d7e0fb0e7a320b3fa52703884f3c970b59521adf68993343e7e50c0b273`
passed the three observer/Netdata modules: 43 passed, 3 skipped, 7.35 seconds
on DL380 CPU with four workers and native threads bounded to one. The skips
are CUDA-only coverage. An earlier wrong-environment submission lacked
compressed_tensors and did not collect; it is not a regression result.
This changes observation recovery, not pricing, sampling values or admission.

## Concurrent progress serialization (issue #460)

A first monitor round added dictionary keys while the main thread could be
serializing progress, raising `RuntimeError: dictionary changed size during
iteration`. Initialize monitor counters and both host failure slots before
starting threads; each failure record also has its complete key set before
publication. Progress remains a non-atomic observation; the final snapshot
follows thread shutdown and retains any errors.

The deterministic regression pauses the actual JSON dictionary iterator before
running the first monitor round, for both monitor kinds. PB
`2093a627f4189ac9360beba0cef59b369783d8b9d3ea271ac13e99e478893b8f`
failed both cases with the dictionary-size error. Final green PB
`550b801d8faaea6d9895201fa0959007a9b3dc3bb3bda7f540a5019c6306f1cd`
passed 45 tests with the same 3 CUDA-only skips in 7.40 seconds (DL380 CPU,
four workers, one native thread each). The terminal record confirms exit 0
and resource cleanup. The CAS result hash is
`b966e20248cd4757ea42a8d4f8593f0e3944122d14993d0b1efa9d8b3feb5261`;
the tested snapshot is `232e6c89ddef8faa177c604d60c19f4d01240458`.
