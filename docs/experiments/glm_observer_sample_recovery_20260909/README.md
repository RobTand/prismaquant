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
