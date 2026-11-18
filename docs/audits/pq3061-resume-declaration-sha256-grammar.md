# Resume declaration SHA-256 grammar: #3315

## Changed sites

`prismaquant/stage_a_chain_resume.py::parse_declaration` (`:275`) checks
each part of `FROM:TO` with `is_sha256hex` instead of a bare regex.
`prismaquant/digests.py:120` owns `is_sha256hex`.
It accepts string subclasses.
These stay: `None`, `str(text).split(":")`, the two-part rule, the explicit
length check, the check order, the `from` and `to` keys and the
`ChainResumeRefused` message.
Callers: `prismaquant/joint_cost_stage_a.py` (`:3041`, `:3043`), for
`--resume-implementation-compatibility`.
Line numbers are for main at `6ca8ada35`.

## Evidence

Merge `a359ffb36` (PR #4656, #4586) made the change.
Flux job `f3AxefUohEF` recorded 12 compatibility cases through the
unedited consumer.
`test_resume_declaration_uses_shared_sha256_grammar` failed by assertion
on base and passed on head.
Main record run 44303024917 on `dbb7f578d` passed
`//tests:test_stage_a_chain_resume` on amd64 and arm64.

## Residual scope

The explicit length check stays, by the parent's rule.
The record-name patterns `_RESUME_RECORD` and `_ROLLING_ENTRY` are not
digest grammar. They stay.
The duplication inventory and its shrink-only baseline stay unchanged.
#3061 retains inventory integration and scanner gaps.
#1301 recipes stay unchanged.
This note makes no speed or serving-quality claim.
