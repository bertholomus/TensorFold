# Receipt

## Before

Current TP2 kit (`kit/aot/aot.json`, 366 records) at 2026-11-20. A record-by-record fit check at `n=600000`, `nt=9375`:

- `_index_score` float-score signature (`HAS_BASE=1`, `KEYS=0`, `OUT=*fp32`): 0 fitting records. The matching record `34247d211cd2...` requires `n.div16` and `nt.div16`; `nt=9375` fails.
- `_score_keys` `HAS_TMAX=1`: 0 fitting records. The matching record `df85d0b39e2d...` requires `n.div16` and `nt.div16`; `nt=9375` fails.

The real dispatcher returns `MissingTritonVariant`; an eager retry hits the same absent variant.

## After

With the two records in this folder appended:

- `_index_score`: exactly one fitting record, `052d6c1ceb0c...` (`n.div16=false`, `nt.div16=false`).
- `_score_keys`: exactly one fitting record, `9ab1ce45d139...` (`n.div16=false`, `nt.div16=false`).

## Device evidence

Candidate kit with these records, actual Zig AOT loader on two GB10 GPUs (2026-11-10):

- 122 full-output validations and 80 negative-dispatch checks per GPU.
- `n=1, 64, 129, 262144, 599936, 599999, 600000, 600001, 600064, 600065`; `R=3/6`.
- Eager, one capture and two replays with changed base/visibility; normal and `TF_AOT_WEAKEST=1` policies.
- Independent CPU reference matched bit-for-bit; prefix/suffix guards held; unsupported alignment, dtype, constexpr and ABI length were refused before launch.

Synthetic kernel and dispatch results, not full-inference qualification. The initiating CUDA error 1/901 is a separate concern.
