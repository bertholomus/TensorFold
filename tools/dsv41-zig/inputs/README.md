# A node's run-time inputs for the `dsv41` lane

Each node of the lane reads a weight file (`TF_DS_RANK_CACHE`: its slices of the checkpoint in the loader's order) and
Engram's compressed token map (`TF_DS_TOKEN_MAP`). Both are made from the EXL3 checkpoint
([Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)), no GPU:

```sh
tools/dsv41-zig/inputs/node_inputs.sh MODEL_DIR CACHE_DIR RANK [WORLD]
```

writes `CACHE_DIR/rank-cache/rank<RANK>of<WORLD>-<hash>.bin` with `tf-dsv41-rank-cache` (`zig build` on Linux installs
it; on four nodes it takes the balanced experts split, `--parity`) and `CACHE_DIR/dsv41_token_map.json` with
`make_token_map.py` (the `tokenizers` package; in the serving container with docker when the host has none).
`tf-dsv41-rank-cache --index-only` prints the index a node's file will hold without writing it, and
`tf-dsv41-check MODEL_DIR CACHE_DIR/rank-cache RANK WORLD --parity --bytes N` checks a written file against the split
plan and the checkpoint.
