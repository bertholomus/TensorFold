# A node's run-time inputs for the `dsv41` lane

Each node of the lane reads a weight file (`TF_DS_RANK_CACHE`: its slices of the checkpoint in the loader's order) and
Engram's compressed token map (`TF_DS_TOKEN_MAP`). Both are made from the EXL3 checkpoint, no GPU:

```sh
tools/dsv41-zig/inputs/node_inputs.sh MODEL_DIR CACHE_DIR RANK [WORLD]
```

writes `CACHE_DIR/tf-ds-rank/rank<RANK>of<WORLD>-<hash>.bin` with `tf-dsv41-rank-cache` (`zig build` on Linux installs
it; it needs no CUDA) and `CACHE_DIR/dsv41_token_map.json` with `make_token_map.py` (the `tokenizers` package; in the
serving container with docker when the host has none). These are the files `../build_inputs.sh` has the Python engine
write on the GPUs: for a 2.9 bpw checkpoint both nodes' files came out byte for byte the same (only the hash in the
file name differs; the loader takes either). WORLD is 2 unless given (RANK 0 on the head node, 1 on the other); this
branch writes one or two nodes, and four nodes or `--parity` are refused (they are on branch `deepseek-v41-zig-tp4`).
`tf-dsv41-rank-cache --index-only` prints the index a node's file will hold without writing it, and
`tf-dsv41-check MODEL_DIR CACHE_DIR/tf-ds-rank RANK 2 --bytes N` checks a written file against the split plan and the
checkpoint.

## Stage a new checkpoint while the lane serves

The GPUs are not needed, so each node can make the new checkpoint's files while the old checkpoint keeps serving. The
switch is then a restart:

1. Put the new checkpoint on both nodes, and pick a new cache folder on each (leave the serving one as it is). The
   weight file needs about 106 GB a node for a 2.9 bpw checkpoint.
2. On the head node run `node_inputs.sh NEW_MODEL NEW_CACHE 0`, and on the other node `node_inputs.sh NEW_MODEL
   NEW_CACHE 1`. The two can run at the same time.
3. If you like, check each file: `tf-dsv41-check NEW_MODEL NEW_CACHE/tf-ds-rank RANK 2 --bytes 64`.
4. Stop the lane and start it again on the new checkpoint, with `TF_DS_RANK_CACHE=NEW_CACHE/tf-ds-rank` and
   `TF_DS_TOKEN_MAP=NEW_CACHE/dsv41_token_map.json` (and the kernel kit that goes with the new checkpoint).
5. Once the new checkpoint serves, the old cache folder can go.

Times we measured: on a 24-core x86 server, with the checkpoint and the new files on one disk array, writing both
nodes' files at the same time took 38 minutes (no GPU). We have not timed it on a GB10 node yet.
