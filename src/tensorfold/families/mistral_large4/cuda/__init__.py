"""Mistral-Large-4 CUDA package.

Phase 1 holds only ``weights.py`` (the pack loader, note D3). Phase 2 adds the engine here — kernels, model,
multi-rank link — per the extension points in note D1 §Extension points (family registration, kernel selection,
engine, allocation) and note D2 §1 (rank organization: rank 0 schedules, followers follow, world 4).
"""
