<!-- `uv run python -m benchmarks.repo_eval_v5 --live --confirm-live` at ad70746, OpenAI text-embedding-3-small, 2026-11-16. -->

# repo-eval-v5 (live_model, 44 cases, corpus efd04a5)

## Ranking

| Chunking | Retriever | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | behavior hit@5 | config hit@5 | identifier hit@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| line_v1 | bm25 (legacy) | 0.432 | 0.409 | 0.216 | 0.568 | 0.557 | 0.333 | 0.333 | 0.875 |
| line_v1 | bm25 | 0.659 | 0.625 | 0.419 | 0.773 | 0.727 | 0.600 | 0.500 | 1.000 |
| line_v1 | semantic | 0.545 | 0.511 | 0.359 | 0.636 | 0.614 | 0.500 | 0.333 | 0.875 |
| line_v1 | hybrid | 0.727 | 0.670 | 0.505 | 0.795 | 0.750 | 0.700 | 0.500 | 1.000 |
| line_v1 | hybrid_symbol | 0.727 | 0.670 | 0.505 | 0.795 | 0.750 | 0.700 | 0.500 | 1.000 |
| python_ast_v1 | bm25 (legacy) | 0.295 | 0.250 | 0.224 | 0.364 | 0.307 | 0.200 | 0.167 | 0.750 |
| python_ast_v1 | bm25 | 0.523 | 0.443 | 0.295 | 0.591 | 0.500 | 0.467 | 0.333 | 0.875 |
| python_ast_v1 | bm25 (legacy) + symbol | 0.273 | 0.239 | 0.218 | 0.341 | 0.295 | 0.167 | 0.167 | 0.750 |
| python_ast_v1 | bm25 + symbol | 0.523 | 0.443 | 0.266 | 0.591 | 0.500 | 0.467 | 0.333 | 0.875 |
| python_ast_v1 | semantic | 0.341 | 0.273 | 0.241 | 0.477 | 0.420 | 0.267 | 0.500 | 0.500 |
| python_ast_v1 | hybrid | 0.523 | 0.455 | 0.307 | 0.614 | 0.523 | 0.433 | 0.500 | 0.875 |
| python_ast_v1 | hybrid_symbol | 0.523 | 0.455 | 0.273 | 0.591 | 0.511 | 0.433 | 0.500 | 0.875 |

## Evidence reaching the model (RAG packing, default 20,000-char budget)

| Chunking | Retriever | top_k | Context | Evidence hit | Evidence recall | Mean context chars | Mean sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| line_v1 | bm25 (legacy) | 5 | seeds_only | 0.341 | 0.330 | 17,531 | 3.9 |
| line_v1 | bm25 (legacy) | 5 | expanded | 0.295 | 0.273 | 14,475 | 3.4 |
| line_v1 | bm25 | 5 | seeds_only | 0.591 | 0.580 | 17,517 | 4.2 |
| line_v1 | bm25 | 5 | expanded | 0.568 | 0.534 | 14,967 | 3.9 |
| line_v1 | semantic | 5 | seeds_only | 0.545 | 0.511 | 17,630 | 4.5 |
| line_v1 | semantic | 5 | expanded | 0.455 | 0.432 | 14,607 | 3.9 |
| line_v1 | hybrid | 5 | seeds_only | 0.705 | 0.648 | 17,354 | 4.3 |
| line_v1 | hybrid | 5 | expanded | 0.682 | 0.636 | 14,891 | 4.0 |
| line_v1 | hybrid_symbol | 5 | seeds_only | 0.705 | 0.648 | 17,354 | 4.3 |
| line_v1 | hybrid_symbol | 5 | expanded | 0.682 | 0.636 | 14,891 | 4.0 |
| python_ast_v1 | bm25 (legacy) | 5 | seeds_only | 0.295 | 0.250 | 7,743 | 5.0 |
| python_ast_v1 | bm25 (legacy) | 5 | expanded | 0.409 | 0.364 | 13,061 | 11.2 |
| python_ast_v1 | bm25 (legacy) | 10 | seeds_only | 0.364 | 0.307 | 13,923 | 9.1 |
| python_ast_v1 | bm25 (legacy) | 10 | expanded | 0.432 | 0.364 | 15,670 | 13.8 |
| python_ast_v1 | bm25 (legacy) | 15 | seeds_only | 0.364 | 0.307 | 16,852 | 11.5 |
| python_ast_v1 | bm25 (legacy) | 15 | expanded | 0.432 | 0.364 | 15,865 | 12.7 |
| python_ast_v1 | bm25 (legacy) | 20 | seeds_only | 0.386 | 0.330 | 18,077 | 12.8 |
| python_ast_v1 | bm25 (legacy) | 20 | expanded | 0.432 | 0.375 | 15,964 | 12.8 |
| python_ast_v1 | bm25 | 5 | seeds_only | 0.523 | 0.443 | 6,581 | 5.0 |
| python_ast_v1 | bm25 | 5 | expanded | 0.636 | 0.580 | 12,092 | 11.2 |
| python_ast_v1 | bm25 | 10 | seeds_only | 0.591 | 0.500 | 12,781 | 9.6 |
| python_ast_v1 | bm25 | 10 | expanded | 0.682 | 0.602 | 15,564 | 14.6 |
| python_ast_v1 | bm25 | 15 | seeds_only | 0.636 | 0.557 | 17,354 | 13.2 |
| python_ast_v1 | bm25 | 15 | expanded | 0.682 | 0.591 | 15,948 | 14.2 |
| python_ast_v1 | bm25 | 20 | seeds_only | 0.705 | 0.602 | 18,710 | 14.8 |
| python_ast_v1 | bm25 | 20 | expanded | 0.682 | 0.580 | 15,958 | 14.2 |
| python_ast_v1 | bm25 + symbol | 5 | seeds_only | 0.523 | 0.443 | 6,588 | 5.0 |
| python_ast_v1 | bm25 + symbol | 5 | expanded | 0.636 | 0.568 | 12,116 | 11.2 |
| python_ast_v1 | bm25 + symbol | 10 | seeds_only | 0.591 | 0.500 | 12,871 | 9.7 |
| python_ast_v1 | bm25 + symbol | 10 | expanded | 0.682 | 0.602 | 15,564 | 14.6 |
| python_ast_v1 | bm25 + symbol | 15 | seeds_only | 0.636 | 0.557 | 17,533 | 13.2 |
| python_ast_v1 | bm25 + symbol | 15 | expanded | 0.682 | 0.591 | 15,942 | 14.0 |
| python_ast_v1 | bm25 + symbol | 20 | seeds_only | 0.705 | 0.602 | 18,764 | 14.8 |
| python_ast_v1 | bm25 + symbol | 20 | expanded | 0.636 | 0.545 | 15,945 | 14.0 |
| python_ast_v1 | semantic | 5 | seeds_only | 0.341 | 0.273 | 5,704 | 5.0 |
| python_ast_v1 | semantic | 5 | expanded | 0.364 | 0.318 | 11,366 | 12.1 |
| python_ast_v1 | semantic | 10 | seeds_only | 0.477 | 0.420 | 10,666 | 9.7 |
| python_ast_v1 | semantic | 10 | expanded | 0.500 | 0.455 | 15,508 | 17.1 |
| python_ast_v1 | semantic | 15 | seeds_only | 0.545 | 0.466 | 14,564 | 13.9 |
| python_ast_v1 | semantic | 15 | expanded | 0.523 | 0.466 | 15,896 | 17.2 |
| python_ast_v1 | semantic | 20 | seeds_only | 0.545 | 0.477 | 17,386 | 17.1 |
| python_ast_v1 | semantic | 20 | expanded | 0.523 | 0.466 | 15,965 | 17.2 |
| python_ast_v1 | hybrid | 5 | seeds_only | 0.523 | 0.455 | 6,559 | 5.0 |
| python_ast_v1 | hybrid | 5 | expanded | 0.591 | 0.523 | 12,009 | 11.4 |
| python_ast_v1 | hybrid | 10 | seeds_only | 0.614 | 0.523 | 11,637 | 9.7 |
| python_ast_v1 | hybrid | 10 | expanded | 0.659 | 0.568 | 15,592 | 15.8 |
| python_ast_v1 | hybrid | 15 | seeds_only | 0.705 | 0.614 | 15,569 | 13.5 |
| python_ast_v1 | hybrid | 15 | expanded | 0.682 | 0.614 | 15,926 | 16.0 |
| python_ast_v1 | hybrid | 20 | seeds_only | 0.727 | 0.648 | 17,679 | 16.2 |
| python_ast_v1 | hybrid | 20 | expanded | 0.705 | 0.648 | 15,950 | 16.1 |
| python_ast_v1 | hybrid_symbol | 5 | seeds_only | 0.523 | 0.455 | 6,589 | 5.0 |
| python_ast_v1 | hybrid_symbol | 5 | expanded | 0.568 | 0.500 | 11,889 | 11.3 |
| python_ast_v1 | hybrid_symbol | 10 | seeds_only | 0.591 | 0.511 | 11,920 | 9.8 |
| python_ast_v1 | hybrid_symbol | 10 | expanded | 0.614 | 0.534 | 15,558 | 15.7 |
| python_ast_v1 | hybrid_symbol | 15 | seeds_only | 0.705 | 0.614 | 15,781 | 13.5 |
| python_ast_v1 | hybrid_symbol | 15 | expanded | 0.682 | 0.614 | 15,932 | 15.7 |
| python_ast_v1 | hybrid_symbol | 20 | seeds_only | 0.727 | 0.648 | 17,842 | 16.1 |
| python_ast_v1 | hybrid_symbol | 20 | expanded | 0.705 | 0.648 | 15,958 | 15.8 |
