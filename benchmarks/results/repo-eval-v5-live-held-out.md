<!-- `uv run python -m benchmarks.repo_eval_v5 --live --confirm-live --held-out` at ad70746, OpenAI text-embedding-3-small, 2026-11-16. -->

# repo-eval-v5-held-out (live_model, 16 cases, corpus efd04a5)

## Ranking

| Chunking | Retriever | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | behavior hit@5 | config hit@5 | identifier hit@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| line_v1 | bm25 (legacy) | 0.812 | 0.812 | 0.460 | 0.812 | 0.812 | 0.818 | 1.000 | 0.667 |
| line_v1 | bm25 | 0.812 | 0.812 | 0.708 | 0.875 | 0.875 | 0.818 | 1.000 | 0.667 |
| line_v1 | semantic | 0.438 | 0.438 | 0.406 | 0.688 | 0.688 | 0.455 | 0.000 | 0.667 |
| line_v1 | hybrid | 0.812 | 0.812 | 0.635 | 0.938 | 0.938 | 0.818 | 1.000 | 0.667 |
| line_v1 | hybrid_symbol | 0.812 | 0.812 | 0.635 | 0.938 | 0.938 | 0.818 | 1.000 | 0.667 |
| python_ast_v1 | bm25 (legacy) | 0.688 | 0.656 | 0.583 | 0.688 | 0.656 | 0.545 | 1.000 | 1.000 |
| python_ast_v1 | bm25 | 0.750 | 0.719 | 0.672 | 0.750 | 0.719 | 0.636 | 1.000 | 1.000 |
| python_ast_v1 | bm25 (legacy) + symbol | 0.625 | 0.594 | 0.594 | 0.688 | 0.656 | 0.455 | 1.000 | 1.000 |
| python_ast_v1 | bm25 + symbol | 0.750 | 0.719 | 0.641 | 0.750 | 0.719 | 0.636 | 1.000 | 1.000 |
| python_ast_v1 | semantic | 0.562 | 0.531 | 0.463 | 0.562 | 0.562 | 0.364 | 1.000 | 1.000 |
| python_ast_v1 | hybrid | 0.625 | 0.594 | 0.531 | 0.875 | 0.844 | 0.455 | 1.000 | 1.000 |
| python_ast_v1 | hybrid_symbol | 0.562 | 0.531 | 0.516 | 0.875 | 0.844 | 0.364 | 1.000 | 1.000 |

## Evidence reaching the model (RAG packing, default 20,000-char budget)

| Chunking | Retriever | top_k | Context | Evidence hit | Evidence recall | Mean context chars | Mean sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| line_v1 | bm25 (legacy) | 5 | seeds_only | 0.750 | 0.750 | 17,401 | 3.9 |
| line_v1 | bm25 (legacy) | 5 | expanded | 0.688 | 0.688 | 14,970 | 3.7 |
| line_v1 | bm25 | 5 | seeds_only | 0.812 | 0.812 | 17,646 | 4.3 |
| line_v1 | bm25 | 5 | expanded | 0.812 | 0.812 | 14,895 | 3.9 |
| line_v1 | semantic | 5 | seeds_only | 0.438 | 0.438 | 16,400 | 4.5 |
| line_v1 | semantic | 5 | expanded | 0.438 | 0.438 | 14,454 | 4.1 |
| line_v1 | hybrid | 5 | seeds_only | 0.812 | 0.812 | 16,521 | 4.4 |
| line_v1 | hybrid | 5 | expanded | 0.812 | 0.812 | 14,638 | 4.0 |
| line_v1 | hybrid_symbol | 5 | seeds_only | 0.812 | 0.812 | 16,521 | 4.4 |
| line_v1 | hybrid_symbol | 5 | expanded | 0.812 | 0.812 | 14,638 | 4.0 |
| python_ast_v1 | bm25 (legacy) | 5 | seeds_only | 0.688 | 0.656 | 7,289 | 5.0 |
| python_ast_v1 | bm25 (legacy) | 5 | expanded | 0.750 | 0.750 | 12,374 | 11.8 |
| python_ast_v1 | bm25 (legacy) | 10 | seeds_only | 0.688 | 0.656 | 12,123 | 9.2 |
| python_ast_v1 | bm25 (legacy) | 10 | expanded | 0.750 | 0.750 | 15,720 | 14.9 |
| python_ast_v1 | bm25 (legacy) | 15 | seeds_only | 0.688 | 0.656 | 16,082 | 12.3 |
| python_ast_v1 | bm25 (legacy) | 15 | expanded | 0.750 | 0.750 | 15,932 | 13.6 |
| python_ast_v1 | bm25 (legacy) | 20 | seeds_only | 0.688 | 0.656 | 17,589 | 13.9 |
| python_ast_v1 | bm25 (legacy) | 20 | expanded | 0.750 | 0.719 | 15,940 | 13.4 |
| python_ast_v1 | bm25 | 5 | seeds_only | 0.750 | 0.719 | 7,942 | 5.0 |
| python_ast_v1 | bm25 | 5 | expanded | 0.875 | 0.875 | 12,608 | 10.8 |
| python_ast_v1 | bm25 | 10 | seeds_only | 0.750 | 0.719 | 12,926 | 9.1 |
| python_ast_v1 | bm25 | 10 | expanded | 0.812 | 0.812 | 15,731 | 14.4 |
| python_ast_v1 | bm25 | 15 | seeds_only | 0.750 | 0.719 | 16,418 | 12.5 |
| python_ast_v1 | bm25 | 15 | expanded | 0.750 | 0.750 | 15,969 | 13.4 |
| python_ast_v1 | bm25 | 20 | seeds_only | 0.750 | 0.719 | 17,991 | 14.3 |
| python_ast_v1 | bm25 | 20 | expanded | 0.750 | 0.719 | 15,967 | 13.2 |
| python_ast_v1 | bm25 + symbol | 5 | seeds_only | 0.750 | 0.719 | 7,498 | 5.0 |
| python_ast_v1 | bm25 + symbol | 5 | expanded | 0.875 | 0.875 | 12,315 | 10.8 |
| python_ast_v1 | bm25 + symbol | 10 | seeds_only | 0.750 | 0.719 | 13,088 | 9.2 |
| python_ast_v1 | bm25 + symbol | 10 | expanded | 0.812 | 0.812 | 15,742 | 14.9 |
| python_ast_v1 | bm25 + symbol | 15 | seeds_only | 0.750 | 0.719 | 16,654 | 12.8 |
| python_ast_v1 | bm25 + symbol | 15 | expanded | 0.750 | 0.750 | 15,982 | 13.5 |
| python_ast_v1 | bm25 + symbol | 20 | seeds_only | 0.750 | 0.719 | 18,318 | 14.6 |
| python_ast_v1 | bm25 + symbol | 20 | expanded | 0.750 | 0.719 | 15,982 | 13.4 |
| python_ast_v1 | semantic | 5 | seeds_only | 0.562 | 0.531 | 5,401 | 5.0 |
| python_ast_v1 | semantic | 5 | expanded | 0.562 | 0.562 | 10,491 | 11.5 |
| python_ast_v1 | semantic | 10 | seeds_only | 0.562 | 0.562 | 10,114 | 9.8 |
| python_ast_v1 | semantic | 10 | expanded | 0.562 | 0.562 | 15,221 | 18.1 |
| python_ast_v1 | semantic | 15 | seeds_only | 0.625 | 0.625 | 13,954 | 14.1 |
| python_ast_v1 | semantic | 15 | expanded | 0.625 | 0.625 | 15,775 | 17.9 |
| python_ast_v1 | semantic | 20 | seeds_only | 0.688 | 0.688 | 16,286 | 17.8 |
| python_ast_v1 | semantic | 20 | expanded | 0.625 | 0.625 | 15,961 | 18.7 |
| python_ast_v1 | hybrid | 5 | seeds_only | 0.625 | 0.594 | 6,003 | 5.0 |
| python_ast_v1 | hybrid | 5 | expanded | 0.688 | 0.688 | 10,963 | 11.4 |
| python_ast_v1 | hybrid | 10 | seeds_only | 0.875 | 0.844 | 12,311 | 9.7 |
| python_ast_v1 | hybrid | 10 | expanded | 0.875 | 0.875 | 15,253 | 16.2 |
| python_ast_v1 | hybrid | 15 | seeds_only | 0.875 | 0.875 | 15,128 | 12.9 |
| python_ast_v1 | hybrid | 15 | expanded | 0.875 | 0.875 | 15,907 | 15.8 |
| python_ast_v1 | hybrid | 20 | seeds_only | 0.938 | 0.938 | 17,727 | 15.6 |
| python_ast_v1 | hybrid | 20 | expanded | 0.938 | 0.938 | 15,958 | 15.8 |
| python_ast_v1 | hybrid_symbol | 5 | seeds_only | 0.562 | 0.531 | 6,160 | 5.0 |
| python_ast_v1 | hybrid_symbol | 5 | expanded | 0.625 | 0.625 | 10,967 | 11.3 |
| python_ast_v1 | hybrid_symbol | 10 | seeds_only | 0.875 | 0.844 | 12,029 | 9.7 |
| python_ast_v1 | hybrid_symbol | 10 | expanded | 0.875 | 0.875 | 15,239 | 16.5 |
| python_ast_v1 | hybrid_symbol | 15 | seeds_only | 0.875 | 0.875 | 15,241 | 13.0 |
| python_ast_v1 | hybrid_symbol | 15 | expanded | 0.875 | 0.875 | 15,904 | 15.6 |
| python_ast_v1 | hybrid_symbol | 20 | seeds_only | 0.938 | 0.938 | 17,852 | 15.7 |
| python_ast_v1 | hybrid_symbol | 20 | expanded | 0.938 | 0.938 | 15,935 | 15.8 |
