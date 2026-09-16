# AgentFlow V2 Retrieval Report

- Status: `FROZEN`
- Dataset SHA-256: `5a6d83f12bb7dd21a21582c2076447c643265496f49e56809adb3afdc797c0ef`

| Chunking | Embedding | Method | All-gold Hit@5 | MRR@10 | nDCG@10 | Candidate all-gold |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| fixed_char | bge-small-zh-v1.5 | bm25 | 0.438 | 0.361 | 0.381 | 0.525 |
| fixed_char | bge-small-zh-v1.5 | vector | 0.250 | 0.167 | 0.196 | 0.450 |
| fixed_char | bge-small-zh-v1.5 | candidate_union | 0.438 | 0.361 | 0.381 | 0.588 |
| fixed_char | bge-small-zh-v1.5 | rrf | 0.300 | 0.262 | 0.287 | 0.588 |
| fixed_char | bge-small-zh-v1.5 | weighted | 0.438 | 0.338 | 0.362 | 0.588 |
| fixed_char | bge-small-zh-v1.5 | rrf_expanded | 0.300 | 0.250 | 0.281 | 0.625 |
| fixed_char | bge-small-zh-v1.5 | weighted_expanded | 0.438 | 0.338 | 0.362 | 0.625 |
