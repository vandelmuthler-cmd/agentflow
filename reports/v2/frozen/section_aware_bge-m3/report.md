# AgentFlow V2 Retrieval Report

- Status: `FROZEN`
- Dataset SHA-256: `5a6d83f12bb7dd21a21582c2076447c643265496f49e56809adb3afdc797c0ef`

| Chunking | Embedding | Method | All-gold Hit@5 | MRR@10 | nDCG@10 | Candidate all-gold |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| section_aware | bge-m3 | bm25 | 0.500 | 0.465 | 0.468 | 0.537 |
| section_aware | bge-m3 | vector | 0.812 | 0.720 | 0.742 | 1.000 |
| section_aware | bge-m3 | candidate_union | 0.500 | 0.465 | 0.468 | 1.000 |
| section_aware | bge-m3 | rrf | 0.625 | 0.500 | 0.537 | 1.000 |
| section_aware | bge-m3 | weighted | 0.650 | 0.536 | 0.588 | 1.000 |
| section_aware | bge-m3 | rrf_expanded | 0.637 | 0.494 | 0.534 | 1.000 |
| section_aware | bge-m3 | weighted_expanded | 0.650 | 0.531 | 0.589 | 1.000 |
| section_aware | bge-m3 | rrf_lexical_expanded | 0.625 | 0.500 | 0.537 | 1.000 |
| section_aware | bge-m3 | weighted_lexical_expanded | 0.650 | 0.530 | 0.585 | 1.000 |
| section_aware | bge-m3 | cross_encoder | 0.950 | 0.914 | 0.900 | 0.975 |
