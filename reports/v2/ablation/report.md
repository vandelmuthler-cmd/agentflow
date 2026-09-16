# AgentFlow V2 Retrieval Report

- Status: `FROZEN`
- Dataset SHA-256: `5a6d83f12bb7dd21a21582c2076447c643265496f49e56809adb3afdc797c0ef`

| Chunking | Embedding | Method | All-gold Hit@5 | MRR@10 | nDCG@10 | Candidate all-gold |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| section_aware | bge-m3 | bm25 | 0.400 | 0.440 | 0.432 | 0.500 |
| section_aware | bge-m3 | vector | 0.725 | 0.720 | 0.712 | 0.975 |
| section_aware | bge-m3 | candidate_union | 0.400 | 0.440 | 0.432 | 0.975 |
| section_aware | bge-m3 | rrf | 0.475 | 0.483 | 0.515 | 0.975 |
| section_aware | bge-m3 | weighted | 0.575 | 0.512 | 0.547 | 0.975 |
| section_aware | bge-m3 | rrf_expanded | 0.525 | 0.493 | 0.530 | 0.975 |
| section_aware | bge-m3 | weighted_expanded | 0.625 | 0.522 | 0.560 | 0.975 |
| section_aware | bge-m3 | rrf_lexical_expanded | 0.475 | 0.482 | 0.512 | 0.975 |
| section_aware | bge-m3 | weighted_lexical_expanded | 0.600 | 0.515 | 0.546 | 0.975 |
| section_aware | bge-m3 | cross_encoder | 0.825 | 0.808 | 0.820 | 0.950 |
