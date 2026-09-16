# AgentFlow V2 Retrieval Report

- Status: `FROZEN`
- Dataset SHA-256: `5a6d83f12bb7dd21a21582c2076447c643265496f49e56809adb3afdc797c0ef`

| Chunking | Embedding | Method | All-gold Hit@5 | MRR@10 | nDCG@10 | Candidate all-gold |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| fixed_char | bge-small-zh-v1.5 | vector | 0.300 | 0.279 | 0.297 | 0.575 |
| fixed_char | bge-small-en-v1.5 | vector | 0.225 | 0.295 | 0.328 | 0.600 |
| fixed_char | bge-m3 | vector | 0.625 | 0.581 | 0.598 | 0.925 |
| recursive_token | bge-small-zh-v1.5 | vector | 0.200 | 0.140 | 0.167 | 0.400 |
| recursive_token | bge-small-en-v1.5 | vector | 0.375 | 0.314 | 0.363 | 0.675 |
| recursive_token | bge-m3 | vector | 0.650 | 0.649 | 0.656 | 0.975 |
| section_aware | bge-small-zh-v1.5 | vector | 0.275 | 0.194 | 0.237 | 0.600 |
| section_aware | bge-small-en-v1.5 | vector | 0.450 | 0.372 | 0.417 | 0.700 |
| section_aware | bge-m3 | vector | 0.725 | 0.720 | 0.712 | 0.975 |
