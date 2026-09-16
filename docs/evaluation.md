# V2 Evaluation

## Scope

- Corpus: 16 documents, 12 English and 4 Chinese.
- Frozen dataset: 80 bilingual intents with 160 query variants.
- Development split: 20 intents, 40 query variants.
- Frozen answerable split: 40 intents, 80 query variants.
- Stress split: 20 primary-language questions, including unanswerable, ambiguous, numerical-conflict, and prompt-injection cases.
- Section-aware index: 934 chunks.
- Dataset SHA-256: `5a6d83f12bb7dd21a21582c2076447c643265496f49e56809adb3afdc797c0ef`.

The dataset was manually inspected against local source pages with AI assistance. This is a source audit, not an independent human blind review. The audit corrected ten cases before freezing: one PDF equation extraction error, one missing numerical answer, and eight cases whose Gold evidence did not cover every required fact. All 80 Gold spans map to all three chunking strategies.

## Metrics

- **All-gold Hit@K**: a query variant scores 1 only when every required Gold claim has at least one mapped chunk in the top K.
- **Claim Recall@K**: retrieved Gold claims divided by all required Gold claims.
- **Candidate all-gold**: whether the pre-reranking candidate pool contains every Gold claim.
- **MRR@10**: reciprocal rank of the first relevant Gold result, averaged over query variants.
- **nDCG@10**: ranking quality with the number of covered Gold claims as relevance.
- **Confidence interval**: 2,000-sample bootstrap 95% interval over query-level All-gold Hit@5.

These are query-level retrieval metrics. They are not classical corpus-level recall over every relevant passage.

## Chunking and Embedding selection

The 3×3 matrix was evaluated only on the development split.

| Chunking | Embedding | All-gold Hit@5 | MRR@10 | Candidate @50 | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed character | bge-small-zh-v1.5 | 30.0% | 0.279 | 57.5% | 16.9 ms |
| fixed character | bge-small-en-v1.5 | 22.5% | 0.295 | 60.0% | 27.6 ms |
| fixed character | BGE-M3 | 62.5% | 0.581 | 92.5% | 201.4 ms |
| recursive token | bge-small-zh-v1.5 | 20.0% | 0.140 | 40.0% | 19.5 ms |
| recursive token | bge-small-en-v1.5 | 37.5% | 0.314 | 67.5% | 26.4 ms |
| recursive token | BGE-M3 | 65.0% | 0.649 | 97.5% | 197.3 ms |
| section aware | bge-small-zh-v1.5 | 27.5% | 0.194 | 60.0% | 16.4 ms |
| section aware | bge-small-en-v1.5 | 45.0% | 0.372 | 70.0% | 26.8 ms |
| **section aware** | **BGE-M3** | **72.5%** | **0.720** | **97.5%** | **195.9 ms** |

Selection rule: maximize development All-gold Hit@5; within one bilingual intent, maximize MRR@10; if still tied, prefer lower latency, parameter memory, and index size. This selected `section_aware + BGE-M3` before the frozen split was run.

## Development retrieval ablation

| Method | All-gold Hit@5 | MRR@10 | Candidate all-gold | Mean latency |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 40.0% | 0.440 | 50.0% | 11.7 ms |
| BGE-M3 Vector | 72.5% | 0.720 | 97.5% | 200.9 ms |
| Candidate union ranking | 40.0% | 0.440 | 97.5% | 213.0 ms |
| RRF | 47.5% | 0.483 | 97.5% | 213.0 ms |
| Linear weighted fusion | 57.5% | 0.512 | 97.5% | 213.1 ms |
| RRF + expansion | 52.5% | 0.493 | 97.5% | 216.1 ms |
| Weighted + expansion | 62.5% | 0.522 | 97.5% | 216.1 ms |
| Lexical-only expansion + weighted | 60.0% | 0.515 | 97.5% | 213.3 ms |
| **Expanded RRF top-30 + Cross-Encoder** | **82.5%** | **0.808** | **95.0%** | **10.01 s** |

Vector retrieval outperformed BM25-based fusion on this bilingual corpus. The final Cross-Encoder path was retained because reranking improved both Hit@5 and MRR, while the Vector path remains the practical low-latency option.

## Frozen retrieval results

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Candidate all-gold | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 41.3% | 48.8% | 50.0% | 0.465 | 53.8% | 10.7 ms |
| BGE-M3 Vector | 60.0% | 78.8% | 81.3% | 0.720 | 100.0% | 192.7 ms |
| RRF | 38.8% | 53.8% | 62.5% | 0.500 | 100.0% | 203.8 ms |
| RRF + expansion | 37.5% | 55.0% | 63.8% | 0.494 | 100.0% | 199.5 ms |
| **Cross-Encoder** | **85.0%** | **95.0%** | **95.0%** | **0.914** | **97.5% @30** | **9.65 s** |

Cross-Encoder All-gold Hit@5 95% bootstrap interval: 90.0% to 98.75%. The exact report is stored in `reports/v2/frozen/section_aware_bge-m3/`.

## End-to-end results

The end-to-end experiment uses one primary-language query per frozen or stress intent: 60 cases total, 44 answerable and 16 unanswerable/safety cases.

| Metric | Vector | Cross-Encoder |
| --- | ---: | ---: |
| Answerable all-gold Hit@5 in final context | 79.5% | 86.4% |
| Gold claim citation coverage | 81.4% | 90.9% |
| Citation ID validity | 100.0% | 100.0% |
| Unanswerable abstention keyword rate | 100.0% | 93.8% |
| Numerical-conflict phrase rate | 75.0% | 100.0% |
| Mean end-to-end latency | 8.20 s | 23.04 s |
| P95 end-to-end latency | 13.53 s | 59.23 s |
| Model tokens | 351,099 | 326,988 |
| Logical retrieval calls | 118 | 102 |

AI-assisted answer review for the Cross-Encoder run: 55 complete, 3 partial, and 2 failed. The five bad cases and reasons are stored in `reports/v2/e2e/cross_encoder/answer_review.json`. This score is not presented as independent human evaluation.

The 60 generations were produced before the final source audit. They were reused because case IDs, primary languages, and question text were byte-for-byte unchanged; only reference answers and Gold evidence were corrected. `scripts/rescore_v2_e2e.py` verifies this identity before recomputing metrics. Both generation and evaluation dataset hashes are stored in each final provenance file.

## Reproduction

After acquiring the files listed in `data/v2_corpus_manifest.json`:

```bash
python -B scripts/prepare_v2_corpus.py --strategy all
python -B scripts/build_v2_indexes.py
python -B scripts/run_v2_retrieval_suite.py
```

Local model directories may be passed to avoid downloads:

```bash
python -B scripts/run_v2_retrieval_suite.py \
  --bge-small-zh-path /models/bge-small-zh-v1.5 \
  --bge-small-en-path /models/bge-small-en-v1.5 \
  --bge-m3-path /models/bge-m3 \
  --reranker-path /models/bge-reranker-base
```

Exact machine-dependent latency should not be expected to reproduce across CPU/GPU environments. Ranking metrics, dataset hashes, document hashes, and index alignment checks are the reproducibility targets.
