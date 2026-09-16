# Bilingual Corpus and Evaluation

AgentFlow uses 16 selected English and Chinese publications as a difficult frozen corpus for evaluating a general-purpose technical-document Agent.

Third-party PDFs are not redistributed. `data/v2_corpus_manifest.json` records the public source, DOI when available, language, role, document type, extraction status, and SHA-256 for each file. Set `AGENTFLOW_CORPUS_DIR` to an authorized local copy of the corpus.

## Prepare the corpus

```bash
python -B scripts/prepare_v2_corpus.py --strategy all
```

This creates three independent document indexes under `data/index/v2/`:

- `fixed_char`: 500 characters with 50-character overlap.
- `recursive_token`: recursive paragraph and sentence splitting, targeting 300 tokens with 50-token overlap.
- `section_aware`: section boundaries followed by the same recursive token splitter.

Tails shorter than 80 estimated tokens are merged into the preceding chunk. Each index includes a report with per-document counts, token statistics, coverage, and parsing errors.

## Build isolated vector indexes

Run the following command for every chunking and embedding combination:

```bash
python -B scripts/build_v2_vector_index.py --strategy section_aware --model bge-m3
```

Available model keys are `bge-small-zh-v1.5`, `bge-small-en-v1.5`, and `bge-m3`. A local model can be supplied without placing its machine-specific path in public configuration:

```bash
python -B scripts/build_v2_vector_index.py \
  --strategy section_aware \
  --model bge-small-zh-v1.5 \
  --model-path /authorized/local/model/path
```

Every model and chunking pair has a separate directory containing vectors, IDs, and build metadata. The loader rejects missing and dimension-mismatched indexes.

## Evaluation-data governance

`data/v2_eval_frozen.json` contains 80 bilingual question intents:

- 20 development intents used for selecting chunking, embedding, and retrieval parameters.
- 40 frozen answerable intents, including 20 questions from documents excluded from parameter selection.
- 20 stress intents covering unanswerable, ambiguous, conflicting-number, and prompt-injection cases.

Each answerable item stores Gold evidence as `document_id + page + evidence_quote + claim_id`. Gold spans are mapped to the current chunks at evaluation time, so a chunking change does not silently redefine correctness.

All 80 cases were checked against the source pages for answerability, bilingual consistency, required-fact coverage, and Gold evidence support. Ten cases were corrected before freezing. The resulting cases use `source_verified`; this records an AI-assisted manual source audit and does not claim an independent human blind review.

Validate the frozen dataset and all Gold-to-Chunk mappings with:

```bash
python -B scripts/audit_v2_eval.py --require-verified
```

Rejected or corrected data must be published as a new version. Frozen files are never overwritten.

## Retrieval experiments

The nine-way development matrix runs both language variants for every intent:

```bash
python -B scripts/evaluate_v2_retrieval.py --mode matrix
python -B scripts/select_v2_retrieval_config.py
```

The selected configuration can then be evaluated with BM25, Vector, candidate union, Reciprocal Rank Fusion, weighted fusion, query expansion, and a Cross-Encoder:

```bash
python -B scripts/evaluate_v2_retrieval.py \
  --mode ablation --strategy section_aware --model bge-m3 --with-reranker \
  --reranker-model /authorized/local/bge-reranker-base \
  --output-dir reports/v2/ablation
```

Reports are written as JSON, CSV, and Markdown under `reports/v2/`. Selection uses development All-gold Hit@5 first, MRR@10 for configurations within one bilingual intent, then latency and memory.

## End-to-end experiment

The online run evaluates the 40 frozen answerable and 20 stress intents using their designated primary language:

```bash
python -B scripts/evaluate_v2_e2e.py \
  --strategy section_aware --model bge-m3 --method cross_encoder \
  --reranker-model /authorized/local/bge-reranker-base \
  --output-dir reports/v2/e2e/cross_encoder-new-run
```

The runner checkpoints completed rows in its details file. `--retry-failures` reruns only failed cases with the same dataset and retrieval configuration, or refreshes derived scores from completed rows without calling DeepSeek again. Keep different retrieval methods in different output directories.

To review generated answers against the question, reference answer, Gold passage, and actual retrieved chunk text, export a worksheet:

```bash
python -B scripts/export_v2_e2e_review.py \
  --details reports/v2/e2e_cross_encoder/details.json \
  --output /private/path/v2_e2e_review.csv
```

Fill `score_0_1_2`, `citation_support`, and `reason`, then run the same evaluation command with `--retry-failures --human-scores /private/path/v2_e2e_review.csv`. Automated citation-ID validity and Gold overlap do not measure whether a passage semantically supports a claim. Independent human review remains the stronger protocol for final answer correctness.
