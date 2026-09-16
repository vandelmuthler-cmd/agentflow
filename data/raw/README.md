# V2 Corpus Setup

AgentFlow does not redistribute third-party PDFs. The frozen benchmark uses the 16 documents listed in `../v2_corpus_manifest.json`.

The manifest records each document ID, title, language, role, source URL, DOI when available, expected file size, and SHA-256. Download an authorized copy of every document and keep the manifest filename unchanged.

By default, place the files under:

```text
data/raw/v2/
```

Alternatively, point to an external directory so the repository stays clean:

```env
AGENTFLOW_CORPUS_DIR=/absolute/path/to/v2-corpus
```

Verify an individual file on PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "path\to\document.pdf"
```

Prepare all three chunking indexes:

```bash
python -B scripts/prepare_v2_corpus.py --strategy all
```

Build the complete 3×3 vector-index matrix:

```bash
python -B scripts/build_v2_indexes.py
```

Available model keys:

- `bge-small-zh-v1.5`
- `bge-small-en-v1.5`
- `bge-m3`

Use `--model-path` for an existing local model directory. Generated document and vector indexes are excluded from Git; published frozen reports remain under `reports/v2/`.
