import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
_trace_dir = Path(os.getenv("AGENTFLOW_TRACE_DIR", "data/traces"))
TRACE_DIR = _trace_dir if _trace_dir.is_absolute() else PROJECT_ROOT / _trace_dir
INDEX_DIR = DATA_DIR / "index"
DOCUMENT_INDEX_PATH = INDEX_DIR / "documents.jsonl"
VECTOR_INDEX_PATH = INDEX_DIR / "vectors.npy"
VECTOR_IDS_PATH = INDEX_DIR / "vector_ids.json"
RETRIEVAL_BACKEND = os.getenv("AGENTFLOW_RETRIEVAL_BACKEND", "local").strip().lower()
DATABASE_URL = os.getenv(
    "AGENTFLOW_DATABASE_URL",
    "postgresql://agentflow:agentflow@postgres:5432/agentflow",
)
ADMIN_API_KEY = os.getenv("AGENTFLOW_ADMIN_API_KEY", "")
EMBED_MODEL_PATH = os.getenv(
    "AGENTFLOW_EMBED_MODEL_PATH",
    "BAAI/bge-m3",
)
RERANKER_MODEL_PATH = os.getenv(
    "AGENTFLOW_RERANKER_MODEL_PATH",
    "BAAI/bge-reranker-base",
)
RETRIEVAL_METHOD = os.getenv(
    "AGENTFLOW_RETRIEVAL_METHOD", "cross_encoder"
).strip().lower()
ONLINE_CHUNKING_STRATEGY = os.getenv(
    "AGENTFLOW_CHUNKING_STRATEGY", "section_aware"
).strip().lower()
EVAL_RUNS_DIR = DATA_DIR / "eval_runs"
_run_workspace_dir = Path(os.getenv("AGENTFLOW_RUN_WORKSPACE_DIR", "data/runs"))
RUN_WORKSPACE_DIR = (
    _run_workspace_dir if _run_workspace_dir.is_absolute() else PROJECT_ROOT / _run_workspace_dir
)
_checkpoint_path = Path(os.getenv("AGENTFLOW_CHECKPOINT_DB_PATH", "data/agentflow_checkpoints.sqlite3"))
CHECKPOINT_DB_PATH = _checkpoint_path if _checkpoint_path.is_absolute() else PROJECT_ROOT / _checkpoint_path
QUERY_EXPANSION_TERMS_PATH = DATA_DIR / "query_expansion_terms.json"
_v2_corpus_dir = Path(os.getenv("AGENTFLOW_CORPUS_DIR", "data/raw/v2"))
V2_CORPUS_DIR = (
    _v2_corpus_dir if _v2_corpus_dir.is_absolute() else PROJECT_ROOT / _v2_corpus_dir
)
V2_MANIFEST_PATH = DATA_DIR / "v2_corpus_manifest.json"
_v2_index_root = Path(os.getenv("AGENTFLOW_V2_INDEX_ROOT", "data/index/v2"))
V2_INDEX_ROOT = (
    _v2_index_root if _v2_index_root.is_absolute() else PROJECT_ROOT / _v2_index_root
)
CORPUS_VERSION = os.getenv("AGENTFLOW_CORPUS_VERSION", "v1").strip().lower()
V2_CHUNKING_STRATEGY = os.getenv("AGENTFLOW_V2_CHUNKING_STRATEGY", "section_aware")
V2_EMBED_MODEL = os.getenv("AGENTFLOW_V2_EMBED_MODEL", "bge-m3")
V2_EMBED_MODEL_PATH = os.getenv("AGENTFLOW_V2_EMBED_MODEL_PATH", "")
V2_RETRIEVAL_METHOD = os.getenv("AGENTFLOW_V2_RETRIEVAL_METHOD", "rrf_expanded")
V2_RERANKER_MODEL = os.getenv("AGENTFLOW_V2_RERANKER_MODEL", "BAAI/bge-reranker-base")

DEFAULT_TOP_K = 5
MAX_RETRIES = 2
MAX_RETRIEVAL_QUERIES = int(os.getenv("AGENTFLOW_MAX_RETRIEVAL_QUERIES", "7"))
MAX_FOLLOW_UP_QUERIES = int(os.getenv("AGENTFLOW_MAX_FOLLOW_UP_QUERIES", "2"))
MAX_RETRIEVAL_DURATION_MS = float(
    os.getenv("AGENTFLOW_MAX_RETRIEVAL_DURATION_MS", "30000")
)
MAX_NO_PROGRESS_ROUNDS = int(
    os.getenv("AGENTFLOW_MAX_NO_PROGRESS_ROUNDS", "1")
)
QUERY_SIMILARITY_THRESHOLD = float(
    os.getenv("AGENTFLOW_QUERY_SIMILARITY_THRESHOLD", "0.95")
)
DEFAULT_CANDIDATE_TOP_K = 50
DEFAULT_RERANK_CANDIDATE_TOP_K = 30
DEFAULT_RERANK_VECTOR_WEIGHT = 0.4

LLM_API_KEY = os.getenv("AGENTFLOW_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("AGENTFLOW_LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("AGENTFLOW_LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TIMEOUT_SECONDS = int(os.getenv("AGENTFLOW_LLM_TIMEOUT_SECONDS", "30"))
TOOL_CALLING_ENABLED = os.getenv("AGENTFLOW_TOOL_CALLING_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
LLM_VERIFIER_ENABLED = os.getenv(
    "AGENTFLOW_LLM_VERIFIER_ENABLED", "true"
).strip().lower() not in {"0", "false", "no", "off"}
INPUT_COST_PER_MILLION_USD = float(os.getenv("AGENTFLOW_INPUT_COST_PER_MILLION_USD", "0"))
OUTPUT_COST_PER_MILLION_USD = float(os.getenv("AGENTFLOW_OUTPUT_COST_PER_MILLION_USD", "0"))

MAX_CONTEXT_TOKENS = int(os.getenv("AGENTFLOW_MAX_CONTEXT_TOKENS", "6000"))
RESERVED_OUTPUT_TOKENS = int(os.getenv("AGENTFLOW_RESERVED_OUTPUT_TOKENS", "1200"))
MAX_TOKENS_PER_EVIDENCE = int(os.getenv("AGENTFLOW_MAX_TOKENS_PER_EVIDENCE", "900"))
