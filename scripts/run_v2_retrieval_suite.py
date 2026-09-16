from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen AgentFlow V2 retrieval benchmark end to end."
    )
    parser.add_argument("--bge-small-zh-path", default="")
    parser.add_argument("--bge-small-en-path", default="")
    parser.add_argument("--bge-m3-path", default="")
    parser.add_argument("--reranker-path", default="BAAI/bge-reranker-base")
    return parser.parse_args()


def run(*arguments: str) -> None:
    command = [sys.executable, "-B", *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def model_overrides(args: argparse.Namespace) -> list[str]:
    values = []
    for key, path in (
        ("bge-small-zh-v1.5", args.bge_small_zh_path),
        ("bge-small-en-v1.5", args.bge_small_en_path),
        ("bge-m3", args.bge_m3_path),
    ):
        if path:
            values.extend(["--model-override", f"{key}={path}"])
    return values


def main() -> None:
    args = parse_args()
    overrides = model_overrides(args)
    run("scripts/audit_v2_eval.py", "--require-verified")
    run("scripts/audit_v2_indexes.py")
    run("scripts/evaluate_v2_retrieval.py", "--mode", "matrix", *overrides)
    run("scripts/select_v2_retrieval_config.py")
    selected = json.loads(
        (PROJECT_ROOT / "reports/v2/selected_config.json").read_text(encoding="utf-8")
    )["selected"]
    run(
        "scripts/evaluate_v2_retrieval.py",
        "--mode",
        "ablation",
        "--strategy",
        selected["strategy"],
        "--model",
        selected["model"],
        "--with-reranker",
        "--with-lexical-expansion",
        "--reranker-model",
        args.reranker_path,
        *overrides,
    )
    run("scripts/select_v2_final_pipeline.py")
    run(
        "scripts/evaluate_v2_retrieval.py",
        "--mode",
        "frozen",
        "--strategy",
        "fixed_char",
        "--model",
        "bge-small-zh-v1.5",
        *overrides,
    )
    run(
        "scripts/evaluate_v2_retrieval.py",
        "--mode",
        "frozen",
        "--strategy",
        selected["strategy"],
        "--model",
        selected["model"],
        "--with-reranker",
        "--with-lexical-expansion",
        "--reranker-model",
        args.reranker_path,
        *overrides,
    )


if __name__ == "__main__":
    main()
