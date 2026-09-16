from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATEGIES = ("fixed_char", "recursive_token", "section_aware")
MODELS = ("bge-small-zh-v1.5", "bge-small-en-v1.5", "bge-m3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all nine isolated V2 vector indexes.")
    parser.add_argument("--bge-small-zh-path", default="")
    parser.add_argument("--bge-small-en-path", default="")
    parser.add_argument("--bge-m3-path", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "bge-small-zh-v1.5": args.bge_small_zh_path,
        "bge-small-en-v1.5": args.bge_small_en_path,
        "bge-m3": args.bge_m3_path,
    }
    for strategy in STRATEGIES:
        for model in MODELS:
            metadata = (
                PROJECT_ROOT
                / "data/index/v2"
                / strategy
                / model
                / "index_metadata.json"
            )
            if metadata.exists() and not args.force:
                print(f"skip existing {strategy} x {model}")
                continue
            command = [
                sys.executable,
                "-B",
                "scripts/build_v2_vector_index.py",
                "--strategy",
                strategy,
                "--model",
                model,
            ]
            if paths[model]:
                command.extend(["--model-path", paths[model]])
            if args.force:
                command.append("--force")
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
