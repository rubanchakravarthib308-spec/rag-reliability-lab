from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_benchmark, to_json, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic RAG reliability benchmark")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--json", dest="json_path", default="benchmark-report.json")
    parser.add_argument("--markdown", dest="markdown_path", default="benchmark-report.md")
    args = parser.parse_args()

    summary = run_benchmark(args.version)
    Path(args.json_path).write_text(to_json(summary) + "\n", encoding="utf-8")
    Path(args.markdown_path).write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary), end="")
    raise SystemExit(0 if summary.passed else 1)


if __name__ == "__main__":
    main()
