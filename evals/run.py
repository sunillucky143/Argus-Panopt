"""Command-line entrypoint for adapter-independent local model evaluations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from evals.models import DatasetError, load_dataset
from evals.reporting import ReportError, evaluate, write_reports
from evals.runners import FixtureRunner, HttpSseRunner, Runner

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET = _ROOT / "evals" / "datasets" / "phase1-seed.json"
_DEFAULT_REPORTS = _ROOT / "evals" / "reports"
_DEFAULT_ENDPOINT = "http://127.0.0.1:8080/api/v1/debug/generate"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run content-safe evaluations against a local Argus model adapter."
    )
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--runner", choices=("fixture", "http"), default="http")
    parser.add_argument("--endpoint")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_REPORTS)
    parser.add_argument("--report-stem")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        runner: Runner
        if args.runner == "fixture":
            runner = FixtureRunner()
        else:
            endpoint = args.endpoint or os.environ.get("ARGUS_EVAL_ENDPOINT", _DEFAULT_ENDPOINT)
            runner = HttpSseRunner(
                endpoint,
                timeout_seconds=args.timeout,
                max_tokens=args.max_tokens,
            )
        report = evaluate(dataset, runner)
        stem = args.report_stem or f"{dataset.identifier}-{runner.name}"
        json_path, markdown_path = write_reports(report, args.output_dir, stem)
    except (DatasetError, ReportError, ValueError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 2

    print(
        f"evaluation complete: {report.passed_cases}/{report.total_cases} passed; "
        f"{report.failed_cases} failed; {report.error_cases} errors"
    )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0 if report.failed_cases == 0 and report.error_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
