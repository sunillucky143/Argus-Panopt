"""Score evaluation cases and write metadata-only reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from evals.models import EvalCase, EvalDataset
from evals.runners import EvaluationError, Runner

_REPORT_STEM = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class ReportError(RuntimeError):
    """A content-safe evaluation report could not be produced."""


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Content-free score and timing metadata for one evaluation case."""

    case_id: str
    category: str
    source_format: str
    status: str
    exact_match: bool
    required_term_coverage: float
    ttft_ms: float | None
    latency_ms: float | None
    finish_reason: str | None
    response_sha256: str | None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate evaluation evidence without prompts, context, or answers."""

    schema_version: int
    dataset_id: str
    synthetic: bool
    runner: str
    generated_at: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    error_cases: int
    exact_match_rate: float
    average_required_term_coverage: float
    average_ttft_ms: float | None
    average_latency_ms: float | None
    cases: tuple[CaseResult, ...]


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _score(case: EvalCase, response: str) -> tuple[bool, float]:
    normalized_response = _normalize(response)
    exact = normalized_response == _normalize(case.expected_answer)
    padded_response = f" {normalized_response} "
    matched = sum(1 for term in case.required_terms if f" {_normalize(term)} " in padded_response)
    return exact, matched / len(case.required_terms)


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def evaluate(dataset: EvalDataset, runner: Runner) -> EvaluationReport:
    """Run every case and return a content-free aggregate report."""

    results: list[CaseResult] = []
    for case in dataset.cases:
        try:
            generation = runner.generate(dataset, case)
            exact, coverage = _score(case, generation.text)
            status = "passed" if exact or coverage == 1.0 else "failed"
            results.append(
                CaseResult(
                    case_id=case.identifier,
                    category=case.category,
                    source_format=case.source_format,
                    status=status,
                    exact_match=exact,
                    required_term_coverage=round(coverage, 3),
                    ttft_ms=round(generation.ttft_ms, 3),
                    latency_ms=round(generation.latency_ms, 3),
                    finish_reason=generation.finish_reason,
                    response_sha256=hashlib.sha256(generation.text.encode("utf-8")).hexdigest(),
                )
            )
        except EvaluationError:
            results.append(
                CaseResult(
                    case_id=case.identifier,
                    category=case.category,
                    source_format=case.source_format,
                    status="error",
                    exact_match=False,
                    required_term_coverage=0.0,
                    ttft_ms=None,
                    latency_ms=None,
                    finish_reason=None,
                    response_sha256=None,
                )
            )

    passed = sum(result.status == "passed" for result in results)
    errors = sum(result.status == "error" for result in results)
    measured = [result for result in results if result.status != "error"]
    if measured:
        exact_match_rate = round(sum(result.exact_match for result in measured) / len(measured), 3)
        average_coverage = round(
            sum(result.required_term_coverage for result in measured) / len(measured), 3
        )
    else:
        exact_match_rate = average_coverage = 0.0
    return EvaluationReport(
        schema_version=1,
        dataset_id=dataset.identifier,
        synthetic=dataset.synthetic,
        runner=runner.name,
        generated_at=datetime.now(UTC).isoformat(),
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=len(results) - passed - errors,
        error_cases=errors,
        exact_match_rate=exact_match_rate,
        average_required_term_coverage=average_coverage,
        average_ttft_ms=_average(
            [result.ttft_ms for result in measured if result.ttft_ms is not None]
        ),
        average_latency_ms=_average(
            [result.latency_ms for result in measured if result.latency_ms is not None]
        ),
        cases=tuple(results),
    )


def _json_bytes(report: EvaluationReport) -> bytes:
    return (json.dumps(asdict(report), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def _markdown_bytes(report: EvaluationReport) -> bytes:
    lines = [
        f"# Evaluation report: {report.dataset_id}",
        "",
        f"- Runner: `{report.runner}`",
        f"- Synthetic dataset: `{str(report.synthetic).lower()}`",
        f"- Generated: `{report.generated_at}`",
        f"- Passed: {report.passed_cases}/{report.total_cases}",
        f"- Failed: {report.failed_cases}",
        f"- Errors: {report.error_cases}",
        f"- Exact-match rate: {report.exact_match_rate:.3f}",
        (f"- Average required-term coverage: {report.average_required_term_coverage:.3f}"),
        "",
        "| Case | Category | Format | Status | Exact | Term coverage | TTFT ms | Latency ms |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for result in report.cases:
        ttft = "-" if result.ttft_ms is None else f"{result.ttft_ms:.3f}"
        latency = "-" if result.latency_ms is None else f"{result.latency_ms:.3f}"
        lines.append(
            f"| {result.case_id} | {result.category} | {result.source_format} | "
            f"{result.status} | {str(result.exact_match).lower()} | "
            f"{result.required_term_coverage:.3f} | {ttft} | {latency} |"
        )
    lines.extend(
        [
            "",
            (
                "This report intentionally excludes prompts, source context, expected "
                "answers, and generated text."
            ),
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ReportError(f"unable to write evaluation report: {type(error).__name__}") from None


def write_reports(
    report: EvaluationReport,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports atomically with private file modes."""

    if not _REPORT_STEM.fullmatch(stem):
        raise ReportError("report stem is invalid")
    try:
        if output_dir.is_symlink():
            raise ReportError("report directory must not be a symbolic link")
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        output_root = output_dir.resolve(strict=True)
        json_path = output_root / f"{stem}.json"
        markdown_path = output_root / f"{stem}.md"
        if json_path.is_symlink() or markdown_path.is_symlink():
            raise ReportError("report path must not be a symbolic link")
        _atomic_write(json_path, _json_bytes(report))
        _atomic_write(markdown_path, _markdown_bytes(report))
        return json_path, markdown_path
    except ReportError:
        raise
    except OSError as error:
        raise ReportError(f"unable to prepare report directory: {type(error).__name__}") from None
