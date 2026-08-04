from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.models import EvalCase, EvalDataset, load_dataset
from evals.reporting import ReportError, _score, evaluate, write_reports
from evals.run import main
from evals.runners import EvaluationError, GenerationResult

_ROOT = Path(__file__).resolve().parents[2]
_SEED = _ROOT / "evals" / "datasets" / "phase1-seed.json"
_DATASET = load_dataset(_SEED)


class _PartialRunner:
    name = "partial"

    def generate(self, dataset: EvalDataset, case: EvalCase) -> GenerationResult:
        del dataset
        if case.identifier == _DATASET.cases[0].identifier:
            raise EvaluationError(f"must not expose {_DATASET.cases[0].context}")
        return GenerationResult(
            text="unrelated response",
            ttft_ms=1.23456,
            latency_ms=2.34567,
            finish_reason="length",
        )


def test_fixture_report_passes_and_excludes_all_content(tmp_path: Path) -> None:
    result = main(
        [
            "--runner",
            "fixture",
            "--dataset",
            str(_SEED),
            "--output-dir",
            str(tmp_path),
            "--report-stem",
            "phase1-test",
        ]
    )

    assert result == 0
    json_text = (tmp_path / "phase1-test.json").read_text(encoding="utf-8")
    markdown = (tmp_path / "phase1-test.md").read_text(encoding="utf-8")
    report = json.loads(json_text)
    assert report["passed_cases"] == 20
    assert report["exact_match_rate"] == 1.0
    assert report["average_required_term_coverage"] == 1.0
    assert len(report["cases"]) == 20
    for case in _DATASET.cases:
        assert case.context not in json_text
        assert case.question not in json_text
        assert case.expected_answer not in json_text
        assert case.context not in markdown
    assert "intentionally excludes prompts" in markdown


def test_report_records_failures_and_errors_without_error_details() -> None:
    report = evaluate(_DATASET, _PartialRunner())

    assert report.passed_cases == 0
    assert report.failed_cases == 19
    assert report.error_cases == 1
    assert report.cases[0].status == "error"
    assert report.cases[0].response_sha256 is None
    assert report.cases[1].status == "failed"
    assert report.cases[1].response_sha256 is not None
    assert report.average_latency_ms == 2.346


def test_required_term_matching_uses_phrase_boundaries() -> None:
    _, coverage = _score(_DATASET.cases[0], "lisinopril 100 mg once daily")

    assert coverage == pytest.approx(2 / 3)


@pytest.mark.parametrize("stem", ["../escape", "UPPER", "", "a" * 65])
def test_report_rejects_unsafe_stems(tmp_path: Path, stem: str) -> None:
    report = evaluate(_DATASET, _PartialRunner())

    with pytest.raises(ReportError, match="stem"):
        write_reports(report, tmp_path, stem)


def test_cli_returns_nonzero_for_invalid_dataset(tmp_path: Path) -> None:
    assert main(["--dataset", str(tmp_path / "missing.json"), "--runner", "fixture"]) == 2
