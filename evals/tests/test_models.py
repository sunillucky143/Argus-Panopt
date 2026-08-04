from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from evals.models import DatasetError, load_dataset

_ROOT = Path(__file__).resolve().parents[2]
_SEED = _ROOT / "evals" / "datasets" / "phase1-seed.json"


def _valid_dataset() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_id": "test-set",
        "description": "Synthetic fixture.",
        "synthetic": True,
        "cases": [
            {
                "id": "case-one",
                "category": "laboratory",
                "source_format": "pdf",
                "context": "Synthetic result is 10.",
                "question": "What is the result?",
                "expected_answer": "The result is 10.",
                "required_terms": ["result", "10"],
            }
        ],
    }


def test_phase1_seed_has_twenty_synthetic_diverse_cases() -> None:
    dataset = load_dataset(_SEED)

    assert dataset.identifier == "phase1-seed"
    assert dataset.synthetic is True
    assert len(dataset.cases) == 20
    assert {case.source_format for case in dataset.cases} == {"pdf", "docx", "xlsx"}
    assert len({case.identifier for case in dataset.cases}) == 20


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update(schema_version=2), "schema_version"),
        (lambda data: data.update(synthetic="yes"), "synthetic"),
        (lambda data: data.update(cases=[]), "cases"),
        (
            lambda data: data["cases"][0].update(source_format="txt"),
            "source_format",
        ),
        (
            lambda data: data["cases"][0].update(required_terms=[]),
            "required_terms",
        ),
        (
            lambda data: data["cases"][0].update(required_terms=["!!!"]),
            "required_terms",
        ),
        (
            lambda data: data["cases"][0].update(required_terms=["Dose", "dose"]),
            "required_terms",
        ),
        (
            lambda data: data["cases"].append(data["cases"][0]),
            "case ids",
        ),
    ],
)
def test_dataset_rejects_schema_violations(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    data = _valid_dataset()
    mutate(data)
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_dataset_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1,"dataset_id":"x",'
        '"description":"x","synthetic":true,"cases":[]}',
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="duplicate key"):
        load_dataset(path)


def test_dataset_rejects_unreadable_and_oversized_inputs(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="unable to read"):
        load_dataset(tmp_path / "missing.json")

    oversized = tmp_path / "large.json"
    oversized.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    with pytest.raises(DatasetError, match="maximum size"):
        load_dataset(oversized)
