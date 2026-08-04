"""Strict schema for synthetic and operator-supplied evaluation datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_DATASET_BYTES = 2 * 1024 * 1024
_MAX_CASES = 100
_MAX_CONTEXT_CHARS = 32_768
_MAX_QUESTION_CHARS = 4_096
_MAX_ANSWER_CHARS = 4_096
_MAX_REQUIRED_TERMS = 16
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_FORMATS = {"pdf", "docx", "xlsx"}
_DATASET_KEYS = {"schema_version", "dataset_id", "description", "synthetic", "cases"}
_CASE_KEYS = {
    "id",
    "category",
    "source_format",
    "context",
    "question",
    "expected_answer",
    "required_terms",
}


class DatasetError(ValueError):
    """The evaluation dataset is malformed or violates safety bounds."""


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One grounded question with synthetic source text and expected evidence."""

    identifier: str
    category: str
    source_format: str
    context: str
    question: str
    expected_answer: str
    required_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A versioned collection of bounded evaluation cases."""

    identifier: str
    description: str
    synthetic: bool
    cases: tuple[EvalCase, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f"duplicate key: {key}")
        result[key] = value
    return result


def _object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DatasetError(f"{label} keys do not match schema")
    return value


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise DatasetError(f"{label} must be bounded non-empty text")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label, 64)
    if not _IDENTIFIER.fullmatch(text):
        raise DatasetError(f"{label} is invalid")
    return text


def _term_key(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _parse_case(value: Any) -> EvalCase:
    data = _object(value, _CASE_KEYS, "case")
    source_format = _text(data.get("source_format"), "source_format", 16)
    if source_format not in _FORMATS:
        raise DatasetError("source_format is unsupported")
    raw_terms = data.get("required_terms")
    if not isinstance(raw_terms, list) or not 0 < len(raw_terms) <= _MAX_REQUIRED_TERMS:
        raise DatasetError("required_terms must be a bounded unique list")
    terms = tuple(_text(term, "required term", 256) for term in raw_terms)
    term_keys = tuple(_term_key(term) for term in terms)
    if any(not key for key in term_keys) or len(set(term_keys)) != len(term_keys):
        raise DatasetError("required_terms must be a bounded unique list")
    return EvalCase(
        identifier=_identifier(data.get("id"), "case id"),
        category=_identifier(data.get("category"), "category"),
        source_format=source_format,
        context=_text(data.get("context"), "context", _MAX_CONTEXT_CHARS),
        question=_text(data.get("question"), "question", _MAX_QUESTION_CHARS),
        expected_answer=_text(data.get("expected_answer"), "expected_answer", _MAX_ANSWER_CHARS),
        required_terms=terms,
    )


def load_dataset(path: Path) -> EvalDataset:
    """Load a strict, bounded dataset without accepting duplicate JSON keys."""

    try:
        if path.stat().st_size > _MAX_DATASET_BYTES:
            raise DatasetError("dataset exceeds the maximum size")
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except DatasetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetError(f"unable to read dataset: {type(error).__name__}") from None

    data = _object(parsed, _DATASET_KEYS, "dataset")
    if data.get("schema_version") != 1 or type(data.get("schema_version")) is not int:
        raise DatasetError("schema_version must be 1")
    if type(data.get("synthetic")) is not bool:
        raise DatasetError("synthetic must be a boolean")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not 0 < len(raw_cases) <= _MAX_CASES:
        raise DatasetError("cases must be a bounded non-empty list")
    cases = tuple(_parse_case(case) for case in raw_cases)
    if len({case.identifier for case in cases}) != len(cases):
        raise DatasetError("case ids must be unique")
    return EvalDataset(
        identifier=_identifier(data.get("dataset_id"), "dataset_id"),
        description=_text(data.get("description"), "description", 1_024),
        synthetic=data["synthetic"],
        cases=cases,
    )
