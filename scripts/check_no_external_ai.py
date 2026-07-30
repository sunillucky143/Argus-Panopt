#!/usr/bin/env python3
"""Fail when source or lockfiles reference prohibited external AI services."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
IGNORED_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".venv", "dist", "node_modules"}
SCANNER_PATH = Path(__file__).resolve()

FORBIDDEN_PACKAGES = (
    "anthropic",
    "cohere",
    "google-generativeai",
    "google-genai",
    "mistralai",
    "openai",
)
FORBIDDEN_HOSTS = (
    "api.anthropic.com",
    "api.cohere.com",
    "api.openai.com",
    "generativelanguage.googleapis.com",
)


def iter_text_files(root: Path) -> list[Path]:
    """Return deterministic candidates while excluding generated content."""

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.resolve() != SCANNER_PATH
        and not any(part in IGNORED_DIRECTORIES for part in path.parts)
        and (path.suffix.lower() in TEXT_SUFFIXES or path.name in {"uv.lock", "package-lock.json"})
    )


def violations(root: Path) -> list[str]:
    """Find prohibited package tokens and API hosts in repository text."""

    package_pattern = re.compile(
        r"(?i)(?:^|[\"'/=\s])(" + "|".join(re.escape(item) for item in FORBIDDEN_PACKAGES) + r")"
        r"(?:[<=>@\"'/,\s]|$)"
    )
    host_pattern = re.compile("|".join(re.escape(item) for item in FORBIDDEN_HOSTS), re.IGNORECASE)
    findings: list[str] = []
    for path in iter_text_files(root):
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if package_pattern.search(line) or host_pattern.search(line):
                findings.append(f"{path.relative_to(root)}:{line_number}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    findings = violations(root)
    if findings:
        print("Prohibited external AI dependency or hostname detected:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1
    print("No prohibited external AI dependencies or hostnames detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
