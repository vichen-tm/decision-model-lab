"""Validate user JSONL before preparing any network operation."""

from ..archive import read_jsonl
from ..scoring import validate_case


def load_dataset(path):
    cases = {}
    for line, case in enumerate(read_jsonl(path), 1):
        if not isinstance(case, dict):
            raise ValueError(f"Dataset row {line} must be an object")
        for name in ("id", "language", "suite"):
            if not isinstance(case.get(name), str) or not case[name].strip():
                raise ValueError(f"Dataset row {line}: non-empty {name} required")
        validate_case(case)
        key = (case["language"], case["id"])
        if key in cases:
            raise ValueError(f"Duplicate dataset identity: {key}")
        cases[key] = case
    if not cases:
        raise ValueError("Dataset is empty")
    return cases
