"""Read and verify portable campaign shards; no provider or research-tree access."""

import base64
import gzip
import json
from pathlib import Path

from .encoding import encode, sha256

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_corpus(root=PACKAGE_ROOT):
    cases = {}
    for path in sorted((Path(root) / "data/corpus").glob("*/*.jsonl.gz")):
        for case in read_jsonl(path):
            key = (case["language"], case["id"])
            if key in cases:
                raise ValueError(f"Duplicate case: {key}")
            cases[key] = case
    return cases


def load_routes(root=PACKAGE_ROOT):
    return json.loads((Path(root) / "data/routes.json").read_text())


def responses(root=PACKAGE_ROOT):
    for path in sorted((Path(root) / "data/responses").glob("*.jsonl.gz")):
        yield from read_jsonl(path)


def response_bytes(raw):
    if raw["body_encoding"] == "base64":
        return base64.b64decode(raw["response_base64"], validate=True)
    if raw["body_encoding"] != "utf8":
        raise ValueError("Unknown body encoding")
    return raw["response_text"].encode("utf-8")


def verify_files(root=PACKAGE_ROOT):
    root = Path(root)
    manifest = json.loads((root / "data/manifest.json").read_text())
    expected = set(manifest["files"])
    actual = {
        str(path.relative_to(root))
        for path in (root / "data").rglob("*")
        if path.is_file() and path != root / "data/manifest.json"
    }
    if actual != expected:
        raise ValueError(
            f"Data inventory mismatch: missing={expected-actual}, extra={actual-expected}"
        )
    for name, info in manifest["files"].items():
        data = (root / name).read_bytes()
        if len(data) != info["bytes"] or sha256(data) != info["sha256"]:
            raise ValueError(f"Data checksum mismatch: {name}")
    return manifest


def verify_archive(root=PACKAGE_ROOT):
    """Validate byte hashes, gold-free requests, joins and the full Cartesian set."""
    root = Path(root)
    manifest = verify_files(root)
    cases = load_corpus(root)
    routes = {r["model"]: r for r in load_routes(root)}
    from .scoring import request_body, validate_case

    for case in cases.values():
        validate_case(case)
    requests = {}
    for path in sorted((root / "data/requests").glob("*/*.jsonl.gz")):
        for record in read_jsonl(path):
            body = record["body"]
            digest = sha256(encode(body))
            if digest != record["body_sha256"] or digest in requests:
                raise ValueError("Invalid or duplicate request hash")
            if set(body) != {"state", "questions", "model", "provider"}:
                raise ValueError("Unexpected request fields / possible gold leakage")
            requests[digest] = body
    expected = {
        (model, language, repeat, case_id)
        for model in routes
        for language, case_id in cases
        for repeat in (range(1, 6) if language == "en" else [1])
    }
    actual = set()
    decisions = 0
    accepted_tasks = {}
    for row in responses(root):
        key = row["key"]
        identity = (key["model"], key["language"], key["repeat"], key["case"])
        if identity in actual:
            raise ValueError("Duplicate accepted request")
        actual.add(identity)
        if row["logical_id"] in accepted_tasks:
            raise ValueError("Duplicate logical task")
        accepted_tasks[row["logical_id"]] = (row["key"], row["body_sha256"])
        case = cases[key["language"], key["case"]]
        decisions += len(case["request"]["questions"])
        body = request_body(case, routes[key["model"]])
        if encode(body) != encode(requests[row["body_sha256"]]):
            raise ValueError("Corpus/request mismatch")
        raw = row["raw"]
        if sha256(response_bytes(raw)) != raw["body_sha256"]:
            raise ValueError("Response bytes mismatch")
        if response_bytes(raw).decode("utf-8", "replace") != raw["response_text"]:
            raise ValueError("Response text mismatch")
    if actual != expected:
        raise ValueError(
            f"Incomplete campaign: missing={len(expected-actual)}, extra={len(actual-expected)}"
        )
    counts = {
        "accepted_requests": len(actual), "decisions": decisions,
        "cases": len(cases), "distinct_requests": len(requests),
    }
    for name, count in counts.items():
        if count != manifest[name]:
            raise ValueError("Manifest count mismatch: " + name)
    return {
        "accepted_requests": len(actual),
        "cases": len(cases),
        "distinct_requests": len(requests),
        "files": len(manifest["files"]),
    }
