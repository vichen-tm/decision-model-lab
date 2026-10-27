"""The historical JSON encoding contract and durable file publication."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def encode(value):
    """Preserve insertion order and emit UTF-8 with no trailing newline."""
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value):
    """Atomically publish a durable JSON document on a local POSIX filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encode(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
