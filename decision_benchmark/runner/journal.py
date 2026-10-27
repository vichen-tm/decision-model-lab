"""One run owns its immutable plan and append-only attempt directories."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from ..encoding import encode, sha256, write_json


class RunJournal:
    def __init__(self, directory):
        self.directory = Path(directory)

    @contextmanager
    def lock(self, creating=False):
        if not creating and not (self.directory / "plan.json").is_file():
            raise ValueError("No prepared run at this path")
        with (self.directory / ".lock").open("a+") as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process owns this run") from error
            file.seek(0)
            file.truncate()
            file.write(str(os.getpid()))
            file.flush()
            try:
                yield
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    def create_plan(self, plan):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.lock(creating=True):
            if any(p.name != ".lock" for p in self.directory.iterdir()):
                raise ValueError("Run directory must be empty")
            path = self.directory / "plan.json"
            if path.exists():
                raise ValueError("Run already exists; prepare a new directory")
            write_json(path, {"sha256": sha256(encode(plan)), "data": plan})

    def load_plan(self):
        return self.read_record(self.directory / "plan.json")

    def attempts(self, task):
        return sorted(
            p
            for p in (self.directory / "attempts" / task["id"]).glob(
                "[0-9][0-9][0-9][0-9]"
            )
            if (p / "intent.json").exists() or (p / "response.json").exists()
        )

    def read_record(self, path):
        envelope = json.loads(path.read_text())
        if sha256(encode(envelope["data"])) != envelope["sha256"]:
            raise ValueError("Changed immutable record")
        return envelope["data"]

    def record(self, path, data):
        if path.exists():
            raise ValueError("Attempt records cannot be overwritten")
        write_json(path, {"sha256": sha256(encode(data)), "data": data})

    def start_attempt(self, task, metadata):
        path = (
            self.directory
            / "attempts"
            / task["id"]
            / f"{len(self.attempts(task))+1:04d}"
        )
        self.record(
            path / "intent.json",
            {
                "task_id": task["id"],
                "body_sha256": sha256(encode(task["body"])),
                **metadata,
            },
        )
        return path
