import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from decision_benchmark.encoding import encode, sha256
from decision_benchmark.runner.execution import (
    run,
    resolve_unknown,
    resolve_attempt,
    task_state,
)
from decision_benchmark.runner.journal import RunJournal
from decision_benchmark.runner.report import report_run
from decision_benchmark.runner.transport import OpenRouterTransport


def task(index=0, model="m"):
    case = {
        "id": str(index),
        "suite": "test",
        "language": "en",
        "request": {
            "state": {"text": "Example"},
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": "A", "b": "B"},
                }
            },
        },
        "gold": {"q": {"type": "choice", "label": "a"}},
    }
    route = {
        "model": model,
        "provider": {"only": ["shared"], "allow_fallbacks": False},
        "expected_provider": "P",
        "expected_version": model + "-v1",
    }
    from decision_benchmark.scoring import request_body

    return {
        "id": sha256(
            encode({"model": model, "language": "en", "case": str(index), "repeat": 1})
        ),
        "key": {"model": model, "language": "en", "case": str(index), "repeat": 1},
        "case": case,
        "route": route,
        "body": request_body(case, route),
    }


class FakeTransport:
    def __init__(self, statuses=None):
        self.statuses = list(statuses or [])
        self.calls = 0
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def send(self, body):
        with self.lock:
            self.calls += 1
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            status = self.statuses.pop(0) if self.statuses else 200
        time.sleep(0.002)
        with self.lock:
            self.active -= 1
        if status == "crash":
            raise RuntimeError("simulated interruption after durable intent")
        if status != 200:
            return {"status": status, "response_text": "", "unknown": status is None}
        response = {
            "model": body["model"] + "-v1",
            "provider": "P",
            "answers": {"q": {"type": "choice", "choice": "a", "confidence": 0.8}},
            "usage": {"cost": 0.001},
        }
        return {"status": 200, "response_text": json.dumps(response), "elapsed_ms": 1}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.journal = RunJournal(Path(self.temp.name) / "run")

    def create(self, tasks=None, concurrency=1):
        self.tasks = tasks or [task()]
        self.journal.create_plan(
            {
                "schema_version": 1,
                "tasks": self.tasks,
                "provider_limits": {"shared": concurrency},
            }
        )

    def test_success_and_resume_do_not_resend(self):
        self.create()
        transport = FakeTransport()
        self.assertEqual(run(self.journal.directory, transport)["status"], "complete")
        run(self.journal.directory, transport)
        self.assertEqual(transport.calls, 1)
        self.assertTrue(report_run(self.journal.directory)["complete"])

    def test_crash_after_intent_requires_resolution(self):
        self.create()
        transport = FakeTransport(["crash"])
        with self.assertRaises(RuntimeError):
            run(self.journal.directory, transport)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            run(self.journal.directory, transport)
        self.assertEqual(transport.calls, 1)
        resolve_unknown(self.journal.directory, self.tasks[0]["id"], "retry")
        run(self.journal.directory, transport)
        self.assertEqual(transport.calls, 2)
        self.assertEqual(len(self.journal.attempts(self.tasks[0])), 2)

    def test_http_retry_is_explicit_and_appended(self):
        self.create()
        transport = FakeTransport([503, 200])
        self.assertEqual(run(self.journal.directory, transport)["status"], "incomplete")
        run(self.journal.directory, transport)
        self.assertEqual(transport.calls, 1)
        self.assertEqual(
            run(self.journal.directory, transport, True)["status"], "complete"
        )
        self.assertEqual(len(self.journal.attempts(self.tasks[0])), 2)

    def test_unknown_504_not_automatic_retry(self):
        self.create()
        transport = FakeTransport([504])
        run(self.journal.directory, transport)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            run(self.journal.directory, transport, True)
        resolve_unknown(self.journal.directory, self.tasks[0]["id"], "abandon")
        run(self.journal.directory, transport, True)
        self.assertEqual(transport.calls, 1)

    def test_shared_provider_limit_across_models(self):
        self.create([task(i, "a" if i % 2 else "b") for i in range(10)])
        transport = FakeTransport()
        run(self.journal.directory, transport)
        self.assertEqual((transport.calls, transport.maximum), (10, 1))

    def test_corrupt_plan_and_attempt_rejected(self):
        self.create()
        path = self.journal.directory / "plan.json"
        envelope = json.loads(path.read_text())
        envelope["data"]["provider_limits"]["shared"] = 3
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, "Changed"):
            run(self.journal.directory, FakeTransport())

    def test_two_owners_and_overwrite_rejected(self):
        self.create()
        with self.journal.lock():
            with self.assertRaises(RuntimeError):
                with RunJournal(self.journal.directory).lock():
                    pass
        with self.assertRaises(ValueError):
            self.journal.create_plan({})

    def test_no_gold_in_body(self):
        self.assertEqual(
            set(task()["body"]), {"model", "provider", "state", "questions"}
        )

    def test_transport_no_deadline_and_secret_redaction(self):
        secret = "private-test-credential"

        class Response:
            status = 500

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return ("echo " + secret).encode()

        class Opener:
            def open(self, request, timeout):
                self_timeout = timeout
                if self_timeout is not None:
                    raise AssertionError("Unexpected deadline")
                return Response()

        with patch("urllib.request.build_opener", return_value=Opener()):
            result = OpenRouterTransport(secret).send(task()["body"])
        self.assertNotIn(secret, json.dumps(result))
        self.assertTrue(result["redacted"])

    def test_missing_plan_does_not_create_directory(self):
        target = Path(self.temp.name) / "missing"
        with self.assertRaises(ValueError):
            report_run(target)
        self.assertFalse(target.exists())

    def test_previous_attempts_cannot_inherit_new_plan(self):
        self.create()
        run(self.journal.directory, FakeTransport())
        (self.journal.directory / "plan.json").unlink()
        with self.assertRaises(ValueError):
            self.create()

    def test_failed_authorization_has_explicit_recovery(self):
        self.create()
        transport = FakeTransport([401, 200])
        run(self.journal.directory, transport)
        run(self.journal.directory, transport, True)
        self.assertEqual(transport.calls, 1)
        resolve_attempt(self.journal.directory, self.tasks[0]["id"], "retry", "failed")
        self.assertEqual(run(self.journal.directory, transport)["status"], "complete")
        self.assertEqual(transport.calls, 2)

    def test_worker_exception_stops_queued_sends(self):
        self.create([task(i) for i in range(20)])
        transport = FakeTransport(["crash"])
        with self.assertRaises(RuntimeError):
            run(self.journal.directory, transport)
        self.assertEqual(transport.calls, 1)

    def test_rehashed_malformed_plan_rejected_before_send(self):
        from decision_benchmark.encoding import write_json

        self.create()
        plan = self.journal.load_plan()
        plan["tasks"][0]["body"]["gold"] = {"answer": "a"}
        write_json(
            self.journal.directory / "plan.json",
            {"data": plan, "sha256": sha256(encode(plan))},
        )
        transport = FakeTransport()
        with self.assertRaises(ValueError):
            run(self.journal.directory, transport)
        self.assertEqual(transport.calls, 0)

    def test_repeated_interrupt_keeps_lock_until_active_send_finishes(self):
        import signal

        self.create([task(i) for i in range(3)])
        transport = FakeTransport()
        original_send = transport.send

        def interrupted_send(body):
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)
            handler(signal.SIGINT, None)
            with self.assertRaises(RuntimeError):
                with RunJournal(self.journal.directory).lock():
                    pass
            return original_send(body)

        transport.send = interrupted_send
        previous = signal.getsignal(signal.SIGINT)
        self.assertEqual(run(self.journal.directory, transport)["status"], "incomplete")
        self.assertEqual(transport.calls, 1)
        self.assertIs(signal.getsignal(signal.SIGINT), previous)
