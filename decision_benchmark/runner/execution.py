"""Pinned plans, shared provider gates, explicit retries and safe resume."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
from datetime import datetime, timezone
import threading
import time
from ..archive import load_corpus, load_routes
from ..encoding import encode, sha256
from ..scoring import request_body, score_response, validate_case
from .journal import RunJournal

RETRYABLE_HTTP = {429, 500, 502, 503}


def prepare(
    root,
    directory,
    models,
    languages,
    suites,
    repeats,
    concurrency,
    routes=None,
    cases=None,
    repeat_by_language=None,
):
    if repeats < 1 or concurrency < 1:
        raise ValueError("Repeats and concurrency must be positive")
    corpus = load_corpus(root) if cases is None else cases
    routes = routes or load_routes(root)
    selected_routes = [route for route in routes if route["model"] in models]
    if {r["model"] for r in selected_routes} != set(models) or len(
        selected_routes
    ) != len(models):
        raise ValueError("Unknown or duplicate model selection")
    if set(languages) - {language for language, _ in corpus}:
        raise ValueError("Unknown language")
    if suites and set(suites) - {case["suite"] for case in corpus.values()}:
        raise ValueError("Unknown suite")
    tasks = []
    for route in selected_routes:
        if not route.get("expected_version") or not route.get("expected_provider"):
            raise ValueError("Route needs pinned response model and provider")
        for (language, case_id), case in sorted(corpus.items()):
            if language not in languages or suites and case["suite"] not in suites:
                continue
            count = (repeat_by_language or {}).get(language, repeats)
            if type(count) is not int or count < 1:
                raise ValueError("Repeat count must be a positive integer")
            for repeat in range(1, count + 1):
                key = {
                    "model": route["model"],
                    "language": language,
                    "case": case_id,
                    "repeat": repeat,
                }
                tasks.append(
                    {
                        "id": sha256(encode(key)),
                        "key": key,
                        "case": case,
                        "route": route,
                        "body": request_body(case, route),
                    }
                )
    if not tasks:
        raise ValueError("Selection contains no tasks")
    providers = {r["provider"]["only"][0]: concurrency for r in selected_routes}
    plan = {
        "schema_version": 1,
        "tasks": tasks,
        "provider_limits": providers,
        "timeout": None,
        "retries": "Explicit retry-technical or resolve-unknown/resolve-failed only",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_plan(plan)
    RunJournal(directory).create_plan(plan)
    return {"prepared_requests": len(tasks), "provider_limits": providers, "sent": 0}


def task_state(journal, task):
    attempts = journal.attempts(task)
    if not attempts:
        return "pending"
    accepted = False
    for directory in attempts:
        intent = journal.read_record(directory / "intent.json")
        if (
            intent["body_sha256"] != sha256(encode(task["body"]))
            or intent["task_id"] != task["id"]
        ):
            raise ValueError("Attempt does not match locked plan")
        response = directory / "response.json"
        if not response.exists():
            if not (directory / "resolution.json").exists():
                return "unknown"
            journal.read_record(directory / "resolution.json")
            continue
        raw = journal.read_record(response)
        if score_response(task["case"], raw, task["route"])["complete"]:
            accepted = True
    if accepted:
        return "accepted"
    last = attempts[-1]
    if (last / "resolution.json").exists():
        resolution = journal.read_record(last / "resolution.json")
        return "resolved_retry" if resolution["action"] == "retry" else "abandoned"
    if not (last / "response.json").exists():
        return "unknown"
    raw = journal.read_record(last / "response.json")
    if raw.get("unknown") or raw.get("status") in {408, 504, 520, 522, 524}:
        return "unknown"
    return "retryable" if raw.get("status") in RETRYABLE_HTTP else "failed"


def validate_plan(plan):
    if plan.get("schema_version") != 1 or not plan.get("tasks"):
        raise ValueError("Invalid plan schema or empty task list")
    identifiers = set()
    providers = set()
    for task in plan["tasks"]:
        if task["id"] != sha256(encode(task["key"])) or task["id"] in identifiers:
            raise ValueError("Invalid or duplicate task identifier")
        identifiers.add(task["id"])
        if not task["route"].get("expected_version") or not task["route"].get(
            "expected_provider"
        ):
            raise ValueError("Route needs pinned response model and provider")
        validate_case(task["case"])
        if encode(task["body"]) != encode(request_body(task["case"], task["route"])):
            raise ValueError("Plan body differs from gold-free request contract")
        if (
            task["key"]["model"] != task["route"]["model"]
            or task["key"]["case"] != task["case"]["id"]
            or task["key"]["language"] != task["case"]["language"]
        ):
            raise ValueError("Task identity does not match case/route")
        providers.add(task["route"]["provider"]["only"][0])
    if providers != set(plan["provider_limits"]) or any(
        type(n) is not int or n < 1 for n in plan["provider_limits"].values()
    ):
        raise ValueError("Invalid provider limits")


def run(directory, transport, retry_technical=False):
    journal = RunJournal(directory)
    with journal.lock():
        plan = journal.load_plan()
        validate_plan(plan)
        states = {task["id"]: task_state(journal, task) for task in plan["tasks"]}
        if "unknown" in states.values():
            raise ValueError(
                "Unknown attempt: inspect and resolve explicitly before continuing"
            )
        pending = [
            task
            for task in plan["tasks"]
            if states[task["id"]] in ("pending", "resolved_retry")
            or retry_technical
            and states[task["id"]] == "retryable"
        ]
        stop = threading.Event()
        executors = {
            provider: ThreadPoolExecutor(max_workers=limit)
            for provider, limit in plan["provider_limits"].items()
        }
        main_thread = threading.current_thread() is threading.main_thread()
        previous_handler = signal.getsignal(signal.SIGINT) if main_thread else None
        if main_thread:
            # Repeated interrupts must not release the run lock while writers live.
            signal.signal(signal.SIGINT, lambda signum, frame: stop.set())

        def execute(task, queued):
            if stop.is_set():
                return
            try:
                provider = task["route"]["provider"]["only"][0]
                attempt = journal.start_attempt(
                    task,
                    {
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "client_queue_to_intent_ms": (time.monotonic() - queued) * 1000,
                        "provider_concurrency_limit": plan["provider_limits"][provider],
                        "client_correlation_id": task["id"],
                    },
                )
                raw = transport.send(task["body"])
                journal.record(attempt / "response.json", raw)
                if not score_response(task["case"], raw, task["route"])["complete"]:
                    stop.set()
            except BaseException:
                stop.set()
                raise

        futures = []
        try:
            for task in pending:
                if stop.is_set():
                    break
                provider = task["route"]["provider"]["only"][0]
                futures.append(
                    executors[provider].submit(execute, task, time.monotonic())
                )
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            try:
                for executor in executors.values():
                    executor.shutdown(wait=True, cancel_futures=True)
            finally:
                if main_thread:
                    signal.signal(signal.SIGINT, previous_handler)
        counts = {}
        for task in plan["tasks"]:
            state = task_state(journal, task)
            counts[state] = counts.get(state, 0) + 1
        return {
            "status": (
                "complete"
                if counts.get("accepted", 0) == len(plan["tasks"])
                else "incomplete"
            ),
            "tasks": len(plan["tasks"]),
            "states": counts,
        }


def resolve_attempt(directory, task_id, action, required_state="unknown"):
    if action not in ("retry", "abandon"):
        raise ValueError("Resolution must be retry or abandon")
    journal = RunJournal(directory)
    with journal.lock():
        plan = journal.load_plan()
        task = next((t for t in plan["tasks"] if t["id"] == task_id), None)
        if task is None or task_state(journal, task) != required_state:
            raise ValueError("Task does not have the required unresolved state")
        journal.record(
            journal.attempts(task)[-1] / "resolution.json",
            {
                "action": action,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "duplicate_billing_possible": action == "retry",
            },
        )
    return {"task_id": task_id, "resolution": action}


def resolve_unknown(directory, task_id, action):
    return resolve_attempt(directory, task_id, action)
