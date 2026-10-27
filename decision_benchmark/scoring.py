"""Versioned request/scoring contract. Pure offline functions; no API client."""

import copy, hashlib, json, math
from collections import Counter

VERSION = "1.2.0"
RESPONSE_SUM_TOLERANCE = 0.01
SUM_BOUNDARY_EPSILON = 1e-12
CATEGORIES = {
    "topic": ["ag_news"],
    "intent": ["banking77"],
    "sentiment": ["sst2", "sst5", "emotion"],
    "reading": ["boolq"],
    "inference": ["xnli_en"],
    "decisions": ["typed-decisions"],
    "proof": ["proofwriter"],
}
SUITE_COUNTS = {
    "ag_news": 200,
    "banking77": 200,
    "sst2": 200,
    "sst5": 200,
    "emotion": 200,
    "boolq": 200,
    "xnli_en": 150,
    "typed-decisions": 400,
    "proofwriter": 300,
}


def encode(value):
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def options(question):
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    if question["type"] == "choice":
        return list(question["criteria"])
    raise ValueError("unknown_question_type")


def finite_number(x):
    try:
        return type(x) in (int, float) and math.isfinite(x)
    except OverflowError:
        return False


def parse_object(text):

    def number(s):
        v = float(s)
        if not math.isfinite(v):
            raise ValueError("non_finite_number")
        return v

    def bad(s):
        raise ValueError("non_finite_number")

    try:
        value = json.loads(text, parse_float=number, parse_constant=bad)
    except RecursionError:
        raise ValueError("response_too_deep")
    if not isinstance(value, dict):
        raise ValueError("response_not_object")
    return value


def distribution(values, tolerance=0.0001):
    if not values or not all((finite_number(v) and 0 <= v <= 1 for v in values)):
        raise ValueError("invalid_probability")
    total = math.fsum(values)
    if total <= 0 or abs(total - 1) > tolerance + SUM_BOUNDARY_EPSILON:
        raise ValueError("probabilities_not_normalized")
    return [v / total for v in values]


def validate_case(case):
    qs = case["request"]["questions"]
    gold = case["gold"]
    if not qs or set(qs) != set(gold):
        raise ValueError("question_gold_mismatch")
    if set(case["request"]) != {"state", "questions"} or set(
        case["request"]["state"]
    ) != {"text"}:
        raise ValueError("input_schema")
    if not isinstance(case["request"]["state"]["text"], str):
        raise ValueError("state_not_text")
    for qid, q in qs.items():
        if not isinstance(q.get("instructions"), str) or not q["instructions"]:
            raise ValueError("missing_instructions")
        keys = options(q)
        g = gold[qid]
        if q["type"] == "score" and (not isinstance(q["criteria"], list)):
            raise ValueError("score_requires_ordered_list")
        if q["type"] == "choice" and (not isinstance(q["criteria"], dict)):
            raise ValueError("choice_requires_mapping")
        if len(keys) < 2 or len(keys) != len(set(keys)):
            raise ValueError("invalid_options")
        if g["type"] != q["type"]:
            raise ValueError("gold_type")
        if q["type"] == "choice":
            if g["label"] not in keys:
                raise ValueError("invalid_gold_label")
        else:
            if type(g["label"]) is not int or not 0 <= g["label"] < len(keys):
                raise ValueError("invalid_gold_label")
        if "dist" in g:
            if len(g["dist"]) != len(keys):
                raise ValueError("gold_distribution_length")
            distribution(g["dist"])
            if "keys" in g:
                if g["keys"] != keys:
                    raise ValueError("gold_distribution_keys")
    return True


def request_body(case, route):
    """Fresh copy: only model/provider differ between models; never include gold."""
    if (
        route["provider"].get("allow_fallbacks") is not False
        or len(route["provider"].get("only", [])) != 1
    ):
        raise ValueError("route_not_pinned")
    if set(case["request"]) != {"state", "questions"}:
        raise ValueError("request_keys")
    for q in case["request"]["questions"].values():
        if set(q) - {"type", "instructions", "criteria"}:
            raise ValueError("unexpected_question_fields")
    return copy.deepcopy(
        {**case["request"], "model": route["model"], "provider": route["provider"]}
    )


def score_answer(question, gold, answer):
    keys = options(question)
    kind = question["type"]
    result = {"type": kind, "valid": False, "correct": False, "error": None}
    try:
        if not isinstance(answer, dict):
            raise ValueError("missing_answer")
        if answer.get("type") != kind:
            raise ValueError("answer_type_mismatch")
        p = None
        if kind == "noul":
            yes = answer.get("noul")
            if not finite_number(yes) or not 0 <= yes <= 1:
                raise ValueError("invalid_noul")
            p = [1 - yes, yes]
            pred = int(yes >= 0.5)
        else:
            probs = answer.get("probabilities")
            if probs is not None:
                if not isinstance(probs, dict) or set(probs) != set(keys):
                    raise ValueError("probability_keys_mismatch")
                values = [probs[k] for k in keys]
                p = distribution(values, RESPONSE_SUM_TOLERANCE)
                total = math.fsum(values)
                result["normalization"] = {
                    "original_sum": total,
                    "absolute_deviation": abs(total - 1),
                    "tolerance": RESPONSE_SUM_TOLERANCE,
                    "applied": total != 1.0,
                    "exceeds_previous_tolerance": abs(total - 1)
                    > 0.0001 + SUM_BOUNDARY_EPSILON,
                    "raw_probabilities": values,
                }
            if kind == "choice":
                selected = answer.get("choice")
                if selected not in keys:
                    raise ValueError("invalid_choice")
                pred = keys.index(selected)
                if p is not None and p[pred] + 1e-08 < max(p):
                    result["choice_probability_conflict"] = {
                        "selected": selected,
                        "selected_probability": p[pred],
                        "argmax": [keys[i] for i, v in enumerate(p) if v == max(p)],
                        "max_probability": max(p),
                    }
            else:
                if p is None:
                    raise ValueError("score_probabilities_required")
                pred = max(range(len(p)), key=p.__getitem__)
                native = answer.get("score")
                if native is not None and (
                    not finite_number(native) or not 0 <= native <= len(keys) - 1
                ):
                    raise ValueError("invalid_native_score")
                legend = answer.get("legend")
                if legend is not None and legend != {
                    str(i): v for i, v in enumerate(question["criteria"])
                }:
                    raise ValueError("score_legend_mismatch")
        native_confidence = answer.get("confidence")
        if native_confidence is not None and (
            not finite_number(native_confidence) or not 0 <= native_confidence <= 1
        ):
            raise ValueError("invalid_native_confidence")
        target = keys.index(gold["label"]) if kind == "choice" else gold["label"]
        result.update(
            valid=True,
            correct=pred == target,
            prediction=keys[pred] if kind == "choice" else pred,
            gold=gold["label"],
            probabilities=p,
            confidence=p[pred] if p is not None else None,
            native_confidence=native_confidence,
            native_score=answer.get("score"),
        )
        if p is not None:
            result.update(
                nll=-math.log(max(p[target], 1e-12)),
                brier=sum(((v - int(i == target)) ** 2 for i, v in enumerate(p))),
            )
        if kind == "score":
            expected = sum((i * v for i, v in enumerate(p)))
            result.update(
                expected_index=expected,
                score_mae=abs(expected - target),
                score_class_mae=abs(pred - target),
            )
        if "dist" in gold:
            gd = distribution(gold["dist"])
            result["soft_gold_at_prediction"] = gd[pred]
            result["soft_kl"] = (
                sum(
                    (
                        v * math.log(v / max(p[i], 1e-12))
                        for i, v in enumerate(gd)
                        if v > 0
                    )
                )
                if p is not None
                else None
            )
    except (ValueError, TypeError, KeyError) as e:
        result.update(valid=False, correct=False, error=str(e))
    return result


def score_response(case, raw, route):
    """HTTP failures, route/version changes and incomplete answers remain explicit."""
    error = None
    data = {}
    if raw is None:
        error = "missing_response"
    elif raw.get("unknown"):
        error = "unknown_outcome"
    elif raw.get("status") != 200:
        error = "http_" + str(raw.get("status"))
    else:
        try:
            data = parse_object(raw["response_text"])
            if not isinstance(data, dict):
                raise ValueError("response_not_object")
            if data.get("provider") != route["expected_provider"]:
                error = "provider_mismatch"
            elif data.get("model") != route["expected_version"]:
                error = "model_version_mismatch"
            elif not isinstance(data.get("answers"), dict):
                error = "answers_not_object"
            elif set(data["answers"]) - set(case["request"]["questions"]):
                error = "unexpected_question_id"
        except (ValueError, KeyError, TypeError) as e:
            error = "parse_error:" + str(e)
    results = {}
    for qid, q in case["request"]["questions"].items():
        results[qid] = (
            {"type": q["type"], "valid": False, "correct": False, "error": error}
            if error
            else score_answer(q, case["gold"][qid], data["answers"].get(qid))
        )
    return {
        "id": case["id"],
        "suite": case["suite"],
        "results": results,
        "complete": all((r["valid"] for r in results.values())),
        "usage": data.get("usage"),
        "elapsed_ms": None if raw is None else raw.get("elapsed_ms"),
        "error": error,
    }
