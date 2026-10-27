"""Exact Jaccard threshold selection and practical confidence utility."""

from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float | None
    retained: int
    correct: int
    total: int
    total_correct: int

    @property
    def errors(self):
        return self.retained - self.correct

    @property
    def j(self):
        denominator = self.total_correct + self.errors
        return Fraction(self.correct, denominator) if denominator else Fraction(1)

    @property
    def utility(self):
        total_errors = self.total - self.total_correct
        if not self.total_correct or not total_errors:
            return None
        return Fraction(self.correct, self.total_correct) - Fraction(
            self.errors, total_errors
        )

    def record(self):
        return {
            "threshold": self.threshold,
            "n": self.retained,
            "correct": self.correct,
            "errors": self.errors,
            "total": self.total,
            "baseline_accuracy": self.total_correct / self.total,
            "accuracy": self.correct / self.retained if self.retained else None,
            "coverage": self.retained / self.total,
            "lost_correct": self.total_correct - self.correct,
            "removed_errors": self.total - self.total_correct - self.errors,
        }


def enumerate_points(levels):
    """Each level is (native confidence, count, correct); never split a tie."""
    merged = {}
    for threshold, count, correct in levels:
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("Native confidence must be numeric and within [0, 1]")
        if (
            type(count) is not int
            or type(correct) is not int
            or count <= 0
            or not 0 <= correct <= count
        ):
            raise ValueError("Invalid level counts")
        old_count, old_correct = merged.get(threshold, (0, 0))
        merged[threshold] = (old_count + count, old_correct + correct)
    total = sum(n for n, _ in merged.values())
    total_correct = sum(g for _, g in merged.values())
    if not total:
        raise ValueError("Empty sample")
    points = [OperatingPoint(None, 0, 0, total, total_correct)]
    retained = correct = 0
    for threshold, (count, good) in sorted(merged.items(), reverse=True):
        retained += count
        correct += good
        points.append(
            OperatingPoint(
                threshold if retained < total else 0,
                retained,
                correct,
                total,
                total_correct,
            )
        )
    return points


def choose_threshold(levels):
    return max(enumerate_points(levels), key=lambda p: (p.j, -p.errors, p.correct))
