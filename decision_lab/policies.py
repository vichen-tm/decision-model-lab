"""Finite policy catalogue. Inputs contain only model outputs, never gold."""
import itertools
import numpy as np

NAMES = ['jev', 'liquid', 'microsoft']


def grid(values):
    return np.unique(np.r_[0, np.quantile(values, np.linspace(0, 1, 7), method='nearest'), 1.000001])


def vote(pred, q=None, tie_primary=False):
    weight = np.ones_like(pred, dtype=float) if q is None else q
    scores = np.column_stack([(weight*(pred == c)).sum(axis=1) for c in range(3)])
    winner = scores.argmax(axis=1)
    ties = (np.abs(scores-scores.max(axis=1)[:, None]) < 1e-12).sum(axis=1) > 1
    winner[ties] = pred[ties, 0] if tie_primary else -1
    strength = scores.max(axis=1)/weight.sum(axis=1)
    return winner, strength


def catalogue(pred, prob, scores, train):
    n = len(pred)
    def emit(family, params, decision, calls):
        return family, params, decision, np.full(n, calls, dtype=np.uint8) if np.isscalar(calls) else calls

    for kind, q in scores.items():
        thresholds = [grid(q[train, m]) for m in range(3)]
        for m in range(3):
            # Single-model family also considers every tie-preserving threshold.
            for t in np.unique(np.r_[0, q[train, m], 1.000001]):
                decision = np.where(q[:, m] >= t, pred[:, m], -1)
                yield emit('single_'+kind, {'model': NAMES[m], 'threshold': float(t)}, decision, 1 << m)
        for length in (2, 3):
            for order in itertools.permutations(range(3), length):
                for ts in itertools.product(*(thresholds[m] for m in order)):
                    decision = np.full(n, -1)
                    calls = np.zeros(n, dtype=np.uint8)
                    remaining = np.ones(n, dtype=bool)
                    for m, t in zip(order, ts):
                        calls[remaining] |= 1 << m
                        accepted = remaining & (q[:, m] >= t)
                        decision[accepted] = pred[accepted, m]
                        remaining[accepted] = False
                    yield emit(f'cascade{length}_{kind}', {'order': [NAMES[m] for m in order], 'thresholds': list(map(float, ts))}, decision, calls)

        if kind == 'raw':
            continue
        for mode in ('weighted_vote', 'best_model'):
            if mode == 'weighted_vote':
                winner, strength = vote(pred, q)
            else:
                # Model index is the predetermined tie breaker: Jev, Liquid, Microsoft.
                best = q.argmax(axis=1)
                winner = pred[np.arange(n), best]
                strength = q.max(axis=1)
            for t in grid(strength[train]):
                yield emit(mode+'_'+kind, {'threshold': float(t)}, np.where(strength >= t, winner, -1), 7)

        for first in range(3):
            for second in range(3):
                if first == second:
                    continue
                for t, u in itertools.product(thresholds[first], thresholds[second]):
                    high = q[:, first] >= t
                    # A low-confidence second answer is only an error-warning signal.
                    # It does not replace the primary answer in this detect-only policy.
                    accept = high | (q[:, second] >= u)
                    calls = np.full(n, 1 << first, dtype=np.uint8)
                    calls[~high] |= 1 << second
                    yield emit('conditional_veto_'+kind,
                               {'primary': NAMES[first], 'verifier': NAMES[second], 'thresholds': [float(t), float(u)]},
                               np.where(accept, pred[:, first], -1), calls)

        majority, majority_strength = vote(pred)
        weighted, weighted_strength = vote(pred, q)
        for mode, winner, strength in [('majority', majority, majority_strength), ('weighted', weighted, weighted_strength)]:
            for first in range(3):
                for t, u in itertools.product(thresholds[first], grid(strength[train])):
                    high = q[:, first] >= t
                    decision = np.where(high, pred[:, first], np.where(strength >= u, winner, -1))
                    calls = np.full(n, 7, dtype=np.uint8)
                    calls[high] = 1 << first
                    yield emit('conditional_'+mode+'_'+kind,
                               {'primary': NAMES[first], 'threshold': float(t), 'vote_threshold': float(u)}, decision, calls)

    # Label agreement requires both calls regardless of the result.
    for a, b in itertools.combinations(range(3), 2):
        yield emit('agreement_pair', {'models': [NAMES[a], NAMES[b]]}, np.where(pred[:, a] == pred[:, b], pred[:, a], -1), (1 << a) | (1 << b))
    yield emit('unanimity', {}, np.where((pred == pred[:, :1]).all(axis=1), pred[:, 0], -1), 7)
    majority, _ = vote(pred)
    yield emit('majority', {'all_different': 'abstain'}, majority, 7)
    majority_primary, _ = vote(pred, tie_primary=True)
    yield emit('majority_primary_tie', {'all_different': 'jev'}, majority_primary, 7)
    pooled = prob.mean(axis=1)
    winner, strength = pooled.argmax(axis=1), pooled.max(axis=1)
    # A separate heuristic: normalized vectors are not calibrated correctness.
    for t in grid(strength[train]):
        yield emit('probability_pool', {'threshold': float(t)}, np.where(strength >= t, winner, -1), 7)
