"""Auditable metrics and finite-choice policies. No network access."""
import collections
import itertools
import json
import math
from pathlib import Path

def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Require nonempty rows with unique IDs')
    names = set(rows[0]['models'])
    if not names:
        raise ValueError('No models')
    for r in rows:
        if set(r['models']) != names or not isinstance(r['gold'], str):
            raise ValueError('Models/gold must align across rows')
        for v in r['models'].values():
            if not isinstance(v['pred'], str):
                raise ValueError('Prediction must be a label')
            for field in ('confidence', 'cost'):
                x = v.get(field)
                if x is not None and (isinstance(x, bool) or not isinstance(x, (int, float)) or (not math.isfinite(x)) or (x < 0)):
                    raise ValueError('Invalid ' + field)
            if v.get('confidence') is not None and v['confidence'] > 1:
                raise ValueError('Confidence exceeds one')
    return rows

def metrics(rows, decisions, calls, referral_cost=None):
    if len(rows) != len(decisions) or len(rows) != len(calls):
        raise ValueError('Length mismatch')
    accepted = sum((x is not None for x in decisions))
    correct = sum((p == r['gold'] for r, p in zip(rows, decisions) if p is not None))
    values = [r['models'][m].get('cost') for r, ms in zip(rows, calls) for m in set(ms)]
    unknown = sum((x is None for x in values))
    known = sum((x for x in values if x is not None))
    n = len(rows)
    p = correct / accepted if accepted else None
    z = 1.959963984540054
    lower = (p + z * z / (2 * accepted) - z * math.sqrt(p * (1 - p) / accepted + z * z / (4 * accepted ** 2))) / (1 + z * z / accepted) if accepted else None
    return dict(n=n, accepted=accepted, correct=correct, errors=accepted - correct, referred=n - accepted, accuracy=p, coverage=accepted / n, wilson_lower95=lower, calls=sum((len(set(x)) for x in calls)), known_cost_usd=known, unknown_cost_calls=unknown, cost_usd=None if unknown else known, total_with_referral_usd=known + (n - accepted) * referral_cost if not unknown and referral_cost is not None else None)

def apply(rows, policy):
    kind = policy['kind']
    order = policy.get('models', [])
    if not order or len(set(order)) != len(order):
        raise ValueError('Policy needs distinct ordered models')
    if any((m not in rows[0]['models'] for m in order)):
        raise ValueError('Unknown model')
    thresholds = policy.get('thresholds', [0] * len(order))
    if len(thresholds) != len(order) or any((not isinstance(t, (float, int)) or not math.isfinite(t) or t < 0 or (t > 1.000001) for t in thresholds)):
        raise ValueError('Invalid thresholds')
    decisions, calls = ([], [])
    for row in rows:
        used = []

        def qualifies(i):
            v = row['models'][order[i]]
            c = v.get('confidence')
            if c is None:
                raise ValueError('Policy requires observed confidence')
            return c >= thresholds[i]
        if kind == 'single':
            if len(order) != 1:
                raise ValueError('single takes one model')
            used = order[:]
            pred = row['models'][order[0]]['pred'] if qualifies(0) else None
        elif kind == 'cascade':
            pred = None
            for i, m in enumerate(order):
                used.append(m)
                if qualifies(i):
                    pred = row['models'][m]['pred']
                    break
        elif kind in ('majority', 'agreement', 'conditional-majority'):
            if kind == 'conditional-majority' and qualifies(0):
                used = order[:1]
                pred = row['models'][order[0]]['pred']
            else:
                used = order[:]
                votes = collections.Counter((row['models'][m]['pred'] for m in order))
                label, count = votes.most_common(1)[0]
                required = len(order) if kind == 'agreement' else len(order) // 2 + 1
                pred = label if count >= required else None
        elif kind == 'veto':
            if len(order) != 2:
                raise ValueError('veto takes primary and signal model')
            used = order[:1]
            good = qualifies(0)
            if not good:
                used.append(order[1])
                good = qualifies(1)
            pred = row['models'][order[0]]['pred'] if good else None
        else:
            raise ValueError('Unknown policy kind')
        decisions.append(pred)
        calls.append(used)
    return (decisions, calls)

def analyze(rows):
    names = sorted(rows[0]['models'])
    result = {'models': {}, 'error_histogram': dict.fromkeys(map(str, range(len(names) + 1)), 0), 'pairs': {}}
    for row in rows:
        k = sum((v['pred'] != row['gold'] for v in row['models'].values()))
        result['error_histogram'][str(k)] += 1
    for m in names:
        pred = [r['models'][m]['pred'] for r in rows]
        base = metrics(rows, pred, [[m] for r in rows])
        base['by_predicted_class'] = {}
        for label in sorted(set(pred)):
            subset = [r for r in rows if r['models'][m]['pred'] == label]
            base['by_predicted_class'][label] = metrics(subset, [label] * len(subset), [[m]] * len(subset))
        base['confidence_curve'] = []
        if all((r['models'][m].get('confidence') is not None for r in rows)):
            from decision_benchmark.confidence import choose_threshold
            point = choose_threshold([[r['models'][m]['confidence'], 1, int(r['models'][m]['pred'] == r['gold'])] for r in rows])
            base['descriptive_J_point'] = dict(point.record(), J=float(point.j), U=float(point.utility) if point.utility is not None else None)
            for t in sorted({0.0, 1.000001} | {r['models'][m]['confidence'] for r in rows}):
                d, c = apply(rows, dict(kind='single', models=[m], thresholds=[t]))
                base['confidence_curve'].append(dict(threshold=t, **metrics(rows, d, c)))
        result['models'][m] = base
    for a, b in itertools.combinations(names, 2):
        result['pairs'][a + ' / ' + b] = dict(both_wrong=sum((r['models'][a]['pred'] != r['gold'] and r['models'][b]['pred'] != r['gold'] for r in rows)), a_wrong_b_right=sum((r['models'][a]['pred'] != r['gold'] and r['models'][b]['pred'] == r['gold'] for r in rows)), b_wrong_a_right=sum((r['models'][b]['pred'] != r['gold'] and r['models'][a]['pred'] == r['gold'] for r in rows)))
    return result

def compare(before, after):
    a = {r['id']: r for r in before}
    b = {r['id']: r for r in after}
    if set(a) != set(b):
        raise ValueError('Comparison requires identical IDs')
    out = {}
    for m in a[next(iter(a))]['models']:
        fixed = []
        broken = []
        classes = {}
        for cid, r in a.items():
            if r['gold'] != b[cid]['gold']:
                raise ValueError('Gold changed')
            old = r['models'][m]['pred'] == r['gold']
            new = b[cid]['models'][m]['pred'] == r['gold']
            if new and (not old):
                fixed.append(cid)
            if old and (not new):
                broken.append(cid)
            z = classes.setdefault(r['gold'], {'fixed': 0, 'broken': 0})
            z['fixed'] += int(new and (not old))
            z['broken'] += int(old and (not new))
        out[m] = {'fixed': fixed, 'broken': broken, 'net': len(fixed) - len(broken), 'by_gold': classes}
    return out
