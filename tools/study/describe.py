"""Descriptive inventory of saved decisions; no fitting or model calls."""
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import decision_benchmark
PACKAGE = Path(decision_benchmark.__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PACKAGE))
from decision_benchmark.confidence import choose_threshold, enumerate_points

LABELS = ['true', 'false', 'unknown']
SELECTED = {'jev': 'candidate', 'liquid': 'candidate', 'microsoft': 'baseline'}


def wilson(g, n):
    if not n:
        return None
    z = 1.959963984540054
    p = g / n
    return (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)


def group_record(mask, correct, cost):
    n = int(mask.sum())
    g = int(correct[mask].sum())
    return {'n': n, 'correct': g, 'errors': n-g, 'accuracy': g/n if n else None,
            'cost_usd': float(cost[mask].sum()), 'wilson95_lower': wilson(g, n)}


def point_record(point):
    return {**point.record(), 'J': float(point.j), 'U': float(point.utility) if point.utility is not None else None,
            'wilson95_lower': wilson(point.correct, point.retained), 'risk': point.errors/point.retained if point.retained else None}


def auc_correctness(conf, correct):
    pos = conf[correct]
    neg = conf[~correct]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg).sum() + 0.5*(pos[:, None] == neg).sum()) / (len(pos)*len(neg)))


def calibration_bins(conf, correct, edges):
    # Identical confidence values always map to the same bin. Endpoints 0 and 1 included.
    assignments = np.searchsorted(edges[1:-1], conf, side='right')
    out = []
    for i in range(len(edges)-1):
        mask = assignments == i
        n = int(mask.sum())
        out.append({'lo': float(edges[i]), 'hi': float(edges[i+1]), 'upper_inclusive': i == len(edges)-2,
                    'n': n, 'correct': int(correct[mask].sum()),
                    'mean_confidence': float(conf[mask].mean()) if n else None,
                    'accuracy': float(correct[mask].mean()) if n else None})
    assert sum(b['n'] for b in out) == len(conf)
    return out


def main():
    source = HERE / 'data.json'
    data = json.loads(source.read_text())
    n = len(data['ids'])
    assert n == len(set(data['ids'])) == 1000
    gold = np.array(data['gold'])
    depth = np.array(data['depth'], dtype=object)
    result = {'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'confidence_py_sha256': hashlib.sha256((PACKAGE / 'decision_benchmark/confidence.py').read_bytes()).hexdigest(),
              'n': n, 'selected': SELECTED,
              'method': {'scope': 'Descriptive in-sample inventory; thresholds are evaluated on the same saved observations used to choose them.',
                         'probability_order': LABELS, 'confidence': 'native_confidence',
                         'wilson': 'Two-sided 95% Wilson interval lower endpoint, z=1.959963984540054; pointwise, without threshold-selection adjustment.',
                         'targets': 'Maximum coverage among complete native-confidence tie blocks; no minimum-support restriction; empty when infeasible.',
                         'brier': 'binary_native_brier compares native confidence with correctness; multiclass_brier sums three squared probability errors.',
                         'quantiles': 'Empirical decile edges, deduplicated; bins left-closed/right-open, final includes upper endpoint; ties stay intact.',
                         'pair_rescue_damage': 'For switching all predictions from A to B: rescue=A wrong/B right; damage=A right/B wrong.',
                         'oracle': 'At least one selected model correct; uses gold and is an upper bound, not an executable selection method.'}, 'models': {}}
    csv_rows = []
    for model, variants in data['models'].items():
        result['models'][model] = {}
        for variant, a in variants.items():
            correct = np.array(a['correct'], dtype=bool)
            conf = np.array(a['confidence'], dtype=float)
            pred = np.array(a['pred'])
            cost = np.array(a['cost'])
            probs = np.array(a['probabilities'])
            assert np.array_equal(correct, pred == gold)
            levels = [(float(c), 1, int(g)) for c, g in zip(conf, correct)]
            points = enumerate_points(levels)
            point_records = [point_record(p) for p in points]
            opt = choose_threshold(levels)
            assert opt == max(points, key=lambda p: (p.j, -p.errors, p.correct))
            assert points[-1].retained == n and points[-1].correct == int(correct.sum())
            for point in points:
                retained_mask = np.zeros(n, dtype=bool) if point.threshold is None else conf >= point.threshold
                assert int(retained_mask.sum()) == point.retained
                assert int(correct[retained_mask].sum()) == point.correct
            equal = calibration_bins(conf, correct, np.linspace(0, 1, 11))
            edges = np.unique(np.quantile(conf, np.linspace(0, 1, 11)))
            if len(edges) == 1:
                edges = np.array([edges[0], edges[0]])
            quantiles = calibration_bins(conf, correct, edges)
            targets = {}
            for target in [0.98, 0.99, 1.0]:
                emp = [p for p in points if p.retained and p.correct/p.retained >= target]
                conservative = [p for p in points if p.retained and wilson(p.correct, p.retained) >= target]
                targets[str(target)] = {'empirical': point_record(max(emp, key=lambda p: p.retained)) if emp else None,
                                        'wilson_lower_constrained': point_record(max(conservative, key=lambda p: p.retained)) if conservative else None}
            record = {**group_record(np.ones(n, dtype=bool), correct, cost),
                      'mean_cost_usd': float(cost.mean()), 'cost_usd_per_1000': float(cost.mean()*1000),
                      'mean_elapsed_ms': float(np.mean(a['elapsed_ms'])),
                      'auroc_correctness_native': auc_correctness(conf, correct),
                      'binary_native_brier': float(np.mean((conf-correct)**2)),
                      'multiclass_brier': float(np.mean(np.sum((probs - (gold[:, None] == np.array(LABELS)))**2, axis=1))),
                      'ece10_equal_width': sum(b['n']/n*abs(b['mean_confidence']-b['accuracy']) for b in equal if b['n']),
                      'equal_width_bins': equal, 'native_confidence_quantile_bins': quantiles,
                      'J_optimum': point_record(opt), 'targets': targets,
                      'by_gold': {label: group_record(gold == label, correct, cost) for label in LABELS},
                      'by_depth': {str(d): group_record(depth == d, correct, cost) for d in [None, 0, 1, 2, 3, 4, 5]},
                      'predicted_category_reliability': {label: group_record(pred == label, correct, cost) for label in LABELS}}
            if SELECTED[model] == variant:
                record['risk_coverage'] = point_records
            for key in ['by_gold', 'by_depth', 'predicted_category_reliability']:
                assert sum(v['n'] for v in record[key].values()) == n
                assert sum(v['correct'] for v in record[key].values()) == int(correct.sum())
            result['models'][model][variant] = record
            csv_rows.append({'model': model, 'variant': variant, **{k: record[k] for k in ['n', 'correct', 'errors', 'accuracy', 'cost_usd', 'cost_usd_per_1000', 'mean_elapsed_ms', 'auroc_correctness_native', 'binary_native_brier', 'multiclass_brier', 'ece10_equal_width']},
                             'J': record['J_optimum']['J'], 'U_at_J': record['J_optimum']['U'],
                             **{f'empirical_{round(t*100)}_coverage': targets[str(t)]['empirical']['coverage'] if targets[str(t)]['empirical'] else 0 for t in [0.98, 0.99, 1.0]}})
    selected_correct = {m: np.array(data['models'][m][v]['correct'], dtype=bool) for m, v in SELECTED.items()}
    stack = np.stack(list(selected_correct.values()), axis=1)
    counts = stack.sum(axis=1)
    histogram = {str(k): int((counts == k).sum()) for k in range(4)}
    assert sum(histogram.values()) == n
    pair = {}
    for a, ac in selected_correct.items():
        for b, bc in selected_correct.items():
            if a == b:
                continue
            rescue, damage = int((~ac & bc).sum()), int((ac & ~bc).sum())
            assert rescue-damage == int(bc.sum())-int(ac.sum())
            pair[f'{a}->{b}'] = {'rescue': rescue, 'damage': damage, 'net_correct': rescue-damage,
                                 'both_wrong': int((~ac & ~bc).sum()), 'both_correct': int((ac & bc).sum()),
                                 'rescue_fraction_of_A_errors': rescue / int((~ac).sum()),
                                 'damage_fraction_of_A_correct': damage / int(ac.sum())}
    result['selected_overlap'] = {'correct_count_histogram': histogram,
                                 'error_count_histogram': {str(k): histogram[str(3-k)] for k in range(4)},
                                 'all_three_correct': histogram['3'], 'at_least_two_correct': histogram['2']+histogram['3'],
                                 'unique_correct': {m: int((c & (counts == 1)).sum()) for m, c in selected_correct.items()},
                                 'pair_rescue_damage': pair,
                                 'oracle_correct': int((counts >= 1).sum()), 'oracle_accuracy': float((counts >= 1).mean()),
                                 'all_three_wrong_ids': [cid for cid, count in zip(data['ids'], counts) if count == 0]}
    assert result['selected_overlap']['oracle_correct'] == n-histogram['0']
    (HERE / 'descriptive.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (HERE / 'descriptive.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({'models': csv_rows, 'overlap': result['selected_overlap']}, indent=2))


if __name__ == '__main__':
    main()
