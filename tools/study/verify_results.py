"""Independent saved-artifact audit, stdlib only; no provider or model fitting.
Run after analyze.py completes. Raises on any mismatch; prints an audit summary.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ['jev', 'liquid', 'microsoft']
checks = 0


def eq(actual, expected, where):
    global checks
    checks += 1
    if isinstance(expected, float):
        assert actual is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (where, actual, expected)
    else:
        assert actual == expected, (where, actual, expected)


def compare(actual, expected, where):
    for key, value in expected.items():
        eq(actual[key], value, where + '/' + key)


def metrics(indices, decisions, data, models):
    n = len(indices)
    accepted = good = 0
    cost = elapsed = 0.
    calls = Counter({m: 0 for m in NAMES})
    for i in indices:
        d = decisions[i]
        mask = d['calls_mask']
        assert isinstance(mask, int) and 0 <= mask <= 7
        p = d['prediction']
        assert p in (None, 'true', 'false', 'unknown')
        accepted += p is not None
        good += p is not None and p == data['gold'][i]
        if p is not None:
            assert any(mask & (1 << j) and models[j]['pred'][i] == p for j in range(3)) or mask == 7
        for j, m in enumerate(NAMES):
            if mask & (1 << j):
                calls[m] += 1
                cost += models[j]['cost'][i]
                elapsed += models[j]['elapsed_ms'][i]
    lower = None
    if accepted:
        z = 1.959963984540054
        p = good / accepted
        lower = (p + z*z/(2*accepted) - z*math.sqrt(p*(1-p)/accepted + z*z/(4*accepted**2)))/(1+z*z/accepted)
    return dict(n=n, accepted=accepted, correct=good, errors=accepted-good,
                referred=n-accepted, accuracy=good/accepted if accepted else None,
                coverage=accepted/n, correct_automatic_share=good/n,
                wilson_lower95=lower, cost_usd=cost, cost_per1000_inputs=1000*cost/n,
                cost_per_correct_auto=cost/good if good else None,
                calls=sum(calls.values()), calls_per_model=dict(calls), mean_serial_sum_ms=elapsed/n)


def confusion(indices, rejected, primary_correct):
    tp = sum(rejected[i] and not primary_correct[i] for i in indices)
    fp = sum(rejected[i] and primary_correct[i] for i in indices)
    fn = sum(not rejected[i] and not primary_correct[i] for i in indices)
    tn = sum(not rejected[i] and primary_correct[i] for i in indices)
    return dict(tp_errors_referred=tp, fp_correct_referred=fp, fn_errors_retained=fn,
                tn_correct_retained=tn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None)


def main():
    data_bytes = (ROOT/'data.json').read_bytes()
    data = json.loads(data_bytes)
    result = json.loads((ROOT/'results.json').read_text())
    records = [json.loads(line) for line in (ROOT/'oof-decisions.jsonl').read_text().splitlines()]
    eq(result['input_sha256'], hashlib.sha256(data_bytes).hexdigest(), 'input hash')
    eq([r['id'] for r in records], data['ids'], 'ordered IDs')
    n = len(records)
    eq(len(set(data['ids'])), n, 'unique IDs')
    eq([r['gold'] for r in records], data['gold'], 'gold')
    models = [data['models'][m][v] for m, v in zip(NAMES, ['candidate', 'candidate', 'baseline'])]
    correct = [[models[j]['pred'][i] == data['gold'][i] for i in range(n)] for j in range(3)]
    folds = {}
    for f in result['folds']:
        indices = [i for i, r in enumerate(records) if r['fold'] == f['fold']]
        eq([data['ids'][i] for i in indices], f['test_ids'], 'fold IDs')
        eq(len(indices), f['test'], 'fold test size')
        eq(n-len(indices), f['train'], 'fold train size')
        folds[f['fold']] = indices
    eq(sorted(i for ix in folds.values() for i in ix), list(range(n)), 'fold partition')
    expected_keys = {r['family']+'/'+r['target'] for r in result['policies']}
    for r in records:
        eq(set(r['policies']), expected_keys, 'policy keys')
    for row in result['policies']:
        key = row['family']+'/'+row['target']
        ds = [r['policies'][key] for r in records]
        compare(row, metrics(range(n), ds, data, models), key)
        for label, report in row['by_gold'].items():
            compare(report, metrics([i for i in range(n) if data['gold'][i] == label], ds, data, models), key+'/'+label)
        compare(row['error_detection_vs_jev'], confusion(range(n), [d['prediction'] is None for d in ds], correct[0]), key+'/detection')
        if row['family'].startswith('J_'):
            m = NAMES.index(row['family'][2:])
            G, E = sum(correct[m]), n-sum(correct[m])
            g, e = row['correct'], row['errors']
            compare(row, dict(G=G, E=E, J=g/(G+e) if G+e else None, U=g/G-e/E if G and E else None), key+'/JU')
        eq(row['infeasible_folds'], sum(not s['feasible'] for s in result['selections'] if s['group'] == row['family'] and s['target'] == row['target']), key+'/infeasible')
    routes_checked = veto_checked = 0
    for selection in result['selections']:
        key = selection['group']+'/'+selection['target']
        ix = folds[selection['fold']]
        ds = [r['policies'][key] for r in records]
        if not selection['feasible']:
            for i in ix:
                eq(ds[i], {'prediction': None, 'calls_mask': 0}, key+'/infeasible route')
            continue
        compare(selection['test'], metrics(ix, ds, data, models), key+'/fold')
        if selection['target'] != 'J_opt':
            target = int(selection['target'].replace('emp', '').replace('lcb', '')) / 100
            for name, metric_name in [('achieved_empirical_target', 'accuracy'), ('achieved_lcb_target', 'wilson_lower95')]:
                value = selection['test'][metric_name]
                eq(selection['test'][name], value is not None and value+1e-12 >= target, key+'/'+name)
        family, params = selection['family'], selection['params']
        if family in ('cascade2_raw', 'cascade3_raw'):
            for i in ix:
                mask, prediction = 0, None
                for m, threshold in zip(params['order'], params['thresholds']):
                    j = NAMES.index(m)
                    mask |= 1 << j
                    if models[j]['confidence'][i] >= threshold:
                        prediction = models[j]['pred'][i]
                        break
                eq(ds[i], {'prediction': prediction, 'calls_mask': mask}, key+'/raw route')
                routes_checked += 1
        if 'detection_test' in selection:
            j = NAMES.index(params['primary'])
            kind = family.removeprefix('conditional_veto_')
            low = [r['calibrated_correctness'][kind][j] < params['thresholds'][0] for r in records]
            compare(selection['detection_test']['primary_only_same_threshold'], confusion(ix, low, correct[j]), key+'/primary detector')
            compare(selection['detection_test']['both_low'], confusion(ix, [d['prediction'] is None for d in ds], correct[j]), key+'/veto detector')
            veto_checked += 1
    descriptive = json.loads((ROOT/'descriptive.json').read_text())['selected_overlap']
    counts = [sum(correct[j][i] for j in range(3)) for i in range(n)]
    histogram = {str(k): counts.count(k) for k in range(4)}
    eq(descriptive['correct_count_histogram'], histogram, 'correct overlap histogram')
    eq(descriptive['error_count_histogram'], {str(k): histogram[str(3-k)] for k in range(4)}, 'error overlap histogram')
    eq(sum(histogram.values()), n, '0..3 sum')
    eq(descriptive['oracle_correct'], n-histogram['0'], 'oracle')
    eq(descriptive['all_three_wrong_ids'], [data['ids'][i] for i in range(n) if counts[i] == 0], 'all wrong IDs')
    print(json.dumps({'status': 'verified', 'checks': checks, 'cases': n, 'policies': len(result['policies']), 'raw_cascade_case_routes': routes_checked, 'veto_fold_reports': veto_checked}, indent=2))


if __name__ == '__main__':
    main()
