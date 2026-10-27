"""Nested offline policy selection on the working 1000 only.

No HTTP, provider SDK, external inference, or closed-final data access.
Run with python3 analyze.py. Outputs are reproducible given data.json.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from calibration import crossfit, stratified_folds
from policies import catalogue, NAMES, vote

ROOT = Path(__file__).resolve().parent
LABELS = ['true', 'false', 'unknown']
TARGETS = [('emp98', .98, False), ('emp99', .99, False),
           ('emp100', 1., False), ('lcb98', .98, True), ('lcb99', .99, True)]
MIN_SUPPORT = 50


def wilson(correct, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = correct/total
    return (p+z*z/(2*total)-z*np.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total)


def metric(decision, calls, gold, costs, elapsed):
    accepted = decision >= 0
    correct = int(((decision == gold) & accepted).sum())
    count = int(accepted.sum())
    used = ((calls[:, None] & (1 << np.arange(3))) > 0)
    cost = float((used*costs).sum())
    return {'n': len(gold), 'accepted': count, 'correct': correct, 'errors': count-correct,
            'referred': len(gold)-count, 'accuracy': correct/count if count else None,
            'coverage': count/len(gold), 'correct_automatic_share': correct/len(gold),
            'wilson_lower95': wilson(correct, count), 'cost_usd': cost,
            'cost_per1000_inputs': cost*1000/len(gold),
            'cost_per_correct_auto': cost/correct if correct else None,
            'calls': int(used.sum()), 'calls_per_model': dict(zip(NAMES, map(int, used.sum(axis=0)))),
            'mean_serial_sum_ms': float((used*elapsed).sum()/len(gold))}


def detection(reject, primary_correct):
    tp = int((reject & ~primary_correct).sum())
    fp = int((reject & primary_correct).sum())
    fn = int((~reject & ~primary_correct).sum())
    tn = int((~reject & primary_correct).sum())
    return {'tp_errors_referred': tp, 'fp_correct_referred': fp, 'fn_errors_retained': fn,
            'tn_correct_retained': tn, 'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None, 'false_positive_rate': fp/(fp+tn) if fp+tn else None}


def main():
    data = json.loads((ROOT/'data.json').read_text())
    gold = np.array([LABELS.index(x) for x in data['gold']])
    variants = ['candidate', 'candidate', 'baseline']
    chosen = [data['models'][m][v] for m, v in zip(NAMES, variants)]
    pred = np.column_stack([[LABELS.index(x) for x in d['pred']] for d in chosen])
    conf = np.column_stack([d['confidence'] for d in chosen])
    correct = pred == gold[:, None]
    costs = np.column_stack([d['cost'] for d in chosen])
    elapsed = np.column_stack([d['elapsed_ms'] for d in chosen])
    prob = np.stack([d['probabilities'] for d in chosen], axis=1)
    folds = stratified_folds(gold, 5, 20261023)
    calibration_oof = {k: np.zeros_like(conf) for k in ('cal', 'cat')}
    accum, selections, audit = {}, [], []
    summary_path = ROOT/'candidate-training-scores.csv'
    with summary_path.open('w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['fold', 'family', 'parameters', 'accepted_train', 'errors_train', 'cost_train_usd', 'wilson_lower95'])
        for fold in range(5):
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            assert not set(train) & set(test)
            fitted = crossfit(conf, pred, correct, gold, train, test, 1000+fold)
            scores = {'raw': conf}
            for name, (q_train, q_test) in fitted.items():
                scores[name] = np.empty_like(conf)
                scores[name][train], scores[name][test] = q_train, q_test
                calibration_oof[name][test] = q_test
            best, families, candidates = {}, set(), 0
            for family, params, decision, calls in catalogue(pred, prob, scores, train):
                families.add(family)
                candidates += 1
                accepted = decision[train] >= 0
                count = int(accepted.sum())
                good = int(((decision[train] == gold[train]) & accepted).sum())
                error = count-good
                used = (calls[train, None] & (1 << np.arange(3))) > 0
                cost = float((used*costs[train]).sum())
                lower = wilson(good, count)
                paramstr = json.dumps(params, sort_keys=True)
                writer.writerow([fold, family, paramstr, count, error, cost, lower])
                if family == 'single_raw':
                    m = NAMES.index(params['model'])
                    total_correct = int(correct[train, m].sum())
                    j = good/(total_correct+error) if total_correct+error else 0
                    key = ('J_'+NAMES[m], 'J_opt')
                    rank = (j, -error, good)
                    if key not in best or rank > best[key]['rank']:
                        best[key] = {'rank': rank, 'family': family, 'params': params,
                                     'decision': decision[test].copy(), 'calls': calls[test].copy(),
                                     'train': {'accepted': count, 'errors': error, 'j': j}}
                for target_name, target, bounded in TARGETS:
                    quality = lower if bounded else good/count if count else 0
                    if count < MIN_SUPPORT or quality is None or quality + 1e-12 < target:
                        continue
                    # Predeclared objective: coverage, then cost, then error count.
                    rank = (count, -cost, -error)
                    for group in (family, 'selected_global'):
                        key = (group, target_name)
                        if key not in best or rank > best[key]['rank']:
                            best[key] = {'rank': rank, 'family': family, 'params': params,
                                         'decision': decision[test].copy(), 'calls': calls[test].copy(),
                                         'train': {'accepted': count, 'errors': error, 'accuracy': good/count,
                                                   'wilson_lower95': lower, 'cost_usd': cost}}
            for family in sorted(families | {'selected_global'}):
                for target_name, target, bounded in TARGETS:
                    key = (family, target_name)
                    if key not in accum:
                        accum[key] = (np.full(len(gold), -1), np.zeros(len(gold), dtype=np.uint8))
                    item = best.get(key)
                    if item:
                        accum[key][0][test], accum[key][1][test] = item['decision'], item['calls']
                        item['test'] = metric(item['decision'], item['calls'], gold[test], costs[test], elapsed[test])
                        item['test']['achieved_empirical_target'] = bool(item['test']['accuracy'] is not None and item['test']['accuracy'] + 1e-12 >= target)
                        item['test']['achieved_lcb_target'] = bool(item['test']['wilson_lower95'] is not None and item['test']['wilson_lower95'] + 1e-12 >= target)
                        if family.startswith('conditional_veto'):
                            a, b = NAMES.index(item['params']['primary']), NAMES.index(item['params']['verifier'])
                            kind = family.removeprefix('conditional_veto_')
                            t, u = item['params']['thresholds']
                            item['detection_test'] = {
                                'primary': NAMES[a],
                                'primary_only_same_threshold': detection(scores[kind][test, a] < t, correct[test, a]),
                                'both_low': detection(item['decision'] < 0, correct[test, a])}
                    selections.append({'fold': fold, 'group': family, 'target': target_name,
                                       'feasible': item is not None,
                                       **({k: item[k] for k in ('family', 'params', 'train', 'test')} if item else {}),
                                       **({'detection_test': item['detection_test']} if item and 'detection_test' in item else {})})
            for m in NAMES:
                key = ('J_'+m, 'J_opt')
                item = best[key]
                if key not in accum:
                    accum[key] = (np.full(len(gold), -1), np.zeros(len(gold), dtype=np.uint8))
                accum[key][0][test], accum[key][1][test] = item['decision'], item['calls']
                item['test'] = metric(item['decision'], item['calls'], gold[test], costs[test], elapsed[test])
                selections.append({'fold': fold, 'group': key[0], 'target': key[1], 'feasible': True,
                                   **{k: item[k] for k in ('family', 'params', 'train', 'test')}})
            audit.append({'fold': fold, 'train': len(train), 'test': len(test), 'candidates': candidates,
                          'test_ids': [data['ids'][i] for i in test]})
            print(f'Fold {fold+1}/5: {candidates} policies; {len(best)} feasible selections', flush=True)

    results = []
    for (family, target), (decision, calls) in sorted(accum.items()):
        row = {'family': family, 'target': target, **metric(decision, calls, gold, costs, elapsed)}
        row['infeasible_folds'] = sum(not s['feasible'] for s in selections if s['group'] == family and s['target'] == target)
        row['error_detection_vs_jev'] = detection(decision < 0, correct[:, 0])
        if family.startswith('J_'):
            m = NAMES.index(family.removeprefix('J_'))
            g, e, G, E = row['correct'], row['errors'], int(correct[:, m].sum()), int((~correct[:, m]).sum())
            row.update({'G': G, 'E': E, 'J': g/(G+e) if G+e else None, 'U': g/G-e/E if G and E else None})
        row['by_gold'] = {}
        for c, label in enumerate(LABELS):
            ix = gold == c
            row['by_gold'][label] = metric(decision[ix], calls[ix], gold[ix], costs[ix], elapsed[ix])
        results.append(row)

    fixed = []
    for m, name in enumerate(NAMES):
        fixed.append({'name': name, **metric(pred[:, m], np.full(len(gold), 1 << m), gold, costs, elapsed)})
    for name, decision, mask in [
            ('majority_abstain', vote(pred)[0], 7),
            ('majority_jev_tie', vote(pred, tie_primary=True)[0], 7),
            ('unanimity', np.where((pred == pred[:, :1]).all(axis=1), pred[:, 0], -1), 7),
            ('probability_average', prob.mean(axis=1).argmax(axis=1), 7)]:
        fixed.append({'name': name, **metric(decision, np.full(len(gold), mask), gold, costs, elapsed)})
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        decision = np.where(pred[:, a] == pred[:, b], pred[:, a], -1)
        fixed.append({'name': f'agree_{NAMES[a]}_{NAMES[b]}', **metric(decision, np.full(len(gold), (1 << a) | (1 << b)), gold, costs, elapsed)})

    calibration = []
    for kind, values in [('raw', conf), *calibration_oof.items()]:
        for m, name in enumerate(NAMES):
            q, y = values[:, m], correct[:, m]
            bins = np.minimum((q*10).astype(int), 9)
            ece = sum(abs(q[bins == b].mean()-y[bins == b].mean()) * (bins == b).sum()/len(y)
                      for b in range(10) if (bins == b).any())
            calibration.append({'kind': kind, 'model': name, 'brier': float(np.mean((q-y)**2)), 'ece10': float(ece)})

    output = {'scope': 'Internal nested cross-validation on prompt-development working1000, not closed final1000',
              'seed': 20261023, 'models': dict(zip(NAMES, variants)), 'folds': audit,
              'fixed': fixed, 'calibration': calibration, 'policies': results, 'selections': selections,
              'input_sha256': hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest()}
    (ROOT/'results.json').write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n')
    with (ROOT/'policy-results.csv').open('w', newline='') as fp:
        fields = ['family', 'target', 'accepted', 'correct', 'errors', 'referred', 'accuracy', 'coverage',
                  'correct_automatic_share', 'wilson_lower95', 'cost_usd', 'calls', 'infeasible_folds']
        writer = csv.DictWriter(fp, fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in results)
    # Per-case outputs make accounting and routing independently verifiable.
    with (ROOT/'oof-decisions.jsonl').open('w') as fp:
        for i, case_id in enumerate(data['ids']):
            row = {'id': case_id, 'fold': int(folds[i]), 'gold': data['gold'][i],
                   'calibrated_correctness': {k: v[i].tolist() for k, v in calibration_oof.items()},
                   'policies': {f'{family}/{target}': {'prediction': LABELS[d[i]] if d[i] >= 0 else None, 'calls_mask': int(c[i])}
                                for (family, target), (d, c) in sorted(accum.items())}}
            fp.write(json.dumps(row)+'\n')
    for row in results:
        assert row['accepted'] == row['correct'] + row['errors']
        assert row['accepted'] + row['referred'] == len(gold)
        assert row['calls'] >= row['accepted']
    print('Completed. Results and per-case routes saved.', flush=True)


if __name__ == '__main__':
    main()
