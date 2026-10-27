"""Opus-review revision: select a family using inner held-out policy outcomes.

Preserves the first procedure; does not inspect outer-fold labels for selection.
Every candidate family's parameters are refitted on outer training by analyze.py.
"""
import json
from pathlib import Path
import numpy as np
from analyze import LABELS, TARGETS, MIN_SUPPORT, metric, wilson
from calibration import stratified_folds, crossfit
from policies import catalogue, NAMES

P = Path(__file__).resolve().parent


def main():
    data = json.loads((P/'data.json').read_text())
    original = json.loads((P/'results.json').read_text())
    old_cases = [json.loads(x) for x in (P/'oof-decisions.jsonl').read_text().splitlines()]
    gold = np.array([LABELS.index(x) for x in data['gold']])
    chosen = [data['models'][m][v] for m, v in zip(NAMES, ['candidate', 'candidate', 'baseline'])]
    pred = np.column_stack([[LABELS.index(x) for x in d['pred']] for d in chosen])
    conf = np.column_stack([d['confidence'] for d in chosen])
    costs = np.column_stack([d['cost'] for d in chosen])
    elapsed = np.column_stack([d['elapsed_ms'] for d in chosen])
    prob = np.stack([d['probabilities'] for d in chosen], axis=1)
    folds = stratified_folds(gold, 5, 20261023)
    global_results = {t: (np.full(1000, -1), np.zeros(1000, dtype=np.uint8)) for t, _, _ in TARGETS}
    selections, inner_report = [], []
    for outer in range(5):
        training, testing = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        y, pr, cf, co, el, pb = gold[training], pred[training], conf[training], costs[training], elapsed[training], prob[training]
        correct = pr == y[:, None]
        inner_folds = stratified_folds(y, 3, 3000+outer)
        pooled = {}
        for inner in range(3):
            tr, va = np.flatnonzero(inner_folds != inner), np.flatnonzero(inner_folds == inner)
            fitted = crossfit(cf, pr, correct, y, tr, va, 5000+outer*3+inner)
            scores = {'raw': cf}
            for name, (a, b) in fitted.items():
                scores[name] = np.empty_like(cf)
                scores[name][tr], scores[name][va] = a, b
            best, families = {}, set()
            for family, params, decision, calls in catalogue(pr, pb, scores, tr):
                families.add(family)
                count = int((decision[tr] >= 0).sum())
                good = int((decision[tr] == y[tr]).sum())
                err = count-good
                used = (calls[tr, None] & (1 << np.arange(3))) > 0
                cost = float((used*co[tr]).sum())
                lower = wilson(good, count)
                for target_name, target, bounded in TARGETS:
                    quality = lower if bounded else good/count if count else 0
                    if count < MIN_SUPPORT or quality is None or quality + 1e-12 < target:
                        continue
                    key = family, target_name
                    rank = count, -cost, -err
                    if key not in best or rank > best[key][0]:
                        best[key] = rank, decision[va].copy(), calls[va].copy()
            for family in families:
                for target_name, _, _ in TARGETS:
                    key = family, target_name
                    if key not in pooled:
                        pooled[key] = np.full(len(y), -1), np.zeros(len(y), dtype=np.uint8)
                    if key in best:
                        pooled[key][0][va], pooled[key][1][va] = best[key][1], best[key][2]
            print(f'Outer {outer+1}/5, inner {inner+1}/3 complete', flush=True)
        # Persist inner held-out decisions as well as aggregate selection evidence.
        ordered_keys = sorted(pooled)
        np.savez_compressed(P/f'inner-oof-{outer}.npz',
                            ids=np.array([data['ids'][i] for i in training]),
                            keys=np.array([family+'/'+target for family, target in ordered_keys]),
                            predictions=np.stack([pooled[k][0] for k in ordered_keys]),
                            calls=np.stack([pooled[k][1] for k in ordered_keys]))
        for target_name, target, bounded in TARGETS:
            winner, winning_rank = None, None
            for (family, t), (decision, calls) in sorted(pooled.items()):
                if t != target_name:
                    continue
                m = metric(decision, calls, y, co, el)
                inner_report.append({'outer_fold': outer, 'family': family, 'target': t, **m})
                quality = m['wilson_lower95'] if bounded else m['accuracy']
                if m['accepted'] < MIN_SUPPORT or quality is None or quality + 1e-12 < target:
                    continue
                rank = m['accepted'], -m['cost_usd'], -m['errors']
                if winning_rank is None or rank > winning_rank:
                    winner, winning_rank = family, rank
            selection = {'fold': outer, 'target': target_name, 'family': winner, 'inner_feasible': winner is not None}
            if winner:
                refit = next(s for s in original['selections'] if s['fold'] == outer and s['group'] == winner and s['target'] == target_name)
                selection['outer_refit_feasible'] = refit['feasible']
                selection['outer_refit_params'] = refit.get('params')
                if refit['feasible']:
                    for i in testing:
                        case = old_cases[i]['policies'][winner+'/'+target_name]
                        global_results[target_name][0][i] = LABELS.index(case['prediction']) if case['prediction'] is not None else -1
                        global_results[target_name][1][i] = case['calls_mask']
            dec, calls = global_results[target_name]
            selection['test'] = metric(dec[testing], calls[testing], gold[testing], costs[testing], elapsed[testing])
            selections.append(selection)
    policies = [{'family': 'nested_selected_global', 'target': t, **metric(dec, calls, gold, costs, elapsed)}
                for t, (dec, calls) in global_results.items()]
    (P/'nested-results.json').write_text(json.dumps({'scope': 'Review-driven exploratory revision, outer test never used in family selection',
            'policies': policies, 'selections': selections, 'inner_family_metrics': inner_report}, indent=2)+'\n')
    with (P/'nested-oof-decisions.jsonl').open('w') as fp:
        for i, case_id in enumerate(data['ids']):
            fp.write(json.dumps({'id': case_id, 'fold': int(folds[i]), 'gold': data['gold'][i], 'policies': {
                t: {'prediction': LABELS[d[i]] if d[i] >= 0 else None, 'calls_mask': int(c[i])} for t, (d,c) in global_results.items()}})+'\n')
    print('Nested family selection complete', flush=True)


if __name__ == '__main__':
    main()
