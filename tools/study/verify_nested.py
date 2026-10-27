"""Independent nested artifact audit; no model fitting or provider calls."""
import hashlib
import json
from pathlib import Path
import verify_results as audit
import numpy as np

P = Path(__file__).resolve().parent


def main():
    data = json.loads((P/'data.json').read_text())
    original = json.loads((P/'results.json').read_text())
    old = [json.loads(x) for x in (P/'oof-decisions.jsonl').read_text().splitlines()]
    result = json.loads((P/'nested-results.json').read_text())
    rows = [json.loads(x) for x in (P/'nested-oof-decisions.jsonl').read_text().splitlines()]
    models = [data['models'][m][v] for m, v in zip(audit.NAMES, ['candidate', 'candidate', 'baseline'])]
    n = len(data['ids'])
    audit.eq([r['id'] for r in rows], data['ids'], 'nested IDs')
    audit.eq([r['gold'] for r in rows], data['gold'], 'nested gold')
    audit.eq([r['id'] for r in old], data['ids'], 'original IDs')
    audit.eq([r['fold'] for r in rows], [r['fold'] for r in old], 'fold partition')
    audit.eq(len(set(r['id'] for r in rows)), n, 'unique IDs')
    targets = ['emp98', 'emp99', 'emp100', 'lcb98', 'lcb99']
    for r in rows:
        audit.eq(set(r['policies']), set(targets), 'policy set')
    audit.eq(len(result['selections']), 25, 'selection count')
    audit.eq(len({(s['fold'], s['target']) for s in result['selections']}), 25, 'unique selections')
    inner_recomputed = 0
    for outer in range(5):
        expected_ix = [i for i, r in enumerate(rows) if r['fold'] != outer]
        with np.load(P/f'inner-oof-{outer}.npz', allow_pickle=False) as archive:
            audit.eq(archive['ids'].tolist(), [data['ids'][i] for i in expected_ix], 'inner heldout IDs')
            keys = archive['keys'].tolist()
            expected_reports = {m['family']+'/'+m['target']: m for m in result['inner_family_metrics'] if m['outer_fold'] == outer}
            audit.eq(set(keys), set(expected_reports), 'inner policy key set')
            audit.eq(len(set(keys)), len(keys), 'inner unique policy keys')
            audit.eq(archive['predictions'].shape, (len(keys), len(expected_ix)), 'inner prediction shape')
            audit.eq(archive['calls'].shape, (len(keys), len(expected_ix)), 'inner calls shape')
            predictions, calls = archive['predictions'], archive['calls']
            for k, key in enumerate(keys):
                ds = [None] * n
                for j, i in enumerate(expected_ix):
                    pred = int(predictions[k, j])
                    assert pred in (-1, 0, 1, 2)
                    ds[i] = {'prediction': ['true', 'false', 'unknown'][pred] if pred >= 0 else None,
                             'calls_mask': int(calls[k, j])}
                audit.compare(expected_reports[key], audit.metrics(expected_ix, ds, data, models), 'inner metric/'+key)
                inner_recomputed += 1
    transferred = 0
    for s in result['selections']:
        target = int(s['target'].replace('emp', '').replace('lcb', ''))/100
        bounded = s['target'].startswith('lcb')
        candidates = sorted([m for m in result['inner_family_metrics'] if m['outer_fold'] == s['fold'] and m['target'] == s['target']], key=lambda m: m['family'])
        assert candidates
        audit.eq(len({m['family'] for m in candidates}), len(candidates), 'unique inner families')
        eligible = []
        for m in candidates:
            audit.eq(m['n'], n-sum(r['fold'] == s['fold'] for r in rows), 'inner n')
            audit.eq(m['correct']+m['errors'], m['accepted'], 'inner accepted accounting')
            audit.eq(m['accepted']+m['referred'], m['n'], 'inner referred accounting')
            quality = m['wilson_lower95'] if bounded else m['accuracy']
            if m['accepted'] >= 50 and quality is not None and quality+1e-12 >= target:
                eligible.append(m)
        winner = max(eligible, key=lambda m: (m['accepted'], -m['cost_usd'], -m['errors'])) if eligible else None
        audit.eq(s['family'], winner['family'] if winner else None, 'inner family winner')
        audit.eq(s['inner_feasible'], bool(winner), 'inner feasibility')
        ix = [i for i, r in enumerate(rows) if r['fold'] == s['fold']]
        ds = [r['policies'][s['target']] for r in rows]
        audit.compare(s['test'], audit.metrics(ix, ds, data, models), 'nested fold metric')
        refit = None
        if winner:
            matches = [r for r in original['selections'] if r['fold'] == s['fold'] and r['group'] == s['family'] and r['target'] == s['target']]
            audit.eq(len(matches), 1, 'unique refit')
            refit = matches[0]
            audit.eq(s['outer_refit_feasible'], refit['feasible'], 'refit feasibility')
            audit.eq(s['outer_refit_params'], refit.get('params'), 'refit params')
        for i in ix:
            expected = old[i]['policies'][s['family']+'/'+s['target']] if refit and refit['feasible'] else {'prediction': None, 'calls_mask': 0}
            audit.eq(ds[i], expected, 'selected route transfer')
            transferred += 1
    audit.eq(len(result['policies']), 5, 'aggregate count')
    for row in result['policies']:
        audit.eq(row['family'], 'nested_selected_global', 'aggregate family')
        audit.compare(row, audit.metrics(range(n), [r['policies'][row['target']] for r in rows], data, models), 'nested aggregate')
    report = {'status': 'verified', 'checks': audit.checks, 'cases': n, 'targets': len(targets), 'outer_selections': 25, 'transferred_case_routes': transferred,
              'inner_metrics_recomputed': inner_recomputed,
              'method': 'Independent outer and inner per-case metric recomputation, family winner selection from verified inner metrics, exact original refit/route matching.',
              'sha256': {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['verify_nested.py', 'verify_results.py', 'nested_select.py', 'data.json', 'results.json', 'oof-decisions.jsonl', 'nested-results.json', 'nested-oof-decisions.jsonl'] + [f'inner-oof-{i}.npz' for i in range(5)]}}
    (P/'verification-nested.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sha256'}, indent=2))


if __name__ == '__main__':
    main()
