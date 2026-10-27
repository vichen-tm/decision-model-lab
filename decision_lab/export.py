"""Convert complete runner journals into aligned analysis rows."""
import json
from decision_benchmark.runner.journal import RunJournal
from decision_benchmark.scoring import score_response, finite_number

def export_run(directory):
    j = RunJournal(directory)
    rows = {}
    with j.lock():
        plan = j.load_plan()
        for t in plan['tasks']:
            if t['key']['repeat'] != 1:
                raise ValueError('Export one repeat per run')
            case = t['case']
            qid = list(case['gold'])
            if len(qid) != 1 or case['gold'][qid[0]]['type'] != 'choice':
                raise ValueError('Analysis export supports one choice question per case')
            found = []
            for a in j.attempts(t):
                if not (a / 'response.json').exists():
                    continue
                raw = j.read_record(a / 'response.json')
                score = score_response(case, raw, t['route'])
                if score['complete']:
                    found.append(score)
            if len(found) != 1:
                raise ValueError('Require exactly one accepted response per task')
            score = found[0]
            v = score['results'][qid[0]]
            cid = json.dumps([case['language'], case['id']], separators=(',', ':'))
            m = t['key']['model']
            gold = case['gold'][qid[0]]['label']
            row = rows.setdefault(cid, {'id': cid, 'gold': gold, 'models': {}})
            if row['gold'] != gold or m in row['models']:
                raise ValueError('Duplicate/mismatched task')
            usage = score.get('usage')
            cost = usage.get('cost') if isinstance(usage, dict) else None
            if not finite_number(cost) or cost < 0:
                cost = None
            row['models'][m] = {'pred': v['prediction'], 'confidence': v['native_confidence'], 'probabilities': v['probabilities'], 'cost': cost, 'elapsed_ms': score.get('elapsed_ms')}
    names = {t['key']['model'] for t in plan['tasks']}
    if any((set(r['models']) != names for r in rows.values())):
        raise ValueError('Incomplete model alignment')
    return sorted(rows.values(), key=lambda r: r['id'])
