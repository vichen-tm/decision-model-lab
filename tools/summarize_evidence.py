"""Rescore all exported response evidence without benchmark source texts."""
import collections,csv,gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from decision_benchmark.scoring import score_response,finite_number

def lines(p):
 with gzip.open(p,'rt') as f:yield from (json.loads(x) for x in f if x.strip())
def main():
 summaries={}
 for campaign in ['ag-news','proofwriter']:
  p=ROOT/'experiments'/campaign;cases={r['case_sha256']:r['skeleton'] for r in json.loads((p/'corpus-index.json').read_text())};groups={};costs=[];states=collections.Counter()
  for r in lines(p/'responses.jsonl.gz'):
   case=cases[r['case_sha256']];score=score_response(case,r['raw'],r['route']);states['accepted' if score['complete'] else score['error'] or 'invalid_answer']+=1
   try:body=json.loads(r['raw'].get('response_text',''));usage=body.get('usage') if isinstance(body,dict) else None;cost=usage.get('cost') if isinstance(usage,dict) else None
   except ValueError:cost=None
   if not finite_number(cost) or cost<0:cost=None
   costs.append(cost)
   key=(r['run'],r['key']['model']);g=groups.setdefault(key,{'responses':0,'accepted':0,'correct':0,'known_cost_usd':0.,'unknown_cost_responses':0})
   g['responses']+=1;g['accepted']+=int(score['complete']);g['correct']+=int(score['complete'] and all(x['correct'] for x in score['results'].values()));g['known_cost_usd']+=cost or 0;g['unknown_cost_responses']+=cost is None
  with (p/'rescored-responses.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=['first_recorded_run','model','responses','accepted','correct','known_cost_usd','unknown_cost_responses']);w.writeheader()
   for (run,m),g in sorted(groups.items()):w.writerow({'first_recorded_run':run,'model':m,**g})
  summaries[campaign]={'unique_response_states':dict(states),'known_cost_usd':sum(c for c in costs if c is not None),'unknown_cost_responses':sum(c is None for c in costs),'scope':'Unique saved response records, including failures. Recovery copies deduplicated; rows assigned to first recorded run. Missing-response intents are in attempt-ledger and not included in these totals. Upstream baseline reuse is separate.'}
 (ROOT/'experiments/cost-and-response-summary.json').write_text(json.dumps(summaries,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
