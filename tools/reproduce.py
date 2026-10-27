"""Reapply frozen rules and independently verify all headline counts and costs."""
import json,math,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from decision_lab.proofwriter import apply
from decision_lab.analysis import metrics
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'experiments/proofwriter'

def main():
 rows=[json.loads(x) for x in (P/'final-decisions.jsonl').read_text().splitlines()]
 if len(rows)!=1000 or len({r['id'] for r in rows})!=1000:raise ValueError('Invalid final population')
 names=['jev','liquid','microsoft'];labels=['true','false','unknown'];frozen=json.loads((P/'final-frozen.json').read_text())
 pred=np.array([[labels.index(r['models'][m]['pred']) for m in names] for r in rows]);conf=np.array([[r['models'][m]['confidence'] for m in names] for r in rows]);out,q=apply(pred,conf,frozen)
 expected=json.loads((P/'final-results.json').read_text());report={}
 assert expected['frozen_sha256']==hashlib.sha256((P/'final-frozen.json').read_bytes()).hexdigest()
 assert json.loads((P/'working-results.json').read_text())['input_sha256']==hashlib.sha256((P/'working-data.json').read_bytes()).hexdigest()
 for k,(d,c) in out.items():
  decisions=[labels[x] if x>=0 else None for x in d];calls=[[m for i,m in enumerate(names) if int(mask)&(1<<i)] for mask in c]
  result=metrics(rows,decisions,calls);e=expected['final'][k]
  for key in ['n','accepted','correct','errors','referred']:assert result[key]==e[key],(k,key)
  assert math.isclose(result['cost_usd'],e['cost_usd'],abs_tol=1e-12)
  for i,r in enumerate(rows):assert decisions[i]==r['policies'][k]['prediction'] and int(c[i])==r['policies'][k]['calls_mask']
  report[k]=result
 work=json.loads((P/'working-data.json').read_text());oof=[json.loads(x) for x in (P/'working-oof-decisions.jsonl').read_text().splitlines()]
 assert len(set(work['ids']))==1000 and not set(work['ids'])&{r['id'] for r in rows}
 assert {r['id'] for r in oof}==set(work['ids'])
 aligned=[]
 for i,cid in enumerate(work['ids']):
  aligned.append({'id':cid,'gold':work['gold'][i],'models':{m:{'pred':work['models'][m][v]['pred'][i],'cost':work['models'][m][v]['cost'][i]} for m,v in zip(names,['candidate','candidate','baseline'])}})
 lookup={r['id']:r for r in oof}
 for k in out:
  if k in names:d=[r['models'][k]['pred'] for r in aligned];c=[[k]]*1000
  else:
   ps=[lookup[r['id']]['policies'][k] for r in aligned];d=[z['prediction'] for z in ps];c=[[m for i,m in enumerate(names) if z['calls_mask']&(1<<i)] for z in ps]
  r=metrics(aligned,d,c);e=expected['development'][k]
  for key in ['n','accepted','correct','errors','referred']:assert r[key]==e[key],('working',k,key)
  assert math.isclose(r['cost_usd'],e['cost_usd'],abs_tol=1e-12)
  for key in ['n','accepted','correct','errors','referred']:assert r[key]+report[k][key]==expected['combined'][k][key]
 print(json.dumps({'verified_final':1000,'verified_working_oof':1000,'policies':report},indent=2))
if __name__=='__main__':main()
