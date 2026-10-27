"""Verify final and working headline predictions against packaged raw responses."""
import gzip,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from decision_benchmark.scoring import score_response
P=ROOT/'experiments/proofwriter'

def main():
 cases={r['case_sha256']:r['skeleton'] for r in json.loads((P/'corpus-index.json').read_text())}
 final={r['id']:r for r in (json.loads(x) for x in (P/'final-decisions.jsonl').read_text().splitlines())}
 work=json.loads((P/'working-data.json').read_text());wi={cid:i for i,cid in enumerate(work['ids'])};seen=set()
 with gzip.open(P/'responses.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);run=r['run'];parts=run.split('/');m=None;variant=None;split=None
   if parts[0]=='final-validation' and len(parts)==2:split='final';m=parts[1]
   elif parts[0] in ['wide-v2','wide-liquid','wide-microsoft'] and parts[-1] in ['baseline','v2']:
    split='working';m={'wide-v2':'jev','wide-liquid':'liquid','wide-microsoft':'microsoft'}[parts[0]];variant='candidate' if parts[-1]=='v2' else 'baseline'
   if split is None:continue
   c=cases[r['case_sha256']];score=score_response(c,r['raw'],r['route'])
   if not score['complete']:continue
   v=score['results']['verdict'];cid=r['case_id'];identity=(split,m,variant,cid)
   if identity in seen:raise ValueError('Duplicate accepted final/working response')
   seen.add(identity)
   if split=='final':expected=final[cid]['models'][m];gold=final[cid]['gold']
   else:
    d=work['models'][m][variant];i=wi[cid];expected={k:d[k][i] for k in ['pred','confidence','cost','elapsed_ms','probabilities']};gold=work['gold'][i]
   assert v['prediction']==expected['pred'] and c['gold']['verdict']['label']==gold
   for actual,value in [(v['native_confidence'],expected['confidence']),(score['usage']['cost'],expected['cost']),(score['elapsed_ms'],expected['elapsed_ms'])]:assert math.isclose(actual,value,abs_tol=1e-12)
   assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(v['probabilities'],expected['probabilities']))
 assert sum(k[0]=='final' for k in seen)==3000
 assert sum(k[0]=='working' for k in seen)==6000
 print(json.dumps({'verified_raw_to_analysis':len(seen),'working':6000,'final':3000}))
if __name__=='__main__':main()
