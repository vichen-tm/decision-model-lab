"""Reconstruct a recorded run from separately obtained benchmark source text.

ProofWriter: --source source.zip. AG News: --source ordered-test.jsonl
AG News JSONL rows must have text (or article), in original test-row order.
No download and no inference. Hash mismatch stops reconstruction.
"""
import argparse,copy,gzip,hashlib,json,zipfile
from pathlib import Path

def enc(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(v):return hashlib.sha256(enc(v)).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign');p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.campaign);idx={r['case_sha256']:r for r in json.loads((root/'corpus-index.json').read_text())}
 with gzip.open(root/'tasks.jsonl.gz','rt') as f:tasks=[json.loads(x) for x in f if json.loads(x)['run']==a.run]
 if not tasks:raise ValueError('Unknown recorded run')
 zipfile_source=zipfile.is_zipfile(a.source);cache={}
 if zipfile_source:
  z=zipfile.ZipFile(a.source)
  if hashlib.sha256(Path(a.source).read_bytes()).hexdigest()!='bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26':raise ValueError('Unexpected ProofWriter archive bytes')
 else:articles=[json.loads(x) for x in Path(a.source).read_text().splitlines() if x.strip()]
 cases={};routes={}
 for t in tasks:
  entry=idx[t['case_sha256']];case=copy.deepcopy(entry['skeleton']);source=case['source'];candidates=[]
  if zipfile_source:
   member=source['file']
   if member not in cache:cache[member]={v['id']:v for v in (json.loads(x) for x in z.read(member).decode().splitlines() if x.strip())}
   theory=cache[member][source['theory_id']];question=theory['questions'][source['question_id']]['question']
   facts=[v['text'] for v in theory['triples'].values()];rules=[v['text'] for v in theory['rules'].values()]
   for newline in ['\n','\\n']:
    candidates.append({'text':newline.join(['Facts:',*facts,'','Rules:',*rules,'','Statement:',question])})
  else:
   article=articles[source['row_index']];text=article.get('text',article.get('article'))
   if not isinstance(text,str):raise ValueError('Missing article text')
   for ensure in [True,False]:
    for separators in [(',',':'),(', ',': ')]:candidates.append({'text':json.dumps({'article':text},ensure_ascii=ensure,separators=separators)})
  states=[v for v in candidates if sha(v)==entry['state_sha256']]
  if not states:raise ValueError('Cannot reconstruct input exactly: '+case['id'])
  case['request']['state']=states[0]
  # Insertion order is part of historic canonical byte hashes; reconstruct request order.
  case['request']={'state':case['request']['state'],'questions':case['request']['questions']}
  # Full object ordering may differ after removing state; input state itself is verified above.
  if sha(case)!=entry['case_sha256']:raise ValueError('Full case hash mismatch: '+case['id'])
  if case['id'] in cases and sha(cases[case['id']])!=sha(case):raise ValueError('Run has per-model prompt variants; restore one model at a time')
  cases[case['id']]=case;routes[t['route']['model']]=t['route']
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 (out/'dataset.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases.values()))
 (out/'routes.json').write_text(json.dumps(list(routes.values()),indent=2)+'\n')
 print(json.dumps({'cases':len(cases),'routes':len(routes),'input_states_verified':True}))
if __name__=='__main__':main()
