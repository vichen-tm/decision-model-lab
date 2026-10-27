"""Export local research journals into portable, source-text-free evidence.

Usage: python tools/package_experiments.py RESEARCH_ROOT OUTPUT
Exact input texts are retained in a separate local-only archive, not public Git.
"""
import argparse,base64,csv,gzip,hashlib,json,re,zipfile
from pathlib import Path


def sha(b):return hashlib.sha256(b).hexdigest()
def encoded(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k.lower() not in ('authorization','api_key','apikey','cookie')}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,str):return re.sub(r'/Users/[^/\s"\']+/(?:Documents/ChatGPT/context-loader/)?','<local>/',v)
    return v

def main():
    p=argparse.ArgumentParser();p.add_argument('research');p.add_argument('output');a=p.parse_args();root=Path(a.research);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    index=[];full={};corpus={};templates={};seen=set();rows=[];members=[];attempt_ledger=[]
    for planpath in sorted(root.rglob('plan.json')):
        envelope=json.loads(planpath.read_text());plan=envelope.get('data',envelope)
        if 'sha256' in envelope and sha(encoded(plan))!=envelope['sha256']:raise ValueError('Changed plan')
        if 'tasks' not in plan:continue
        run=planpath.parent;runid=run.relative_to(root).as_posix();counts={'tasks':len(plan['tasks']),'responses':0,'unique_responses':0}
        for task in plan['tasks']:
            case=task['case'];cid=case['id'];h=sha(encoded(case));full[h]=case
            corpus[h]={'id':cid,'case_sha256':h,'language':case.get('language'),'suite':case.get('suite'),'gold':case.get('gold'),'source':clean(case.get('source')),'group_id':case.get('group_id'),'depth':case.get('depth')}
            skeleton=json.loads(json.dumps(case));skeleton['request'].pop('state',None)
            corpus[h]['skeleton']=clean(skeleton)
            corpus[h]['state_sha256']=sha(encoded(case['request']['state']))
            body=task['body'];template={k:v for k,v in body.items() if k!='state'};th=sha(encoded(template));templates[th]=clean(template)
            members.append({'run':runid,'task_id':task['id'],'case_sha256':h,'template_sha256':th,'key':task['key'],'route':clean(task['route'])})
            for attempt in sorted((run/'attempts'/task['id']).glob('*')):
                if not attempt.is_dir():continue
                intent_path=attempt/'intent.json';response_path=attempt/'response.json'
                if not intent_path.exists():continue
                envelope=json.loads(intent_path.read_text());intent=envelope.get('data',envelope)
                if 'sha256' in envelope and sha(encoded(intent))!=envelope['sha256']:raise ValueError('Changed intent')
                attempt_ledger.append({'run':runid,'task_id':task['id'],'attempt':attempt.name,'intent':clean(intent),'response_present':response_path.exists(),'response_record_sha256':sha(encoded(json.loads(response_path.read_text()).get('data'))) if response_path.exists() else None})
            for response in sorted((run/'attempts'/task['id']).glob('*/response.json')):
                envelope=json.loads(response.read_text());raw=envelope.get('data',envelope)
                if 'sha256' in envelope and sha(encoded(raw))!=envelope['sha256']:raise ValueError('Changed response')
                if clean(raw)!=raw:raise ValueError('Raw response contains private metadata; quarantine before export rather than altering evidence')
                counts['responses']+=1
                # Canonical response dedup across recovery journals, exact bytes preserved.
                rh=sha(encoded(raw));identity=(task['id'],rh)
                if identity not in seen:
                    seen.add(identity);counts['unique_responses']+=1
                    rows.append({'run':runid,'task_id':task['id'],'case_sha256':h,'case_id':cid,'key':task['key'],'route':clean(task['route']),'request_template_sha256':th,'request_body_sha256':sha(encoded(body)),'raw':raw})
        index.append({'run':runid,**counts})
    with gzip.open(out/'responses.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with gzip.open(out/'attempt-ledger.jsonl.gz','wt') as f:
        for r in attempt_ledger:f.write(json.dumps(r)+'\n')
    with gzip.open(out/'tasks.jsonl.gz','wt') as f:
        for r in members:f.write(json.dumps(r)+'\n')
    (out/'corpus-index.json').write_text(json.dumps(list(corpus.values()),indent=2))
    (out/'request-templates.json').write_text(json.dumps(templates,indent=2))
    with (out/'runs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['run','tasks','responses','unique_responses']);w.writeheader();w.writerows(index)
    # Canonical full corpus stays private until third-party text redistribution is resolved.
    private=out.parents[1]/'private';private.mkdir(exist_ok=True)
    with gzip.open(private/(root.name+'-corpus.json.gz'),'wt') as f:json.dump(full,f)
    (out/'manifest.json').write_text(json.dumps({'source_campaign':root.name,'runs':len(index),'unique_responses':len(rows),'cases_and_prompt_variants':len(corpus),'dedup_key':['task_id','canonical_raw_sha256'],'text_distribution':'Input state excluded from public files; full case hashes and local private corpus retained.','files':{f.name:sha(f.read_bytes()) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}},indent=2))
    print(root.name,len(index),len(rows))
if __name__=='__main__':main()
