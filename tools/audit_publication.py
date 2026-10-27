"""Scan publishable files, including decompressed/decoded responses; build manifest."""
import base64,gzip,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
IGNORED={'private','runs','output','.git','.venv','__pycache__','build','dist'}
def public_files():
 roots={'decision_lab','decision_benchmark','docs','skills','examples','experiments','results','tests','tools','releases'}
 top={'README.md','AGENTS.md','ATTRIBUTION.md','LICENSE','pyproject.toml','.gitignore','.env.example'}
 for p in sorted(ROOT.rglob('*')):
  rel=p.relative_to(ROOT)
  if not p.is_file() or any(x in IGNORED or x.endswith('.egg-info') for x in rel.parts):continue
  if len(rel.parts)==1 and rel.name not in top:continue
  if len(rel.parts)>1 and rel.parts[0] not in roots:continue
  if p.name=='.env' or (p.name.startswith('.env.') and p.name!='.env.example'):continue
  if p.name=='.DS_Store' or p.suffix=='.pyc':continue
  if rel.parts[0]=='releases' and (p.name.endswith(('.zip','.tar.gz','.tgz')) or p.name=='manifest.json'):continue
  yield p

def main():
 findings=[];files={}
 for campaign in (ROOT/"experiments").iterdir():
  manifest=campaign/"manifest.json"
  if not manifest.is_file():continue
  meta=json.loads(manifest.read_text());meta["files"]={str(p.relative_to(campaign)):hashlib.sha256(p.read_bytes()).hexdigest() for p in public_files() if campaign in p.parents and p!=manifest}
  manifest.write_text(json.dumps(meta,indent=2)+"\n")
 patterns=[('private home path',r'/Users/[A-Za-z0-9_.-]+/'),('API key',r'sk-(?:or-v1-|ant-api\w*-)?[A-Za-z0-9_-]{24,}'),('credential header',r'Bearer [A-Za-z0-9_-]{20,}')]
 for p in public_files():
  b=p.read_bytes();files[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();data=gzip.decompress(b) if p.suffix=='.gz' else b
  text=data.decode('utf8',errors='replace')
  for label,pattern in patterns:
   if re.search(pattern,text):findings.append((str(p.relative_to(ROOT)),label))
  if p.name in ('responses.jsonl.gz','upstream-baselines.jsonl.gz'):
   for line in text.splitlines():
    row=json.loads(line);raw=row['raw'];encoded=raw.get('response_base64')
    if encoded:
     decoded=base64.b64decode(encoded,validate=True)
     if hashlib.sha256(decoded).hexdigest()!=raw['body_sha256']:raise ValueError('Raw response hash mismatch')
     try: payload=json.loads(decoded)
     except (ValueError,UnicodeError): payload=None
     def inspect_metadata(value):
      if isinstance(value,dict):
       for key,item in value.items():
        if key.lower() in {'user_id','account_id','email','api_key','authorization','password','secret','client_ip','ip_address'} and item not in (None,''):findings.append((str(p.relative_to(ROOT)),'private metadata field: '+key))
        inspect_metadata(item)
      elif isinstance(value,list):
       for item in value:inspect_metadata(item)
     inspect_metadata(payload)
     for label,pattern in patterns:
      if re.search(pattern,decoded.decode('utf8',errors='replace')):findings.append((str(p.relative_to(ROOT)),label+' in decoded response'))
 if findings:print(json.dumps(findings,indent=2));raise SystemExit(1)
 (ROOT/'releases/manifest.json').write_text(json.dumps({'format_version':1,'publication':'local prepared; no remote release URL yet','input_texts':'excluded; see ATTRIBUTION.md','files':files},indent=2)+'\n')
 print(json.dumps({'files':len(files),'secret_scan_findings':0}))
if __name__=='__main__':main()
