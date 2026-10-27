"""Re-run the original working-set policy search in a new directory (no API).

Uses only the working thousand. Optional --nested repeats family selection in
outer folds and takes longer. Published evidence remains unchanged.
"""
import argparse,shutil,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--nested',action='store_true');a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copy2(ROOT/'experiments/proofwriter/working-data.json',out/'data.json')
for name in ['calibration.py','policies.py']:shutil.copy2(ROOT/'decision_lab'/name,out/name)
for path in (ROOT/'tools/study').glob('*.py'):shutil.copy2(path,out/path.name)
steps=['describe.py','analyze.py','verify_results.py']+(['nested_select.py','verify_nested.py'] if a.nested else [])
env=dict(os.environ);env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
for step in steps:subprocess.run([sys.executable,str(out/step)],cwd=out,check=True,env=env)
