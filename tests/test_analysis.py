import unittest
from decision_lab.analysis import apply,metrics,compare
class AnalysisTests(unittest.TestCase):
 def setUp(self):
  self.rows=[{'id':'a','gold':'yes','models':{'a':{'pred':'no','confidence':.2,'cost':1},'b':{'pred':'yes','confidence':.9,'cost':2}}},{'id':'b','gold':'no','models':{'a':{'pred':'no','confidence':.95,'cost':1},'b':{'pred':'yes','confidence':.8,'cost':2}}}]
 def test_conditional_calls_and_referral(self):
  d,c=apply(self.rows,{'kind':'cascade','models':['a','b'],'thresholds':[.9,.95]})
  self.assertEqual(d,[None,'no']);self.assertEqual(c,[['a','b'],['a']]);r=metrics(self.rows,d,c,4)
  self.assertEqual((r['accuracy'],r['coverage'],r['cost_usd'],r['total_with_referral_usd']),(1,.5,4,8))
 def test_agreement_does_not_claim_referrals_correct(self):
  d,c=apply(self.rows,{'kind':'agreement','models':['a','b']});r=metrics(self.rows,d,c)
  self.assertEqual(r['accepted'],0);self.assertIsNone(r['accuracy'])
 def test_unknown_cost(self):
  self.rows[0]['models']['a']['cost']=None
  r=metrics(self.rows,['no','no'],[['a'],['a']]);self.assertIsNone(r['cost_usd']);self.assertEqual(r['known_cost_usd'],1)
 def test_no_invented_confidence(self):
  self.rows[0]['models']['a']['confidence']=None
  with self.assertRaises(ValueError):apply(self.rows,{'kind':'single','models':['a']})
 def test_changed_gold_rejected(self):
  import copy
  after=copy.deepcopy(self.rows);after[0]['gold']='no'
  with self.assertRaises(ValueError):compare(self.rows,after)

class JournalIntegrationTests(unittest.TestCase):
 def test_unknown_usage_and_language_identity(self):
  import tempfile,copy,json
  from pathlib import Path
  from test_runner import task,FakeTransport
  from decision_benchmark.encoding import encode,sha256
  from decision_benchmark.runner.journal import RunJournal
  from decision_benchmark.runner.execution import run
  from decision_lab.export import export_run
  from decision_benchmark.runner.report import report_run
  a=task();b=copy.deepcopy(a);b['case']['language']='fr';b['key']['language']='fr';b['id']=sha256(encode(b['key']))
  class MissingUsage(FakeTransport):
   def send(self,body):
    r=super().send(body);data=json.loads(r['response_text']);data['usage']='unavailable';r['response_text']=json.dumps(data);return r
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'run';RunJournal(p).create_plan({'schema_version':1,'tasks':[a,b],'provider_limits':{'shared':1}})
   self.assertEqual(run(p,MissingUsage())['status'],'complete')
   rows=export_run(p);self.assertEqual(len(rows),2);self.assertEqual(len({r['id'] for r in rows}),2)
   self.assertTrue(all(r['models']['m']['cost'] is None for r in rows));self.assertEqual(report_run(p)['unknown_cost_attempts'],2)
 def test_rejected_billed_attempt_remains_in_total(self):
  import tempfile,json
  from pathlib import Path
  from test_runner import task,FakeTransport
  from decision_benchmark.runner.journal import RunJournal
  from decision_benchmark.runner.execution import run,resolve_attempt
  from decision_benchmark.runner.report import report_run
  class BadVersion(FakeTransport):
   def send(self,body):
    r=super().send(body);d=json.loads(r['response_text']);d['model']='other';r['response_text']=json.dumps(d);return r
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'run';t=task();RunJournal(p).create_plan({'schema_version':1,'tasks':[t],'provider_limits':{'shared':1}})
   run(p,BadVersion());resolve_attempt(p,t['id'],'retry','failed');run(p,FakeTransport());r=report_run(p)
   self.assertEqual(r['all_attempts_cost_usd'],.002);self.assertEqual(r['accepted_reported_cost_usd'],.001)
 def test_cli_incomplete_is_nonzero(self):
  from unittest.mock import patch
  from decision_lab.cli import main
  with patch('sys.argv',['decision-lab','run','--run-dir','dummy','--send']),patch.dict('os.environ',{'OPENROUTER_API_KEY':'fake'}),patch('decision_benchmark.runner.execution.run',return_value={'status':'incomplete'}):
   with self.assertRaises(SystemExit) as error:main()
   self.assertEqual(error.exception.code,2)
