import tempfile,unittest
from pathlib import Path
import numpy as np
from pai_ros.fleet import Fleet
from pai_ros.service_budget import configure,orders,summary,BudgetDispatch,BudgetNetwork

class BudgetTest(unittest.TestCase):
 def test_score_requires_correct_receipt_but_ignores_both_cutoffs(self):
  f=Fleet(package_count=5);configure(f,100)
  jobs=list(f.jobs.values())
  for j in jobs:j.priority=3;j.destination=j.color;j.deadline=90;j.receipt_confirmed=True;j.received_at=80
  jobs[1].received_at=95 # order late
  jobs[2].deadline=120;jobs[2].received_at=110 # operation late
  jobs[3].destination='wrong'
  jobs[4].receipt_confirmed=False
  self.assertEqual(summary(f)['score'],27);self.assertEqual(summary(f)['by_priority'][0]['scored'],3)
 def test_cutoff_leaves_queue_unstarted_but_finishes_active(self):
  f=Fleet(package_count=3);configure(f,1)
  for _ in range(5000):
   f.tick(.1)
   for j in f.jobs.values():
    if j.phase=='observe':f.decide(j.name,j.color)
   if f.done:break
  self.assertTrue(f.done);self.assertEqual(sum(j.phase=='queued' for j in f.jobs.values()),2)
  self.assertEqual(sum(j.phase=='delivered' for j in f.jobs.values()),1)
  self.assertTrue(all(a.task is None for a in f.agents.values()));self.assertEqual(summary(f)['score'],1)
 def test_policy_inputs_ignore_true_visual_labels(self):
  f=Fleet(2,2,8);configure(f,180);orders(f,1);d=BudgetDispatch('value')
  obs,rs=d.observation(f)
  for j in f.jobs.values():j.color='secret';j.address='hidden';j.contents='private'
  other,rr=d.observation(f)
  np.testing.assert_array_equal(obs['candidates'],other['candidates']);self.assertEqual(rs,rr)
  f.time=180;self.assertIsNone(d.select(f))
 def test_late_estimated_arrival_is_allowed_until_dispatch_cutoff(self):
  f=Fleet(2,2,12);configure(f,180);orders(f,1);f.time=179
  for policy in ('value','priority','fifo','random','untrained'):
   self.assertIsNotNone(BudgetDispatch(policy).select(f))
  f.time=180
  self.assertIsNone(BudgetDispatch('value').select(f))
 def test_order_deadlines_do_not_change_model_inputs_or_selection(self):
  f=Fleet(2,2,8);configure(f,180);orders(f,1);d=BudgetDispatch('untrained')
  before,rows=d.observation(f);selected=d.select(f)
  for j in f.jobs.values():j.deadline=-1000
  after,other=d.observation(f)
  np.testing.assert_array_equal(before['candidates'],after['candidates'])
  self.assertEqual(rows,other);self.assertEqual(selected,d.select(f))
 def test_checkpoint_roundtrip_and_schema_validation(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'model.npz';m=BudgetNetwork(7);m.save(p)
   n=BudgetNetwork.load(p)
   for k in m.params:np.testing.assert_array_equal(m.params[k],n.params[k])
   with np.load(p) as d:data={k:d[k] for k in d.files}
   data['features']=np.array(['wrong']);np.savez(p,**data)
   with self.assertRaises(ValueError):BudgetNetwork.load(p)
 def test_budget_validation_and_reproducible_orders(self):
  for bad in (0,-1,float('nan'),float('inf')):
   with self.assertRaises(ValueError):configure(Fleet(),bad)
  a=Fleet(package_count=12);b=Fleet(package_count=12)
  for f in (a,b):configure(f,180);orders(f,17)
  self.assertEqual([(j.priority,j.deadline,j.release) for j in a.jobs.values()],[(j.priority,j.deadline,j.release) for j in b.jobs.values()])
if __name__=='__main__':unittest.main()
