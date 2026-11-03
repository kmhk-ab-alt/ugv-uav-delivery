"""Frozen dispatch BC versus matched baselines on unseen time-bounded deliveries.
Kinematic / inspection oracle benchmark, separate from Gazebo validation.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from .fleet import Fleet
from .city import generate_city
from .rendezvous import configure as meeting
from .service_budget import BudgetDispatch,BudgetNetwork,configure,orders,summary
from .dispatch_model import Adam

def run(seed,policy='value',model=None,collect=False,packages=12,limit=180.,layout='lanes'):
    d=BudgetDispatch(policy,model=model,seed=seed)
    f=Fleet(2,2,packages,'relay',seed=seed+100000,randomize=True,dispatcher=d)
    generate_city(seed,packages,layout=layout).apply(f);meeting(f,'eta');configure(f,limit);orders(f,seed+100000)
    samples=[];events=[];distance=0.;initial={k:v.copy() for k,v in model.params.items()} if model else None
    for _ in range(int((limit+500)/.25)):
        if collect:
            obs,rs=d.observation(f)
            if rs:samples.append((obs,d.choose(obs,rs)))
        before={a.name:a.pos.copy() for a in f.agents.values()}
        f.tick(.25)
        for j in f.jobs.values():
            if j.phase=='observe' and j.destination is None:f.decide(j.name,__import__('pai_ros.fleet',fromlist=['PRIORITY_TOKENS']).PRIORITY_TOKENS[j.priority] if j.numbered_destination else j.color,'inspection oracle fixture')
        distance+=sum(math.dist(a.pos,before[a.name]) for a in f.agents.values())
        events.extend(f.events);f.events.clear()
        if f.done:break
    if model:assert all(np.array_equal(initial[k],v) for k,v in model.params.items())
    s=summary(f)
    return dict(seed=seed,policy=policy,**s,all_returned=f.done,distance_m=distance,end_s=f.time,orders=[dict(name=j.name,priority=j.priority,release=j.release,deadline=j.deadline,received_at=j.received_at,phase=j.phase) for j in f.jobs.values()],assignments=[e for e in events if e['event']=='assigned'],geometry_sha256=f.city_manifest['geometry_sha256']),samples

def main():
    p=argparse.ArgumentParser();p.add_argument('--city-layout',choices=('lanes','blocks'),default='lanes');p.add_argument('--output-dir',required=True);p.add_argument('--train-episodes',type=int,default=24);p.add_argument('--eval-episodes',type=int,default=12);p.add_argument('--epochs',type=int,default=35);p.add_argument('--packages',type=int,default=12);p.add_argument('--limit',type=float,default=180.)
    a=p.parse_args()
    if min(a.train_episodes,a.eval_episodes,a.epochs,a.packages)<1 or not math.isfinite(a.limit) or a.limit<=0:p.error('positive finite arguments required')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(layout=a.city_layout,train=list(range(81000,81000+a.train_episodes)),validation=list(range(82000,82004)),test=list(range(83000,83000+a.eval_episodes)),packages=a.packages,limit_s=a.limit,teacher='feasible value / estimated full cycle time',objective='correct confirmed receipt regardless of time; operation cutoff stops new assignments; priority 1/2/3 = 1/3/9 points',meeting='eta rule shared by all arms',perception='inspection oracle fixture',control='kinematic; not a Gazebo benchmark',training='behavior cloning of dispatch only; no test fitting')
    samples=[]
    for seed in protocol['train']:
        _,ss=run(seed,collect=True,packages=a.packages,limit=a.limit,layout=a.city_layout);samples+=ss;print('teacher',seed,len(samples),flush=True)
    model=BudgetNetwork(42);initial=BudgetNetwork(42);opt=Adam(model.params,.002);rng=np.random.default_rng(42);loss=[]
    for ep in range(a.epochs):
        ls=[]
        for i in rng.permutation(len(samples)):
            obs,action=samples[i];l,g=model.imitation(obs,action);opt.step(model.params,g);ls.append(l)
        loss.append(float(np.mean(ls)));print('epoch',ep+1,loss[-1],flush=True)
    model.save(out/'best.npz',dict(protocol=protocol,samples=len(samples),epochs=a.epochs,loss=loss))
    validation=[run(seed,'learned',model,packages=a.packages,limit=a.limit,layout=a.city_layout)[0] for seed in protocol['validation']]
    rows=[]
    for seed in protocol['test']:
        for name,m in [('fifo',None),('priority',None),('value',None),('untrained',initial),('learned',model)]:
            r,_=run(seed,'learned' if m else name,m,packages=a.packages,limit=a.limit,layout=a.city_layout);r['policy']=name;rows.append(r);print(seed,name,r['score'],r['scored'],flush=True)
    summaries=[]
    for name in ('fifo','priority','value','untrained','learned'):
        rs=[r for r in rows if r['policy']==name]
        summaries.append(dict(policy=name,episodes=len(rs),mean_score=float(np.mean([r['score'] for r in rs])),scored=sum(r['scored'] for r in rs),urgent_scored=sum(r['by_priority'][0]['scored'] for r in rs),urgent_total=sum(r['by_priority'][0]['total'] for r in rs),total=sum(sum(b['total'] for b in r['by_priority']) for r in rs),mean_distance_m=float(np.mean([r['distance_m'] for r in rs]))))
    report=dict(protocol=protocol,checkpoint_sha256=hashlib.sha256((out/'best.npz').read_bytes()).hexdigest(),loss=loss,validation=validation,summary=summaries,rows=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'benchmark.json').write_text(json.dumps({k:report[k] for k in ('protocol','checkpoint_sha256','summary')},indent=2));print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
