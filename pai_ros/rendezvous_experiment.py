"""Train rendezvous-site BC; compare fixed sites, ETA rule and frozen learned sites.
Kinematic rollouts with explicit inspection oracle; not a VLM/physical benchmark.
"""
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from .fleet import Fleet
from .city import generate_city,configure_orders
from .live_scenario import LiveScenario
from .rendezvous import configure,MeetingNetwork
from .dispatch_model import Adam
from .generalization import digest

def run(seed,policy='eta',model=None,collect=False,packages=6):
    f=Fleet(2,2,packages,'relay',seed=seed+100000,randomize=True);generate_city(seed,packages).apply(f);configure_orders(f,seed+100000)
    configure(f,policy,model=model);f.meeting.collect=collect;sc=LiveScenario(f)
    distance=0;events=[];before_hash=digest(model) if model is not None else None;errors=[]
    for _ in range(3250):
        sc.tick(f);before={a.name:a.pos.copy() for a in f.agents.values()}
        try:f.tick(.2)
        except ValueError as e:errors.append(str(e));break
        for j in f.jobs.values():
            if j.phase=='observe' and j.destination is None:f.decide(j.name,j.color,'inspection oracle fixture')
        distance+=sum(math.dist(a.pos,before[a.name]) for a in f.agents.values())
        events.extend(f.events);f.events.clear()
        if f.done:break
    if before_hash!=(digest(model) if model is not None else None):raise RuntimeError('evaluation changed weights')
    jobs=list(f.jobs.values());delivered=sum(j.receipt_confirmed for j in jobs);ontime=sum(j.receipt_confirmed and j.received_at<=j.deadline for j in jobs)
    wait=sum(j.meeting_wait_s for j in jobs)
    result=dict(city_seed=seed,policy=policy,delivered=delivered,total=len(jobs),on_time=ontime,complete=f.done,meeting_wait_s=wait,distance_m=distance,makespan_s=f.time,
        handoffs=sum(e['event']=='handoff_complete' for e in events),meeting_replans=sum(e['event']=='meeting_selected' and e['revision']>1 for e in events),
        selected_sites=dict(Counter(e['site'] for e in events if e['event']=='meeting_selected')),geometry_sha256=f.city_manifest['geometry_sha256'],checkpoint_weights_sha256=before_hash,
        control='kinematic',perception='inspection_oracle_fixture',errors=errors)
    return result,f.meeting.examples

def train(samples,epochs):
    model=MeetingNetwork(42);opt=Adam(model.params,.002);rng=np.random.default_rng(42);history=[]
    for _ in range(epochs):
        losses=[]
        for i in rng.permutation(len(samples)):
            obs,a=samples[i];l,g=model.imitation(obs,a);opt.step(model.params,g);losses.append(l)
        history.append(float(np.mean(losses)))
    return model,history

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--train-cities',type=int,default=24);p.add_argument('--test-cities',type=int,default=20);p.add_argument('--epochs',type=int,default=50);p.add_argument('--split-offset',type=int,default=0)
    args=p.parse_args()
    if not (2<=args.train_cities<=100 and 2<=args.test_cities<=100 and 1<=args.epochs<=200):p.error('invalid episode/epoch count')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(train=list(range(70000,70000+args.train_cities)),validation=list(range(71000,71005)),test=list(range(72000,72000+args.test_cities)),teacher='minimum receipt ETA + .25 waiting + .03 distance + priority-weighted lateness penalty',
        perception='inspection oracle; destination available only after inspection',control='kinematic',training='BC for meeting site only; dispatch is fixed priority',scenario='KDDI-inspired ground/UAV relay without a road vehicle',epochs=args.epochs)
    for split in ('train','validation','test'):protocol[split]=[s+args.split_offset for s in protocol[split]]
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));samples=[]
    for s in protocol['train']:
        _,examples=run(s,collect=True);samples.extend(examples);print('teacher',s,'samples',len(samples),flush=True)
    model,loss=train(samples,args.epochs);metadata=dict(protocol=protocol,samples=len(samples),loss=loss)
    model.save(out/'best.npz',metadata);initial=MeetingNetwork(42)
    validation=[run(s,'learned',model=model)[0] for s in protocol['validation']]
    (out/'validation.json').write_text(json.dumps(validation,indent=2))
    rows=[]
    for s in protocol['test']:
        for name,m in [('fixed',None),('eta',None),('untrained',initial),('learned',model)]:
            r,_=run(s,'learned' if m is not None else name,model=m);r['policy']=name;rows.append(r)
            print(s,name,r['delivered'],r['on_time'],round(r['meeting_wait_s'],1),flush=True)
    summary=[]
    for pol in ('fixed','eta','untrained','learned'):
        rs=[r for r in rows if r['policy']==pol]
        summary.append(dict(policy=pol,episodes=len(rs),delivered=sum(r['delivered'] for r in rs),total=sum(r['total'] for r in rs),on_time=sum(r['on_time'] for r in rs),completed=sum(r['complete'] for r in rs),
            mean_wait_s=float(np.mean([r['meeting_wait_s'] for r in rs])),mean_distance_m=float(np.mean([r['distance_m'] for r in rs])),mean_makespan_s=float(np.mean([r['makespan_s'] for r in rs]))))
    (out/'report.json').write_text(json.dumps(dict(protocol=protocol,loss=loss,summary=summary,rows=rows),indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
