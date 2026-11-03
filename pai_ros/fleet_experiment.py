"""Native 3D-task rollout comparisons and BC training (no renderer or ROS).
Perception is an explicit oracle fixture, not an evaluation of VLM accuracy.
Training/validation/test seeds are disjoint. Gazebo physics validation is separate.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np
from .fleet import Fleet
from .fleet_dispatch import FleetDispatch
from .dispatch_model import ScoringPolicy,Adam
from .live_scenario import LiveScenario
from .safety import Obstacle

def run(seed,policy='priority',checkpoint='',model=None,collect=False,mode='auto'):
    dispatch=FleetDispatch(policy if model is None else 'priority',checkpoint,mode=='auto',seed)
    if model is not None:dispatch.policy=model;dispatch.name='learned'
    fleet=Fleet(2,2,6,'relay' if mode=='auto' else mode,seed=seed,randomize=True,dispatcher=dispatch)
    rng=random.Random(seed)
    for i,j in enumerate(fleet.jobs.values()):
        j.priority=rng.randint(1,3);j.deadline=60+i*16+rng.randint(0,55);j.release=0 if i<2 else rng.randint(15,55)
    scenario=LiveScenario(fleet)
    fleet.motion.set_obstacles([Obstacle('trolley',(1.3,-.2,.04),(1.8,.2,.60))])
    demos=[];distance=0;decisions=[];last={a.name:a.pos.copy() for a in fleet.agents.values()};events=[]
    for _ in range(10000):
        scenario.tick(fleet)
        if collect:
            obs,actions=dispatch.observation(fleet)
            if actions:
                action=dispatch.policy.select(obs)
                # Supervise the chosen mode against every legal alternative,
                # rather than masking all competing modes out of the loss.
                full_obs,_=FleetDispatch('priority',adaptive_modes=True).observation(fleet)
                demos.append((full_obs,action))
        fleet.tick(.1)
        for j in fleet.jobs.values():
            if j.phase=='observe' and j.destination is None:fleet.decide(j.name,j.color,'oracle fixture for dispatch experiment')
        for a in fleet.agents.values():distance+=__import__('math').dist(a.pos,last[a.name]);last[a.name]=a.pos.copy()
        events.extend(fleet.events);fleet.events.clear()
        if fleet.done:break
    ontime=sum(j.received_at is not None and j.received_at<=j.deadline for j in fleet.jobs.values())
    delivered=sum(j.receipt_confirmed for j in fleet.jobs.values())
    weighted=sum(j.priority for j in fleet.jobs.values() if j.received_at is not None and j.received_at<=j.deadline)
    late=sum(j.priority*max(0,(j.received_at or fleet.time)-j.deadline) for j in fleet.jobs.values())
    report=dict(seed=seed,policy=dispatch.name,delivered=delivered,total=len(fleet.jobs),on_time=ontime,weighted_on_time=weighted,distance=distance,makespan=fleet.time,
        reward=20*weighted-.05*late-.01*distance-.05*fleet.time-100*sum(j.priority for j in fleet.jobs.values() if not j.receipt_confirmed),control='kinematic_native_3d',perception='oracle_fixture',
        detours=fleet.motion.detour_count,holds=fleet.motion.hold_count,disruptions=sum(e['event']=='disruption_started' for e in events),
        urgent_total=sum(j.priority==3 for j in fleet.jobs.values()),urgent_on_time=sum(j.priority==3 and j.received_at is not None and j.received_at<=j.deadline for j in fleet.jobs.values()))
    return report,demos

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--train-episodes',type=int,default=24);p.add_argument('--epochs',type=int,default=25);p.add_argument('--eval-episodes',type=int,default=10)
    args=p.parse_args()
    if min(args.train_episodes,args.epochs,args.eval_episodes)<1 or args.train_episodes>100 or args.eval_episodes>100:p.error('episode counts must be 1..100')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    seeds=dict(train=list(range(args.train_episodes)),validation=list(range(600,605)),test=list(range(3000,3000+args.eval_episodes)))
    model=ScoringPolicy(42);initial=model.clone();samples=[]
    for s in seeds['train']:
        candidates=[(run(s,pol,mode=mode)[0]['reward'],pol,mode) for pol in ('priority','deadline') for mode in ('relay','scout','joint')]
        _,teacher,mode=max(candidates)
        _,d=run(s,teacher,collect=True,mode=mode);samples.extend(d)
    print(f'Native BC samples: {len(samples)}',flush=True)
    opt=Adam(model.params,.002);rng=np.random.default_rng(42);history=[]
    for epoch in range(args.epochs):
        losses=[]
        for i in rng.permutation(len(samples)):
            obs,action=samples[i];loss,grad=model.imitation(obs,action);opt.step(model.params,grad);losses.append(loss)
        history.append(float(np.mean(losses)))
    metadata=dict(adapter='fleet_3d_v1',training_control='kinematic_native_3d',teacher='best of priority/deadline x relay/scout/joint on each training seed',seeds=seeds,perception='oracle_fixture')
    model.save(out/'bc.npz',metadata)
    # Model selection uses validation; final test is read only afterwards.
    scores=[]
    for candidate in (initial,model):scores.append(float(np.mean([run(s,model=candidate)[0]['reward'] for s in seeds['validation']])))
    chosen=model;chosen.save(out/'best.npz',dict(**metadata,validation_scores=scores,selected='bc',note='Untrained model is a comparison, not a learned checkpoint candidate'))
    rows=[]
    for s in seeds['test']:
        for name in ('priority','deadline','random','untrained','learned'):
            r,_=run(s,model=chosen if name=='learned' else initial if name=='untrained' else None,policy=name if name in ('priority','deadline','random') else 'priority')
            r['policy']=name;rows.append(r)
            print(f"{name} seed={s}: on_time={r['on_time']}/{r['total']} reward={r['reward']:.2f}",flush=True)
    (out/'report.json').write_text(json.dumps(dict(metadata=metadata,bc_loss=history,validation_scores=scores,rows=rows),indent=2))
if __name__=='__main__':main()
