"""Reproducible held-out-city BC and support-only adaptation evaluation.
Map provided by default; ideal local-map option affects dispatch only. No SLAM.
Fast kinematic evaluation with oracle label fixture; Gazebo/VLM validated separately.
"""
import argparse,hashlib,json,math,random,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from .city import generate_city,LocalObstacleMap,configure_orders
from .fleet import Fleet
from .fleet_dispatch import FleetDispatch
from .dispatch_model import ScoringPolicy,Adam
from .live_scenario import LiveScenario
from .safety import EXTENTS,swept_conflict

def digest(model):
    h=hashlib.sha256()
    for k in sorted(model.params):h.update(model.params[k].tobytes())
    return h.hexdigest()

def run(city_seed,order_seed,policy='priority',model=None,mode='auto',family='train',collect=False,local_map=False,max_time=650):
    dispatch=FleetDispatch(policy,adaptive_modes=mode=='auto',seed=order_seed)
    if model is not None:dispatch.name='learned';dispatch.policy=model
    fleet=Fleet(2,2,6,'relay' if mode=='auto' else mode,seed=order_seed,randomize=True,dispatcher=dispatch)
    city=generate_city(city_seed,6,family);city.apply(fleet);configure_orders(fleet,order_seed)
    scenario=LiveScenario(fleet);mapper=LocalObstacleMap(fleet) if local_map else None
    examples=[];distance=0;contacts=0;active_contacts=set();events=[];errors=[]
    start_hash=digest(model) if model is not None else None
    for _ in range(int(max_time/.2)):
        scenario.tick(fleet)
        if mapper:mapper.tick(fleet)
        if collect:
            obs,actions=dispatch.observation(fleet)
            if actions:
                action=dispatch.policy.select(obs)
                full_obs,_=FleetDispatch('priority',adaptive_modes=True).observation(fleet)
                examples.append((full_obs,action))
        before={a.name:a.pos.copy() for a in fleet.agents.values()}
        try:fleet.tick(.2)
        except ValueError as e:errors.append(str(e));break
        for j in fleet.jobs.values():
            if j.phase=='observe' and j.destination is None:fleet.decide(j.name,j.color,'oracle label fixture, not VLM accuracy')
        hits=set();agents=list(fleet.agents.values())
        for i,a in enumerate(agents):
            distance+=math.dist(before[a.name],a.pos)
            for o in fleet.motion.obstacles:
                if o.intersects(before[a.name],a.pos,EXTENTS[a.kind]):hits.add(('static',a.name,o.name))
            for b in agents[i+1:]:
                if swept_conflict(before[a.name],a.pos,before[b.name],b.pos,a.kind,b.kind):hits.add(('agent',a.name,b.name))
        contacts+=len(hits-active_contacts);active_contacts=hits
        events.extend(fleet.events);fleet.events.clear()
        if fleet.done:break
    end_hash=digest(model) if model is not None else None
    if start_hash!=end_hash:raise RuntimeError('Evaluation modified model weights')
    jobs=list(fleet.jobs.values());on_time=lambda j:j.receipt_confirmed and j.received_at<=j.deadline
    delivered=sum(j.receipt_confirmed for j in jobs);weighted=sum(j.priority for j in jobs if on_time(j))
    late=sum(j.priority*max(0,(j.received_at if j.received_at is not None else fleet.time)-j.deadline) for j in jobs)
    report=dict(mode_counts={m:sum(e['event']=='assigned' and e.get('mode')==m for e in events) for m in ('relay','scout','joint')},city_seed=city_seed,order_seed=order_seed,family=family,geometry_sha256=city.manifest()['geometry_sha256'],policy=dispatch.name,mode=mode,
        delivered=delivered,total=len(jobs),on_time=sum(on_time(j) for j in jobs),weighted_on_time=weighted,priority_weighted_lateness=late,
        urgent_total=sum(j.priority==3 for j in jobs),urgent_on_time=sum(on_time(j) and j.priority==3 for j in jobs),
        distance_m=distance,makespan_s=fleet.time,contacts=contacts,complete=fleet.done,errors=errors,
        reward=20*weighted-.05*late-.01*distance-.05*fleet.time-100*sum(j.priority for j in jobs if not j.receipt_confirmed),
        control='kinematic',perception='oracle_label_fixture',map_access=fleet.metrics['city']['map_access'],checkpoint_weights_sha256=start_hash,weights_unchanged=start_hash==end_hash,
        discovered_obstacles=len(mapper.known) if mapper else None,disruptions=sum(e['event']=='disruption_started' for e in events))
    return report,examples

def train(samples,epochs=30,initial=None,seed=42):
    if not samples:raise ValueError('no demonstrations')
    model=initial.clone() if initial is not None else ScoringPolicy(seed);opt=Adam(model.params,.002);rng=np.random.default_rng(seed);loss=[]
    for _ in range(epochs):
        ls=[]
        for i in rng.permutation(len(samples)):
            obs,action=samples[i];l,g=model.imitation(obs,action);opt.step(model.params,g);ls.append(l)
        loss.append(float(np.mean(ls)))
    return model,loss

RULES=[(p,m) for p in ('priority','deadline') for m in ('relay','scout','joint','auto')]
def demonstrations(city_seed,order_seed,family='train',local_map=False):
    trials=[(run(city_seed,order_seed,p,mode=m,family=family,local_map=local_map)[0],p,m) for p,m in RULES]
    _,p,m=max(trials,key=lambda t:t[0]['reward'])
    return run(city_seed,order_seed,p,mode=m,family=family,collect=True,local_map=local_map)[1]

def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['condition'],r['policy'])].append(r)
    result=[]
    for (condition,policy),rs in groups.items():
        a=dict(condition=condition,policy=policy,episodes=len(rs),delivered=sum(r['delivered'] for r in rs),total=sum(r['total'] for r in rs),on_time=sum(r['on_time'] for r in rs),contacts=sum(r['contacts'] for r in rs),completed=sum(r['complete'] for r in rs))
        for key in ('reward','distance_m','makespan_s','priority_weighted_lateness'):a['mean_'+key]=float(np.mean([r[key] for r in rs]))
        a['reward_standard_error']=float(np.std([r['reward'] for r in rs],ddof=1)/math.sqrt(len(rs))) if len(rs)>1 else None
        result.append(a)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--train-cities',type=int,default=24);p.add_argument('--test-cities',type=int,default=30);p.add_argument('--epochs',type=int,default=30);p.add_argument('--adaptation-cities',type=int,default=6);p.add_argument('--split-offset',type=int,default=0)
    args=p.parse_args()
    if not (2<=args.train_cities<=100 and 2<=args.test_cities<=100 and 1<=args.epochs<=100 and 1<=args.adaptation_cities<=30):p.error('counts outside allowed range')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    split=dict(train=list(range(100,100+args.train_cities)),validation=list(range(10000,10005)),test=list(range(20000,20000+args.test_cities)),adaptation_support=list(range(30000,30000+args.adaptation_cities)),adaptation_query=list(range(40000,40000+args.test_cities)))
    split={k:[s+args.split_offset for s in values] for k,values in split.items()}
    # Order randomness is also disjoint; fixed-city ablation receives the same orders/sample budget.
    metadata=dict(splits=split,order_seed_offset=100000,map_access='provided',control='kinematic',perception='oracle_label_fixture',city_family='procedural_delivery_corridors_v1',
        split_offset=args.split_offset,feature_adapter='map_routes_v1',epochs=args.epochs,training='BC from best of 8 rule policies on training worlds only',adaptation='supervised BC on separate shifted support cities; not autonomous online RL or test-world adaptation',
        limits=['not arbitrary city topology','not SLAM','not VLM evaluation','contacts are geometric proxies, not Gazebo contact sensors'])
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2));models={};losses={}
    for name in ('fixed_city','diverse_cities'):
        samples=[]
        for s in split['train']:
            samples.extend(demonstrations(split['train'][0] if name=='fixed_city' else s,s+100000))
            print('demonstrations',name,s,len(samples),flush=True)
        model,loss=train(samples,args.epochs);models[name]=model;losses[name]=loss
        model.save(out/(name+'.npz'),dict(**metadata,training_condition=name,samples=len(samples)))
    initial=ScoringPolicy(42);rows=[];selection=[]
    for pol,mode in RULES:
        rs=[run(s,s+100000,pol,mode=mode)[0] for s in split['validation']]
        selection.append(dict(policy=pol,mode=mode,reward=float(np.mean([r['reward'] for r in rs]))))
    chosen=max(selection,key=lambda r:r['reward']);(out/'rule_selection.json').write_text(json.dumps(dict(candidates=selection,selected=chosen),indent=2))
    # Freeze selection before any test-world rollout.
    for condition,seeds,family,local in [('held_out_map_provided',split['test'],'train',False),('held_out_local_map_fixture',split['test'],'train',True),('shift_before_adaptation',split['adaptation_query'],'shift',False)]:
        for s in seeds:
            for name in ('selected_rule','random','untrained','fixed_city','diverse_cities'):
                model=models.get(name,initial if name=='untrained' else None)
                pol=chosen['policy'] if name=='selected_rule' else 'random' if name=='random' else 'priority';mode=chosen['mode'] if name=='selected_rule' else 'auto'
                r,_=run(s,s+100000,pol,model=model,mode=mode,family=family,local_map=local);r.update(policy=name,condition=condition);rows.append(r)
            print('evaluation',condition,s,flush=True)
    support=[]
    for s in split['adaptation_support']:support.extend(demonstrations(s,s+100000,family='shift'))
    adapted,loss=train(support,max(1,args.epochs//3),initial=models['diverse_cities'],seed=43);losses['support_adaptation']=loss
    adapted.save(out/'adapted.npz',dict(**metadata,training_condition='diverse_plus_shift_support',support_samples=len(support)))
    for s in split['adaptation_query']:
        r,_=run(s,s+100000,model=adapted,family='shift');r.update(policy='adapted_diverse_cities',condition='shift_after_support_adaptation');rows.append(r)
    # Geometry manifests allow split checks independent of the PRNG seed labels.
    cities={k:[generate_city(s,6,'shift' if k.startswith('adaptation') else 'train').manifest() for s in seeds] for k,seeds in split.items()}
    sets={k:{c['geometry_sha256'] for c in v} for k,v in cities.items()}
    keys=list(sets)
    if any(sets[a]&sets[b] for i,a in enumerate(keys) for b in keys[i+1:]):raise RuntimeError('geometry leakage between splits')
    (out/'cities.json').write_text(json.dumps(cities,indent=2))
    report=dict(protocol=metadata,rule_selection=chosen,losses=losses,summary=aggregate(rows),rows=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    from .city_report import save_analysis
    save_analysis(out/'report.json')
    print(json.dumps(report['summary'],indent=2),flush=True)
if __name__=='__main__':main()
