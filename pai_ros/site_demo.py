"""CPU training and causal smoke test for image-gated cooperative dispatch; no baseline comparison."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from .fleet import Fleet,PRIORITY_TOKENS
from .city import generate_city
from .rendezvous import configure as meeting
from .service_budget import BudgetDispatch,BudgetNetwork,configure,orders
from .site_perception import SiteSurvey
from .dispatch_model import Adam

def episode(seed=75100,policy='learned',model=None,collect=False,ground_count=2,uav_count=2):
    f=Fleet(ground_count,uav_count,12,dispatcher=BudgetDispatch(policy,model=model));f.configure_speeds()
    generate_city(seed,12,layout='blocks').apply(f);meeting(f);configure(f,180);orders(f,seed+100000)
    survey=SiteSurvey(f,'fixture');f.site_survey=survey;events=[];samples=[]
    try:
        for _ in range(12000):
            while not survey.poll():time.sleep(.001)
            if collect:
                obs,records=f.dispatcher.observation(f)
                if records:samples.append((obs,f.dispatcher.choose(obs,records)))
            f.tick(.1)
            for j in f.jobs.values():
                if j.phase=='observe' and j.destination is None:f.decide(j.name,PRIORITY_TOKENS[j.priority],'importance image fixture')
            events.extend(f.events);f.events.clear()
            if f.done:break
        return f,events,samples
    finally:survey.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--episodes',type=int,default=12);p.add_argument('--epochs',type=int,default=35);p.add_argument('--checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--ground-count',type=int,default=2);p.add_argument('--uav-count',type=int,default=2)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.train:
        samples=[]
        for seed in range(85100,85100+a.episodes):
            _,_,items=episode(seed,'value',collect=True);samples+=items
            print('teacher episode',seed,'samples',len(samples),flush=True)
        model=BudgetNetwork(42);opt=Adam(model.params);rng=np.random.default_rng(98)
        for epoch in range(a.epochs):
            losses=[]
            for i in rng.permutation(len(samples)):
                obs,action=samples[i];loss,gradient=model.imitation(obs,action);opt.step(model.params,gradient);losses.append(loss)
            if epoch%10==0:print('BC epoch',epoch+1,'loss',round(float(np.mean(losses)),4),flush=True)
        model.save(out/'best.npz',dict(layout='numbered_mixed_sites_v1',source='site images fixture -> candidate mask -> teacher value/time -> behavior cloning',episodes=a.episodes,epochs=a.epochs))
    else:model=BudgetNetwork.load(a.checkpoint)
    f,events,_=episode(75100,'learned',model,ground_count=a.ground_count,uav_count=a.uav_count)
    assigned=[e for e in events if e.get('event')=='assigned'];replans=[e for e in events if e.get('event')=='meeting_selected' and e.get('revision',0)>1]
    assert f.done and assigned and f.city.pickup_index(f.jobs[assigned[0]['package']].source)==1,'Image closure did not redirect initial pickup to B'
    assert replans,'Image closure did not change a meeting point'
    assert all(f.correct_delivery(j) for j in f.jobs.values() if j.receipt_confirmed)
    evidence=dict(perception='synthetic camera fixture, not a real VLM/Gazebo benchmark',model='trained candidate-scoring MLP',checkpoint_sha256=hashlib.sha256((out/'best.npz' if a.train else Path(a.checkpoint)).read_bytes()).hexdigest(),initial_pickup='B',meeting_replanned=bool(replans),meeting_site_changed=any(e.get('site_changed',False) for e in replans),returned=f.done,received=sum(j.receipt_confirmed for j in f.jobs.values()),events=events,final=f.snapshot())
    (out/'flow_verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('events','final')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
