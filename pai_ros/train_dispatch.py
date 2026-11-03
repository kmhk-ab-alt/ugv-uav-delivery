"""CPU training: behavior cloning, then REINFORCE + value baseline and BC regularization.

Not PPO, ACT, a VLM fine-tune, or physical-controller learning.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
from .dispatch_env import DispatchConfig,DispatchEnv,FEATURES
from .dispatch_policy import PriorityDeadlinePolicy,EarliestDeadlinePolicy,RandomPolicy
from .dispatch_model import ScoringPolicy,Adam
from .dispatch_cli import run_episode

def evaluate(model,config,seeds):
    return [run_episode(config,seed,model) for seed in seeds]

def summary(rows):
    return dict(episodes=len(rows),mean_reward=float(np.mean([r['reward'] for r in rows])),
        delivery_rate=sum(r['delivered'] for r in rows)/sum(r['total'] for r in rows),
        on_time_rate=sum(r['on_time'] for r in rows)/sum(r['total'] for r in rows),
        mean_distance=float(np.mean([r['distance'] for r in rows])),
        priority_on_time={str(p):dict(
            on_time=sum(r['priority_metrics'][str(p)]['on_time'] for r in rows),
            total=sum(r['priority_metrics'][str(p)]['total'] for r in rows)) for p in (1,2,3)})

def train(output,bc_episodes=24,bc_epochs=30,rl_episodes=64,eval_episodes=10,seed=42):
    if min(bc_episodes,bc_epochs,eval_episodes)<1 or rl_episodes<0:raise ValueError('invalid training counts')
    # Freeze before test evaluation; all model selection uses validation only.
    bc_seeds=list(range(bc_episodes));rl_seeds=list(range(100,100+rl_episodes))
    val_seeds=list(range(500,505));test_seeds=list(range(1000,1000+eval_episodes))
    if (set(bc_seeds+rl_seeds)&set(val_seeds+test_seeds)) or set(val_seeds)&set(test_seeds):
        raise ValueError('training/validation/test seed overlap; reduce episode counts')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=DispatchConfig();rng=np.random.default_rng(seed);teacher=PriorityDeadlinePolicy()
    model=ScoringPolicy(seed);untrained=model.clone();started=time.perf_counter()
    metadata=dict(numpy=np.__version__,python=platform.python_version(),config=asdict(config),features=FEATURES,seed=seed,
        bc_seeds=bc_seeds,rl_seeds=rl_seeds,validation_seeds=val_seeds,test_seeds=test_seeds,
        teacher=teacher.name,algorithm='BC -> REINFORCE with learned linear value baseline + BC regularization',
        rl_discount=.99,rl_bc_coefficient=.1,hidden=32,backend='abstract_network_v1',
        source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
            for n in ('dispatch_env.py','dispatch_model.py','train_dispatch.py','dispatch_policy.py')})
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2))
    demonstrations=[]
    for s in bc_seeds:
        env=DispatchEnv(config);obs,_=env.reset(seed=s)
        while True:
            action=teacher.select(obs)
            if int(obs['action_mask'].sum())>1:demonstrations.append((obs,action))
            obs,_,t,x,_=env.step(action)
            if t or x:break
    print(f'BC: {len(demonstrations)} decision samples from {bc_episodes} episodes',flush=True)
    opt=Adam(model.params,lr=.003);history=[]
    for epoch in range(bc_epochs):
        losses=[]
        for i in rng.permutation(len(demonstrations)):
            obs,action=demonstrations[i];loss,grad=model.imitation(obs,action)
            opt.step(model.params,grad);losses.append(loss)
        history.append(dict(stage='bc',epoch=epoch+1,loss=float(np.mean(losses))))
        if epoch==0 or (epoch+1)%10==0:print(f'BC epoch {epoch+1}: loss={np.mean(losses):.4f}',flush=True)
    untrained.save(output/'untrained.npz',dict(stage='untrained',**metadata))
    bc=model.clone();bc.save(output/'bc.npz',dict(stage='bc',**metadata))
    best=bc.clone();best_score=summary(evaluate(best,config,val_seeds))['mean_reward'];best_stage='bc'
    opt=Adam(model.params,lr=.0005);critic=np.zeros(7)
    for episode,s in enumerate(rl_seeds):
        env=DispatchEnv(config);obs,_=env.reset(seed=s);trajectory=[]
        while True:
            decision=int(obs['action_mask'].sum())>1
            action=model.sample(obs,rng) if decision else 0
            nxt,reward,t,x,_=env.step(action)
            trajectory.append((obs,action,reward,decision));obs=nxt
            if t or x:break
        returns=[];value=0.
        for _,_,reward,_ in reversed(trajectory):value=reward+.99*value;returns.append(value)
        returns.reverse()
        grads={k:np.zeros_like(v) for k,v in model.params.items()};count=0
        for (obs,action,_,decision),ret in zip(trajectory,returns):
            if not decision:continue
            vf=np.append(obs['global_state'].astype(float),1.)
            advantage=float(np.clip(ret/100.-critic@vf,-3.,3.))
            critic+=.01*advantage*vf
            pg=model.reinforce(obs,action,advantage)
            _,anchor=model.imitation(obs,teacher.select(obs))
            for k in grads:grads[k]+=pg[k]+.1*anchor[k]
            count+=1
        if count:opt.step(model.params,{k:g/count for k,g in grads.items()})
        history.append(dict(stage='rl',episode=episode+1,seed=s,reward=env.total_reward))
        if (episode+1)%16==0 or episode+1==rl_episodes:
            score=summary(evaluate(model,config,val_seeds))['mean_reward']
            history[-1]['validation_mean_reward']=score
            if score>best_score:best=model.clone();best_score=score;best_stage=f'rl_{episode+1}'
            print(f'RL episode {episode+1}: validation reward={score:.2f}, best={best_stage}',flush=True)
    model.save(output/'rl_last.npz',dict(stage='rl_last',**metadata))
    best.save(output/'best.npz',dict(stage=best_stage,validation_mean_reward=best_score,**metadata))
    # Test metrics are computed after choosing best.npz, never used above.
    results={}
    for name,policy in [('untrained',untrained),('priority_rule',teacher),('deadline_rule',EarliestDeadlinePolicy()),
                        ('bc',bc),('rl_last',model),('selected',best)]:
        rows=evaluate(policy,config,test_seeds);results[name]=dict(summary=summary(rows),episodes=rows)
        print('TEST '+name+': '+json.dumps(results[name]['summary']),flush=True)
    random_rows=[run_episode(config,s,RandomPolicy(s)) for s in test_seeds]
    results['random']=dict(summary=summary(random_rows),episodes=random_rows)
    report=dict(training_seconds=time.perf_counter()-started,selected_stage=best_stage,
                validation_mean_reward=best_score,history=history,test=results)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(f'checkpoint: {output/"best.npz"}',flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--bc-episodes',type=int,default=24);p.add_argument('--bc-epochs',type=int,default=30)
    p.add_argument('--rl-episodes',type=int,default=64);p.add_argument('--eval-episodes',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();train(a.output_dir,a.bc_episodes,a.bc_epochs,a.rl_episodes,a.eval_episodes,a.seed)

if __name__=='__main__':main()
