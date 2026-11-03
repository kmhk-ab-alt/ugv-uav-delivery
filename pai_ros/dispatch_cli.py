"""CPU-only baseline comparison. No external API, rendering, or learned weights."""
import argparse
from dataclasses import asdict
import json
import hashlib
import platform
from pathlib import Path
import time
from .dispatch_env import DispatchConfig,DispatchEnv,FEATURES
from .dispatch_policy import PriorityDeadlinePolicy,EarliestDeadlinePolicy,RandomPolicy


def run_episode(config,seed,policy,trace=None):
    env=DispatchEnv(config);obs,info=env.reset(seed=seed)
    elapsed=time.perf_counter();steps=0
    while True:
        action=policy.select(obs)
        nxt,reward,terminated,truncated,info=env.step(action)
        if trace:
            trace.write(json.dumps(dict(seed=seed,policy=policy.name,step=steps,
                observation={k:v.tolist() for k,v in obs.items()},action=action,
                action_decoded=env.decode(action),reward=reward,
                next_observation={k:v.tolist() for k,v in nxt.items()},
                terminated=terminated,truncated=truncated,events=info['events']))+'\n')
        obs=nxt;steps+=1
        if terminated or truncated:break
    return dict(policy=policy.name,seed=seed,steps=steps,wall_seconds=time.perf_counter()-elapsed,
        terminated=terminated,truncated=truncated,
        **{k:v for k,v in info.items() if k not in ('events','seed')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',choices=('all','priority','deadline','random','learned'),default='all')
    p.add_argument('--checkpoint',help='trained .npz checkpoint; implies --policy learned')
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--seed-start',type=int,default=1000,help='evaluation seed range; keep disjoint from training')
    p.add_argument('--ground-count',type=int,default=2);p.add_argument('--uav-count',type=int,default=2)
    p.add_argument('--orders',type=int,default=8);p.add_argument('--horizon',type=int,default=240)
    p.add_argument('--no-closures',action='store_true');p.add_argument('--output',required=True)
    p.add_argument('--trace-dir',help='optional baseline demonstration logs, NOT LeRobot format')
    a=p.parse_args()
    if a.episodes<1:p.error('episodes must be positive')
    config=DispatchConfig(a.ground_count,a.uav_count,a.orders,a.horizon,closures=not a.no_closures)
    learned=None
    if a.checkpoint:
        from .dispatch_model import ScoringPolicy
        learned=ScoringPolicy.load(a.checkpoint);a.policy='learned'
    elif a.policy=='learned':p.error('--policy learned requires --checkpoint')
    results=[];trace=None
    try:
        if a.trace_dir:
            directory=Path(a.trace_dir);directory.mkdir(parents=True,exist_ok=True)
            # Exclusive create prevents accidentally mixing or overwriting datasets.
            trace=(directory/'transitions.jsonl').open('x')
            (directory/'manifest.json').write_text(json.dumps(dict(schema='dispatch_transition_v1',
                config=asdict(config),features=FEATURES,seed_start=a.seed_start,episodes=a.episodes,
                source='scripted baselines, not human demonstrations or learned policies',
                policy_selection=a.policy,python=platform.python_version(),
                source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in ('dispatch_env.py','dispatch_policy.py','dispatch_cli.py')},
                environment='abstract_network_v1; not physical simulation'),indent=2))
        for seed in range(a.seed_start,a.seed_start+a.episodes):
            policies=dict(priority=PriorityDeadlinePolicy(),deadline=EarliestDeadlinePolicy(),random=RandomPolicy(seed))
            if learned is not None:policies['learned']=learned
            selected=policies.values() if a.policy=='all' else [policies[a.policy]]
            for policy in selected:
                row=run_episode(config,seed,policy,trace);results.append(row)
                print(f'{policy.name} seed={seed}: delivered={row["delivered"]}/{row["total"]}, '
                      f'on_time={row["on_time"]}, distance={row["distance"]:.0f}m, reward={row["reward"]:.2f}',flush=True)
    finally:
        if trace:trace.close()
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(config=asdict(config),environment='abstract_network_v1',
        trained=learned is not None,checkpoint=a.checkpoint,results=results),indent=2))

if __name__=='__main__':main()
