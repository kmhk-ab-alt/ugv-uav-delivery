"""Deterministic coordination benchmark; ground-truth fixture, NOT VLM accuracy."""
import argparse
import json
from .fleet import Fleet

def evaluate(mode, ground, uav, packages, seed, horizon=1200):
    f=Fleet(ground,uav,packages,mode,seed=seed,randomize=True)
    for _ in range(int(horizon/.04)):
        f.tick(.04)
        for j in f.jobs.values():
            if j.phase=='observe': f.decide(j.name,j.color,'privileged fixture; not AI inference')
        f.events.clear()
        if f.done: break
    return dict(mode=mode,ground=ground,uav=uav,packages=packages,seed=seed,
                perception='ground_truth_fixture',backend=f.motion.name,complete=f.done,
                mission_seconds=f.time,received=sum(j.receipt_confirmed for j in f.jobs.values()),
                on_time=sum(j.received_at is not None and j.received_at<=j.deadline for j in f.jobs.values()),
                airspace_warnings=f.warning_count,collision_holds=getattr(f.motion,'hold_count',0),
                obstacle_detours=getattr(f.motion,'detour_count',0))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.episodes<1: p.error('episodes must be positive')
    rows=[evaluate(mode,2,2,3,seed) for mode in ('relay','scout','joint') for seed in range(a.episodes)]
    from pathlib import Path
    Path(a.output).write_text(json.dumps(rows,indent=2))
    print(f'{len(rows)} coordination evaluations; NOT a VLM or physical-control benchmark')

if __name__=='__main__': main()
