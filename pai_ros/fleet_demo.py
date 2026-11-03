"""Standalone smoke runner for machines without ROS 2 (same coordination core)."""
import argparse
import json
import time
from pathlib import Path
from .fleet import Fleet, MODES, COLORS
from .fleet_world import FleetWorld
from .recording import EpisodeRecorder
from .scheduling import parse_priorities
from .cargo_vlm import offline_decide, vlm_decide, api_key_for, DEFAULT_MODEL, DEFAULT_BASE_URL

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ground-count', type=int, default=1)
    ap.add_argument('--uav-count', type=int, default=1)
    ap.add_argument('--package-count', type=int, default=3)
    ap.add_argument('--mode', choices=MODES, default='relay')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--stub', action='store_true', help='explicit pixel fixture, no VLM/rendering')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--base-url', default=DEFAULT_BASE_URL)
    ap.add_argument('--max-time', type=float, default=600.)
    ap.add_argument('--report')
    ap.add_argument('--priorities',default='',help='one priority per parcel, e.g. 1,3,2; 3 is highest')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--randomize',action='store_true')
    ap.add_argument('--record-dir',help='optional versioned observation/action/image logs')
    ap.add_argument('--completion-hold', type=float, default=5., help='seconds to show completed scene')
    args = ap.parse_args()
    api_key = None if args.stub else api_key_for(args.base_url)
    import os
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    fleet = Fleet(args.ground_count, args.uav_count, args.package_count, args.mode, seed=args.seed, randomize=args.randomize,priorities=parse_priorities(args.priorities,args.package_count))
    world = FleetWorld(fleet, args.headless, render=not args.stub)
    config={k:getattr(args,k) for k in ('ground_count','uav_count','package_count','mode','stub','model','base_url','seed','randomize','priorities')}
    recorder=EpisodeRecorder(args.record_dir,config) if args.record_dir else None
    if recorder: print('Episode log: '+str(recorder.path),flush=True)
    pool = ThreadPoolExecutor(max_workers=1)
    pending, decisions, events, errors = {}, set(), [], []
    wall_start = time.monotonic()
    try:
        while not fleet.done and fleet.time < args.max_time and time.monotonic()-wall_start < args.max_time:
            tick_start = time.monotonic()
            if world.viewer and not world.viewer.is_running():
                break
            before=fleet.observation()
            fleet.tick(.02)
            if recorder: recorder.step(fleet,before)
            world.sync()
            for j in fleet.jobs.values():
                if j.phase == 'observe' and j.name not in decisions and j.name not in pending:
                    if args.stub:
                        image = np.full((32,32,3), np.array(COLORS[j.color])*255, dtype=np.uint8)
                        if recorder: recorder.image(fleet,j.name,image)
                        pending[j.name] = pool.submit(offline_decide, image)
                    else:
                        image = world.image(j.name)
                        if recorder: recorder.image(fleet,j.name,image)
                        pending[j.name] = pool.submit(vlm_decide, image,
                            [x.strip() for x in args.model.split(',') if x.strip()], args.base_url,
                            api_key)
            for name, future in list(pending.items()):
                if future.done():
                    try:
                        destination, reason = future.result()
                        fleet.decide(name, destination, reason)
                        decisions.add(name)
                    except Exception as exc:
                        errors.append(dict(package=name, error=str(exc)))
                    del pending[name]
            for event in fleet.events:
                if event['event'] in ('assigned', 'decision', 'handoff_complete', 'receipt_confirmed', 'agents_available', 'simulated_flight_authorized', 'airspace_warning', 'airspace_clear', 'collision_hold', 'obstacle_detour'):
                    print('[Delivery] ' + json.dumps(event), flush=True)
            events.extend(fleet.events)
            fleet.events.clear()
            if errors:
                break
            if world.viewer or (pending and not args.stub):
                time.sleep(max(0., .02 - (time.monotonic()-tick_start)))
            elif pending:
                time.sleep(.0001)
        if fleet.done and world.viewer:
            until = time.monotonic() + max(0., args.completion_hold)
            while world.viewer.is_running() and time.monotonic() < until:
                world.sync()
                time.sleep(.05)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        world.close()
        if recorder: recorder.close(fleet,errors)
    report = fleet.snapshot()
    report.update(success=fleet.done and all(j.destination==j.color for j in fleet.jobs.values()),
                  stub=args.stub, errors=errors, events=events)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(mode=args.mode, ground=args.ground_count, uav=args.uav_count,
                         delivered=sum(j.phase=='delivered' for j in fleet.jobs.values()),
                         total=len(fleet.jobs), success=report['success'], errors=errors)))
    if not report['success']:
        raise SystemExit(1)

if __name__ == '__main__': main()
