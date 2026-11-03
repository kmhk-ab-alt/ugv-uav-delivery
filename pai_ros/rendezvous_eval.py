"""Re-evaluate a frozen meeting model and baselines on fresh city seeds."""
import argparse,json
from pathlib import Path
import numpy as np
from .rendezvous import MeetingNetwork
from .rendezvous_experiment import run

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--city-start',type=int,default=79000);p.add_argument('--episodes',type=int,default=20);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists() or not 1<=args.episodes<=100:p.error('use a new output file and 1..100 episodes')
    with np.load(args.checkpoint,allow_pickle=False) as data:training=json.loads(str(data['metadata']))['protocol']
    seeds=list(range(args.city_start,args.city_start+args.episodes))
    if set(seeds)&set(training['train']+training['validation']):p.error('evaluation cities overlap training or validation')
    model=MeetingNetwork.load(args.checkpoint);initial=MeetingNetwork(42);rows=[]
    for s in seeds:
        for name,m in [('fixed',None),('eta',None),('untrained',initial),('learned',model)]:
            r,_=run(s,'learned' if m is not None else name,model=m);r['policy']=name;rows.append(r)
            print(s,name,r['delivered'],r['on_time'],flush=True)
    summary=[]
    for name in ('fixed','eta','untrained','learned'):
        rs=[r for r in rows if r['policy']==name]
        summary.append(dict(policy=name,delivered=sum(r['delivered'] for r in rs),total=sum(r['total'] for r in rs),on_time=sum(r['on_time'] for r in rs),mean_wait_s=float(np.mean([r['meeting_wait_s'] for r in rs])),mean_makespan_s=float(np.mean([r['makespan_s'] for r in rs]))))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(arguments=vars(args),training_protocol=training,summary=summary,rows=rows),indent=2))
if __name__=='__main__':main()
