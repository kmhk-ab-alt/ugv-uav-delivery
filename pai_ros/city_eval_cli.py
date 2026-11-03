"""Evaluate a frozen checkpoint on newly generated cities, with failures retained."""
import argparse,json
from pathlib import Path
from .dispatch_model import ScoringPolicy
from .generalization import run,aggregate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--episodes',type=int,default=30);p.add_argument('--city-start',type=int,default=20000);p.add_argument('--city-family',choices=('train','shift'),default='train');p.add_argument('--local-map',action='store_true');p.add_argument('--output',required=True)
    p.add_argument('--rule-policy',choices=('priority','deadline'));p.add_argument('--rule-mode',choices=('auto','relay','scout','joint'));p.add_argument('--rule-selection')
    args=p.parse_args()
    if not 1<=args.episodes<=100:p.error('--episodes must be 1..100')
    out=Path(args.output)
    if out.exists():p.error('output already exists')
    selection=Path(args.rule_selection) if args.rule_selection else Path(args.checkpoint).with_name('rule_selection.json')
    chosen=json.loads(selection.read_text())['selected'] if selection.exists() else dict(policy='priority',mode='auto')
    args.rule_policy=args.rule_policy or chosen['policy'];args.rule_mode=args.rule_mode or chosen['mode']
    model=ScoringPolicy.load(args.checkpoint);rows=[]
    for seed in range(args.city_start,args.city_start+args.episodes):
        for label,policy,m,mode in [('learned','priority',model,'auto'),('rule',args.rule_policy,None,args.rule_mode)]:
            r,_=run(seed,seed+100000,policy,model=m,mode=mode,family=args.city_family,local_map=args.local_map);r.update(policy=label,condition='fresh_cities');rows.append(r)
            print(f"{label} city={seed}: delivered={r['delivered']}/{r['total']} on_time={r['on_time']} contacts={r['contacts']}",flush=True)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(arguments=vars(args),summary=aggregate(rows),rows=rows),indent=2))
if __name__=='__main__':main()
