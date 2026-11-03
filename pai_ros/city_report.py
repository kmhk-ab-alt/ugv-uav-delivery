"""Paired, episode-level bootstrap comparisons; no model selection on test data."""
import json
from pathlib import Path
import numpy as np

def paired_analysis(report):
    rows=report['rows'];rng=np.random.default_rng(7021);results=[]
    pairs=[('held_out_map_provided','diverse_cities','held_out_map_provided',b) for b in ('selected_rule','untrained','fixed_city')]
    pairs.append(('shift_after_support_adaptation','adapted_diverse_cities','shift_before_adaptation','diverse_cities'))
    for ca,pa,cb,pb in pairs:
        a={(r['city_seed'],r['order_seed']):r for r in rows if r['condition']==ca and r['policy']==pa}
        b={(r['city_seed'],r['order_seed']):r for r in rows if r['condition']==cb and r['policy']==pb}
        if a.keys()!=b.keys():raise ValueError('paired worlds do not match')
        keys=sorted(a);item=dict(a=pa,b=pb,condition_a=ca,condition_b=cb,episodes=len(keys),difference='a minus b',metrics={})
        for metric in ('reward','on_time','priority_weighted_lateness','distance_m','makespan_s','contacts'):
            delta=np.array([a[k][metric]-b[k][metric] for k in keys],dtype=float)
            boot=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(axis=1)
            item['metrics'][metric]=dict(mean_difference=float(delta.mean()),bootstrap_95_percent_ci=[float(x) for x in np.quantile(boot,[.025,.975])])
        results.append(item)
    return dict(method='paired episode bootstrap, 2000 resamples; descriptive intervals, not multiplicity-adjusted hypothesis tests',comparisons=results)

def save_analysis(path):
    path=Path(path);report=json.loads(path.read_text());a=paired_analysis(report)
    path.with_name('paired_analysis.json').write_text(json.dumps(a,indent=2));return a
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('report');args=p.parse_args();save_analysis(args.report)
