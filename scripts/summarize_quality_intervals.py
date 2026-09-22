"""Paper-cluster paired bootstrap. Reads completed public reports; no model calls."""
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
ROOT=Path(__file__).resolve().parents[1]
SEED=20260922
REPEATS=2000


def interval(rows):
    groups=defaultdict(list)
    for paper,value in rows:
        if value is not None:groups[paper].append(value)
    if not groups:return {'questions':0,'papers':0,'mean_delta':None,'ci95':None}
    blocks=[groups[k] for k in sorted(groups)]
    rng=random.Random(SEED)
    replicates=[]
    for _ in range(REPEATS):
        sampled=[v for group in rng.choices(blocks,k=len(blocks)) for v in group]
        replicates.append(statistics.mean(sampled))
    replicates.sort()
    return {'questions':len(rows),'papers':len(groups),'mean_delta':statistics.mean(v for group in blocks for v in group),
            'ci95':[replicates[int(.025*REPEATS)],replicates[int(.975*REPEATS)]]}


def main():
    rrf=json.loads((ROOT/'reports/p2b_rrf_grid_v3.json').read_text())
    gen=json.loads((ROOT/'reports/p2b_generation_quality_v3.json').read_text())
    assert rrf['complete'] and gen['complete']
    output={'seed':SEED,'replicates':REPEATS,'method':'paired percentile bootstrap resampling paper clusters; each sample recomputes question mean; no multiplicity correction',
            'generation_v2_minus_v1':{},'rrf_equal_minus_dense':{}}
    for pool,p in rrf['pools'].items():
        output['rrf_equal_minus_dense'][pool]={key:interval([(c['paper_id'],c['arms']['K60_alpha0.5'][key]-c['arms']['K60_alpha1'][key]) for c in p['cases']]) for key in ('hit','recall','mrr')}
    for pool,p in gen['pools'].items():
        pairs=defaultdict(dict)
        for c in p['cases']:pairs[c['question_id']][c['profile']]=c
        assert len(pairs)==80 and all(set(v)=={'v1','v2'} for v in pairs.values())
        output['generation_v2_minus_v1'][pool]={}
        for category in ('all','extractive','abstractive','boolean','unanswerable'):
            selected=[v for v in pairs.values() if category=='all' or v['v1']['category']==category]
            output['generation_v2_minus_v1'][pool][category]={}
            for metric in selected[0]['v1']['metrics']:
                rows=[(v['v1']['paper_id'],v['v2']['metrics'][metric]-v['v1']['metrics'][metric]) for v in selected
                      if v['v1']['metrics'][metric] is not None and v['v2']['metrics'][metric] is not None]
                output['generation_v2_minus_v1'][pool][category][metric]=interval(rows)
    path=ROOT/'reports/p2b_quality_intervals_v3.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    print('Intervals written; no additional model calls')

if __name__=='__main__':main()
