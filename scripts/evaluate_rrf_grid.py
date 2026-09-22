"""Frozen K (RRF smoothing) × weight experiment; no paid API calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from research_agent.dataset import read_jsonl, sha256_file
from research_agent.embeddings import LocalEncoder, CONFIG as EMBEDDING
from research_agent.evaluate import gold_references, retrieval_metrics
from research_agent.reranking import LocalReranker, CONFIG as RERANKER
from research_agent.retrieval import reciprocal_rank_fusion
from research_agent.service import PersistentEvidenceService
from research_agent.settings import Settings
from research_agent.storage import Repository

GRID = [(k, a) for k in (10, 30, 60, 100) for a in (0, .25, .5, .75, 1)]

def arm_key(k, alpha):
    return f'K{k}_alpha{alpha:g}'


def summarize(rows, arm):
    return {metric: statistics.mean(r['arms'][arm][metric] for r in rows)
            for metric in ('hit', 'recall', 'mrr')}


def main():
    path = ROOT / 'reports/p2b_rrf_grid_v3.json'
    if path.exists(): raise SystemExit('Frozen report exists; do not overwrite')
    load_dotenv(ROOT / '.env')
    manifest = json.loads((ROOT / 'reports/p2b_generation_strata_v2.json').read_text())
    for name, checksum in manifest['data_sha256'].items():
        assert sha256_file(ROOT/'data/processed'/name)==checksum
    questions = {q['question_id']:q for q in read_jsonl(ROOT/'data/processed/questions.jsonl')}
    repo=Repository(Settings.from_env()); revision=repo.revision()
    reranker=LocalReranker('mps'); encoder=LocalEncoder('cpu')
    service=PersistentEvidenceService(repo, encoder=encoder, reranker=reranker)
    started=time.perf_counter()
    report={'protocol':'rrf-smoothing-weight-v3','manifest_sha256':sha256_file(ROOT/'reports/p2b_generation_strata_v2.json'),
            'corpus_revision':revision,'config':{'grid':GRID,'candidate_pool':20,'top_k':5,
              'embedding':EMBEDDING,'reranker':RERANKER,'embedding_device':'cpu','reranker_device':'mps'},
            'source_sha256':hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'src/research_agent').glob('*.py')))).hexdigest(),
            'pools':{},'selected':None,'timing_note':'Offline cached per-query paragraph logits; not online per-arm latency',
            'paid_model_calls':0}
    selected=None
    for pool_name, pool in manifest['pools'].items():
        cases=[]
        metas=[m for m in pool['cases'] if m['retrieval_metric_eligible']]
        vectors=encoder.encode_queries([questions[m['question_id']]['question'] for m in metas])
        grid=GRID if pool_name=='train_tuning' else list(dict.fromkeys([selected,(60,.5),(60,1),(60,0)]))
        for num,(meta,vector) in enumerate(zip(metas,vectors),1):
            q=questions[meta['question_id']]
            refs,_=gold_references(q)
            branches=[service.retrieve(q['question'],q['paper_id'],20,mode=mode,query_vector=vector)
                      for mode in ('bm25','dense')]
            ids=[[c['chunk_id'] for c in r['citations']] for r in branches]
            facts={c['chunk_id']:c for r in branches for c in r['citations']}
            union=sorted(facts)
            scores={}
            # Each union paragraph scored exactly once. Batches respect the same model/window budget.
            for offset in range(0,len(union),20):
                chunk_ids=union[offset:offset+20]
                values,_=reranker.score(q['question'],[facts[c]['text'] for c in chunk_ids])
                scores.update(zip(chunk_ids,values))
            row={'question_id':q['question_id'],'paper_id':q['paper_id'],'category':meta['category'],'arms':{}}
            for k,alpha in grid:
                fused=reciprocal_rank_fusion(ids,20,k,[2*(1-alpha),2*alpha])
                candidates=[c for c,_ in fused]
                ranked=sorted(candidates,key=lambda c:(-scores[c],c))[:5]
                row['arms'][arm_key(k,alpha)]={**retrieval_metrics(ranked,refs),
                    'ranked_ids':ranked,'candidate_metrics':retrieval_metrics(candidates,refs)}
            # Validate the cache replay against the actual service for one case per split.
            if num==1:
                k,alpha=(60,.5) if selected is None else selected
                live=service.retrieve(q['question'],q['paper_id'],5,mode='hybrid',rerank=True,
                                      query_vector=vector,rrf_constant=k,dense_weight=alpha)
                assert [c['chunk_id'] for c in live['citations']]==row['arms'][arm_key(k,alpha)]['ranked_ids']
            assert repo.revision()==revision
            cases.append(row)
            if num%20==0: print(pool_name,num,'/',len(metas),flush=True)
        metrics={arm_key(k,a):summarize(cases,arm_key(k,a)) for k,a in grid}
        report['pools'][pool_name]={'questions':len(cases),'excluded_from_80':80-len(cases),
                                  'metrics':metrics,'cases':cases,'service_replay_checks':1}
        if pool_name=='train_tuning':
            selected=max(GRID,key=lambda config:(metrics[arm_key(*config)]['recall'],metrics[arm_key(*config)]['mrr'],
                                                config[1]==.5,config[0]==60,-GRID.index(config)))
            report['selected']={'rrf_constant':selected[0],'dense_weight':selected[1],
                                'selected_on':'train_tuning_only','online_default_changed':False}
            print('Selected on tuning:',report['selected'],flush=True)
        report['elapsed_seconds']=round(time.perf_counter()-started,2)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    report['complete']=True
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({name:pool['metrics'] for name,pool in report['pools'].items()},indent=2),flush=True)

if __name__=='__main__':main()
