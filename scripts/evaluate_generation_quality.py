"""Paired, bounded v1/v2 English generation evaluation on frozen three pools."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from research_agent.dataset import read_jsonl,sha256_file
from research_agent.generation import AnswerService,GenerationSettings,DeepSeekClient,ModelFailure
from research_agent.evaluation_budget import EvaluationBudget
from research_agent.quality_metrics import score_answer
from research_agent.service import PersistentEvidenceService
from research_agent.settings import Settings
from research_agent.storage import Repository
from research_agent.reranking import LocalReranker


def summaries(cases):
    output={}
    for profile in ('v1','v2'):
        rows=[c for c in cases if c['profile']==profile]
        output[profile]={}
        for category in ('all','extractive','abstractive','boolean','unanswerable'):
            subset=[c for c in rows if category=='all' or c['category']==category]
            if not subset: continue
            metrics={}
            for key in subset[0]['metrics']:
                values=[c['metrics'][key] for c in subset if c['metrics'][key] is not None]
                metrics[key]={'mean':statistics.mean(values) if values else None,'denominator':len(values)}
            times=sorted(c['generation']['latency_ms'] for c in subset)
            output[profile][category]={'n':len(subset),'metrics':metrics,
                'statuses':{status:sum(c['status']==status for c in subset) for status in sorted({c['status'] for c in subset})},
                'latency_ms':{'p50':statistics.median(times),'p95':times[int(.95*(len(times)-1))]},
                'support_rate':None,'support_rate_reason':'Independent human labels pending'}
    return output


class RecordedBudgetClient:
    def __init__(self,client,budget):
        self.client,self.budget,self.records=client,budget,[]
    def complete(self,prompt,payload,schema):
        try:
            result,meta=self.budget.complete(self.client,prompt,payload,schema)
            self.records.append({'schema':schema.__name__,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                                 'payload':payload,'output':result.model_dump(),'meta':meta})
            return result,meta
        except ModelFailure as exc:
            self.records.append({'schema':schema.__name__,'error':exc.code,'meta':exc.metadata})
            raise


class FixedEvidence:
    def __init__(self,repository,result):self.repository,self.result=repository,result
    def retrieve(self,*args,**kwargs):return deepcopy(self.result)


def main():
    load_dotenv(ROOT/'.env')
    config=GenerationSettings.from_env()
    if not config.configured:raise SystemExit('Generation configuration missing')
    if config.model != 'deepseek-flash':raise SystemExit('Budget rate requires deepseek-flash')
    manifest_path=ROOT/'reports/p2b_generation_strata_v2.json'
    manifest=json.loads(manifest_path.read_text())
    for name,sha in manifest['data_sha256'].items():assert sha256_file(ROOT/'data/processed'/name)==sha
    local=ROOT/'.local/generation-quality-v3';local.mkdir(parents=True,exist_ok=True)
    questions={q['question_id']:q for q in read_jsonl(ROOT/'data/processed/questions.jsonl')}
    repo=Repository(Settings.from_env());revision=repo.revision()
    service=PersistentEvidenceService(repo,reranker=LocalReranker('mps'))
    budget=EvaluationBudget(local/'budget.jsonl')
    source=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'src/research_agent').glob('*.py')))).hexdigest()
    identity={'manifest_sha256':sha256_file(manifest_path),'source_sha256':source,'revision':revision,'model':config.model}
    identity_path=local/'identity.json'
    if identity_path.exists():assert json.loads(identity_path.read_text())==identity,'Resume code/data changed'
    else:identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    path=ROOT/'reports/p2b_generation_quality_v3.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):raise SystemExit('Completed report exists; refusing another paid run')
    report={'protocol':'generation-quality-v3','identity':identity,'created_at':datetime.now(timezone.utc).isoformat(),
            'config':{'retrieval':'dense+rerank','candidate_pool':20,'top_k':5,'language':'en','profiles':['v1','v2'],
                      'model':config.model,'concurrent_offline_jobs':4},
            'limitations':['Balanced 80-question pools, not population accuracy; dev already inspected for retrieval',
                           'v1 full-claims answer vs v2 short-answer format is part of intervention, not pure semantic-quality delta',
                           'Semantic support rate pending independent human labels; model verdict is not gold',
                           'Model timing excludes shared offline retrieval preparation; no production latency SLA'],
            'pools':{},'complete':False}
    for pool_name,pool in manifest['pools'].items():
        tasks=[];cases=[]
        for meta in pool['cases']:
            q=questions[meta['question_id']]
            cache=local/(q['question_id']+'-retrieval.json')
            if cache.exists():retrieved=json.loads(cache.read_text())
            else:
                retrieved=service.retrieve(q['question'],q['paper_id'],5,mode='dense',rerank=True)
                cache.write_text(json.dumps(retrieved,ensure_ascii=False)+'\n')
            for profile in ('v1','v2'):
                saved=local/(q['question_id']+'-'+profile+'.json')
                if saved.exists():cases.append(json.loads(saved.read_text())['public'])
                else:tasks.append((meta,q,profile,retrieved,saved))
        print(pool_name,'prepared',len(pool['cases']),'questions; pending runs',len(tasks),flush=True)
        def run(task):
            meta,q,profile,retrieved,saved=task
            client=RecordedBudgetClient(DeepSeekClient(config),budget)
            result=AnswerService(config,client,profile).answer(FixedEvidence(repo,retrieved),q['question'],q['paper_id'],
                                                           5,'dense',True,language='en')
            public={**meta,'profile':profile,'status':result['status'],'generation':result['generation'],
                    'metrics':score_answer(result,q),'output_sha256':hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()}
            saved.write_text(json.dumps({'public':public,'result':result,'model_records':client.records,
                                       'review_status':'pending','annotations':q['annotations']},ensure_ascii=False,indent=2)+'\n')
            return public
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(run,t) for t in tasks]
            for future in as_completed(futures):
                cases.append(future.result())
                cases.sort(key=lambda c:(c['question_id'],c['profile']))
                report['pools'][pool_name]={'cases':cases,'summary':summaries(cases)}
                report['budget']=budget.snapshot()
                path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                if len(cases)%20==0:print(pool_name,len(cases),'/160',report['budget'],flush=True)
        report['pools'][pool_name]={'cases':cases,'summary':summaries(cases)}
        assert repo.revision()==revision
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    report['budget']=budget.snapshot();report['complete']=all(len(p['cases'])==160 for p in report['pools'].values())
    report['budget_exhausted']=any(call['status']=='evaluation_budget_exhausted' for pool in report['pools'].values()
        for row in pool['cases'] for call in row['generation']['calls'])
    if report['budget_exhausted']:report['complete']=False
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Complete',report['complete'],report['budget'],flush=True)

if __name__=='__main__':main()
