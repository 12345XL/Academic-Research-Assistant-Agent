"""Frozen, synthetic counterfactual verifier pairs. Run after generation (shared ledger)."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from research_agent.generation import (Draft,Verification,DetailedVerification,VERIFIER_PROMPT,VERIFIER_V2,
    verification_passed,DeepSeekClient,GenerationSettings,ModelFailure)
from research_agent.evaluation_budget import EvaluationBudget

# Own short synthetic text, not QASPER labels or paper excerpts. No challenge outcomes used in prompts.
PAIRS=[
('number','How many documents were annotated?','We annotated 120 documents.','120 documents were annotated.','1,200 documents were annotated.'),
('percentage','What was the accuracy?','The system achieved 82% accuracy on the held-out set.','Accuracy was 82% on the held-out set.','Accuracy was 92% on the held-out set.'),
('negation','Was the encoder frozen?','The encoder was not frozen during training.','No, the encoder was not frozen.','Yes, the encoder was frozen.'),
('condition','When did accuracy improve?','Accuracy improved only when labeled data exceeded 500 examples.','Accuracy improved when labeled data exceeded 500 examples.','Accuracy improved when labeled data was below 500 examples.'),
('universal','Did all datasets improve?','Two of the five datasets improved; the other three did not.','No, only two of the five datasets improved.','Yes, all five datasets improved.'),
('language','Is the dataset English-only?','The dataset contains English examples. We do not report whether other languages are present.','The reported information does not establish whether the dataset is English-only.','Yes, the dataset is English-only.'),
('planned','Was the multilingual model evaluated?','We evaluated a monolingual model. Evaluation of a multilingual model is left to future work.','No, evaluation of the multilingual model is left to future work.','Yes, the multilingual model was evaluated.'),
('baseline','Was Model B evaluated as a baseline?','We discuss Model B as related work but do not evaluate it. Our only evaluated baseline is Model A.','No, Model B was discussed but not evaluated.','Yes, Model B was an evaluated baseline.'),
('comparison','Which model was faster?','Model A took 10 seconds; Model B took 20 seconds on the same device.','Model A was faster than Model B.','Model B was faster than Model A.'),
('units','How much memory was used?','Peak memory usage was 8 GB.','Peak memory usage was 8 GB.','Peak memory usage was 8 MB.'),
('split','What was the test score?','The development score was 90, whereas the test score was 75.','The test score was 75.','The test score was 90.'),
('causality','Does the study establish causality?','The observational study found a correlation, but does not establish a causal relationship.','No, the study reports correlation without establishing causality.','Yes, the study establishes a causal relationship.'),
('certainty','Is the improvement statistically significant?','The improvement was not statistically significant at the specified threshold.','No, the improvement was not statistically significant.','Yes, the improvement was statistically significant.'),
('scope','Does the guarantee apply to long inputs?','The guarantee applies only to inputs of at most 100 tokens. It does not cover longer inputs.','No, inputs longer than 100 tokens are not covered.','Yes, the guarantee applies to inputs of any length.'),
('direction','Did latency rise or fall?','Latency fell from 40 milliseconds to 30 milliseconds.','Latency fell by 10 milliseconds.','Latency rose by 10 milliseconds.'),
('subset','How many annotated items were used for training?','We annotated 100 items and used 80 of them for training.','80 annotated items were used for training.','100 annotated items were used for training.'),
('attribution','Who proposed the current method?','Smith proposed the baseline. Jones proposed the current method.','Jones proposed the current method.','Smith proposed the current method.'),
('coverage','Were human evaluations performed?','We performed automatic evaluations only and did not perform human evaluations.','No, human evaluations were not performed.','Yes, human evaluations were performed.'),
('definition','How is robustness defined?','Robustness is the fraction of predictions unchanged after perturbation.','Robustness is the fraction of predictions unchanged after perturbation.','Robustness is the fraction of predictions changed after perturbation.'),
]


def frozen_cases():
    cases=[]
    for name,question,paragraph,good,bad in PAIRS:
        for label,text in ((True,good),(False,bad)):
            cases.append({'id':name+('-supported' if label else '-corrupted'),'pair':name,'expected_publish':label,
                'question':question,'claims':[{'text':text,'evidence':[{'chunk_id':'a','quote':paragraph}]}],
                'cited_paragraphs':[{'chunk_id':'a','text':paragraph}]})
    for supported in (True,False):
        # Both IDs exist, but only b supports the second claim. a's existence is insufficient.
        cases.append({'id':'ownership-'+str(supported).lower(),'pair':'ownership','expected_publish':supported,
            'question':'How many training and test examples were used?',
            'claims':[{'text':'There were 50 training examples.','evidence':[{'chunk_id':'a','quote':'There were 50 training examples.'}]},
                      {'text':'There were 20 test examples.','evidence':[{'chunk_id':'b' if supported else 'a',
                          'quote':'There were 20 test examples.' if supported else 'There were 50 training examples.'}]}],
            'cited_paragraphs':[{'chunk_id':'a','text':'There were 50 training examples.'},
                                {'chunk_id':'b','text':'There were 20 test examples.'}]})
    assert len(cases)==40
    return cases


def main():
    load_dotenv(ROOT/'.env')
    config=GenerationSettings.from_env()
    assert config.configured and config.model=='deepseek-flash'
    manifest={'label_source':'engineering constructed synthetic pairs, not independent human gold','cases':frozen_cases()}
    frozen=ROOT/'reports/p2b_verifier_challenges_manifest_v3.json'
    if frozen.exists():assert json.loads(frozen.read_text())==manifest
    else:frozen.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    generation=json.loads((ROOT/'reports/p2b_generation_quality_v3.json').read_text())
    assert generation['complete'],'Finish generation first: do not concurrently write the budget ledger'
    local=ROOT/'.local/generation-quality-v3';budget=EvaluationBudget(local/'budget.jsonl')
    client=DeepSeekClient(config)
    path=ROOT/'reports/p2b_verifier_challenges_v3.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):raise SystemExit('Completed report exists')
    report={'manifest_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
            'prompt_sha256':{v:hashlib.sha256(p.encode()).hexdigest() for v,p in [('v1',VERIFIER_PROMPT),('v2',VERIFIER_V2)]},
            'label_source':manifest['label_source'],'cases':[],'complete':False}
    for case in manifest['cases']:
        draft=Draft.model_validate({'answerable':True,'claims':case['claims']})
        payload={'question':case['question'],'claims':[{'claim_index':i,**c} for i,c in enumerate(case['claims'])],
                 'cited_paragraphs':case['cited_paragraphs']}
        for profile,prompt,schema in [('v1',VERIFIER_PROMPT,Verification),('v2',VERIFIER_V2,DetailedVerification)]:
            saved=local/('challenge-'+case['id']+'-'+profile+'.json')
            if saved.exists():row=json.loads(saved.read_text())
            else:
                row={'id':case['id'],'pair':case['pair'],'profile':profile,'expected_publish':case['expected_publish']}
                try:
                    output,meta=budget.complete(client,prompt,payload,schema)
                    row.update(published=verification_passed(draft,output,{c['chunk_id']:c for c in case['cited_paragraphs']}),
                               status='completed',verdict=output.model_dump(),meta=meta)
                except ModelFailure as exc:
                    row.update(published=False,status=exc.code,meta=exc.metadata)
                saved.write_text(json.dumps(row,indent=2)+'\n')
            report['cases'].append(row)
            report['budget']=budget.snapshot();path.write_text(json.dumps(report,indent=2)+'\n')
    report['summary']={}
    for profile in ('v1','v2'):
        rows=[r for r in report['cases'] if r['profile']==profile]
        valid=[r for r in rows if r['status']=='completed']
        report['summary'][profile]={'total':len(rows),'completed':len(valid),'faults':len(rows)-len(valid),
            'supported_published':sum(r['expected_publish'] and r['published'] for r in rows),'supported_total':20,
            'unsupported_published':sum(not r['expected_publish'] and r['published'] for r in rows),'unsupported_total':20,
            'unsupported_rejected_valid_verdict':sum(not r['expected_publish'] and not r['published'] for r in valid)}
    report['complete']=len(report['cases'])==80 and all(r['status']!='evaluation_budget_exhausted' for r in report['cases'])
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary']),flush=True)

if __name__=='__main__':main()
