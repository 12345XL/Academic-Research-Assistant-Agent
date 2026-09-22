from copy import deepcopy

import pytest
from pydantic import ValidationError
from research_agent.generation import (GroundedDraft,DetailedVerification,verification_passed,AnswerService,
                                      GenerationSettings,ModelFailure)
from research_agent.quality_metrics import score_answer,token_f1
from research_agent.evaluation_budget import EvaluationBudget
from research_agent.retrieval import reciprocal_rank_fusion
from research_agent.service import PersistentEvidenceService
from test_persistent_service import RepositoryDouble

D={'answerable':True,'answer_type':'extractive','short_answer':'alpha evidence',
   'claims':[{'text':'alpha evidence','evidence':[{'chunk_id':'a','quote':'alpha evidence'}]}]}
V={'addresses_question':True,'verdicts':[{'claim_index':0,'label':'supported','relevant':True,
                                       'support':[{'chunk_id':'a','quote':'alpha evidence'}]}]}


def test_weighted_rrf_defaults_preserve_scores_and_endpoints_do_not_pad():
    rankings=[['a','b','a'],['b','c']]
    baseline=reciprocal_rank_fusion(rankings,5)
    assert reciprocal_rank_fusion(rankings,5,60,[1,1])==baseline
    assert baseline[0][0]=='b'
    assert baseline[0][1]==pytest.approx(1/61+1/62)
    assert [x[0] for x in reciprocal_rank_fusion(rankings,5,10,[0,2])]==['b','c']
    assert [x[0] for x in reciprocal_rank_fusion(rankings,5,100,[2,0])]==['a','b']


@pytest.mark.parametrize('constant,weights',[(0,[1,1]),(1.2,[1,1]),(True,[1,1]),(60,[1]),
        (60,[1,float('nan')]),(60,[-1,1]),(60,[0,0]),(60,[True,1])])
def test_invalid_rrf_configuration_rejected(constant,weights):
    with pytest.raises(ValueError):reciprocal_rank_fusion([['a'],['b']],5,constant,weights)


def test_smoothing_changes_rank_sharpness_instead_of_number_returned():
    small=reciprocal_rank_fusion([['a','b'],['b','a']],2,10)
    large=reciprocal_rank_fusion([['a','b'],['b','a']],2,100)
    assert [x[0] for x in small]==[x[0] for x in large]
    assert small[0][1]>large[0][1]


@pytest.mark.parametrize('change', ['short','type','boolean','refusal'])
def test_v2_short_answer_contract(change):
    d=deepcopy(D)
    if change=='short':d['short_answer']='unverified summary'
    if change=='type':d['answer_type']='unanswerable'
    if change=='boolean':d['answer_type']='boolean'
    if change=='refusal':d.update(answerable=False,claims=[])
    with pytest.raises(ValidationError):GroundedDraft.model_validate(d)


@pytest.mark.parametrize('fault',['label','relevant','missing','index','duplicate','quote','foreign'])
def test_verifier_v2_cannot_pass_bad_or_cross_claim_support(fault):
    v=deepcopy(V)
    if fault=='label':v['verdicts'][0]['label']='insufficient'
    if fault=='relevant':v['verdicts'][0]['relevant']=False
    if fault=='missing':v['verdicts'][0]['support']=[]
    if fault=='index':v['verdicts'][0]['claim_index']=1
    if fault=='duplicate':v['verdicts']*=2
    if fault=='quote':v['verdicts'][0]['support'][0]['quote']='invented'
    if fault=='foreign':v['verdicts'][0]['support'][0]['chunk_id']='b'
    assert not verification_passed(GroundedDraft.model_validate(D),DetailedVerification.model_validate(v),
                                   {'a':{'text':'alpha evidence'},'b':{'text':'alpha evidence'}})


def test_v2_end_to_end_publishes_only_verified_short_answer():
    class Client:
        def complete(self,prompt,payload,schema):
            return schema.model_validate(D if schema is GroundedDraft else V),{}
    result=AnswerService(GenerationSettings('x','deepseek-flash',True),Client(),'v2').answer(
        PersistentEvidenceService(RepositoryDouble()),'alpha','p1',language='en')
    assert result['status']=='answered' and result['short_answer']==result['claims'][0]['text']
    assert result['generation']['prompt_version']=='paper-claims-v2'
    assert result['generation']['answer_language']=='en'


def test_english_metrics_do_not_reward_provider_or_verifier_failures_as_abstention():
    q={'annotations':[{'unanswerable':True}]}
    base={'generation':{'answer_language':'en'},'claims':[]}
    for status in ('model_unavailable','verification_failed','not_configured'):
        scores=score_answer({**base,'status':status},q)
        assert scores['answer_f1']==0 and scores['evidence_f1_mapped']==0
    assert score_answer({**base,'status':'evidence_insufficient'},q)['answer_f1']==1
    with pytest.raises(ValueError):score_answer({**base,'generation':{'answer_language':'zh'},'status':'answered'},q)
    assert token_f1('The alpha model.','alpha model')==1
    assert token_f1('alpha alpha beta','alpha beta')==pytest.approx(.8)


def test_multiple_gold_references_and_published_evidence_denominator():
    q={'annotations':[{'unanswerable':False,'extractive_spans':['alpha'],'matched_chunk_ids':['a'],
                      'evidence_mappings':[]},
                     {'unanswerable':False,'free_form_answer':'beta','matched_chunk_ids':['b'],'evidence_mappings':[]}]}
    result={'status':'answered','generation':{'answer_language':'en'},'short_answer':'beta',
            'claims':[{'text':'beta','evidence':[{'chunk_id':'b','quote':'beta'}]}]}
    scores=score_answer(result,q)
    assert scores['answer_f1']==scores['evidence_f1_mapped']==1


def test_budget_limits_and_crash_reservations(tmp_path):
    class Client:
        def complete(self,*args):return GroundedDraft.model_validate(D),{'usage':{'prompt_tokens':10,'completion_tokens':5}}
    path=tmp_path/'ledger.jsonl';budget=EvaluationBudget(path,max_calls=1)
    budget.complete(Client(),'test',{},GroundedDraft)
    assert budget.snapshot()['calls']==1
    assert budget.snapshot()['estimated_charged_usd']==pytest.approx(9e-6)
    resumed=EvaluationBudget(path,max_calls=1)
    with pytest.raises(ModelFailure,match='evaluation_budget_exhausted'):
        resumed.complete(Client(),'test',{},GroundedDraft)
    empty=EvaluationBudget(tmp_path/'tiny.jsonl',max_usd=0)
    with pytest.raises(ModelFailure):empty.complete(Client(),'test',{},GroundedDraft)
    failed=EvaluationBudget(tmp_path/'failed.jsonl')
    class Broken:
        def complete(self,*args):raise ModelFailure('timeout')
    with pytest.raises(ModelFailure):failed.complete(Broken(),'test',{},GroundedDraft)
    assert failed.snapshot()['estimated_charged_usd']>0


def test_conditional_rates_exclude_the_other_population():
    unanswerable={'annotations':[{'unanswerable':True}]}
    answerable={'annotations':[{'unanswerable':False,'extractive_spans':['alpha'], 'matched_chunk_ids':['a'], 'evidence_mappings':[]}]}
    result={'status':'evidence_insufficient','generation':{'answer_language':'en'},'claims':[]}
    u=score_answer(result,unanswerable);a=score_answer(result,answerable)
    assert u['answerable_abstention'] is None and u['unanswerable_explicit_abstention']==1
    assert a['answerable_abstention']==1 and a['unanswerable_false_publication'] is None


@pytest.mark.parametrize('field,value',[('rrf_constant',True),('rrf_constant',0),('rrf_constant',1.5),('dense_weight',True),('dense_weight',-0.1),('dense_weight',1.1),('dense_weight',float('nan'))])
def test_api_rejects_invalid_experiment_values(field,value):
    from research_agent.api import AnswerRequest
    with pytest.raises(ValidationError):AnswerRequest(paper_id='p1',query='alpha',**{field:value})


def test_api_v2_dispatch_and_language_are_effective(tmp_path):
    from fastapi.testclient import TestClient
    from research_agent.api import create_app
    from test_api import write_corpus
    write_corpus(tmp_path)
    class Client:
        def complete(self,prompt,payload,schema):
            if schema is GroundedDraft:
                assert 'English.' in prompt
                data=deepcopy(D)
                data['claims'][0]['evidence'][0]['chunk_id']='p1:s0:p0'
            else:
                assert schema is DetailedVerification
                data=deepcopy(V)
                data['verdicts'][0]['support'][0]['chunk_id']='p1:s0:p0'
            return schema.model_validate(data),{}
    with TestClient(create_app(tmp_path,generation_settings=GenerationSettings('x','deepseek-flash',True),model_client=Client())) as client:
        r=client.post('/api/v1/answer',json={'paper_id':'p1','query':'alpha','profile':'v2','language':'en'})
        assert r.status_code==200 and r.json()['status']=='answered'
        assert r.json()['short_answer']=='alpha evidence'
        assert r.json()['generation']['prompt_version']=='paper-claims-v2'
        assert client.post('/api/v1/answer',json={'paper_id':'p1','query':'alpha','profile':'v3'}).status_code==422


def test_synthetic_challenges_are_balanced_and_have_real_quotes():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_verifier_challenges.py'
    spec=importlib.util.spec_from_file_location('challenges',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cases=module.frozen_cases()
    assert len(cases)==40 and sum(c['expected_publish'] for c in cases)==20
    assert len({c['id'] for c in cases})==40
    for c in cases:
        by_id={p['chunk_id']:p['text'] for p in c['cited_paragraphs']}
        for claim in c['claims']:
            for ref in claim['evidence']:assert ref['quote'] in by_id[ref['chunk_id']]
