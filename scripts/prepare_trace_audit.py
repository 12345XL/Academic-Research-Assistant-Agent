"""Freeze a small audit cohort and masked local packets from existing outputs. No model/network imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 'trace-audit-v1-20260922'
KNOWN = {
    '1db37e98768f09633dfbc78616992c9575f6dba4', '0bd992a6a218331aa771d922e3c7bb60b653949a',
    '50690b72dc61748e0159739a9a0243814d37f360', '8d074aabf4f51c8455618c5bf7689d3f62c4da1d',
    'ad1be65c4f0655ac5c902d17f05454c0d4c4a15d', 'a96a1a354cb3a2a434b085e4d9c8844d0b672ec4',
    '1bdf7e9f3f804930b2933ebd9207a3e000b27742',
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def freeze(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == text, f'Frozen data differs: {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(text)


def choose(pairs):
    available = sorted((q for q in pairs if q not in KNOWN), key=lambda q:digest(SEED+':sample:'+q))
    rules = [
        ('unanswerable_published',5,lambda a,b:a['category']=='unanswerable' and 'answered' in (a['status'],b['status'])),
        ('answerable_generator_abstained',5,lambda a,b:a['category']!='unanswerable' and 'evidence_insufficient' in (a['status'],b['status'])),
        ('verification_blocked',5,lambda a,b:'verification_failed' in (a['status'],b['status'])),
        ('changed',5,lambda a,b:a['status']!=b['status'] or abs(a['metrics']['answer_f1']-b['metrics']['answer_f1'])>=.2),
        ('both_published_control',4,lambda a,b:a['status']==b['status']=='answered'),
    ]
    chosen=[];papers=set()
    for bucket,count,predicate in rules:
        rows=[q for q in available if q not in {r['question_id'] for r in chosen} and pairs[q]['v1']['paper_id'] not in papers and predicate(pairs[q]['v1'],pairs[q]['v2'])]
        added=0
        for q in rows:
            if pairs[q]['v1']['paper_id'] in papers:continue
            papers.add(pairs[q]['v1']['paper_id'])
            chosen.append({'question_id':q,'paper_id':pairs[q]['v1']['paper_id'],'bucket':bucket})
            added+=1
            if added==count:break
        assert added==count, f'Not enough eligible cases for {bucket}'
    return sorted(chosen,key=lambda r:digest(SEED+':display:'+r['question_id']))


def main():
    report_path=ROOT/'reports/p2b_generation_quality_v3.json'
    report=json.loads(report_path.read_text());assert report['complete']
    pairs={}
    for pool,p in report['pools'].items():
        for c in p['cases']:pairs.setdefault(c['question_id'],{})[c['profile']]={**c,'pool':pool}
    assert len(pairs)==240 and all(set(v)=={'v1','v2'} for v in pairs.values())
    selected=choose(pairs)
    origin=ROOT/'.local/generation-quality-v3';local=ROOT/'.local/trace-audit-v1'
    packets_a=[];packets_b=[];reveal=[];private_hashes={}
    for i,row in enumerate(selected,1):
        row['audit_id']=f'T{i:02}'
        q=row['question_id'];records={v:json.loads((origin/f'{q}-{v}.json').read_text()) for v in ('v1','v2')}
        for v in records:
            assert records[v]['public']=={k:val for k,val in pairs[q][v].items() if k!='pool'}
            assert digest(json.dumps(records[v]['result'],sort_keys=True))==records[v]['public']['output_sha256']
            private_hashes[f'{q}-{v}.json']=hashlib.sha256((origin/f'{q}-{v}.json').read_bytes()).hexdigest()
        contexts={v:next(r['payload'] for r in records[v]['model_records'] if r.get('schema') in ('Draft','GroundedDraft')) for v in records}
        assert contexts['v1']==contexts['v2'],'Paired evidence differs'
        packets_a.append({'audit_id':row['audit_id'],**contexts['v1']})
        order=('v1','v2') if int(digest(SEED+':arms:'+q),16)%2==0 else ('v2','v1')
        arms={}
        for alias,profile in zip(('A','B'),order):
            generation=next(r for r in records[profile]['model_records'] if r.get('schema') in ('Draft','GroundedDraft'))
            arms[alias]={'answerable':generation['output']['answerable'],'claims':generation['output']['claims']}
        packets_b.append({'audit_id':row['audit_id'],'arms':arms})
        reveal.append({**row,'pool':pairs[q]['v1']['pool'],'category':pairs[q]['v1']['category'],
            'arm_mapping':dict(zip(('A','B'),order)),
            'annotations':records['v1']['annotations'],
            'candidate_ids':records['v1']['result']['trace']['rerank_candidates'],
            'context_ids':[e['chunk_id'] for e in contexts['v1']['evidence']],
            'arms':{v:{'status':records[v]['result']['status'],'metrics':pairs[q][v]['metrics'],
                       'checks':records[v]['result']['generation']['checks'],
                       'verifier_records':[m for m in records[v]['model_records'] if m.get('schema') in ('Verification','DetailedVerification')]} for v in records}})
    manifest={'protocol':'trace-audit-v1','seed':SEED,'source_report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
              'selection':'diagnostic enriched sample, unique papers, ordered quotas; not population estimate',
              'known_exclusions':sorted(KNOWN),'question_count':24,'output_count':48,'cases':selected,
              'local_source_sha256':private_hashes,'review_source':'model_assisted_staged_review','independent_human_review':'pending','new_paid_model_calls':0}
    freeze(ROOT/'reports/p2b_trace_audit_manifest_v1.json',manifest)
    freeze(local/'stage_a_contexts.json',packets_a)
    freeze(local/'stage_b_drafts.json',packets_b)
    freeze(local/'stage_c_reveal.json',reveal)
    template=[{'audit_id':r['audit_id'],'reviewer':None,'source':'human_review','status':'pending',
               'context_sufficient':None,'arms':{a:{'supported':None,'direct':None,'complete':None,'publishable':None,'notes':None} for a in ('A','B')}} for r in selected]
    freeze(local/'human_review_blank.json',template)
    print('Frozen: 24 questions / 48 outputs; contexts, masked drafts, reveal, blank human form saved locally. No model calls.')

if __name__=='__main__':main()
