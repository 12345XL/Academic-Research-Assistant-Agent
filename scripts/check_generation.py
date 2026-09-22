"""Freeze four dev cases, then optionally run a bounded paid HTTP smoke check."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from research_agent.dataset import read_jsonl, sha256_file


def category(annotation):
    if annotation.get('figure_evidence') or annotation.get('missing_answer_fields'):
        return None
    kinds = []
    if annotation.get('unanswerable') is True: kinds.append('unanswerable')
    if type(annotation.get('yes_no')) is bool: kinds.append('boolean')
    if annotation.get('extractive_spans'): kinds.append('extractive')
    if annotation.get('free_form_answer'): kinds.append('abstractive')
    return kinds[0] if len(kinds) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Spend at most 8 model requests through localhost API')
    args = parser.parse_args()
    questions = sorted((q for q in read_jsonl(ROOT / 'data/processed/questions.jsonl') if q['split']=='dev'),
                       key=lambda q:q['question_id'])
    selected = {}
    for q in questions:
        kinds = {category(a) for a in q['annotations']}
        if len(kinds) != 1 or None in kinds: continue
        kind = next(iter(kinds))
        if kind not in selected: selected[kind] = q
    assert set(selected) == {'unanswerable','boolean','extractive','abstractive'}
    frozen = {'protocol':'paper-claims-v1-four-case-smoke', 'split':'dev',
              'config':{'mode':'dense','rerank':True,'top_k':5},
              'data_sha256': {name:sha256_file(ROOT / 'data/processed' / name)
                             for name in ('questions.jsonl','paragraphs.jsonl')},
              'cases':[{'category':kind,'question_id':q['question_id'],'paper_id':q['paper_id']}
                       for kind,q in sorted(selected.items())]}
    path = ROOT / 'reports/p2b_generation_selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == frozen, 'Frozen case selection changed'
    else:
        path.write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(frozen['cases']),flush=True)
    if not args.run: return
    # Questions are sent as user queries only; annotations remain offline in this process.
    out = ROOT / 'reports/p2b_generation_runtime.json'
    if out.exists(): raise SystemExit('Report exists: archive it explicitly before another paid run')
    local = ROOT / '.local/generation-smoke'; local.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'src/research_agent').glob('*.py')))).hexdigest()
    report={'protocol':frozen['protocol'],'generated_at':datetime.now(timezone.utc).isoformat(),
            'source_sha256':source_hash,'selection_sha256':sha256_file(path),'cases':[],
            'limitations':['Four-case smoke check only; not an answer accuracy or refusal benchmark',
                           'Same-model semantic judge can make correlated errors; manual review required']}
    with httpx.Client(base_url='http://127.0.0.1:8011',timeout=180) as client:
        for meta in frozen['cases']:
            q = selected[meta['category']]
            response=client.post('/api/v1/answer',json={'paper_id':q['paper_id'],'query':q['question'],**frozen['config']})
            response.raise_for_status()
            result=response.json()
            (local / (meta['category']+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            case={**meta,'status':result['status'],'trace_id':result['trace_id'],'trace':result['trace'],
                  'generation':result['generation'],'claim_count':len(result['claims']),
                  'published_claims_sha256':hashlib.sha256(json.dumps(result['claims'],sort_keys=True).encode()).hexdigest()}
            report['cases'].append(case)
            usage=[c.get('usage',{}) for c in result['generation']['calls']]
            print(json.dumps({'category':meta['category'],'status':result['status'],'usage':usage}),flush=True)
            out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            if result['status'] in ('model_unavailable','not_configured'): break
    usages=[c.get('usage',{}) for row in report['cases'] for c in row['generation']['calls']]
    report['usage']={k:sum(u.get(k,0) for u in usages) for k in ('prompt_tokens','completion_tokens')}
    report['model_calls']=sum(r['generation']['model_calls'] for r in report['cases'])
    report['estimated_peak_uncached_usd']=round((report['usage']['prompt_tokens']*.3+report['usage']['completion_tokens']*1.2)/1e6,6)
    report['cost_note']='2026-09-22 deepseek-flash peak uncached list rates; estimate, not an invoice; failed calls may lack usage'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__': main()
