"""Build an ignored, offline human-review page; no model or business API access."""
import hashlib
import json
from pathlib import Path

from prepare_trace_audit import main as validate_frozen_packets

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.local/trace-audit-v1'


def build():
    validate_frozen_packets()
    manifest_bytes = (ROOT / 'reports/p2b_trace_audit_manifest_v1.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    a = json.loads((LOCAL / 'stage_a_contexts.json').read_text())
    b = {r['audit_id']: r for r in json.loads((LOCAL / 'stage_b_drafts.json').read_text())}
    c = {r['audit_id']: r for r in json.loads((LOCAL / 'stage_c_reveal.json').read_text())}
    expected = {r['audit_id'] for r in manifest['cases']}
    assert {r['audit_id'] for r in a} == set(b) == set(c) == expected
    assert len(a) == len(expected) == 24
    cases = []
    for context in a:
        aid = context['audit_id']
        reveal = c[aid]
        cases.append({
            'audit_id': aid, 'question_id': reveal['question_id'], 'context': context,
            'drafts': b[aid]['arms'],
            'reveal': {
                'arm_mapping': reveal['arm_mapping'], 'category': reveal['category'],
                'annotations': reveal['annotations'], 'candidate_ids': reveal['candidate_ids'],
                'context_ids': reveal['context_ids'],
                'arms': {v: {'status': r['status'], 'checks': r['checks'],
                             'verdicts': [m['output'] for m in r['verifier_records']]}
                         for v, r in reveal['arms'].items()}}})
    data = {'schema': 'human-trace-review-v1', 'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'rubric_sha256': hashlib.sha256((ROOT / 'docs/P2B-6人工审阅口径与操作.md').read_bytes()).hexdigest(),
            'cases': cases}
    # The page never reads model_stage_*.json or the model review report.
    data['packet_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    encoded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    template = (ROOT / 'scripts/human_trace_review.html').read_text()
    assert template.count('__REVIEW_DATA__') == 1
    html = template.replace('__REVIEW_DATA__', encoded)
    target = LOCAL / 'review/index.html'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html)
    return target


if __name__ == '__main__':
    print(build())
