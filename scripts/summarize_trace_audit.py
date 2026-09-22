"""Check and summarize frozen local reviews. No inference, database, or network calls.

Requires the original ignored .local audit packets. Public output contains labels,
IDs and reviewer rationale, never the source questions, drafts or paper paragraphs.
"""
from collections import Counter
import hashlib
import json

from prepare_trace_audit import ROOT, freeze


LOCAL = ROOT / '.local/trace-audit-v1'


def read(name):
    return json.loads((LOCAL / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cases, *, amendments=False):
    counts = {p: Counter() for p in ('v1', 'v2', 'all')}
    decisions = {p: Counter() for p in counts}
    refusals = {p: Counter() for p in counts}
    contexts = Counter()
    for case in cases:
        context = case['stage_a']['context_sufficiency']
        if amendments:
            for edit in case['stage_c']['amendments']:
                if edit['target'] == 'context_sufficiency':
                    assert context == edit['before']
                    context = edit['after']
        contexts[context] += 1
        for profile, arm in case['arms'].items():
            label = arm['stage_b']['publishable']
            if amendments:
                for edit in case['stage_c']['amendments']:
                    if edit['target'] == 'both_arms.publishable':
                        assert label == edit['before']
                        label = edit['after']
            for key in (profile, 'all'):
                counts[key][arm['status']] += 1
                if label == 'abstained':
                    assert not arm['semantic_verifier_called']
                    refusals[key][context] += 1
                    continue
                assert arm['semantic_verifier_called']
                decision = 'published' if arm['status'] == 'answered' else 'blocked'
                meaning = {'yes': 'acceptable', 'no': 'unacceptable', 'uncertain': 'uncertain'}[label]
                decisions[key][f'{meaning}_{decision}'] += 1
    return {'context_sufficiency': dict(contexts),
            'actual_statuses': {p: dict(c) for p, c in counts.items()},
            'semantic_gate_decisions': {p: dict(c) for p, c in decisions.items()},
            'generator_abstentions_by_context': {p: dict(c) for p, c in refusals.items()}}


def main():
    manifest_path = ROOT / 'reports/p2b_trace_audit_manifest_v1.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha(ROOT / 'reports/p2b_generation_quality_v3.json') == manifest['source_report_sha256']
    for name, expected in manifest['local_source_sha256'].items():
        assert sha(ROOT / '.local/generation-quality-v3' / name) == expected

    phase_hashes = {}
    for phase in ('a', 'b'):
        actual = sha(LOCAL / f'model_stage_{phase}.json')
        assert actual == (LOCAL / f'stage_{phase}_frozen.sha256').read_text().strip()
        phase_hashes[phase] = actual

    contexts = {r['audit_id']: r for r in read('stage_a_contexts.json')}
    drafts = {r['audit_id']: r for r in read('stage_b_drafts.json')}
    revealed = {r['audit_id']: r for r in read('stage_c_reveal.json')}
    stages = {s: {r['audit_id']: r for r in read(f'model_stage_{s}.json')} for s in ('a', 'b', 'c')}
    selected = {r['audit_id'] for r in manifest['cases']}
    assert len(selected) == len({r['paper_id'] for r in manifest['cases']}) == 24
    assert not {r['question_id'] for r in manifest['cases']} & set(manifest['known_exclusions'])
    for rows in (contexts, drafts, revealed, *stages.values()):
        assert set(rows) == selected

    human = read('human_review_blank.json')
    assert len(human) == 24 and {r['audit_id'] for r in human} == selected
    assert all(r['status'] == 'pending' and r['reviewer'] is None
               and r['context_sufficient'] is None
               and all(v is None for arm in r['arms'].values() for v in arm.values()) for r in human)

    cases = []
    quotes_checked = 0
    for item in manifest['cases']:
        aid = item['audit_id']
        context, draft, reveal = contexts[aid], drafts[aid], revealed[aid]
        # These allowlists also check the masking contract.
        assert set(context) == {'audit_id', 'question', 'paper_id', 'evidence'}
        assert set(draft) == {'audit_id', 'arms'}
        assert set(draft['arms']) == set(reveal['arm_mapping']) == {'A', 'B'}
        assert set(reveal['arm_mapping'].values()) == {'v1', 'v2'}
        text_by_id = {e['chunk_id']: e['text'] for e in context['evidence']}
        assert len(text_by_id) == 5 and 5 <= len(reveal['candidate_ids']) <= 20
        assert list(text_by_id) == reveal['context_ids']
        arms = {}
        for alias, profile in reveal['arm_mapping'].items():
            masked = draft['arms'][alias]
            assert set(masked) == {'answerable', 'claims'}
            actual = reveal['arms'][profile]
            model_label = stages['b'][aid]['arms'][alias]
            called = len(actual['verifier_records']) == 1
            if masked['answerable']:
                assert actual['checks']['citation_integrity'] == 'passed' and called
                assert model_label['publishable'] in ('yes', 'no', 'uncertain')
                for claim in masked['claims']:
                    for ref in claim['evidence']:
                        assert ref['quote'].strip() and ref['quote'] in text_by_id[ref['chunk_id']]
                        quotes_checked += 1
            else:
                assert not masked['claims'] and not actual['verifier_records']
                assert model_label['publishable'] == 'abstained'
                assert actual['status'] == 'evidence_insufficient'
            arms[profile] = {'masked_alias': alias, 'status': actual['status'],
                             'checks': actual['checks'], 'answer_f1': actual['metrics']['answer_f1'],
                             'semantic_verifier_called': called, 'stage_b': model_label}
        gold_ids = sorted({c for a in reveal['annotations'] for c in a['matched_chunk_ids']})
        cases.append({**item, 'pool': reveal['pool'], 'dataset_label': reveal['category'],
                      'stage_a': stages['a'][aid], 'arms': arms, 'stage_c': stages['c'][aid],
                      'actual_candidate_count': len(reveal['candidate_ids']),
                      'actual_context_count': len(text_by_id),
                      'gold_mapping_diagnostic_only': [
                          {'chunk_id': c,
                           'candidate_position': reveal['candidate_ids'].index(c) + 1 if c in reveal['candidate_ids'] else None,
                           'in_actual_context': c in text_by_id} for c in gold_ids],
                      'human_review': {'status': 'pending'}, 'synthetic_label': None})
    summary = summarize(cases)
    sensitivity = summarize(cases, amendments=True)
    report = {
        'protocol': 'trace-audit-v1', 'review_source': 'model_assisted_staged_review',
        'independent_human_review': 'pending', 'new_paid_model_calls': 0,
        'source_manifest_sha256': sha(manifest_path), 'pre_reveal_review_sha256': phase_hashes,
        'post_reveal_review_sha256': sha(LOCAL / 'model_stage_c.json'),
        'question_count': 24, 'output_count': 48,
        'selection_warning': 'Failure-enriched exposed development sample; no population quality estimate or causal v1/v2 comparison.',
        'label_policy': 'Pre-reveal judgments are primary. Explicit post-reveal amendments are separate sensitivity counts; neither is human ground truth.',
        'quoted_spans_rechecked': quotes_checked,
        'summary_before_reveal': summary, 'summary_with_explicit_amendments': sensitivity,
        'offline_no_semantic_gate': {
            'published_if_only_hard_citation_check_retained': sum(a['semantic_verifier_called'] for c in cases for a in c['arms'].values()),
            'actual_published': summary['actual_statuses']['all']['answered'],
            'additional_unacceptable_publications_by_pre_reveal_model_review': summary['semantic_gate_decisions']['all'].get('unacceptable_blocked', 0),
            'additional_uncertain_publications': summary['semantic_gate_decisions']['all'].get('uncertain_blocked', 0),
            'additional_acceptable_publications': summary['semantic_gate_decisions']['all'].get('acceptable_blocked', 0),
            'limitation': 'Decision-only counterfactual on existing drafts, not a rerun or deployment; corpus version/security checks still required.'},
        'cases': cases}
    freeze(ROOT / 'reports/p2b_trace_audit_review_v1.json', report)
    print(json.dumps({k: report[k] for k in ('question_count', 'output_count', 'quoted_spans_rechecked',
                                           'summary_before_reveal', 'summary_with_explicit_amendments',
                                           'offline_no_semantic_gate')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
