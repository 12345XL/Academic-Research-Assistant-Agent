"""Offline QASPER-style English metrics; failures are not successful abstentions.

Normalization and max-over-references follow the public allenai evaluator.
Paragraph IDs replace texts only for complete unambiguous mapped references.
"""
from collections import Counter
import re
import string

from .evaluate import gold_references


def normalized_tokens(text):
    without_punctuation = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\b(a|an|the)\b', ' ', without_punctuation).split()


def token_f1(prediction, reference):
    p, r = normalized_tokens(prediction), normalized_tokens(reference)
    common = sum((Counter(p) & Counter(r)).values())
    return 2 * common / (len(p) + len(r)) if common else 0.0


def reference_answer(a):
    if a['unanswerable']: return 'unanswerable'
    if a.get('extractive_spans'): return ', '.join(a['extractive_spans'])
    if a.get('free_form_answer'): return a['free_form_answer']
    if type(a.get('yes_no')) is bool: return 'Yes' if a['yes_no'] else 'No'
    raise ValueError('No valid reference answer')


def score_answer(result, question):
    if result.get('generation', {}).get('answer_language') != 'en':
        raise ValueError('English reference F1 requires an explicitly English run')
    status = result['status']
    if status == 'answered':
        prediction = result.get('short_answer') or ' '.join(c['text'] for c in result['claims'])
    elif status == 'evidence_insufficient':
        prediction = 'unanswerable'
    else:
        prediction = ''
    f1 = max(token_f1(prediction, reference_answer(a)) for a in question['annotations'])
    gold_unanswerable = all(a['unanswerable'] for a in question['annotations'])
    boolean = all(type(a.get('yes_no')) is bool for a in question['annotations'])
    direct_boolean_correct = (float(any(normalized_tokens(prediction) == normalized_tokens(reference_answer(a))
                                       for a in question['annotations'])) if boolean else None)
    cited = {r['chunk_id'] for c in result.get('claims', []) for r in c['evidence']}
    refs, _ = gold_references(question)
    if gold_unanswerable:
        evidence_f1 = float(status == 'evidence_insufficient')
    elif refs:
        evidence_f1 = max(2 * len(cited & ref) / (len(cited) + len(ref)) for ref in refs)
    else:
        evidence_f1 = None
    return {'answer_f1':f1, 'evidence_f1_mapped':evidence_f1,
            'direct_boolean_accuracy':direct_boolean_correct, 'published':int(status=='answered'),
            'answerable_abstention':int(status=='evidence_insufficient') if not gold_unanswerable else None,
            'unanswerable_false_publication':int(status=='answered') if gold_unanswerable else None,
            'unanswerable_explicit_abstention':int(status=='evidence_insufficient') if gold_unanswerable else None}
