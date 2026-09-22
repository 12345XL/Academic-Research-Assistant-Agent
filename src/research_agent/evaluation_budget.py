"""Local paid-evaluation ledger, not a production distributed budget service."""
import json
from pathlib import Path
import threading
import uuid

from .generation import ModelFailure

INPUT_USD_PER_M = .3
OUTPUT_USD_PER_M = 1.2


class EvaluationBudget:
    def __init__(self, path: Path, max_usd=1.0, max_calls=1000):
        self.path, self.max_usd, self.max_calls = path, max_usd, max_calls
        self.lock = threading.Lock()
        self.entries = {}
        if path.exists():
            for line in path.read_text().splitlines():
                event = json.loads(line)
                self.entries[event['id']] = event

    def append(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        self.entries[event['id']] = event

    def snapshot(self):
        return {'calls':len(self.entries), 'estimated_charged_usd':sum(e['charged_usd'] for e in self.entries.values()),
                'unsettled_calls':sum(e['state']=='reserved' for e in self.entries.values()),
                'limits':{'max_calls':self.max_calls,'max_estimated_usd':self.max_usd},
                'rate_basis':'2026-09-22 deepseek-flash peak uncached; conservative estimate, not invoice'}

    def complete(self, client, prompt, payload, schema):
        # UTF-8 bytes conservatively upper-bound BPE token counts; include schema/role overhead.
        byte_count = len((prompt + json.dumps(payload,ensure_ascii=False) +
                          json.dumps(schema.model_json_schema(),ensure_ascii=False)).encode()) + 2048
        reserve = (byte_count * INPUT_USD_PER_M + 2048 * OUTPUT_USD_PER_M) / 1e6
        ident=uuid.uuid4().hex
        with self.lock:
            state=self.snapshot()
            if state['calls']>=self.max_calls or state['estimated_charged_usd']+reserve>self.max_usd:
                raise ModelFailure('evaluation_budget_exhausted')
            self.append({'id':ident,'state':'reserved','charged_usd':reserve})
        meta={}
        try:
            parsed,meta=client.complete(prompt,payload,schema)
            return parsed,meta
        except ModelFailure as exc:
            meta=exc.metadata
            raise
        finally:
            usage=meta.get('usage',{})
            has_usage=all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens'))
            cost=(usage['prompt_tokens']*INPUT_USD_PER_M+usage['completion_tokens']*OUTPUT_USD_PER_M)/1e6 if has_usage else reserve
            with self.lock:
                self.append({'id':ident,'state':'settled' if has_usage else 'usage_unknown',
                             'charged_usd':cost,'usage':usage})
