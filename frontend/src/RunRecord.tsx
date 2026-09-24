import { useEffect, useRef, useState } from 'react';
import { errorMessage, isAbort, request } from './api';
import type { RunRecord, StoredRun } from './api';
import { useRunMonitor } from './useRunMonitor';

export const RUN_STATES: Record<RunRecord['state'], string> = {
  running: '运行中', interrupted: '运行中断', completed: '已完成', abstained: '未作答', blocked: '已拦截', failed: '运行失败',
};
const RUN_STAGES: Record<string, string> = {
  retrieve: '检索证据', context: '组装上下文', generate: '生成回答',
  citation_check: '检查引用', verify: '核验语义支持', publication_check: '检查发布条件', publish: '发布回答',
};
const STAGE_STATES: Record<RunRecord['stages'][number]['status'], string> = {
  running: '进行中', completed: '已完成', stopped: '在此停止', failed: '失败', not_run: '未执行',
};
const RUN_REASONS: Record<string, string> = {
  cancelled: '运行已取消', deadline_exceeded: '任务截止时间已到',
  tool_not_allowed: '当前工具不在允许范围内', call_budget_exceeded: '模型调用次数预算已用尽', input_budget_exceeded: '模型输入字符预算已用尽',
  repair_exhausted: '一次修复后仍未通过检查', access_denied: '当前身份无权访问',
  run_store_unavailable: '运行记录存储暂不可用', server_restarted: '服务重启，未完成运行已中断',
  published: '回答已通过当前检查并发布', generation_not_configured: '生成模型尚未配置',
  no_retrieved_evidence: '检索未找到可用证据', context_budget_excluded_all: '上下文预算未容纳任何证据',
  generator_abstained: '生成模型判断证据不足，未作答', citation_invalid: '引用完整性检查未通过',
  semantic_verification_failed: '语义支持核验未通过', provider_timeout: '模型服务请求超时',
  provider_http_error: '模型服务返回错误', provider_connection_error: '无法连接模型服务',
  invalid_output: '模型输出格式不符合要求', incomplete_output: '模型输出不完整', model_refused: '模型拒绝回答',
  corpus_changed: '运行期间论文语料发生变化', evidence_integrity_failed: '原文证据完整性检查未通过',
  vector_unavailable: '向量检索暂不可用', reranker_unavailable: '重排模型暂不可用',
  paper_not_found: '未找到当前论文', invalid_request: '请求参数无效',
  dependency_unavailable: '依赖服务暂不可用', internal_error: '运行出现内部错误',
};
const runDuration = (value: number | null | undefined) => value == null ? '—' : `${value.toLocaleString('zh-CN', { maximumFractionDigits: 1 })} ms`;

export function RunRecordView({ run, historical = false, live = false }: { run: RunRecord; historical?: boolean; live?: boolean }) {
  return <details className="run-record">
    <summary><span>运行记录</span><span className={`run-state ${run.state}`}>{RUN_STATES[run.state]}</span><span>{runDuration(run.latency_ms)}</span></summary>
    <div className="run-record-body">
      <p className="run-record-hint">{live ? '这是最近一次读取的服务端阶段快照，可能略有延迟；耗时截至该快照，不是持续计时。' : historical ? '这是历史运行快照，不包含答案全文，也不代表实时进度。' : '这是请求结束后返回的记录，不是实时进度。'}记录反映执行与检查结果，回答质量仍需对照原文判断。</p>
      <p className="run-reason">{RUN_REASONS[run.reason || ''] || (run.state === 'running' ? '运行尚未结束' : '请查看原因码')}<code>{run.reason}</code></p>
      <dl className="run-meta"><div><dt>{run.state === 'running' ? '当前状态' : '最终状态'}</dt><dd>{RUN_STATES[run.state]} <code>{run.state}</code></dd></div><div><dt>{run.state === 'running' ? '当前阶段' : '结束阶段'}</dt><dd>{RUN_STAGES[run.terminal_stage || ''] || run.terminal_stage || '尚未开始'} <code>{run.terminal_stage}</code></dd></div><div><dt>调用记录</dt><dd className="mono">Trace {run.trace_id}</dd></div></dl>
      {run.limits && <dl className="run-meta" aria-label="运行预算上限"><div><dt>任务截止时间</dt><dd>{run.limits.deadline_seconds} 秒</dd></div><div><dt>模型调用上限</dt><dd>{run.limits.max_model_calls} 次</dd></div><div><dt>单次输入字符上限</dt><dd>{run.limits.max_prompt_chars.toLocaleString('zh-CN')}</dd></div><div><dt>单次输出 Token 上限</dt><dd>{run.limits.max_completion_tokens.toLocaleString('zh-CN')}</dd></div><div><dt>修复上限</dt><dd>{run.limits.max_repairs} 次</dd></div></dl>}
      {run.budget && <><dl className="run-meta" aria-label="运行预算使用"><div><dt>模型调用预留次数</dt><dd>{run.budget.model_calls} 次</dd></div><div><dt>已预留输出 Token</dt><dd>{run.budget.completion_tokens_reserved.toLocaleString('zh-CN')}</dd></div><div><dt>模型报告输入 / 输出 Token</dt><dd>{run.budget.reported_prompt_tokens.toLocaleString('zh-CN')} / {run.budget.reported_completion_tokens.toLocaleString('zh-CN')}</dd></div></dl><p className="run-record-hint">预留额度不是实际消耗或账单。{run.budget.usage_unknown_calls > 0 ? `有 ${run.budget.usage_unknown_calls} 次调用未返回用量，以上用量不完整。` : '用量以模型服务返回值为准。'}</p></>}
      <div className="table-scroll"><table className="run-stages" aria-label="运行阶段记录"><thead><tr><th scope="col">阶段</th><th scope="col">结果</th><th scope="col">耗时</th></tr></thead><tbody>{run.stages.map(stage => <tr key={stage.name}><th scope="row">{RUN_STAGES[stage.name] || stage.name}<code>{stage.name}</code></th><td>{STAGE_STATES[stage.status]}</td><td>{runDuration(stage.latency_ms)}</td></tr>)}</tbody></table></div>
      {run.attempts?.some(attempt => attempt.attempt > 0) && <div className="table-scroll"><table className="run-stages" aria-label="阶段尝试记录"><thead><tr><th>阶段</th><th>尝试</th><th>结果</th><th>耗时</th></tr></thead><tbody>{run.attempts.map((attempt, index) => <tr key={`${attempt.name}-${index}`}><th scope="row">{RUN_STAGES[attempt.name] || attempt.name}</th><td>{attempt.attempt === 0 ? '初次' : `修复 ${attempt.attempt}`}</td><td>{STAGE_STATES[attempt.status as keyof typeof STAGE_STATES] || attempt.status}</td><td>{runDuration(attempt.latency_ms)}</td></tr>)}</tbody></table></div>}
    </div>
  </details>;
}

export function RunProgress({ runId, finalRun, waiting = false, notice = '', initial }: {
  runId: string; finalRun?: RunRecord | null; waiting?: boolean; notice?: string; initial?: StoredRun;
}) {
  const finished = Boolean(finalRun && finalRun.state !== 'running');
  const { record, phase, error, retry } = useRunMonitor(runId, finished, initial);
  if (finished) return <RunRecordView run={finalRun!} />;
  const run = record?.snapshot.run;
  const terminal = phase === 'terminal';
  const label = terminal ? RUN_REASONS[record?.reason || ''] || '运行已结束'
    : phase === 'retrying' || phase === 'unavailable' ? '运行状态尚未确认'
    : record?.cancel_requested ? '取消请求已接受，正在确认终态'
    : phase === 'pending' ? '正在等待服务端运行记录'
    : run?.terminal_stage ? `当前阶段：${RUN_STAGES[run.terminal_stage] || run.terminal_stage}` : '运行已受理，等待开始';
  return <section className="run-progress" aria-label="当前运行状态">
    <p className="run-progress-label" role="status">{label}</p>
    {terminal ? <p className="run-record-hint">{record?.reason === 'published' && waiting
      ? '服务端已完成发布检查，正在接收回答；状态记录不会代替答案正文。'
      : '已读取服务端终态，自动查询已停止。运行记录不包含答案全文。'}</p>
      : <p className="run-record-hint">{phase === 'unavailable' ? '自动查询已停止。' : '执行阶段约每秒更新；查询异常时会降低频率。'}回答仅在核验通过后展示。</p>}
    {notice && !terminal && <p className="generation-hint">{notice}</p>}
    {error && <p className="error-notice">{error}</p>}
    {phase === 'unavailable' && <button className="secondary-button" onClick={retry}>重新查询状态</button>}
    {run && <RunRecordView run={run} historical={terminal} live={!terminal} />}
  </section>;
}

// The caller remounts this component when the paper or credential changes.
export function RunHistory({ paperId }: { paperId: string }) {
  const [items, setItems] = useState<StoredRun[]>([]);
  const [selected, setSelected] = useState<StoredRun | null>(null);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);

  async function load(runId?: string) {
    controller.current?.abort();
    const next = new AbortController(); controller.current = next;
    setLoading(true); setError('');
    try {
      if (runId) {
        const record = await request<StoredRun>(`/api/v1/runs/${encodeURIComponent(runId)}`, next.signal);
        if (!next.signal.aborted) setSelected(record);
      } else {
        const response = await request<{ items: StoredRun[] }>(`/api/v1/runs?limit=20&paper_id=${encodeURIComponent(paperId)}`, next.signal);
        if (!next.signal.aborted) { setItems(response.items); setLoaded(true); setSelected(null); }
      }
    } catch (failure) {
      if (!next.signal.aborted && !isAbort(failure)) setError(errorMessage(failure));
    } finally { if (!next.signal.aborted) setLoading(false); }
  }

  return <details className="run-history"><summary>当前论文的最近运行</summary><div className="run-history-body">
    <div className="history-heading"><p>最近 20 条记录 · 列表手动刷新，打开未完成运行后自动查询状态。历史不包含答案全文。</p><button type="button" className="secondary-button" onClick={() => void load()} disabled={loading}>{loading ? '读取中…' : '刷新运行记录'}</button></div>
    {error && <p className="error-notice" role="alert">{error}</p>}
    {!loaded && <p className="run-record-hint">点击刷新，读取当前身份可见的运行记录。</p>}
    {loaded && !items.length && <p className="run-record-hint">当前论文暂无运行记录。</p>}
    {!!items.length && <ul className="history-items">{items.map(item => <li key={item.run_id}><button type="button" disabled={loading} onClick={() => void load(item.run_id)} aria-label={`查看运行 ${item.run_id}`}><code>{item.run_id.slice(0, 12)}</code><span>{new Date(item.created_at).toLocaleString('zh-CN', { hour12: false })}</span><span>{RUN_STATES[item.state] || item.state}</span><span>{item.cancel_requested && item.state === 'running' ? '已请求取消' : RUN_REASONS[item.reason || ''] || item.reason}</span></button></li>)}</ul>}
    {selected && <div className="history-detail"><p className="run-record-hint">运行 {selected.run_id} · 打开时记录版本 {selected.revision}</p>{selected.state === 'running' ? <RunProgress key={selected.run_id} runId={selected.run_id} initial={selected} /> : selected.snapshot.run ? <RunRecordView key={selected.run_id} run={selected.snapshot.run} historical /> : <p className="run-record-hint">此记录暂未保存阶段快照。当前状态：{RUN_STATES[selected.state]}</p>}</div>}
  </div></details>;
}
