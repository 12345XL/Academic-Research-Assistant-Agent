# 科研论文助手 Agent

面向 AI 论文阅读的可信 RAG 与受控反馈改进系统，使用 **QASPER + 自选论文**作为应用场景，分阶段实现并学习其原理。

**当前交付：P0 规划 + P1 证据检索基线。** 已实现官方数据导入、数据审计、单篇论文 BM25 检索、原文定位、CLI、FastAPI 和离线评测。当前 API 返回原文证据，不生成答案；完整 Agent、混合检索、权限、记忆、反馈闭环与正式前端还未完成。

## 先看什么

- [项目规划](项目规划.md)：目标、选型理由、替代方案、P0–P8 阶段和验收规则。
- [面试题库](科研论文助手Agent面试问题.md)：8 个维度、56 题，当前不开始提问。
- [P1 验收记录与学习路线](docs/P1验收与学习路线.md)：实际运行结果、边界和建议阅读顺序。
- [原项目参考与差异](docs/原项目参考与差异.md)：原代码可以借鉴什么，哪些叙述不能直接沿用。
- [数据契约](docs/DATASET.md)：正文与答案隔离、数据来源、许可、证据映射和审计。

## 本地运行

需要 Python 3.11+，本阶段在 Python 3.12 验证。P1 无模型 API 调用，不需要密钥、GPU、Docker 或服务型数据库。

在本仓库根目录执行（macOS / Linux）：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps .
.venv/bin/python scripts/prepare_qasper.py
.venv/bin/python -m uvicorn research_agent.api:app --app-dir src --host 127.0.0.1 --port 8010
```

下载脚本只读取固定官方 train/dev 包，并校验已记录的 SHA256；原始数据和处理文件被 Git 忽略。网络不可用时不能完成首次数据准备；已有缓存也会重新校验。

打开 [API 交互文档](http://127.0.0.1:8010/docs)，先通过 `GET /api/v1/papers` 选论文，再执行 `POST /api/v1/retrieve`：

```json
{
  "paper_id": "1503.00841",
  "query": "What method and dataset are used?",
  "top_k": 5
}
```

响应包含原文、论文标题、章节、段落 ID、数据版本与 BM25 分数。位置指向 QASPER 结构化文本，不是 PDF 页码。BM25 分数不是概率；`no_lexical_match` 也不表示已经判定论文不可回答。

`/health` 的 HTTP 200 表示进程存活，JSON 中的 `status=ready` 才表示数据已加载。若启动时缺数据，准备数据后需重启。当前服务只绑定本机，未提供身份认证或个人论文上传入口。

命令行演示：

```bash
PYTHONPATH=src .venv/bin/python -m research_agent.cli papers --limit 3
PYTHONPATH=src .venv/bin/python -m research_agent.cli search --paper-id 1503.00841 --query 'What method and dataset are used?' --top-k 5
```

源码运行显式指定 `PYTHONPATH=src` 或 `--app-dir src`，改完代码立即生效，不依赖 editable 安装的 `.pth` 文件。普通安装后的 `research-agent` 命令使用安装时的代码；修改源码后要重新安装或使用上述源码命令。

## 测试与基线评测

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m research_agent.evaluate
```

本地验收已通过。当前 GitHub 登录缺少 `workflow` scope，远端 CI 尚未启用；配置保存在 [工作流示例](docs/github-actions-test.yml.example)。获得相应权限后，才可将其放入 `.github/workflows/test.yml` 并验证运行结果。

数据审计见 [data_audit.json](reports/data_audit.json)，评测逐题记录见 [p1_retrieval_dev.json](reports/p1_retrieval_dev.json)。

| P1 实测项 | 结果 |
|---|---:|
| train + dev 论文 | 1,169 |
| train + dev 问题 | 3,598 |
| 可检索摘要与正文段落 | 61,317 |
| dev 纳入检索评测的问题 | 745 / 1,005 |
| Hit@5 | 58.52% |
| 证据 Recall@5 | 50.64% |
| MRR@5 | 0.3330 |

协议是**给定论文 ID、纯文本证据子集、BM25 top-5**；全语料正文用于 IDF，不读取答案建索引。多参考按每个指标独立取最佳完整文本参考。排除 75 道可回答性有分歧的问题、60 道一致标为不可回答的问题、125 道没有完整无歧义文本参考的问题。数据仍保留这些样本。图表型参考与文本型参考并存的题目可能纳入文本子集，因此成绩不代表图表理解。

这些指标是检索基线，**不是 QASPER 官方问答分数，也不是回答正确率或拒答率**。官方 test 尚未下载；没有根据此结果调参或挑选题目。

## 当前代码结构

```text
src/research_agent/
  dataset.py       # 官方数据获取、转换与审计
  retrieval.py     # 可读的 Okapi BM25 基线
  service.py       # 只读取论文与段落，返回原文引用
  api.py           # FastAPI 请求契约、检索与错误状态
  cli.py           # 命令行入口
  evaluate.py      # 离线读取标准标注，计算检索指标
scripts/prepare_qasper.py
tests/
docs/
reports/           # 小型审计和评测产物，可公开复现
```

目标技术栈为 React + TypeScript + Vite、FastAPI、PostgreSQL + pgvector、S3 兼容对象存储，Redis 按需求引入。当前 JSONL 与内存索引属于 P1 适配层，不代表这些目标组件已经接入。

## 数据署名与公开范围

QASPER: Dasigi et al. (2021), [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://aclanthology.org/2021.naacl-main.365/)。[官方数据卡](https://huggingface.co/datasets/allenai/qasper)标注 CC BY 4.0；本项目的转换规则见数据契约。个人论文和原项目附件未随仓库发布。

每阶段工程验收通过后提交代码、测试和结果；用户的学习验收单独记录。后续先讲解本阶段，再继续增加能力。
