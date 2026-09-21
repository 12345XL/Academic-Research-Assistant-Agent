# QASPER 数据契约与 P1 审计

本项目使用 QASPER v0.3 的官方 train/dev 包作为科研论文问答的基础数据。当前实现只处理英文结构化正文与摘要；不调用模型、不需要 API 密钥、不下载 test，也不下载图表图像。数据来源为 [QASPER 官方项目页](https://allenai.org/data/qasper)和[官方数据卡](https://huggingface.co/datasets/allenai/qasper)。论文为 Dasigi et al. (2021), [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://aclanthology.org/2021.naacl-main.365/)。

## 1. 获取与复现

在项目根目录运行：

```bash
python scripts/prepare_qasper.py
```

脚本使用 Python 标准库，下载固定地址：

[qasper-train-dev-v0.3.tgz](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz)

2026-09-18 实际下载文件的 SHA256 已固定在代码中：

```text
a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a
```

该值是本项目记录的官方下载文件校验值，并非发布者签名或官方独立发布的校验值。后续重新运行会验证缓存；不匹配时停止，不能默默接受不同内容。

归档仅按两个明确成员名读取：`qasper-train-v0.3.json`、`qasper-dev-v0.3.json`。不使用 `extractall`；拒绝同名重复成员、符号链接和超出大小限制的成员。数据集也包含 README，但程序不会执行其中的任何内容。

生成结果：

| 文件 | 用途 | 是否可进入检索语料 |
| --- | --- | --- |
| `data/raw/qasper-train-dev-v0.3.tgz` | 原始快照 | 否 |
| `data/raw/manifest.json` | 来源 URL、归档与成员 SHA256 | 否 |
| `data/processed/papers.jsonl` | 论文目录、标题、摘要 | 是，作为论文元数据 |
| `data/processed/paragraphs.jsonl` | 原始正文段落及摘要、定位字段 | 是，唯一段落检索输入 |
| `data/processed/questions.jsonl` | 问题、标准答案、多参考标注、证据映射 | **否，仅离线评测** |
| `reports/data_audit.json` | 各 split 计数、边界与输出 SHA256 | 否，可提交审计记录 |

原始数据及处理结果由 Git 忽略；仓库保留下载脚本和审计报告。数据卡标示 CC BY 4.0，保留上述来源、作者和转换说明。个人上传论文的使用与分享范围不能由 QASPER 的许可推导。

## 2. 论文与段落契约

`load_papers(Path)`、`load_paragraphs(Path)` 返回 JSON 对象列表。训练与开发论文合存在文件中，但每条记录保留 `split`（`train` 或 `dev`）。

论文字段：`paper_id, title, abstract, split, source, version`。

段落字段：`chunk_id, paper_id, title, section_name, section_index, paragraph_index, text, text_sha256, split, source, version`。

- `source` 为 `qasper`，`version` 为 `qasper-v0.3`。
- 正文 ID 为 `论文ID:s章节原始序号:p段落原始序号`，例如 `1909.00694:s4:p1`。
- 空白正文段落跳过，但原始章节/段落序号不重排，便于回到源文件验证。
- 每篇非空摘要形成一个段落，ID 为 `论文ID:abstract`，`section_name=Abstract`，`section_index=-1`，`paragraph_index=0`。
- `text` 保留原始文字；`text_sha256` 是该原始文字 UTF-8 字节的 SHA256（包括原始空白），便于验证内容完整性；分词或展示不能覆盖事实源。
- 摘要参加检索，所以指标须说明“摘要 + 正文”的语料范围。

这些位置是 QASPER 结构化数据中的位置，**不是 PDF 页码**。当前阶段不能生成虚假的页码引用。

### 2.1 浏览与排序元数据

QASPER 原始论文记录没有发布日期、会议或 CCF 等级。为支持论文库排序，项目把补充信息保存在独立的 `paper_metadata` 表；它不进入 BM25 语料、向量或离线标准答案。仓库冻结了 2026-09-21 从 [arXiv API](https://info.arxiv.org/help/api/user-manual.html) 获取的 1,169 篇论文元数据快照 `metadata/qasper_arxiv.json`，并记录来源和获取时间。

- `arxiv_submitted_at` 是 arXiv `<published>`，含义是第一版提交时间，不是会议或期刊正式出版时间。
- `arxiv_journal_ref` 和 DOI 原样保存为来源字段。CCF 匹配只读取 `journal_ref`，不根据标题、作者或模型猜测发表场所。
- CCF 级别来自 CCF 官方的[人工智能](https://www.ccf.org.cn/Academic_Evaluation/AI/)、[数据库/数据挖掘/内容检索](https://www.ccf.org.cn/Academic_Evaluation/DM_CS/)和[交叉/综合/新兴](https://www.ccf.org.cn/Academic_Evaluation/Cross_Compre_Emerging/)目录，目录核对日期为 2026-09-21。
- CCF 对会议通常只认可 full/regular paper。单个 arXiv `journal_ref` 未必能证明投稿类型，因此程序排除 workshop、findings、short paper、demo、poster 等明显非正式场次；剩余标签也只表示**发表载体的目录级别**，不作为论文质量评分，也不宣称已证明 full/regular 身份。
- 无来源、来源歧义或未在已实现目录规则中命中的论文显示“未分级”。未分级不等于 CCF C 或论文质量低。

排序在 PostgreSQL 查询中先作用于全量结果，再做分页。可选值是论文编号升/降序、标题升/降序、arXiv 首次提交时间新/旧顺序，以及 CCF A → B → C → 未分级；同级再按时间和编号稳定排序。刷新快照需要显式运行 `scripts/prepare_arxiv_metadata.py`，脚本遵守 arXiv API 的批量与间隔要求；随后运行 `scripts/ingest_paper_metadata.py`。完整快照覆盖、快照 SHA-256 和等级统计由 `reports/paper_sort_metadata.json` 审计。

## 3. 标注与证据契约

问题字段：`question_id, paper_id, question, split, annotations`。每条标注独立保留以下原始字段和值：

`unanswerable, yes_no, extractive_spans, free_form_answer, evidence, highlighted_evidence`。

另外记录 `annotation_id`、`missing_answer_fields`、`matched_chunk_ids`、`unmapped_evidence`、`figure_evidence`、`evidence_mappings`。最后一项记录每条文本证据对应的一个或多个段落 ID，便于处理重复段落，不把重复位置静默选成其中一个。

证据处理规则：

1. 原始 `evidence` 保留不变。只在匹配时统一空白字符；不进行模糊匹配或自动改写。
2. 文本证据只能映射到同一篇论文的正文或摘要。找不到的保留在 `unmapped_evidence`。
3. `FLOAT SELECTED` 开头的图表证据单独进入 `figure_evidence`，不伪装为已经识别的正文。
4. `highlighted_evidence` 是句子级信息，本阶段原样保留，不以句子替代段落金标。
5. 多参考答案全部保留，包括回答是否可能的分歧；评测时需要显式说明多参考聚合策略。
6. 缺失的答案字段保留为 `null` 并列入 `missing_answer_fields`；不能把未知标签当成 `false`。
7. `yes_no=false` 表示答案为“No”，不是缺失；`unanswerable=false` 也必须与字段缺失区分。

**空证据不等于不可回答。** 不可回答只由明确的 `unanswerable=true` 表达。图表、空证据和映射失败也不能自动改标成不可回答。

## 4. 2026-09-18 实测审计结果

本节数字来自实际转换，与 `reports/data_audit.json` 对应；不包含 test。

| 项目 | train | dev |
| --- | ---: | ---: |
| 论文 | 888 | 281 |
| 问题 | 2,593 | 1,005 |
| 答案标注 | 2,675 | 1,764 |
| 正文非空段落 | 46,882 | 13,266 |
| 摘要段落 | 888 | 281 |
| 检索段落合计 | 47,770 | 13,547 |
| 跳过的空白正文段落 | 805 | 328 |
| 含多个参考标注的问题 | 82 | 744 |
| 多参考可回答性不一致的问题 | 7 | 75 |
| 明确不可回答标注 | 281 | 163 |
| 标为可回答但证据为空的标注 | 86 | 49 |
| 含图表证据的问题 | 320 | 164 |
| 能映射到正文/摘要的文本证据项 | 3,676 | 2,426 |
| 无法映射的文本证据项 | 147 | 129 |
| 同一证据匹配多个原始位置的项 | 6 | 12 |

train/dev 合计 1,169 篇论文、3,598 个问题、4,439 条标注、61,317 个检索段落。两个 split 的论文 ID、问题 ID 均无交叉。未把 test 的论文和答案下载或并入开发数据。

部分无法映射的证据类似章节标题或短标签；本阶段保留这些异常，不在不知道原始意图时自动修复。它们不会被删除后隐藏在统计结果外。

## 5. 评测边界与后续工作

- QASPER 原始任务给定论文，本阶段检索评测应限定 `paper_id`。全库找论文、跨论文比较需另立测试，不能冒充相同基准。
- 官方 dev 可用于开发选择；最终验收在冻结配置后再启用 test。不能一边反复调参、一边把同一集合称作未见测试集。
- 段落检索指标只适用于具有可映射文本证据的问题。报告必须列出纳入与排除的分母以及原因；图表题、空证据题仍保留在数据集中。
- 图表题可能同时包含文本证据。命中文本不代表已经读取表格或图像，不能据此宣称完整解答。
- 段落召回好，不等于最终回答正确；P1 未证明答案生成、引用支持、拒答、多轮、权限或反馈进化效果。
- 自选 PDF 的解析、章节定位、表格抽取、论文版本、权限与多轮会话将在后续阶段独立实现和验收。
- 基准标准答案不能进入段落语料、回答提示词或在线 memory；离线评测代码才可读取 `questions.jsonl`。

## 6. 自动检查

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_dataset.py -v
```

当前 12 个离线测试覆盖：QA 与语料隔离、原始段落定位、原文空白保留及 UTF-8 内容哈希、空证据语义、多个参考答案、缺失字段、图表与未映射证据、重复段落、必要标识校验、train/dev 交叉校验、恶意归档路径与符号链接、manifest 与输出分离。
