# 科研论文助手 Agent

面向 AI 论文阅读的可信 RAG 与受控反馈改进系统，使用 QASPER 正文建立可核对的证据链。P2A、P2B-1/2 已完成；当前已完成 **P2B-3 第一轮生成与核验基础工程**，模型质量仍待改进。工作台保留方向/排序/可调整分栏/PDF 入口，支持三种检索方式、可选重排，以及 DeepSeek 生成、逐字引用检查、模型支持关系核验。真实四题联调已发现核验漏判，不能将“模型核验通过”当作答案正确保证。

已完成 **P2B-4 参数与生成质量对照**：RRF 20 组网格仍选择 K60/等权；240 题双策略评测已完成，新版仍存在不可答题误发布，因此只作为实验选项。最新 **P2B-5** 复核了其中 24 题/48 份既有输出，发现回答目标、标注粒度和评分格式问题；新增项目模型 API 调用为 0，独立人工审阅待完成，未改变在线策略。详见下方报告。

当前已完成 **P3 指定论文、单实例 Harness 工程**：Python 显式状态机、任务时限与调用预算、服务端论文授权、运行元数据持久化、取消与重启中断、一次可选修正及重新核验。继续使用原问答工作台；默认不修正，未配置授权策略时明确为本机公共语料。人工审阅扩展暂停，24 题标注不是继续开发前提；P2B 质量问题继续保留。

## 先看什么

- [P4-3 文本 PDF 与学习路线](docs/P4-3文本PDF与学习路线.md)：已接入本机公共文本 PDF 上传、页序引用、幂等入库及原问答/反馈；339 项后端、69 项前端工程测试通过。当前 PDF 仅支持英文词法检索，不含 OCR、复杂版面识别或私有上传。
- [P4-2 反馈追溯与学习路线](docs/P4-2反馈追溯与学习路线.md)：已完成有帮助/存在问题反馈、修改与历史查看；绑定已发布回答快照，复用权限和 PostgreSQL。321 项后端、62 项前端测试及构建通过，新增付费调用 0；PDF 后续进展见 P4-3，策略改进留在 P6。
- [P4-1 运行交互与学习路线](docs/P4-1运行交互与学习路线.md)：阶段自动更新、取消确认、查询与执行失败的区别。
- [P3 验收与学习路线](docs/P3验收与学习路线.md)：当前链路、预算/权限/持久化/修正实现、295 项后端与 37 项前端工程测试、学习顺序与简历表述。
- [P3-1 运行状态与学习路线](docs/P3-1运行状态与学习路线.md)：第一步增量的历史盘点、阶段控制、失败原因和旧调用讲解。
- [P2B-6 人工审阅口径与操作](docs/P2B-6人工审阅口径与操作.md)：本地分阶段复核页面、部分回答口径与未完成的人工裁定。
- [P2B-5 轨迹审阅结果与决策](docs/P2B-5轨迹审阅结果与决策.md)：真实失败诊断、审阅不确定性、下一步假设及停止条件。
- [P2B-5 冻结审阅协议](docs/P2B-5轨迹审阅协议.md)：抽样、分阶段记录、标签来源与统计分母。
- [P2B-4 实验结果与学习路线](docs/P2B-4实验结果与学习路线.md)：候选数与 RRF 常数的区别、真实分层指标、费用和未解决的漏判。
- [P2B-3 验收与学习路线](docs/P2B-3验收与学习路线.md)：DeepSeek 接入、真实用量、核验漏判和下一步质量评测。
- [P2B 后续优化与原项目对照](docs/P2B后续优化计划与原项目对照.md)：分层评测、K/加权融合实验设计及原行政助手源码差异。
- [项目规划](项目规划.md)：范围、阶段和工程/学习双重验收。
- [架构选型与数据流](docs/架构选型与数据流.md)：为什么选 React、FastAPI、PostgreSQL、S3，以及替代方案和失败边界。
- [P2B-2 验收与学习路线](docs/P2B-2验收与学习路线.md)：重排取舍、六组消融、输入预算与失败边界。
- [P2B-1 验收与学习路线](docs/P2B-1验收与学习路线.md)：三组实测指标、源码阅读顺序、索引恢复与边界。
- [P2B 检索实验协议](docs/P2B检索实验协议.md)：模型、窗口、三组对照、分阶段范围和评测边界。
- [P2A 验收与运行](docs/P2A验收与运行.md)：实测结果、复现步骤、故障处理与源码阅读顺序。
- [P1 验收与学习路线](docs/P1验收与学习路线.md)：冻结的 BM25 基线。
- [数据契约](docs/DATASET.md)：QASPER 来源、正文与评测标注隔离、转换规则。
- [科研论文助手Agent面试问题](科研论文助手Agent面试问题.md)：预备问题；用户开始面试前不直接提问。

## 本机启动

本阶段在 macOS ARM、Python 3.12、Node 20、PostgreSQL 18、SeaweedFS S3 上实测。需要先安装 Python、Node、PostgreSQL、pgvector 扩展和 SeaweedFS；可参考上述验收文档的具体安装与恢复说明。所有命令从本仓库根目录执行：

```bash
brew install postgresql@18 pgvector seaweedfs
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
npm --prefix frontend ci
.venv/bin/python scripts/prepare_qasper.py
.venv/bin/python scripts/local_services.py up
.venv/bin/python scripts/ingest_qasper.py --migrate
.venv/bin/python scripts/ingest_paper_metadata.py
```

`local_services.py up` 首次运行会创建被 Git 忽略的 `.env`，生成本机随机密码，并将数据库和对象文件保存在 `.local/`。准备脚本下载并校验官方 train/dev 数据；首次运行需要网络。`ingest_paper_metadata.py` 把仓库中冻结的 arXiv 元数据快照和保守的 CCF 发表载体匹配结果写入独立表，不改变 RAG 正文。不要提交 `.env`、`.local/` 或语料全文。

首次启用向量检索时，额外安装本地推理依赖并下载固定模型；以下操作不需要付费 API：

```bash
.venv/bin/python -m pip install -r requirements-retrieval-lock.txt
.venv/bin/python scripts/prepare_embedding_model.py
.venv/bin/python scripts/build_vector_index.py --device mps
```

首次启用“模型重排”还需下载独立的交叉编码器，已安装 P2B-1 推理依赖即可使用：

```bash
.venv/bin/python scripts/prepare_reranker_model.py
```

`mps` 用于 Apple GPU；其他设备先用 `--device cpu`。API 默认 CPU 推理，可用环境变量 `RESEARCH_EMBEDDING_DEVICE=mps` 选择向量编码设备，`RESEARCH_RERANKER_DEVICE` 单独选择重排设备。论文导入或换版后重新运行索引脚本；同版本完整索引会复用，未完成任务会继续。没有向量模型仍可使用 BM25。

需要生成回答时，在本机 `.env` 添加 `DEEPSEEK_API_KEY`，设置 `DEEPSEEK_MODEL=deepseek-flash` 和 `RESEARCH_GENERATION_ENABLED=true` 后重启 API。密钥仅在后端使用；不要放进前端 VITE 环境变量或提交到仓库。未配置仍可检索。生成会产生模型 API 费用，默认每题最多两次请求；显式开启一次修正后最多四次，仍受服务端预算约束。页面默认保持证据检索模式。

已有数据升级 P3 / P4 时先执行 `.venv/bin/python scripts/migrate_database.py`，只更新表结构，无需重新导入或重建索引。授权文件格式及预算配置见 P3 文档与 `.env.example`。PostgreSQL 运行存储限制单 worker，不能同时启动多个 API 实例。

在两个终端分别启动：

```bash
.venv/bin/python scripts/run_api.py
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

打开 [论文工作台](http://127.0.0.1:5173/)；API 文档在 [127.0.0.1:8011/docs](http://127.0.0.1:8011/docs)。前端开发服务器将 `/api`、`/health` 等请求转发到本机后端。单独检查依赖可运行 `.venv/bin/python scripts/local_services.py status`，结束本机数据库与对象服务可运行 `.venv/bin/python scripts/local_services.py down`；数据目录会保留。

若项目服务已启动，通常无需重复导入；再次执行导入会核对并复用同一内容。数据库被清空时，重新执行 `ingest_qasper.py --migrate`。代码中不会把评测问题或标准答案导入在线数据库、对象桶或检索索引。

## 验证

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
RUN_STORAGE_INTEGRATION=1 PYTHONPATH=src .venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
.venv/bin/python scripts/verify_workbench.py
.venv/bin/python scripts/check_markdown.py
```

集成测试要求本机 PostgreSQL 与 S3 已启动，并使用临时数据库/桶，不修改主语料。完整回归比较冻结 P1 的 745 道 dev 文本证据题的检索段落 ID；其问题只由离线验证脚本读取。报告见 [P2A 存储回归](reports/p2a_storage_regression.json)。向量和混合检索对照运行 `.venv/bin/python scripts/evaluate_hybrid.py --device cpu`，完整结果见 [P2B 三组评测](reports/p2b_retrieval_dev.json)。重排六组对照运行 `.venv/bin/python scripts/evaluate_reranker.py --device mps`，结果见 [P2B 重排消融](reports/p2b_reranker_dev.json)。

| 已实测项目 | 结果 |
|---|---:|
| train + dev 论文 | 1,169 |
| 可检索正文段落（含摘要） | 61,317 |
| 结构化论文正文对象 | 1,169 |
| arXiv 首次提交时间 | 1,169 / 1,169 |
| arXiv 主分类与官方 PDF 链接 | 1,169 / 1,169 |
| 聚合研究方向 | 10（自然语言处理 953，其他方向合计 216） |
| 可保守映射的 CCF 发表载体 | 33（A 12、B 20、C 1） |
| 与 P1 冻结检索排名比较 | 745 题，0 处差异 |
| P1 基线 Hit@5 / 证据 Recall@5 / MRR@5 | 58.52% / 50.64% / 0.3330 |
| BGE 向量 Hit@5 / 证据 Recall@5 / MRR@5 | 66.17% / 58.23% / 0.4170 |
| RRF 混合 Hit@5 / 证据 Recall@5 / MRR@5 | 65.91% / 58.11% / 0.4148 |
| BM25 + 重排 Hit@5 / 证据 Recall@5 / MRR@5 | 70.60% / 62.47% / 0.4547 |
| 向量 + 重排 Hit@5 / 证据 Recall@5 / MRR@5 | 74.09% / 65.38% / 0.4808 |
| 混合 + 重排 Hit@5 / 证据 Recall@5 / MRR@5 | 73.29% / 64.73% / 0.4710 |
| 窗口向量 / 维度 | 61,467 / 384 |

这些是指定论文范围内的**检索**指标，不是答案正确率、拒答率或官方 QASPER 问答分数。官方 test 未用于开发调参。

## 代码与边界

```text
frontend/src/               React + TypeScript 工作台
src/research_agent/
  api.py                     FastAPI 契约与错误状态
  harness.py / runtime.py     显式阶段流、时限、调用预算与执行约束
  access.py / run_store.py    服务端论文授权、运行元数据与重启中断
  generation.py              草稿、引用检查、语义核验与一次可选修正
  service.py                 三种检索路线、数据库事实回查与版本刷新
  embeddings.py              固定本地模型、token 窗口与归一化编码
  reranking.py               问题-段落联合评分、长度预算与窗口聚合
  vector_store.py            pgvector 精确检索、分批构建与就绪发布
  storage.py                 PostgreSQL 与 S3 适配器
  ingestion.py               白名单数据校验、对象写入与版本发布
  paper_metadata.py          arXiv 主分类方向映射与 CCF 发表载体规则
  migrations/                正文、浏览元数据与运行记录数据库结构
  dataset.py / retrieval.py  QASPER 准备与 P1 BM25 基线
scripts/                     本地服务、导入、API 与回归入口
tests/                       单元和真实存储集成测试
```

方向筛选和排序由 PostgreSQL 在分页前执行。方向是对 arXiv 官方主分类的确定性聚合，不根据标题猜测；时间字段取自 arXiv API 的首次提交时间，不是期刊出版时间；CCF 标签来自 arXiv `journal_ref` 与 CCF 官方目录的保守匹配，只表示发表载体目录级别。原论文 PDF 由 arXiv 官方链接提供，未复制进对象存储；对象存储中的结构化正文仍会单独做哈希校验。来源、刷新方式和限制见[数据契约](docs/DATASET.md)。

当前是本机单实例研发环境，已接入模型回答、引用和支持关系核验，以及可选 Bearer 论文授权；模型核验通过仍不保证回答正确。已支持本机公共文本 PDF 上传；尚无私有上传、OCR、跨论文问答、完整账号/组织管理、分层记忆或反馈自动部署。运行历史只保存元数据，不自动恢复付费调用；明确提交反馈时另存问题、已发布回答及引用快照，支持修改与追溯，不自动用于模型学习。QASPER 结构化正文可定位到章节和段落，不代表已定位到 PDF 页码。

QASPER: Dasigi et al. (2021), [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://aclanthology.org/2021.naacl-main.365/)。[官方数据卡](https://huggingface.co/datasets/allenai/qasper)标注 CC BY 4.0。个人论文和原项目附件未随仓库发布。
