# 科研论文助手 Agent

面向 AI 论文阅读的可信 RAG 与受控反馈改进系统，使用 QASPER 正文建立可核对的证据链。按阶段实现，当前交付 **P2A：持久化论文检索工作台**。它能按 arXiv 主分类选择研究方向，再按编号、标题、首次提交时间或 CCF 发表载体等级排序论文；可以打开/下载官方 arXiv PDF、检索正文段落、查看相邻上下文、下载经过哈希校验的 RAG 结构化正文，并显示真实导入与存储状态。当前只返回 BM25 原文证据，**尚未生成答案或完成答案核验**。

## 先看什么

- [项目规划](项目规划.md)：范围、阶段和工程/学习双重验收。
- [架构选型与数据流](docs/架构选型与数据流.md)：为什么选 React、FastAPI、PostgreSQL、S3，以及替代方案和失败边界。
- [P2A 验收与运行](docs/P2A验收与运行.md)：实测结果、复现步骤、故障处理与源码阅读顺序。
- [P1 验收与学习路线](docs/P1验收与学习路线.md)：冻结的 BM25 基线。
- [数据契约](docs/DATASET.md)：QASPER 来源、正文与评测标注隔离、转换规则。
- [科研论文助手Agent面试问题](科研论文助手Agent面试问题.md)：预备问题；用户开始面试前不直接提问。

## 本机启动 P2A

本阶段在 macOS ARM、Python 3.12、Node 20、PostgreSQL 18、SeaweedFS S3 上实测。需要先安装 Python、Node、PostgreSQL 和 SeaweedFS；可参考上述验收文档的具体安装与恢复说明。所有命令从本仓库根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
npm --prefix frontend ci
.venv/bin/python scripts/prepare_qasper.py
.venv/bin/python scripts/local_services.py up
.venv/bin/python scripts/ingest_qasper.py --migrate
.venv/bin/python scripts/ingest_paper_metadata.py
```

`local_services.py up` 首次运行会创建被 Git 忽略的 `.env`，生成本机随机密码，并将数据库和对象文件保存在 `.local/`。准备脚本下载并校验官方 train/dev 数据；首次运行需要网络。`ingest_paper_metadata.py` 把仓库中冻结的 arXiv 元数据快照和保守的 CCF 发表载体匹配结果写入独立表，不改变 RAG 正文。不要提交 `.env`、`.local/` 或语料全文。

在两个终端分别启动：

```bash
.venv/bin/python scripts/run_api.py
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

打开 [论文工作台](http://127.0.0.1:5173/)；API 文档在 [127.0.0.1:8011/docs](http://127.0.0.1:8011/docs)。前端开发服务器将 `/api`、`/health` 等请求转发到本机后端。单独检查依赖可运行 `.venv/bin/python scripts/local_services.py status`，结束本机数据库与对象服务可运行 `.venv/bin/python scripts/local_services.py down`；数据目录会保留。

若项目服务已启动，通常无需重复导入；再次执行导入会核对并复用同一内容。数据库被清空时，重新执行 `ingest_qasper.py --migrate`。代码中不会把评测问题或标准答案导入在线数据库、对象桶或检索索引。

## 验证

```bash
.venv/bin/python -m pytest -q
RUN_STORAGE_INTEGRATION=1 .venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
.venv/bin/python scripts/verify_workbench.py
```

集成测试要求本机 PostgreSQL 与 S3 已启动，并使用临时数据库/桶，不修改主语料。完整回归比较冻结 P1 的 745 道 dev 文本证据题的检索段落 ID；其问题只由离线验证脚本读取。报告见 [P2A 存储回归](reports/p2a_storage_regression.json)。

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

这些是指定论文范围内的**检索**指标，不是答案正确率、拒答率或官方 QASPER 问答分数。官方 test 未用于开发调参。

## 代码与边界

```text
frontend/src/               React + TypeScript 工作台
src/research_agent/
  api.py                     FastAPI 契约与错误状态
  service.py                 BM25 候选、数据库事实回查与版本刷新
  storage.py                 PostgreSQL 与 S3 适配器
  ingestion.py               白名单数据校验、对象写入与版本发布
  paper_metadata.py          arXiv 主分类方向映射与 CCF 发表载体规则
  migrations/                正文与浏览元数据数据库结构
  dataset.py / retrieval.py  QASPER 准备与 P1 BM25 基线
scripts/                     本地服务、导入、API 与回归入口
tests/                       单元和真实存储集成测试
```

方向筛选和排序由 PostgreSQL 在分页前执行。方向是对 arXiv 官方主分类的确定性聚合，不根据标题猜测；时间字段取自 arXiv API 的首次提交时间，不是期刊出版时间；CCF 标签来自 arXiv `journal_ref` 与 CCF 官方目录的保守匹配，只表示发表载体目录级别。原论文 PDF 由 arXiv 官方链接提供，未复制进对象存储；对象存储中的结构化正文仍会单独做哈希校验。来源、刷新方式和限制见[数据契约](docs/DATASET.md)。

当前是本机单实例研发环境；没有个人 PDF 上传、模型答案、向量或混合检索、跨论文问答、用户权限、分层记忆、反馈自动部署。它们在后续阶段逐项实现并验收。QASPER 结构化正文可定位到章节和段落，不代表已定位到 PDF 页码。

QASPER: Dasigi et al. (2021), [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://aclanthology.org/2021.naacl-main.365/)。[官方数据卡](https://huggingface.co/datasets/allenai/qasper)标注 CC BY 4.0。个人论文和原项目附件未随仓库发布。
