# P2A 验收与运行：持久化论文检索工作台

验收日期：2026-09-21。阶段工程验收通过；用户学习验收尚未开展。本记录只证明本机 QASPER 结构化正文工作台，不把未来的 Agent Harness、生成式回答、混合检索或反馈自进化写成已实现能力。

## 本阶段交付

页面使用 React + TypeScript + Vite，API 使用 FastAPI。PostgreSQL 保存论文、版本、段落、独立浏览元数据、对象引用、导入任务和发布状态；SeaweedFS S3 保存每篇论文经过白名单筛选的结构化正文 JSON。进程内 BM25 从数据库已发布正文构建候选索引，每次检索检查发布版本，并将命中 ID 回到 PostgreSQL 读取事实。下载正文时从 S3 读取字节并核对数据库登记的 SHA-256。

页面支持论文列表/标题搜索、全库服务端排序、单篇论文证据检索、top-k 选择、原文相邻段落查看、正文文件下载，以及数据库、对象存储、导入任务状态。排序可按编号、标题、arXiv 首次提交时间或 CCF 发表载体等级选择，并在分页前执行。切换论文或排序会清除前一篇的检索结果；较早的请求可取消。检索状态明确区分空结果与服务错误。页面写明 BM25 分数不是可信概率、词项未命中不等于不可回答。

QASPER 的论文正文是在线检索内容，问题、答案和人工证据标签只在离线评测目录中。导入程序只读取 `papers.jsonl` 与 `paragraphs.jsonl`，并拒绝混入 gold 字段；在线对象不含 `qas` 或标注。官方 test 未用于开发。

## 本机复现

本次实测环境为 macOS ARM、Python 3.12、Node 20、PostgreSQL 18.6（Homebrew）、SeaweedFS 4.47。首次安装依赖可在 macOS 执行 `brew install postgresql@18 seaweedfs`；若已安装，只需检查可执行文件可用。

在仓库根目录：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
npm --prefix frontend ci
.venv/bin/python scripts/prepare_qasper.py
.venv/bin/python scripts/local_services.py up
.venv/bin/python scripts/ingest_qasper.py --migrate
.venv/bin/python scripts/ingest_paper_metadata.py
```

首次 `up` 自动生成仅本机使用的随机认证值，存于被 Git 忽略的 `.env`；数据库和对象文件位于 `.local/`。两个服务仅绑定 `127.0.0.1`。已导入时重复执行导入会检查同一内容并复用对象/版本，故障后可安全重试。服务关闭和重启不会删除这些文件：

```bash
.venv/bin/python scripts/local_services.py status
.venv/bin/python scripts/local_services.py down
.venv/bin/python scripts/local_services.py up
```

后端和前端分别在两个终端启动：

```bash
.venv/bin/python scripts/run_api.py
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

打开 [论文工作台](http://127.0.0.1:5173/) 或 [API 文档](http://127.0.0.1:8011/docs)。`GET /health` 表示 API 进程状态；`GET /ready` 同时要求 PostgreSQL 与 S3 可用。若文件下载失败而数据库仍可用，已发布正文的检索可继续，页面会分别显示对象存储故障。

## 工程验收证据

| 检查 | 本次结果 |
|---|---|
| 全量导入 | 1,169 篇论文、61,317 段正文（含摘要）、1,169 个派生 JSON 对象；数据库发布版本 1 |
| 排序元数据 | 1,169 篇均有 arXiv 首次提交时间；33 篇保守映射到 CCF 发表载体（A 12、B 20、C 1） |
| 离线全量回归 | 冻结 P1 的 745 道 dev 文本证据题逐题比较 top-5 段落 ID，排名差异 0 |
| 后端单元测试 | 75 通过；5 个需真实存储的用例按设计跳过 |
| 真实 PostgreSQL/S3 集成测试 | 合计 80 通过；覆盖元数据写入与 API 返回、导入幂等、缺失对象修复、数据库失败回滚、孤儿记录、锁竞争、进程中断恢复、历史版本复用、发布版本刷新和依赖故障 |
| 前端测试与构建 | 11 通过；TypeScript 检查和 Vite 生产构建成功 |
| 浏览器联调 | 实测编号、时间与 CCF 排序；实际论文检索得到 5 条原文引用；第三条可打开并关闭含前后段落的上下文弹窗；存储页读取真实数量和导入记录 |
| 重启恢复 | 停止并重新启动本机 PostgreSQL 与 SeaweedFS 后，已导入数据、下载和检索仍可用 |

排序元数据审计见 [paper_sort_metadata.json](../reports/paper_sort_metadata.json)。浏览器实测中，`ccf_best` 首屏均为 CCF A，`submitted_oldest` 首篇为 `1503.00841`（2015-03-03）。CCF 结果来自 arXiv `journal_ref` 与官方目录的保守规则，不将等级描述为论文质量；具体边界见[数据契约](DATASET.md#21-浏览与排序元数据)。

机器可读报告：[P2A 存储回归](../reports/p2a_storage_regression.json)。复核命令：

```bash
.venv/bin/python -m pytest -q
RUN_STORAGE_INTEGRATION=1 .venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
.venv/bin/python scripts/verify_workbench.py
```

真实存储测试使用随机测试数据库/桶，并在结束时清理；运行前必须先执行 `local_services.py up`。全量回归脚本读取离线 gold 问题，用它们对照 P1 已冻结的检索段落 ID；在线服务不读取 gold。两处上游依赖弃用警告不影响本次测试结果。

P1 的 Hit@5 58.52%、证据 Recall@5 50.64%、MRR@5 0.3330 没有因为换存储而提高。它们只衡量指定论文 ID、文本证据子集的检索，不是答案准确率、引用支持率或不可回答拒答率。

## 失败边界和恢复

- S3 对象先写、PostgreSQL 事务后提交。数据库失败时旧版本保持可见，已写对象可能暂时成为孤儿；记录失败并允许按内容键重试。这两个系统没有跨服务原子事务。
- 同一份输入重复导入不会增加当前正文段落与已发布版本；若对象丢失，重试会检查并修复，不因有任务记录就盲目跳过。
- 检索候选由可重建索引给出，但返回前回查当前 PostgreSQL 正文、版本和段落哈希；数据库不可用、证据缺失或内容哈希不一致时不会用旧缓存伪装成功。
- 正文下载逐字节核对 SHA-256；文件不可用或内容不符时返回失败。S3 健康不等于每个对象都存在，导入与下载会按对象验证。
- 当前是本机单节点、单 API 进程的开发工作台。没有用户身份与跨用户权限隔离，没有 PDF 解析与页码定位，没有生产备份、高可用或负载测试。个人论文上传、pgvector、RRF、Reranker、生成、Verifier、Harness、记忆和反馈发布属于后续阶段。

## 如何学习这一阶段

建议沿一条请求读代码：`frontend/src/App.tsx` 的检索动作 → `src/research_agent/api.py` 的请求校验 → `service.py` 的候选与事实回查 → `storage.py` 的版本/段落查询。再沿导入流程读 `ingestion.py` 与迁移 SQL，找出 S3 与 PostgreSQL 各自保存什么、数据库事务从哪里开始和结束、发布版本何时可见。最后用真实存储测试观察一次失败恢复，而不是只记住“有事务”这个词。

设计理由与替代方案见 [架构选型与数据流](架构选型与数据流.md)。阶段完成后再进行用户学习验收；[面试题库](../科研论文助手Agent面试问题.md)已准备，但不会在用户尚未开始面试时直接提问。
