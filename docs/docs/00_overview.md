# GPT Researcher 项目架构概览

> 本文档面向新开发者，从项目目录结构和文件命名出发，对 GPT Researcher 进行架构级解读。

---

## 1. 项目整体结构与分层逻辑

项目采用**核心库 + 应用层 + 前端**的三层架构：

```
┌─────────────────────────────────────────────────────────┐
│  frontend/          前端展示层 (Next.js / 静态HTML)       │
├─────────────────────────────────────────────────────────┤
│  backend/           应用服务层 (FastAPI + WebSocket)      │
│  multi_agents/      多智能体编排层 (LangGraph)            │
├─────────────────────────────────────────────────────────┤
│  gpt_researcher/    核心研究引擎 (Python 包)              │
├─────────────────────────────────────────────────────────┤
│  evals/             评估框架                              │
│  tests/             测试套件                              │
│  docs/              文档站点                              │
│  terraform/         基础设施即代码                         │
└─────────────────────────────────────────────────────────┘
```

**入口点有四个**：

- `cli.py` — 命令行直接使用
- `main.py` — FastAPI 服务启动
- `multi_agents/main.py` — LangGraph 多智能体模式
- Python 包导入：`from gpt_researcher import GPTResearcher`

---

## 2. 各主要目录/模块的功能定位

### 2.1 核心引擎：`gpt_researcher/`

这是项目的心脏，是一个独立可安装的 Python 包（版本 0.14.6）。

| 模块 | 职责 |
|---|---|
| `agent.py` | **主编排器** `GPTResearcher` 类，协调整个研究工作流 |
| `prompts.py` | 所有 LLM 提示词模板，决定研究质量的关键文件 |
| `config/` | 配置管理，从环境变量/文件加载设置 |
| `actions/` | 离散操作单元：查询处理、检索、报告生成、网页抓取 |
| `skills/` | 智能体能力模块：`researcher.py`（核心研究）、`writer.py`、`browser.py`、`deep_research.py`、`image_generator.py` 等 |
| `retrievers/` | **16 种搜索提供者**的适配器（Tavily、Google、Bing、DuckDuckGo、Arxiv、PubMed、Semantic Scholar 等） |
| `scraper/` | **9 种网页抓取方式**（BeautifulSoup、Selenium/Playwright、Firecrawl、PyMuPDF 等） |
| `llm_provider/` | LLM 提供者抽象层（OpenAI、Claude、Ollama、LiteLLM 支持 30+ 后端） |
| `document/` | 文档处理（PDF、DOCX、在线文档） |
| `context/` | 跨操作的研究上下文累积与管理 |
| `memory/` | 状态持久化与上下文记忆 |
| `vector_store/` | 向量数据库集成，用于嵌入存储与检索 |
| `mcp/` | Model Context Protocol 客户端，连接外部数据源 |
| `utils/` | 工具集：日志、成本追踪、速率限制、枚举常量、验证器 |

### 2.2 应用服务层：`backend/`

基于 FastAPI 的 Web 服务，通过 WebSocket 提供实时研究进度推送。

| 子目录 | 职责 |
|---|---|
| `server/app.py` | FastAPI 主应用，路由和 WebSocket 端点 |
| `server/websocket_manager.py` | WebSocket 连接管理 |
| `report_type/` | 三种报告模板：`basic_report/`、`detailed_report/`、`deep_research/` |
| `chat/` | 对话式研究接口 |
| `memory/` | 后端状态管理 |

### 2.3 多智能体系统：`multi_agents/`

基于 **LangGraph** 的多代理协作系统（灵感来自 STORM 论文），包含 8 个专业角色：

```
用户 → Chief Editor → 规划 → Editor（大纲）
                        ↓
                   并行处理各主题:
                   Researcher → 初稿
                   Reviewer → 审核反馈
                   Revisor → 修订
                        ↓
                   Writer → 最终报告
                        ↓
                   Publisher → PDF/DOCX/Markdown
```

### 2.4 前端：`frontend/`

| 子目录 | 技术栈 | 说明 |
|---|---|---|
| `nextjs/` | Next.js + TypeScript + Tailwind | 生产级前端，含 components、hooks、actions 等标准 Next.js 结构 |
| `static/` | 纯 HTML/CSS/JS | 轻量级替代方案，由 FastAPI 直接提供静态文件服务 |

### 2.5 评估框架：`evals/`

| 子目录 | 职责 |
|---|---|
| `simple_evals/` | 基于 OpenAI SimpleQA 的事实准确性评估 |
| `hallucination_eval/` | 通过对比报告与源材料检测幻觉 |

### 2.6 基础设施与配置

| 文件/目录 | 职责 |
|---|---|
| `terraform/` | AWS ECR 和 GitHub Actions 的 IaC 配置 |
| `docker-compose.yml` / `Dockerfile` / `Dockerfile.fullstack` | 容器化部署 |
| `pyproject.toml` | Poetry 项目配置，定义所有依赖 |
| `langgraph.json` | LangGraph 部署配置 |

---

## 3. 模块间依赖与调用关系

从目录结构和命名可推断出以下调用链路：

```
                    ┌──────────┐
                    │ cli.py   │  命令行入口
                    │ main.py  │  Web 服务入口
                    └────┬─────┘
                         │
              ┌──────────▼──────────┐
              │  backend/server/    │  FastAPI + WebSocket
              │  (app.py)           │  ← 调用 report_type/ 选择报告模式
              └──────────┬──────────┘
                         │
         ┌───────────────▼───────────────┐
         │     gpt_researcher/agent.py   │  核心编排器
         │        GPTResearcher          │
         └───┬───┬───┬───┬───┬───┬──────┘
             │   │   │   │   │   │
    ┌────────┘   │   │   │   │   └────────┐
    ▼            ▼   │   ▼   ▼            ▼
 config/    skills/  │  actions/    llm_provider/
 (配置)    (能力)    │  (操作)      (LLM抽象)
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
      retrievers/          scraper/
      (16种搜索源)         (9种抓取器)
           │                   │
           ▼                   ▼
      [外部搜索API]       [目标网页/文档]
           │                   │
           └─────────┬─────────┘
                     ▼
               context/ + memory/
              (上下文聚合与记忆)
                     │
                     ▼
               document/ + vector_store/
              (文档处理与向量存储)
                     │
                     ▼
            actions/report_generation.py
              (最终报告生成 → PDF/DOCX/MD)
```

### 关键依赖关系总结

1. **`agent.py` 是中心节点** — 它导入并协调 `actions/`、`skills/`、`config/`、`llm_provider/` 所有模块
2. **`skills/researcher.py` 是最大的技能模块** — 调用 `retrievers/` 搜索 + `scraper/` 抓取，构成研究的主循环
3. **`retrievers/` 和 `scraper/` 是插件式架构** — 每个搜索源/抓取器各自独立，通过统一接口被 `actions/retriever.py` 调度
4. **`llm_provider/` 解耦了 LLM 选择** — 上层代码无需关心具体使用哪个 LLM
5. **`backend/` 依赖 `gpt_researcher/`** — 后端是核心库的 Web 封装
6. **`multi_agents/` 独立于 `backend/`** — 有自己的 `requirements.txt`，是另一种使用核心库的方式
7. **`frontend/nextjs/` 通过 WebSocket 与 `backend/` 通信** — 前后端分离

---

## 4. 新开发者上手建议

1. 从 `gpt_researcher/agent.py` 开始阅读，它是整个研究流程的入口
2. 顺着它导入的 `skills/researcher.py` 和 `actions/` 理解核心研究循环
3. 查看 `retrievers/` 和 `scraper/` 了解数据获取层的插件式设计
4. 阅读 `prompts.py` 理解 LLM 交互的提示词工程
5. 运行 `cli.py` 做一次完整研究，结合日志追踪代码执行路径
