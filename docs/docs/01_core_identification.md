# GPT Researcher 核心运行流程分析

> 本文档从代码运行流程出发，追踪从程序启动到研究完成的完整调用链。

---

## 1. 程序入口文件与入口函数

项目提供 **四个启动入口**，均最终汇入同一个核心类 `GPTResearcher`。

### 1.1 CLI 命令行入口 — `cli.py`

```
cli.py : main(args)
  └── asyncio.run(main(args))          # __main__ 启动
        ├── GPTResearcher(query=..., ...)
        │     ├── await researcher.conduct_research()
        │     └── await researcher.write_report()
        └── write_md_to_pdf / write_md_to_word  # 输出文件
```

- **入口函数**: `if __name__ == "__main__"` → `asyncio.run(main(args))`
- 通过 `argparse` 解析参数（query、report_type、tone、report_source 等）
- 对 `detailed_report` 类型走 `DetailedReport` 分支，其余直接实例化 `GPTResearcher`
- 核心两步调用：`conduct_research()` → `write_report()`

### 1.2 FastAPI Web 服务入口 — `main.py`

```
main.py
  └── uvicorn.run(app, host="0.0.0.0", port=8000)
        └── app = backend/server/app.py 中定义的 FastAPI 实例
```

- **入口函数**: `if __name__ == "__main__"` → `uvicorn.run(app)`
- `app` 对象在 `backend/server/app.py` 中创建
- 提供两种研究触发方式：
  - **WebSocket** (`/ws`): 实时流式推送研究进度
  - **REST API** (`POST /report/`): 同步或后台生成报告

### 1.3 WebSocket 研究流程 — `backend/server/websocket_manager.py`

```
WebSocket /ws 连接
  └── handle_websocket_communication()
        └── WebSocketManager.start_streaming()
              └── run_agent(task, report_type, ...)
                    ├── report_type == "multi_agents"  → run_research_task()
                    ├── report_type == "detailed_report" → DetailedReport(...).run()
                    └── 其他类型                        → BasicReport(...).run()
```

`run_agent()` 是 WebSocket 路径的核心调度函数，根据 `report_type` 选择不同的报告生成策略，最终都会创建 `GPTResearcher` 实例。

### 1.4 Python 包导入

```python
from gpt_researcher import GPTResearcher

researcher = GPTResearcher(query="...", report_type="research_report")
await researcher.conduct_research()
report = await researcher.write_report()
```

`gpt_researcher/__init__.py` 仅导出一个类：`GPTResearcher`。

---

## 2. 核心业务类：`GPTResearcher`

**文件**: `gpt_researcher/agent.py`

`GPTResearcher` 是整个项目的核心编排器（Orchestrator），负责协调研究的全生命周期。它不直接执行搜索、抓取、生成等操作，而是将具体工作委派给内部的 skill 组件。

---

## 3. `__init__` 中初始化的关键组件

```python
class GPTResearcher:
    def __init__(self, query, report_type=..., ...):
```

### 3.1 配置与基础设施

| 属性 | 类型 | 来源 | 职责 |
|---|---|---|---|
| `self.cfg` | `Config` | `config/config.py` | 加载所有配置项（LLM 模型、搜索提供者、限速参数等） |
| `self.retrievers` | `list[class]` | `actions.get_retrievers()` | 根据配置解析出可用的搜索提供者类列表 |
| `self.memory` | `Memory` | `memory/` | 嵌入向量管理，用于语义相似度检索 |
| `self.prompt_family` | `PromptFamily` | `prompts.py` | 提示词模板族，控制 LLM 交互风格 |

### 3.2 六大 Skill 组件

这是 `GPTResearcher` 的核心设计 —— **通过组合模式将职责拆分到六个 skill 对象中**，每个 skill 在初始化时接收 `self`（即 researcher 实例）作为参数，从而能访问所有共享状态。

| 属性 | 类 | 文件 | 职责 |
|---|---|---|---|
| `self.research_conductor` | `ResearchConductor` | `skills/researcher.py` | **研究执行主引擎**：规划子查询、调度搜索、聚合上下文 |
| `self.report_generator` | `ReportGenerator` | `skills/writer.py` | **报告撰写器**：生成引言、正文、结论 |
| `self.context_manager` | `ContextManager` | `skills/context_manager.py` | **上下文管理**：语义压缩、相似内容检索 |
| `self.scraper_manager` | `BrowserManager` | `skills/browser.py` | **网页抓取管理**：并行抓取 URL、提取图片 |
| `self.source_curator` | `SourceCurator` | `skills/curator.py` | **来源审核**：用 LLM 评估来源的可信度和相关性 |
| `self.deep_researcher` | `DeepResearchSkill` | `skills/deep_research.py` | **深度研究**（可选）：递归树状探索，仅在 report_type 为 deep_research 时创建 |

### 3.3 可选组件

| 属性 | 类型 | 职责 |
|---|---|---|
| `self.image_generator` | `ImageGenerator` | AI 图片生成（Gemini 等），在配置启用时工作 |
| `self.vector_store` | `VectorStoreWrapper` | 外部向量数据库集成（用户传入时激活） |
| `self.websocket` | `WebSocket` | 实时进度推送通道 |

### 3.4 状态追踪

| 属性 | 用途 |
|---|---|
| `self.context` | 累积的研究上下文（list） |
| `self.visited_urls` | 已访问的 URL（set，避免重复抓取） |
| `self.research_sources` | 抓取的原始来源（含标题、内容、图片） |
| `self.research_images` | 筛选出的研究图片 |
| `self.research_costs` | API 累计花费（float） |
| `self.agent` / `self.role` | LLM 自动选择的研究代理身份和角色描述 |

---

## 4. 主执行方法如何串联整个研究流程

`GPTResearcher` 对外暴露两个核心异步方法，按顺序调用即完成一次完整研究：

```
conduct_research()  →  write_report()
```

### 4.1 `conduct_research()` — 研究阶段

```python
async def conduct_research(self):
```

完整调用链路：

```
conduct_research()
│
├── [分支1] report_type == "deep_research"
│   └── self.deep_researcher.run()
│       └── 递归树状探索（breadth × depth），每层并行执行子研究
│
├── [分支2] 常规研究类型
│   │
│   ├── Step 1: choose_agent()               ← actions/agent_creator.py
│   │   └── 用 LLM 根据 query 自动选择最适合的研究代理角色
│   │       → 设置 self.agent (代理名) 和 self.role (角色 prompt)
│   │
│   ├── Step 2: self.research_conductor.conduct_research()  ← skills/researcher.py
│   │   │
│   │   ├── 2a. plan_research()
│   │   │   ├── get_search_results()          ← 用第一个 retriever 初步搜索
│   │   │   └── plan_research_outline()       ← LLM 生成子查询列表
│   │   │
│   │   ├── 2b. 根据 report_source 分派:
│   │   │   ├── "web"    → _get_context_by_web_search()
│   │   │   ├── "local"  → DocumentLoader + web search
│   │   │   ├── "hybrid" → 本地文档 + web search 合并
│   │   │   └── "azure" / "langchain_*" / "vectorstore" → 相应加载器
│   │   │
│   │   ├── 2c. _get_context_by_web_search() 内部:
│   │   │   ├── MCP 策略处理（fast/deep/disabled）
│   │   │   ├── plan_research() → 生成子查询
│   │   │   └── asyncio.gather(*[_process_sub_query(q) for q in sub_queries])
│   │   │       └── _process_sub_query():
│   │   │           ├── _scrape_data_by_urls()
│   │   │           │   ├── _search_relevant_source_urls()  ← 调用所有 retrievers 搜索
│   │   │           │   └── self.scraper_manager.browse_urls()  ← 并行抓取网页
│   │   │           └── self.context_manager.get_similar_content_by_query()
│   │   │               └── ContextCompressor → 语义压缩 + 去噪
│   │   │
│   │   └── 2d. source_curator.curate_sources()  ← 可选：LLM 对来源排序打分
│   │
│   └── Step 3: self.image_generator.plan_and_generate_images()  ← 可选：AI生图
│
└── return self.context
```

### 4.2 `write_report()` — 报告生成阶段

```python
async def write_report(self, existing_headers=[], ...):
```

调用链路：

```
write_report()
│
└── self.report_generator.write_report()    ← skills/writer.py
    │
    └── generate_report(**params)           ← actions/report_generation.py
        └── LLM 调用：基于 context + role + tone 生成最终报告
```

### 4.3 完整流程时序图

```
用户查询
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│                  conduct_research()                          │
│                                                             │
│  1. choose_agent()                                          │
│     LLM 选择最优代理角色                                      │
│         │                                                   │
│  2. research_conductor.conduct_research()                    │
│     ├── plan_research()                                     │
│     │   ├── get_search_results()     初步搜索了解主题         │
│     │   └── plan_research_outline()  LLM 规划子查询           │
│     │                                                       │
│     ├── 并行处理每个子查询:                                    │
│     │   ├── retrievers[*].search()   多引擎搜索               │
│     │   ├── scraper_manager.browse_urls()  抓取网页            │
│     │   └── context_manager.get_similar_content()  语义压缩    │
│     │                                                       │
│     └── source_curator.curate_sources()  来源可信度排序         │
│                                                             │
│  3. image_generator.plan_and_generate_images() (可选)         │
│                                                             │
│  → self.context 已填充完毕                                    │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│                  write_report()                              │
│                                                             │
│  report_generator.write_report()                             │
│  └── generate_report()                                      │
│      └── LLM: context + role + tone → 最终 Markdown 报告     │
│                                                             │
│  → 返回报告字符串                                             │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
输出: Markdown / PDF / DOCX
```

---

## 5. 关键设计模式总结

| 模式 | 体现 |
|---|---|
| **Facade（门面）** | `GPTResearcher` 对外只暴露 `conduct_research()` + `write_report()` 两个方法，隐藏了内部全部复杂性 |
| **Composition（组合）** | 六大 skill 组件各司其职，通过持有 researcher 引用共享状态 |
| **Strategy（策略）** | `report_source` 决定数据获取策略（web/local/hybrid/azure/vectorstore）；`report_type` 决定研究深度 |
| **Plugin（插件）** | `retrievers/` 和 `scraper/` 均为插件式架构，新增搜索源或抓取器无需修改核心代码 |
| **Async Parallel（异步并行）** | 子查询通过 `asyncio.gather()` 并行执行，多个 retriever 并行搜索 |
