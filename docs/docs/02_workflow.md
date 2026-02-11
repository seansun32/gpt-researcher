# GPT Researcher 完整研究流程追踪

> 以一次具体查询 **"What is the future of AI?"** 为例，追踪从用户输入到最终报告生成的完整代码执行路径。
> 本文重点分析三个核心环节：搜索查询生成、上下文检索与压缩、最终报告生成。

---

## 完整调用时序

以 CLI 入口（`report_type="research_report"`、`report_source="web"`）为例：

```
用户输入: "What is the future of AI?"
│
│  cli.py: asyncio.run(main(args))
│
├── [1] GPTResearcher.__init__(query="What is the future of AI?", ...)
│   ├── Config(config_path)                      # 加载配置
│   ├── get_retrievers(headers, cfg)              # 解析搜索引擎列表 → [TavilySearch]
│   ├── Memory(embedding_provider, model)         # 初始化嵌入向量模型
│   ├── ResearchConductor(self)                   # 研究执行器
│   ├── ReportGenerator(self)                     # 报告生成器
│   ├── ContextManager(self)                      # 上下文管理器
│   ├── BrowserManager(self)                      # 网页抓取器
│   └── SourceCurator(self)                       # 来源审核器
│
├── [2] researcher.conduct_research()              ← 研究阶段
│   ├── [2.1] choose_agent()                       ← 自动选择代理角色
│   │
│   ├── [2.2] research_conductor.conduct_research()
│   │   ├── [2.2.1] plan_research()                ← 规划子查询
│   │   │   ├── get_search_results()               ← 初步搜索
│   │   │   └── plan_research_outline()            ← LLM 生成子查询
│   │   │       └── generate_sub_queries()
│   │   │
│   │   ├── [2.2.2] _get_context_by_web_search()   ← Web 搜索上下文
│   │   │   └── asyncio.gather(                     ← 并行处理子查询
│   │   │       *[_process_sub_query(q) for q in sub_queries])
│   │   │       ├── _scrape_data_by_urls()
│   │   │       │   ├── _search_relevant_source_urls()  ← 检索 URL
│   │   │       │   └── scraper_manager.browse_urls()   ← 抓取网页
│   │   │       └── context_manager.get_similar_content_by_query()  ← 压缩
│   │   │
│   │   └── [2.2.3] source_curator.curate_sources() ← 来源排序（可选）
│   │
│   └── [2.3] image_generator.plan_and_generate_images()  ← 生成图片（可选）
│
└── [3] researcher.write_report()                  ← 报告生成阶段
    └── report_generator.write_report()
        └── generate_report()                      ← LLM 生成最终 Markdown
```

---

## 1. 搜索查询的生成逻辑

**核心问题**：原始问题 "What is the future of AI?" 如何被拆解为多个搜索查询？

### 1.1 Step 1 — 自动选择代理角色

**文件**: `actions/agent_creator.py` → `choose_agent()`

系统首先用 LLM 根据查询自动选择最匹配的"研究代理"身份：

```python
# prompts.py: auto_agent_instructions()
# 提示词要求 LLM 返回 JSON: {"server": "代理名", "agent_role_prompt": "角色描述"}
response = await create_chat_completion(
    messages=[
        {"role": "system", "content": prompt_family.auto_agent_instructions()},
        {"role": "user", "content": f"task: {query}"},
    ],
    temperature=0.15,  # 低温度，确保稳定选择
)
```

对于 "What is the future of AI?"，LLM 大概率返回：

```json
{
  "server": "🤖 AI Research Agent",
  "agent_role_prompt": "You are a seasoned AI technology analyst..."
}
```

该 `agent_role_prompt` 将作为后续所有 LLM 调用的 **system prompt**，确保研究风格一致。

### 1.2 Step 2 — 初步搜索获取上下文

**文件**: `skills/researcher.py` → `plan_research()` → `actions/query_processing.py` → `get_search_results()`

在生成子查询之前，系统先执行一次初步搜索，获取实时信息作为子查询生成的参考上下文：

```python
# 用第一个 retriever（默认 TavilySearch）对原始查询做一次搜索
search_results = await get_search_results(
    query,                          # "What is the future of AI?"
    self.researcher.retrievers[0],  # TavilySearch
    query_domains                   # 可选的域名限制
)
```

搜索结果是一组 `[{"title": ..., "href": ..., "body": ...}, ...]` 字典列表。

### 1.3 Step 3 — LLM 生成子查询

**文件**: `actions/query_processing.py` → `generate_sub_queries()`

将原始查询 + 初步搜索结果 + report_type 信息组装成提示词，交给**策略模型**（`strategic_llm_model`，通常是推理能力更强的模型）生成子查询：

```python
# prompts.py: generate_search_queries_prompt()
prompt = f"""Write {max_iterations} google search queries to search online
that form an objective opinion from the following task: "{task}"

Assume the current date is {当前日期} if required.

Context: {初步搜索结果}   ← 用实时信息引导子查询生成

You must respond with a list of strings in the following format:
["query 1", "query 2", "query 3"].
"""
```

**关键设计**：
- `max_iterations` 由配置控制（默认 3），决定生成几个子查询
- 搜索结果作为 context 传入，让 LLM 基于实时信息生成更精准的子查询
- 使用 `json_repair.loads()` 解析 LLM 返回，容错能力强

对于 "What is the future of AI?"，LLM 可能生成：

```json
[
  "AI technology trends and predictions 2025 2026",
  "impact of artificial intelligence on jobs and economy future",
  "AGI artificial general intelligence timeline and feasibility"
]
```

### 1.4 追加原始查询

```python
# skills/researcher.py: _get_context_by_web_search()
if self.researcher.report_type != "subtopic_report":
    sub_queries.append(query)  # 把原始查询也加入列表
```

最终查询列表变为 4 个：3 个子查询 + 1 个原始查询。

---

## 2. 上下文的检索与压缩流程

### 2.1 数据从哪里检索？

检索过程按 **子查询粒度** 并行执行，每个子查询经过 **搜索 → 抓取 → 压缩** 三步。

#### 2.1.1 多引擎搜索 URL

**文件**: `skills/researcher.py` → `_search_relevant_source_urls()`

```python
for retriever_class in self.researcher.retrievers:
    # 跳过 MCP 类型的 retriever
    retriever = retriever_class(query, query_domains=query_domains)
    search_results = await asyncio.to_thread(
        retriever.search,
        max_results=cfg.max_search_results_per_query
    )
    search_urls = [url.get("href") for url in search_results]
    new_search_urls.extend(search_urls)
```

**关键设计**：
- 遍历所有配置的 retriever（可以是 Tavily + Google + Bing 等多个并用）
- 每个 retriever 返回 `max_search_results_per_query` 个结果
- 通过 `_get_new_urls()` 去重，避免重复访问已经抓取过的 URL
- URL 列表 `random.shuffle()` 打乱，避免单一来源偏见

#### 2.1.2 并行抓取网页内容

**文件**: `skills/browser.py` → `BrowserManager.browse_urls()`

```python
scraped_content, images = await scrape_urls(urls, cfg, self.worker_pool)
```

实际调用 `actions/web_scraping.py` → `Scraper` 类：
- 使用 `WorkerPool` 控制并发数（`max_scraper_workers` 配置）
- 支持 9 种抓取方式（BeautifulSoup、Playwright、Firecrawl 等，由 `cfg.scraper` 决定）
- 同时提取页面中的图片 URL 及其评分
- 返回格式: `[{"url": ..., "raw_content": ..., "title": ..., "image_urls": [...]}, ...]`

#### 2.1.3 数据源类型分派

`ResearchConductor.conduct_research()` 根据 `report_source` 选择不同的数据获取策略：

| report_source | 策略 |
|---|---|
| `"web"` | 纯 Web 搜索 → `_get_context_by_web_search()` |
| `"local"` | `DocumentLoader` 加载本地文件 + Web 搜索 |
| `"hybrid"` | 本地文档 + Web 搜索分别执行后合并 |
| `"azure"` | `AzureDocumentLoader` 从 Azure Blob 加载 |
| `"langchain_documents"` | 用户传入的 LangChain Document 对象 |
| `"langchain_vectorstore"` | 直接查询用户传入的向量数据库 |

### 2.2 上下文压缩的具体策略

#### 2.2.1 核心压缩器 — ContextCompressor

**文件**: `context/compression.py` → `ContextCompressor`

这是最关键的压缩组件，采用 **LangChain 的 ContextualCompressionRetriever 管道**：

```python
def __get_contextual_retriever(self):
    # Step 1: 文本分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # 每块 1000 字符
        chunk_overlap=100      # 100 字符重叠，防止截断关键信息
    )

    # Step 2: 嵌入向量相似度过滤
    relevance_filter = EmbeddingsFilter(
        embeddings=self.embeddings,
        similarity_threshold=0.35  # 默认阈值，可通过环境变量调整
    )

    # Step 3: 组装管道
    pipeline = DocumentCompressorPipeline(
        transformers=[splitter, relevance_filter]
    )

    # Step 4: 包装原始文档为 LangChain Retriever
    base_retriever = SearchAPIRetriever(pages=self.documents)

    return ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=base_retriever
    )
```

**压缩管道执行流程**：

```
原始抓取页面 (每页可能数千字)
  │
  ▼
SearchAPIRetriever._get_relevant_documents()
  → 将 scraped pages 转为 LangChain Document 对象
  │
  ▼
RecursiveCharacterTextSplitter
  → 按 1000 字符切块，100 字符重叠
  → 一页可能被切成 5-20 个块
  │
  ▼
EmbeddingsFilter (similarity_threshold=0.35)
  → 计算每个文本块与 query 的嵌入向量余弦相似度
  → 丢弃相似度 < 0.35 的块
  → 仅保留语义相关的块
  │
  ▼
pretty_print_docs(relevant_docs, max_results=10)
  → 格式化: "Source: url\nTitle: title\nContent: content\n"
  → 最多保留 10 个最相关的块
  │
  ▼
返回压缩后的上下文字符串
```

#### 2.2.2 上下文汇总流程

每个子查询的压缩结果通过 `asyncio.gather()` 并行获取后，汇总合并：

```python
# skills/researcher.py: _get_context_by_web_search()
context = await asyncio.gather(
    *[self._process_sub_query(q, scraped_data, query_domains)
      for q in sub_queries]
)
# 过滤空结果并合并
context = [c for c in context if c]
combined_context = " ".join(context)
```

#### 2.2.3 可选：来源审核 (Source Curation)

**文件**: `skills/curator.py` → `SourceCurator.curate_sources()`

如果配置了 `curate_sources=True`，合并后的上下文会经过额外的 LLM 评估：

```python
if self.researcher.cfg.curate_sources:
    self.researcher.context = await self.researcher.source_curator.curate_sources(research_data)
```

审核用的提示词要求 LLM 按以下维度评估每个来源：
- **相关性**: 与查询的直接/间接关联度
- **可信度**: 是否来自权威来源
- **时效性**: 优先近期信息
- **客观性**: 是否存在偏见
- **数据价值**: 是否包含统计数字、具体数据

LLM 返回经过排序和筛选的来源 JSON 列表（最多 `max_results` 条）。

---

## 3. 最终报告的生成过程

### 3.1 基础报告（research_report）— 单轮生成

**文件**: `actions/report_generation.py` → `generate_report()`

#### 3.1.1 选择提示词模板

```python
# 通过 report_type_mapping 查找对应的 prompt 方法
report_type_mapping = {
    "research_report":  "generate_report_prompt",
    "resource_report":  "generate_resource_report_prompt",
    "outline_report":   "generate_outline_report_prompt",
    "subtopic_report":  "generate_subtopic_report_prompt",
    "deep_research":    "generate_deep_research_prompt",
    ...
}

generate_prompt = get_prompt_by_report_type(report_type, prompt_family)
```

#### 3.1.2 组装最终提示词

对于 `research_report`，使用 `generate_report_prompt()`:

```python
# prompts.py: generate_report_prompt()
content = f"""
Information: "{context}"          ← 所有压缩后的研究上下文
---
Using the above information, answer the following query or task:
"{question}" in a detailed report --

The report should focus on the answer to the query, should be well structured,
informative, in-depth, and comprehensive, with facts and numbers if available
and at least {total_words} words.                    ← 默认 1000 词

Please follow all of the following guidelines in your report:
- You MUST determine your own concrete and valid opinion based on the given information.
- You MUST write the report with markdown syntax and {report_format} format.
- Structure your report with clear markdown headers: # for main title, ## for major sections, ### for subsections.
- Use markdown tables when presenting structured data.
- You MUST prioritize the relevance, reliability, and significance of the sources.
- Use in-text citation references in {report_format} format...
- {reference_prompt}                                 ← 引用格式要求
- {tone_prompt}                                      ← 语气要求 (如 Objective)
You MUST write the report in the following language: {language}.
"""
```

如果有预生成的 AI 图片，会追加图片嵌入指令。

#### 3.1.3 LLM 生成报告

```python
report = await create_chat_completion(
    model=cfg.smart_llm_model,          # 使用 "smart" 模型（通常是 GPT-4 级别）
    messages=[
        {"role": "system", "content": agent_role_prompt},  # 步骤 1 选择的代理角色
        {"role": "user", "content": content},               # 上面组装的完整提示词
    ],
    temperature=0.35,                   # 中等创造性
    stream=True,                        # 流式输出（通过 WebSocket 实时推送）
    websocket=websocket,
    max_tokens=cfg.smart_token_limit,
)
```

**基础报告是单轮 LLM 调用，一次生成完整 Markdown 报告。**

### 3.2 详细报告（detailed_report）— 多轮分段生成

**文件**: `backend/report_type/detailed_report/detailed_report.py` → `DetailedReport`

详细报告采用 **多阶段、分主题** 的生成策略，流程更复杂：

```
DetailedReport.run()
│
├── [Phase 1] _initial_research()
│   └── gpt_researcher.conduct_research()
│       → 对主查询执行完整的搜索+压缩流程
│       → 获得 global_context（全局研究上下文）
│
├── [Phase 2] _get_all_subtopics()
│   └── gpt_researcher.get_subtopics()
│       └── construct_subtopics()          ← LLM 从上下文中提取子主题列表
│           → 返回 [{"task": "AI in Healthcare"}, {"task": "AI Ethics"}, ...]
│
├── [Phase 3] gpt_researcher.write_introduction()
│   └── write_report_introduction()        ← LLM 生成引言
│
├── [Phase 4] _generate_subtopic_reports(subtopics)
│   └── 对每个子主题 **顺序** 执行:
│       │
│       ├── 创建子 GPTResearcher 实例（继承主代理角色和已访问 URL）
│       ├── subtopic_assistant.conduct_research()  ← 子主题独立研究
│       ├── get_draft_section_titles()              ← LLM 生成章节标题
│       ├── get_similar_written_contents_by_draft_section_titles()
│       │   └── 从已写内容中找相似段落，避免重复
│       ├── subtopic_assistant.write_report(
│       │     existing_headers=...,                 ← 已有的章节标题
│       │     relevant_written_contents=...,        ← 已有的相关内容
│       │   )
│       │   → LLM 生成子主题报告，自动避开已有内容
│       │
│       └── 更新全局状态:
│           ├── global_written_sections ← 追加新写内容
│           ├── global_context ← 合并新上下文
│           ├── global_urls ← 合并新 URL
│           └── existing_headers ← 追加新标题
│
├── [Phase 5] _construct_detailed_report()
│   ├── table_of_contents(report_body)     ← 从 Markdown 标题生成目录
│   ├── write_report_conclusion()          ← LLM 生成结论
│   ├── add_references()                   ← 添加引用列表
│   └── 拼接: 引言 + 目录 + 正文 + 结论 + 引用
│
└── return report
```

**关键去重机制**：每写完一个子主题，系统将其章节标题和内容加入 `existing_headers` 和 `global_written_sections`。下一个子主题写作时，通过 `WrittenContentCompressor` 进行语义相似度检索，找到已写过的相关段落传给 LLM，提示词明确指示"避免重复已有内容"。

### 3.3 深度研究（deep_research）— 递归树状探索

**文件**: `skills/deep_research.py` → `DeepResearchSkill`

深度研究采用 **递归树状展开**：

```
depth=0: 原始查询
   ├── breadth=4 个子查询 (并行, concurrency_limit=2)
   │   depth=1: 子查询 1
   │      ├── breadth=4 个孙查询 (并行)
   │      │   depth=2: 基线 — 执行搜索+抓取+压缩
   │      ├── ...
   │   depth=1: 子查询 2
   │      ├── ...
   │   ...
```

- `breadth`: 每层展开的子查询数（默认 4）
- `depth`: 递归深度（默认 2）
- `concurrency_limit`: 同层最大并行数（默认 2）
- 每层用 LLM 生成下一层查询，同时累积 learnings
- 最终将所有层级的 context 汇总

### 3.4 报告类型与提示词映射

| report_type | 提示词方法 | 特征 |
|---|---|---|
| `research_report` | `generate_report_prompt` | 单轮生成，至少 1000 词，含引用 |
| `detailed_report` | `generate_subtopic_report_prompt` | 多轮分段，5-6 页，含目录+结论 |
| `resource_report` | `generate_resource_report_prompt` | 侧重资源列表与评价 |
| `outline_report` | `generate_outline_report_prompt` | 仅生成大纲 |
| `deep_research` | `generate_deep_research_prompt` | 递归树状探索后的综合报告 |

---

## 总结：一次研究的关键度量

以 "What is the future of AI?" + `research_report` + `web` 为例的典型数据：

| 阶段 | 操作 | 典型数量 |
|---|---|---|
| 选择代理 | LLM 调用 | 1 次 |
| 初步搜索 | API 调用 | 1 次 |
| 生成子查询 | LLM 调用 | 1 次，产出 3 个子查询 |
| 搜索 URL | 每个子查询 × 每个 retriever | 4 × 1 = 4 次搜索 API 调用 |
| 网页抓取 | 并行抓取 | ~20-40 个 URL |
| 上下文压缩 | 嵌入向量计算 + 相似度过滤 | 4 次（每个子查询一次） |
| 来源审核 | LLM 调用（可选） | 0-1 次 |
| 报告生成 | LLM 调用 | 1 次 |
| **总 LLM 调用** | | **~4 次** |
| **总搜索 API 调用** | | **~5 次** |
