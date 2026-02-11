# 内部 Deep Research 平台构建方案

> 基于 `pip install gpt-researcher` 的方式，不修改源码，通过环境变量、自定义 Retriever、
> 内部 LLM 和 Elasticsearch 向量库，构建完全私有化的 Deep Research 能力。

---

## 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     你的项目 (my-deep-research/)                  │
│                                                                  │
│  main.py                    ← 启动入口                            │
│  config.json                ← 覆盖 gpt-researcher 默认配置         │
│  .env                       ← 环境变量（API 地址、密钥等）          │
│  retrievers/                                                     │
│    └── internal_search.py   ← 适配内部搜索引擎的自定义 Retriever    │
│  vector_store/                                                   │
│    └── es_store.py          ← Elasticsearch 向量库适配             │
├──────────────────────────────────────────────────────────────────┤
│                pip install gpt-researcher                        │
│                (作为黑盒依赖, 不修改任何源码)                       │
└──────────────────────────────────────────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
   内部搜索引擎 API      内部 LLM API           Elasticsearch
```

需要解决的 **4 个适配问题** 和对应方案：

| 问题 | gpt-researcher 扩展点 | 你的方案 |
|------|----------------------|---------|
| 替换外部搜索引擎 | `RETRIEVER=custom` + `RETRIEVER_ENDPOINT` 环境变量 | 方案 A: 用内置 CustomRetriever<br/>方案 B: 写适配层 API 网关 |
| 替换外部 LLM | `SMART_LLM=openai:model-name` + `OPENAI_BASE_URL` | 将内部 LLM 包装为 OpenAI 兼容 API |
| 替换外部 Embedding | `EMBEDDING=openai:model-name` + `OPENAI_BASE_URL` | 同上，复用 OpenAI 兼容协议 |
| 接入 Elasticsearch | 构造函数 `vector_store=` 参数 | 传入 LangChain ElasticsearchStore |

---

## 方案一：替换搜索引擎

gpt-researcher 内置了 `CustomRetriever`（`retrievers/custom/custom.py`），它的行为很简单：
- 读取环境变量 `RETRIEVER_ENDPOINT` 作为 API 地址
- 将 query 作为参数发 GET 请求
- 期望返回格式: `[{"url": "...", "raw_content": "..."}, ...]`

### 方案 A — 直接用内置 CustomRetriever（推荐，零代码）

如果你的内部搜索引擎可以调整返回格式，这是最简单的方式。

**你需要做的**：在内部搜索引擎前面加一个轻量 API 适配层，将返回值转为 gpt-researcher 期望的格式：

```python
# adapter_api.py — 内部搜索引擎适配层（FastAPI 示例）
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.get("/search")
async def search(query: str):
    """将内部搜索引擎的返回格式转换为 gpt-researcher 期望的格式"""

    # 调用你的内部搜索引擎
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://internal-search.yourcompany.com/api/search",
            json={"query": query, "top_k": 10},
            headers={"Authorization": "Bearer xxx"}
        )
    results = resp.json()

    # 转换为 gpt-researcher 期望的格式
    return [
        {
            "url": item["doc_url"],           # 必需
            "raw_content": item["content"],    # 必需 — 页面全文内容
            "title": item.get("title", ""),    # 可选
        }
        for item in results["hits"]
    ]
```

**关键环境变量**：

```bash
# .env
RETRIEVER=custom
RETRIEVER_ENDPOINT=http://localhost:8001/search    # 适配层地址
# 如果需要额外参数，用 RETRIEVER_ARG_ 前缀
RETRIEVER_ARG_TOPK=10
RETRIEVER_ARG_LANG=zh
```

### 方案 B — 写独立适配层搜索 + 直接喂内容（跳过网页抓取）

如果你的内部搜索引擎已经返回了文档全文（不需要再抓取网页），`raw_content` 字段填入完整内容即可。gpt-researcher 在拿到 `raw_content` 后会直接进入压缩阶段，**不会再去抓取 URL**。

> **原理**：`BrowserManager.browse_urls()` 抓取的结果格式就是
> `[{"url": ..., "raw_content": ..., "title": ...}]`，
> 和 CustomRetriever 返回的格式完全一致。搜索结果中的 URL 只是用来标注来源，
> 实际内容来自 `raw_content` 字段。

---

## 方案二：替换 LLM

gpt-researcher 的 LLM 调用走 LangChain 抽象层，支持多种 provider。其中有两条路径可以接入内部大模型：

### 路径 1 — OpenAI 兼容协议（推荐）

如果你的内部 LLM 提供 OpenAI 兼容的 API（大多数私有化部署如 vLLM、TGI、LocalAI 都支持），只需设置：

```bash
# .env — 将 openai provider 指向内部 LLM
OPENAI_API_KEY=your-internal-api-key       # 如果内部无鉴权可填任意字符串
OPENAI_BASE_URL=https://internal-llm.yourcompany.com/v1

# gpt-researcher 有 3 个 LLM 角色，格式为 "provider:model"
FAST_LLM=openai:your-fast-model-name       # 快速模型（子查询生成、摘要）
SMART_LLM=openai:your-smart-model-name     # 智能模型（报告撰写）
STRATEGIC_LLM=openai:your-smart-model-name  # 策略模型（查询规划）
```

**原理**：`base.py` 中 `provider == "openai"` 分支会创建 `ChatOpenAI`，而 `ChatOpenAI` 会读取 `OPENAI_BASE_URL` 环境变量作为 API 地址。

### 路径 2 — vLLM 专用 provider

如果你用 vLLM 部署，gpt-researcher 也内置了 `vllm_openai` provider：

```bash
# .env
VLLM_OPENAI_API_KEY=your-key
VLLM_OPENAI_API_BASE=https://internal-vllm.yourcompany.com/v1

FAST_LLM=vllm_openai:your-model-name
SMART_LLM=vllm_openai:your-model-name
STRATEGIC_LLM=vllm_openai:your-model-name
```

### 路径 3 — Ollama（本地部署）

如果你的内部 LLM 通过 Ollama 提供服务：

```bash
# .env
OLLAMA_BASE_URL=https://internal-ollama.yourcompany.com

FAST_LLM=ollama:qwen2.5:14b
SMART_LLM=ollama:qwen2.5:72b
STRATEGIC_LLM=ollama:qwen2.5:72b
```

### 路径 4 — LiteLLM 万能适配

如果以上都不适用，LiteLLM 可以适配几乎所有 LLM API：

```bash
# .env
FAST_LLM=litellm:your-provider/your-model
SMART_LLM=litellm:your-provider/your-model
STRATEGIC_LLM=litellm:your-provider/your-model
```

---

## 方案三：替换 Embedding 模型

gpt-researcher 的上下文压缩管道（`ContextCompressor`）依赖嵌入向量做相似度过滤。同样支持 OpenAI 兼容协议：

```bash
# .env — 方式 1: 复用 OpenAI 兼容协议（推荐）
EMBEDDING=openai:your-embedding-model-name
OPENAI_BASE_URL=https://internal-llm.yourcompany.com/v1
# 注意: OPENAI_BASE_URL 同时影响 LLM 和 Embedding

# .env — 方式 2: 如果 Embedding 服务地址和 LLM 不同，用 custom provider
EMBEDDING=custom:your-embedding-model-name
OPENAI_BASE_URL=https://internal-embedding.yourcompany.com/v1
OPENAI_API_KEY=your-key

# .env — 方式 3: Ollama 本地模型
EMBEDDING=ollama:nomic-embed-text
OLLAMA_BASE_URL=https://internal-ollama.yourcompany.com

# .env — 方式 4: HuggingFace 本地模型（完全离线，无需 API）
EMBEDDING=huggingface:sentence-transformers/all-MiniLM-L6-v2
```

> **注意**：如果 LLM 和 Embedding 用不同的服务地址，推荐 LLM 用 `openai` provider + `OPENAI_BASE_URL`，
> Embedding 用 `ollama` 或 `huggingface` provider 指向独立地址，这样不会冲突。

---

## 方案四：接入 Elasticsearch 向量数据库

gpt-researcher 的构造函数接受 `vector_store` 参数，类型是任何 LangChain `VectorStore` 实例。
内部会用 `VectorStoreWrapper` 包装它，提供 `load()` 和 `asimilarity_search()` 两个方法。

```python
# vector_store/es_store.py
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch

def create_es_vector_store(
    es_url: str = "https://internal-es.yourcompany.com:9200",
    index_name: str = "research_vectors",
    embedding_model=None,  # LangChain Embeddings 实例
):
    """创建 Elasticsearch 向量数据库实例"""

    es_client = Elasticsearch(
        es_url,
        # 根据你的 ES 鉴权方式选择:
        # basic_auth=("user", "password"),
        # api_key="your-api-key",
        verify_certs=False,  # 内网自签证书
    )

    store = ElasticsearchStore(
        index_name=index_name,
        embedding=embedding_model,
        es_connection=es_client,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(),  # 近似 KNN
    )

    return store
```

---

## 完整集成代码

```python
# main.py — 你的项目入口
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env

from gpt_researcher import GPTResearcher

# ---- [可选] 接入 Elasticsearch 向量库 ----
from vector_store.es_store import create_es_vector_store
from langchain_openai import OpenAIEmbeddings  # 或用其他 Embedding

embedding = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "your-embedding-model"),
    openai_api_base=os.getenv("EMBEDDING_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
es_store = create_es_vector_store(
    es_url=os.getenv("ES_URL"),
    index_name="research_vectors",
    embedding_model=embedding,
)


async def run_research(query: str, report_type: str = "research_report"):
    """执行一次完整研究"""

    researcher = GPTResearcher(
        query=query,
        report_type=report_type,       # "research_report" | "detailed_report" | "deep_research"
        report_source="web",           # 仍用 "web"，但实际走的是 custom retriever
        tone="objective",
        verbose=True,

        # ---- 接入 Elasticsearch（可选）----
        # 研究过程中抓取到的内容会自动写入 ES
        # 后续研究可以从 ES 中检索历史知识
        vector_store=es_store,

        # ---- 传入 JSON 配置文件路径（可选）----
        # config_path="config.json",
    )

    # 执行研究
    context = await researcher.conduct_research()

    # 生成报告
    report = await researcher.write_report()

    # 输出
    print(f"\n{'='*60}")
    print(f"Research Cost: ${researcher.get_costs()}")
    print(f"Sources: {len(researcher.get_source_urls())}")
    print(f"{'='*60}\n")
    print(report)

    return report


if __name__ == "__main__":
    query = "人工智能在医疗领域的最新应用进展"
    asyncio.run(run_research(query, report_type="deep_research"))
```

---

## 完整 .env 配置模板

```bash
# ============================================================
# 搜索引擎配置
# ============================================================
RETRIEVER=custom
RETRIEVER_ENDPOINT=http://localhost:8001/search
# 自定义参数（会以 query string 形式传给适配层）
# RETRIEVER_ARG_TOPK=10
# RETRIEVER_ARG_LANG=zh

# ============================================================
# LLM 配置（指向内部大模型，OpenAI 兼容协议）
# ============================================================
OPENAI_API_KEY=your-internal-api-key
OPENAI_BASE_URL=https://internal-llm.yourcompany.com/v1

FAST_LLM=openai:your-fast-model
SMART_LLM=openai:your-smart-model
STRATEGIC_LLM=openai:your-smart-model

# Token 限制（根据内部模型能力调整）
FAST_TOKEN_LIMIT=4000
SMART_TOKEN_LIMIT=8000
STRATEGIC_TOKEN_LIMIT=4000

# ============================================================
# Embedding 配置
# ============================================================
# 方式1: 和 LLM 共用 OPENAI_BASE_URL（如果同一个服务同时提供 chat 和 embedding）
EMBEDDING=openai:your-embedding-model

# 方式2: 使用 HuggingFace 本地模型（完全离线，不依赖任何外部 API）
# EMBEDDING=huggingface:sentence-transformers/all-MiniLM-L6-v2

# ============================================================
# Elasticsearch 配置（在代码中传入，非环境变量驱动）
# ============================================================
ES_URL=https://internal-es.yourcompany.com:9200
ES_INDEX=research_vectors
EMBEDDING_BASE_URL=https://internal-embedding.yourcompany.com/v1

# ============================================================
# 研究参数
# ============================================================
LANGUAGE=chinese
TOTAL_WORDS=2000
MAX_ITERATIONS=3
MAX_SEARCH_RESULTS_PER_QUERY=5
SIMILARITY_THRESHOLD=0.35
REPORT_FORMAT=APA

# Deep Research 特有参数
DEEP_RESEARCH_BREADTH=3
DEEP_RESEARCH_DEPTH=2
DEEP_RESEARCH_CONCURRENCY=4

# ============================================================
# 禁用不需要的功能
# ============================================================
IMAGE_GENERATION_ENABLED=False
CURATE_SOURCES=False
SCRAPER=bs
```

---

## 可选 config.json（覆盖默认配置）

```json
{
  "RETRIEVER": "custom",
  "LANGUAGE": "chinese",
  "TOTAL_WORDS": 2000,
  "MAX_ITERATIONS": 3,
  "REPORT_FORMAT": "APA",
  "SCRAPER": "bs",
  "CURATE_SOURCES": false,
  "IMAGE_GENERATION_ENABLED": false,
  "DEEP_RESEARCH_BREADTH": 3,
  "DEEP_RESEARCH_DEPTH": 2,
  "DEEP_RESEARCH_CONCURRENCY": 4
}
```

> 使用方式: `GPTResearcher(query=..., config_path="config.json")`
> config.json 和 .env 均可设置参数，**环境变量优先级更高**。

---

## 项目目录结构

```
my-deep-research/
├── .env                           # 所有环境变量配置
├── config.json                    # gpt-researcher 配置覆盖（可选）
├── main.py                        # 主入口
├── adapter_api.py                 # 内部搜索引擎 → CustomRetriever 适配层
├── vector_store/
│   └── es_store.py                # Elasticsearch 向量库适配
├── requirements.txt               # 依赖
│   # gpt-researcher
│   # langchain-elasticsearch
│   # elasticsearch
│   # python-dotenv
└── README.md
```

---

## 关键注意事项

### 1. 网页抓取行为

CustomRetriever 返回 URL 后，gpt-researcher 仍会尝试**抓取这些 URL 的网页内容**（通过 `BrowserManager.browse_urls()`）。如果你的内部搜索引擎已经返回了全文内容，有两种处理方式：

- **推荐方式**：在适配层返回的 `raw_content` 中填入完整文档内容，URL 填内部链接。gpt-researcher 抓取后会用 `raw_content` 覆盖，不会丢失内容。
- **替代方式**：如果内部 URL 无法从外部访问，确保 `raw_content` 有内容，抓取失败时 gpt-researcher 会跳过该 URL 继续处理其他结果。

### 2. Token 限制

内部模型的上下文窗口可能比 GPT-4 小，需要注意：

```bash
# 根据内部模型实际能力调整
SMART_TOKEN_LIMIT=4000     # 默认 6000，如果内部模型窗口小就降低
BROWSE_CHUNK_MAX_LENGTH=4096  # 单个文档最大长度
MAX_SEARCH_RESULTS_PER_QUERY=3  # 减少搜索结果数量以控制上下文长度
MAX_ITERATIONS=2               # 减少子查询数量
```

### 3. Deep Research 的资源消耗

`deep_research` 模式会递归生成 `breadth × depth` 个子研究任务，每个任务都会调用 LLM + 搜索引擎。默认 `breadth=3, depth=2` 意味着 ~12 次子研究。如果内部 LLM 资源有限：

```bash
DEEP_RESEARCH_BREADTH=2     # 降低每层宽度
DEEP_RESEARCH_DEPTH=1       # 降低递归深度
DEEP_RESEARCH_CONCURRENCY=2 # 控制并发数
```

### 4. 中文支持

```bash
LANGUAGE=chinese   # 报告生成使用中文
```

提示词模板中 `{language}` 会被替换为 "chinese"，LLM 生成报告时会使用中文。但子查询生成默认是英文提示词，可能产出英文查询。如果需要中文查询，可以在 query 中明确加上中文指令，如 "请用中文搜索: ..."。
