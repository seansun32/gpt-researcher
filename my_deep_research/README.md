# 内部 Deep Research 应用

基于 [gpt-researcher](https://github.com/assafelovic/gpt-researcher) 构建的私有化深度研究能力，
对接内部搜索引擎、LLM 和 Embedding API，不依赖任何外部服务。

## 架构

```
                    ┌─────────────────────────┐
                    │     main.py             │
                    │  GPTResearcher 编排器    │
                    └──────┬──────┬───────────┘
                           │      │
              ┌────────────┘      └────────────┐
              ▼                                ▼
┌──────────────────────┐           ┌───────────────────────┐
│   CustomRetriever    │           │   InternalEmbeddings  │
│  (gpt-researcher     │           │  (custom_embeddings   │
│   内置, GET 协议)    │           │   .py)                │
└──────────┬───────────┘           └───────────┬───────────┘
           │ GET /search?query=...             │ POST /get_feature
           ▼                                   ▼
┌──────────────────────┐           ┌───────────────────────┐
│   搜索适配服务        │           │   内部 Embedding API  │
│  adapter/app.py      │           │  www.xxx.com          │
│  :8001               │           │  /get_feature         │
└──────────┬───────────┘           └───────────────────────┘
           │ POST (转发)
           ▼                                ┌───────────────────────┐
┌──────────────────────┐                    │   内部 LLM API        │
│   内部搜索引擎        │                    │  www.xxx.com          │
│  www.xxx.com/search  │                    │  /modelops/v1         │
└──────────────────────┘                    │  (OpenAI 兼容)        │
                                            └───────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
cd my_deep_research
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入实际的内部 API 地址和凭证
```

### 3. 启动搜索适配服务

```bash
# 终端 1
uvicorn adapter.app:app --host 0.0.0.0 --port 8001
```

### 4. 运行研究

```bash
# 终端 2 — 基础研究报告
python main.py "AI在医疗领域的最新应用进展"

# 详细报告（多阶段、分主题）
python main.py "AI在医疗领域的最新应用进展" --type detailed_report

# 深度研究（递归树状探索）
python main.py "AI在医疗领域的最新应用进展" --type deep_research

# 输出到文件
python main.py "AI在医疗领域的最新应用进展" --type deep_research --output report.md
```

### 5. 在代码中使用

```python
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from main import run_research

report = asyncio.run(run_research(
    query="人工智能在金融风控领域的应用",
    report_type="deep_research",
))
print(report)
```

## 组件说明

| 组件 | 文件 | 作用 |
|------|------|------|
| 搜索适配服务 | `adapter/app.py` | 将内部搜索 POST API 转换为 gpt-researcher 期望的 GET 接口，同时缓存文档内容供 scraper 抓取 |
| Embedding 适配 | `custom_embeddings.py` | 将内部 Embedding API 包装为 LangChain Embeddings 接口 |
| 研究主入口 | `main.py` | 配置 GPTResearcher，替换 Embedding，执行研究并生成报告 |

## 配置参考

参见 `.env.example`，关键配置项：

- `INTERNAL_SEARCH_URL` — 内部搜索引擎 API 地址
- `INTERNAL_EMBEDDING_URL` — 内部 Embedding API 地址
- `OPENAI_BASE_URL` — 内部 LLM API 地址（OpenAI 兼容）
- `FAST_LLM` / `SMART_LLM` / `STRATEGIC_LLM` — LLM 模型名称
- `DEEP_RESEARCH_BREADTH` / `DEEP_RESEARCH_DEPTH` — 深度研究参数
