"""
搜索适配服务

将内部搜索引擎的 POST 接口适配为 gpt-researcher CustomRetriever 期望的 GET 接口。
同时缓存文档内容，提供 /doc/{doc_id} 端点供 gpt-researcher 的 scraper 抓取。

启动方式:
    uvicorn adapter.app:app --host 0.0.0.0 --port 8001
"""

import os

# 避免代理拦截本地请求（必须在所有网络库 import 之前设置）
# 同时设置大小写两种形式，兼容 requests / urllib3 / httpx
_no_proxy = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = _no_proxy
os.environ['no_proxy'] = _no_proxy

import time
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Search Adapter for GPT-Researcher")

# ------------------------------------------------------------------
# 配置项（从环境变量读取）
# ------------------------------------------------------------------
SEARCH_URL = os.getenv("INTERNAL_SEARCH_URL", "http://www.xxx.com/search")
SEARCH_APP_ID = os.getenv("INTERNAL_SEARCH_APP_ID", "my-app-id")
SEARCH_TENANT_ID = os.getenv("INTERNAL_SEARCH_TENANT_ID", "my-tenant-id")
ADAPTER_HOST = os.getenv("ADAPTER_HOST", "http://127.0.0.1:8001")

# ------------------------------------------------------------------
# 文档内容缓存
# ------------------------------------------------------------------
# key: doc_id (str) → value: {"title": str, "content": str, "url": str}
doc_cache: dict[str, dict] = {}

# 缓存过期时间（秒），默认 1 小时
CACHE_TTL = int(os.getenv("ADAPTER_CACHE_TTL", "3600"))


def _clean_expired_cache():
    """清理过期的缓存条目"""
    now = time.time()
    expired = [k for k, v in doc_cache.items() if now - v.get("_ts", 0) > CACHE_TTL]
    for k in expired:
        del doc_cache[k]


# ------------------------------------------------------------------
# GET /search?query=xxx
# CustomRetriever 会调用此接口
# ------------------------------------------------------------------
@app.get("/search")
async def search(query: str = Query(..., description="搜索关键词")):
    """
    接收 gpt-researcher CustomRetriever 的 GET 请求，
    转发到内部搜索引擎的 POST 接口，
    将返回结果转换为 gpt-researcher 期望的格式。

    gpt-researcher 期望返回格式:
    [
        {"href": "url", "body": "摘要内容", "title": "标题"},
        ...
    ]

    其中 href 会被 scraper 抓取，所以指向本服务的 /doc/{doc_id} 端点。
    """
    _clean_expired_cache()

    # 构造内部搜索 API 请求体
    payload = {
        "app_id": SEARCH_APP_ID,
        "tenant_id": SEARCH_TENANT_ID,
        "user_role": "",
        "user_id": "",
        "search_txt": query,
        "sort_order": "desc",
        "page_index": "1",
        "page_size": "10",
        "tenant_attrib": "",
        "custom_params": [],
    }

    logger.info(f"[/search] Received query: '{query}'")
    logger.info(f"[/search] Calling internal API: POST {SEARCH_URL}")
    logger.debug(f"[/search] Payload: {payload}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(SEARCH_URL, json=payload)
            logger.info(f"[/search] Internal API response status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"[/search] Internal search API error: {e}", exc_info=True)
        return []

    # 解析内部搜索结果
    hits = data.get("data", {}).get("data", [])
    logger.info(f"[/search] Parsed {len(hits)} hits from response")
    if not hits:
        # 打印响应结构帮助诊断
        top_keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        logger.warning(f"[/search] No results for query: '{query}'. Response top-level keys: {top_keys}")
        return []

    results = []
    for item in hits:
        doc_id = str(item.get("DOC_ID", ""))
        title = item.get("DOC_TITLE", "")
        content = item.get("DOC_CONTENT", "")
        highlight = item.get("HIGHLIGHT", [])
        doc_url = item.get("DOC_URL", "")
        source_url = item.get("DOC_SOURCE_URL", "")

        # 优先用 DOC_CONTENT，若为空则拼接 HIGHLIGHT
        if not content and highlight:
            content = "\n".join(highlight)

        # 缓存文档内容，供 /doc/{doc_id} 端点使用
        doc_cache[doc_id] = {
            "title": title,
            "content": content,
            "url": doc_url or source_url,
            "_ts": time.time(),
        }

        # 返回 gpt-researcher 期望的格式
        # href 指向本服务的 /doc/{doc_id}，确保 scraper 可以抓取到内容
        results.append({
            "href": f"{ADAPTER_HOST}/doc/{doc_id}",
            "body": content[:500] if content else title,  # body 用于子查询规划
            "title": title,
        })

    logger.info(f"[/search] Query '{query}' → returning {len(results)} results")
    if results:
        logger.info(f"[/search]   First result: href={results[0].get('href')}, title={results[0].get('title', '')[:50]}")
    return results


# ------------------------------------------------------------------
# GET /doc/{doc_id}
# gpt-researcher 的 BeautifulSoup scraper 会抓取这些 URL
# ------------------------------------------------------------------
@app.get("/doc/{doc_id}", response_class=HTMLResponse)
async def get_document(doc_id: str):
    """
    提供缓存的文档内容，供 gpt-researcher 的 scraper 抓取。
    返回简洁的 HTML，便于 BeautifulSoup 提取文本。
    """
    doc = doc_cache.get(doc_id)
    if not doc:
        return HTMLResponse(
            "<html><body><p>Document not found</p></body></html>",
            status_code=404,
        )

    title = doc.get("title", "")
    content = doc.get("content", "")
    source_url = doc.get("url", "")

    html = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
<h1>{title}</h1>
<article>{content}</article>
<footer>Source: {source_url}</footer>
</body>
</html>"""
    return HTMLResponse(html)


# ------------------------------------------------------------------
# 健康检查
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(doc_cache)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
