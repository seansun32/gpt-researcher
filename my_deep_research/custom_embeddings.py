"""
内部 Embedding API 适配器

将内部的 Embedding API 包装为 LangChain Embeddings 接口，
用于替换 gpt-researcher 默认的 OpenAI Embeddings。

内部 API 协议:
    POST http://www.xxx.com/get_feature
    Body: {"app_id": "...", "search": "文本", "model": "posong"}
    Response: {"embedding": [0.1, 0.2, ...], "status": "success"}
"""

import os

# 避免代理拦截本地请求（必须在所有网络库 import 之前设置）
_no_proxy = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = _no_proxy
os.environ['no_proxy'] = _no_proxy

import logging
from typing import List

import httpx
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class InternalEmbeddings(Embeddings):
    """适配内部 Embedding API 的 LangChain Embeddings 实现。

    Attributes:
        api_url: 内部 embedding API 地址
        app_id: 应用标识
        model: 模型名称
        timeout: 请求超时时间（秒）
    """

    def __init__(
        self,
        api_url: str = None,
        app_id: str = None,
        model: str = None,
        timeout: float = 30.0,
    ):
        self.api_url = api_url or os.getenv(
            "INTERNAL_EMBEDDING_URL", "http://www.xxx.com/get_feature"
        )
        self.app_id = app_id or os.getenv("INTERNAL_EMBEDDING_APP_ID", "app_id")
        self.model = model or os.getenv("INTERNAL_EMBEDDING_MODEL", "posong")
        self.timeout = timeout

    def _call_api(self, text: str) -> List[float]:
        """调用内部 Embedding API 获取单条文本的向量。"""
        payload = {
            "app_id": self.app_id,
            "search": text,
            "model": self.model,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.api_url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            if data.get("status") != "success":
                logger.error(f"Embedding API error: {data.get('message')}")
                return []

            embedding = data.get("embedding", [])
            if not embedding:
                logger.warning(f"Empty embedding for text: {text[:50]}...")
            return embedding

        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """对单条查询文本生成向量（LangChain 接口）。"""
        return self._call_api(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对多条文档文本批量生成向量（LangChain 接口）。

        内部 API 仅支持单条调用，此处逐条请求。
        如果内部 API 未来支持批量，可在此优化。
        """
        results = []
        for i, text in enumerate(texts):
            vec = self._call_api(text)
            results.append(vec)
            if (i + 1) % 50 == 0:
                logger.info(f"Embedding progress: {i + 1}/{len(texts)}")
        return results
