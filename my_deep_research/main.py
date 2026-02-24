"""
内部 Deep Research 应用主入口

基于 gpt-researcher (pip install) 构建的私有化 deep research 能力。
通过环境变量 + 搜索适配层 + 自定义 Embedding，对接内部搜索引擎、LLM 和 Embedding API。

使用前请确保:
    1. 搜索适配服务已启动: uvicorn adapter.app:app --port 8001
    2. .env 配置正确
    3. 内部 LLM、搜索、Embedding 服务可访问

使用方式:
    python main.py "AI在医疗领域的最新应用进展"
    python main.py "AI在医疗领域的最新应用进展" --type deep_research
    python main.py "AI在医疗领域的最新应用进展" --type detailed_report
"""

import os

# 避免代理拦截本地请求（必须在所有网络请求之前设置）
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# 防止 tiktoken 尝试从外网下载编码文件（安全兜底）
os.environ.setdefault('TIKTOKEN_CACHE_DIR', os.path.join(os.path.dirname(__file__), '.tiktoken_cache'))

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

# 加载 .env（必须在 import gpt_researcher 之前, 因为 Config 在 import 时读环境变量）
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from gpt_researcher import GPTResearcher  # noqa: E402

from custom_embeddings import InternalEmbeddings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _check_adapter():
    """检查搜索适配服务是否可用"""
    import httpx

    endpoint = os.getenv("RETRIEVER_ENDPOINT", "http://127.0.0.1:8001/search")
    health_url = endpoint.rsplit("/", 1)[0] + "/health"
    try:
        resp = httpx.get(health_url, timeout=5.0)
        if resp.status_code == 200:
            logger.info(f"Search adapter is running: {health_url}")
            return True
    except Exception:
        pass
    logger.error(
        f"Search adapter not available at {health_url}\n"
        "Please start it first: cd my_deep_research && uvicorn adapter.app:app --port 8001"
    )
    return False


async def run_research(
    query: str,
    report_type: str = "research_report",
    tone: str = "objective",
):
    """
    执行一次完整研究。

    Args:
        query: 研究问题
        report_type: 报告类型
            - "research_report": 基础研究报告（单轮生成，~1000词）
            - "detailed_report": 详细报告（多阶段分主题，~3000词）
            - "deep_research": 深度研究（递归树状探索）
        tone: 报告语气 ("objective", "analytical", "informative" 等)

    Returns:
        str: 生成的 Markdown 报告
    """

    # ------------------------------------------------------------------
    # Step 1: 创建 GPTResearcher 实例
    # ------------------------------------------------------------------
    # 环境变量已设置好:
    #   RETRIEVER=custom → 使用 CustomRetriever
    #   RETRIEVER_ENDPOINT=http://127.0.0.1:8001/search → 指向搜索适配层
    #   OPENAI_BASE_URL=http://www.xxx.com/modelops/v1 → 指向内部 LLM
    #   EMBEDDING=custom:dummy → 占位，下面会替换
    researcher = GPTResearcher(
        query=query,
        report_type=report_type,
        report_source="web",  # 仍用 "web" 流程，但实际通过 CustomRetriever 走内部搜索
        tone=tone,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Step 2: 替换 Embedding 为内部实现
    # ------------------------------------------------------------------
    # gpt-researcher 的 Memory 对象在 __init__ 时会创建一个 OpenAIEmbeddings 占位
    # 此处替换为我们的 InternalEmbeddings，直接调用内部 Embedding API
    internal_embeddings = InternalEmbeddings(
        api_url=os.getenv("INTERNAL_EMBEDDING_URL"),
        app_id=os.getenv("INTERNAL_EMBEDDING_APP_ID"),
        model=os.getenv("INTERNAL_EMBEDDING_MODEL"),
    )
    researcher.memory._embeddings = internal_embeddings
    logger.info("Replaced default embeddings with InternalEmbeddings")

    # ------------------------------------------------------------------
    # Step 3: 执行研究
    # ------------------------------------------------------------------
    logger.info(f"Starting research: '{query}' (type={report_type})")
    await researcher.conduct_research()

    # ------------------------------------------------------------------
    # Step 4: 生成报告
    # ------------------------------------------------------------------
    logger.info("Generating report...")
    report = await researcher.write_report()

    # ------------------------------------------------------------------
    # Step 5: 输出统计
    # ------------------------------------------------------------------
    costs = researcher.get_costs()
    sources = researcher.get_source_urls()
    logger.info(f"Research completed. Cost: ${costs:.4f}, Sources: {len(sources)}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Internal Deep Research Application")
    parser.add_argument("query", help="Research query")
    parser.add_argument(
        "--type",
        choices=["research_report", "detailed_report", "deep_research"],
        default="research_report",
        help="Report type (default: research_report)",
    )
    parser.add_argument(
        "--tone",
        default="objective",
        help="Report tone (default: objective)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: print to stdout)",
    )
    args = parser.parse_args()

    # 预检查
    if not _check_adapter():
        sys.exit(1)

    # 执行研究
    report = asyncio.run(run_research(args.query, args.type, args.tone))

    # 输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {args.output}")
    else:
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)


if __name__ == "__main__":
    main()
