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

# ======================================================================
# 必须在所有网络库 import 之前设置，避免代理拦截本地请求
# 同时设置大小写两种形式，兼容 requests / urllib3 / httpx 等库
# ======================================================================
_no_proxy = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = _no_proxy
os.environ['no_proxy'] = _no_proxy

# 防止 tiktoken 尝试从外网下载编码文件（指向本地空目录即可）
_tiktoken_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.tiktoken_cache')
os.makedirs(_tiktoken_cache, exist_ok=True)
os.environ['TIKTOKEN_CACHE_DIR'] = _tiktoken_cache

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

# 加载 .env（必须在 import gpt_researcher 之前, 因为 Config 在 import 时读环境变量）
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from gpt_researcher import GPTResearcher  # noqa: E402

from custom_embeddings import InternalEmbeddings  # noqa: E402

# ======================================================================
# Monkey-patch: 替换 gpt_researcher.utils.costs 中的 tiktoken 依赖
# pip 安装的包不应直接修改，因此在运行时替换费用估算函数，
# 用简单的字符级近似取代 tiktoken 的精确 token 计数。
# ======================================================================
def _patch_costs():
    """用不依赖 tiktoken 的近似实现替换 costs 模块中的函数"""
    try:
        from gpt_researcher.utils import costs as _costs_mod

        def _approx_token_count(text: str) -> int:
            if not text:
                return 0
            cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff'
                      or '\u3040' <= ch <= '\u30ff'
                      or '\uac00' <= ch <= '\ud7af')
            if len(text) > 0 and cjk / len(text) > 0.3:
                return max(1, int(len(text) / 1.5))
            return max(1, len(text) // 4)

        def patched_estimate_llm_cost(input_content: str, output_content: str) -> float:
            inp = _approx_token_count(input_content) * _costs_mod.INPUT_COST_PER_TOKEN
            out = _approx_token_count(output_content) * _costs_mod.OUTPUT_COST_PER_TOKEN
            return inp + out

        def patched_estimate_embedding_cost(model: str, docs: list) -> float:
            total = sum(_approx_token_count(str(d)) for d in docs)
            return total * _costs_mod.EMBEDDING_COST

        _costs_mod.estimate_llm_cost = patched_estimate_llm_cost
        _costs_mod.estimate_embedding_cost = patched_estimate_embedding_cost
    except Exception:
        pass  # 万一 costs 模块结构变了，不影响主流程

_patch_costs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _check_adapter():
    """检查搜索适配服务是否可用，并做一次真实搜索测试"""
    import httpx

    endpoint = os.getenv("RETRIEVER_ENDPOINT", "http://127.0.0.1:8001/search")
    health_url = endpoint.rsplit("/", 1)[0] + "/health"

    # 1) 健康检查
    try:
        resp = httpx.get(health_url, timeout=5.0)
        if resp.status_code == 200:
            logger.info(f"Search adapter is running: {health_url}")
        else:
            logger.error(f"Search adapter returned status {resp.status_code}")
            return False
    except Exception as e:
        logger.error(
            f"Search adapter not available at {health_url}: {e}\n"
            "Please start it first: cd my_deep_research && uvicorn adapter.app:app --port 8001"
        )
        return False

    # 2) 搜索连通性测试：发一次真实搜索请求，验证整个链路畅通
    try:
        test_resp = httpx.get(endpoint, params={"query": "test"}, timeout=15.0)
        test_resp.raise_for_status()
        test_data = test_resp.json()
        logger.info(f"Search connectivity test: got {len(test_data)} results for query 'test'")
        if test_data:
            logger.info(f"  Sample result: title='{test_data[0].get('title', '')[:50]}', "
                        f"href='{test_data[0].get('href', '')}'")
    except Exception as e:
        logger.warning(f"Search connectivity test failed: {e}")
        logger.warning("The adapter is running but the internal search API may not be reachable.")

    return True


def _check_proxy_env():
    """打印当前代理相关环境变量，方便诊断"""
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY',
                  'http_proxy', 'https_proxy', 'no_proxy']
    for var in proxy_vars:
        val = os.environ.get(var, '')
        if val:
            logger.info(f"  {var}={val}")


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

    # 诊断信息
    logger.info("=== Proxy environment ===")
    _check_proxy_env()

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
