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
import sys
import types

# ======================================================================
# 必须在所有网络库 import 之前设置，避免代理拦截本地请求
# 同时设置大小写两种形式，兼容 requests / urllib3 / httpx 等库
# ======================================================================
_no_proxy = 'localhost,127.0.0.1'
os.environ['NO_PROXY'] = _no_proxy
os.environ['no_proxy'] = _no_proxy

# ======================================================================
# 注入 fake tiktoken 到 sys.modules，彻底阻止外网下载
#
# gpt-researcher 的 costs.py 会 import tiktoken，tiktoken 在调用
# get_encoding() 时会尝试从 openaipublic.blob.core.windows.net 下载
# 编码文件。在内网环境中无法访问，导致 ConnectTimeout。
#
# 解决方案：在 import gpt_researcher 之前，将一个 fake tiktoken 模块
# 注入 sys.modules。这样 costs.py 的 `import tiktoken` 拿到的是我们的
# mock，encode() 返回近似 token 列表，不会发起任何网络请求。
# ======================================================================
_fake_tiktoken = types.ModuleType('tiktoken')
_fake_tiktoken.__package__ = 'tiktoken'


class _FakeEncoding:
    """假编码器：用字符数近似 token 数，避免外网下载真实编码表"""
    def encode(self, text):
        if not text:
            return []
        cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff'
                  or '\u3040' <= ch <= '\u30ff'
                  or '\uac00' <= ch <= '\ud7af')
        ratio = cjk / len(text) if text else 0
        count = int(len(text) / 1.5) if ratio > 0.3 else len(text) // 4
        return [0] * max(1, count)


_fake_tiktoken.get_encoding = lambda name: _FakeEncoding()
_fake_tiktoken.encoding_for_model = lambda model: _FakeEncoding()
_fake_tiktoken.Encoding = _FakeEncoding

sys.modules['tiktoken'] = _fake_tiktoken

import argparse
import asyncio
import logging

from dotenv import load_dotenv

# 加载 .env（必须在 import gpt_researcher 之前, 因为 Config 在 import 时读环境变量）
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from gpt_researcher import GPTResearcher  # noqa: E402
from gpt_researcher.memory.embeddings import Memory  # noqa: E402
from gpt_researcher.prompts import PromptFamily  # noqa: E402
from gpt_researcher.skills.deep_research import DeepResearchSkill  # noqa: E402
from gpt_researcher.utils.llm import create_chat_completion  # noqa: E402

from custom_embeddings import InternalEmbeddings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ######################################################################
#                      Monkey-patch 区域
#
# 以下 patch 均为类级别替换，对所有 researcher 实例（主+子）自动生效，
# 无需修改 gpt-researcher 上游源码。
# ######################################################################

# ======================================================================
# Patch 1: Memory 类 —— 使 "custom" provider 使用 InternalEmbeddings
#
# 默认 "custom" provider 会创建 OpenAIEmbeddings 向 OPENAI_BASE_URL/embeddings
# 发请求，但内部 LLM 服务没有 /embeddings 端点，导致 404。
# ======================================================================
_original_memory_init = Memory.__init__


def _patched_memory_init(self, embedding_provider, model, **kwargs):
    if embedding_provider == "custom":
        self._embeddings = InternalEmbeddings(
            api_url=os.getenv("INTERNAL_EMBEDDING_URL"),
            app_id=os.getenv("INTERNAL_EMBEDDING_APP_ID"),
            model=os.getenv("INTERNAL_EMBEDDING_MODEL"),
        )
    else:
        _original_memory_init(self, embedding_provider, model, **kwargs)


Memory.__init__ = _patched_memory_init
logger.info("Patched Memory.__init__: 'custom' provider → InternalEmbeddings")

# ======================================================================
# Patch 2: DeepResearchSkill.process_research_results
#
# 原始实现的两个结构性缺陷：
#   a) 让 LLM 同时做"提取 learning"和"关联 URL"两件事，LLM 常张冠李戴
#   b) 用正则从 LLM 输出中提取 URL，格式稍有偏差就丢失
#
# 新方案：职责分离
#   - LLM 只负责提取 learning（它擅长的事）
#   - citation 匹配由代码完成：先把 context 解析回结构化 source 块，
#     再用 embedding 余弦相似度把每条 learning 匹配到最相关的源文档 URL
# ======================================================================
import math  # noqa: E402
import re  # noqa: E402
from gpt_researcher.llm_provider.generic.base import ReasoningEfforts  # noqa: E402


def _parse_context_to_sources(context_str):
    """将 pretty_print_docs 输出的扁平文本解析回结构化 source 块。

    输入格式（由 prompts.py 的 pretty_print_docs 生成）：
        Source: https://example.com
        Title: Page Title
        Content: extracted text...

    返回: [{'url': str, 'title': str, 'content': str}, ...]
    """
    sources = []
    if not context_str:
        return sources
    # 按 "Source: " 标记拆分（兼容开头和中间）
    blocks = re.split(r'(?:^|\n)Source: ', context_str)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split('\n', 2)
        url = lines[0].strip()
        if not url.startswith('http'):
            continue
        title = ''
        content = ''
        remaining = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', remaining)
        if title_match:
            title = title_match.group(1).strip()
        content_match = re.search(r'Content:\s*(.*)', remaining, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
        sources.append({'url': url, 'title': title, 'content': content})
    return sources


def _cosine_similarity(vec_a, vec_b):
    """计算两个向量的余弦相似度。"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _patched_process_research_results(self, query, context, num_learnings=3):
    # ------------------------------------------------------------------
    # Phase 1: 解析 context → 结构化 source 块（保留 URL-content 映射）
    # ------------------------------------------------------------------
    sources = _parse_context_to_sources(context)
    logger.debug(f"Parsed {len(sources)} source blocks from context")

    # ------------------------------------------------------------------
    # Phase 2: LLM 只负责提取 learning 和后续问题（不要求关联 URL）
    # ------------------------------------------------------------------
    messages = [
        {"role": "system", "content": "你是一位专业的研究分析师，负责从搜索结果中提取关键发现。"},
        {"role": "user", "content": f"""请根据以下研究结果，针对查询 "{query}" 提取关键发现和后续问题。

要求：
- 每条发现必须来自下面提供的研究结果，不得编造内容
- 不需要标注来源URL（系统会自动匹配）
- 每条发现应简洁、具体、包含关键数据或事实

输出格式（严格遵循，每行一条）：
Learning: 发现内容
Question: 后续问题

研究结果：
{context}"""}
    ]

    response = await create_chat_completion(
        messages=messages,
        llm_provider=self.researcher.cfg.strategic_llm_provider,
        model=self.researcher.cfg.strategic_llm_model,
        temperature=0.4,
        reasoning_effort=ReasoningEfforts.High.value,
        max_tokens=1000
    )

    learnings = []
    questions = []
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('Learning'):
            # 兼容 "Learning: xxx" 和 "Learning 1: xxx" 等格式
            content = re.sub(r'^Learning\s*\d*[:：]\s*', '', line).strip()
            if content:
                learnings.append(content)
        elif line.startswith('Question'):
            content = re.sub(r'^Question\s*\d*[:：]\s*', '', line).strip()
            if content:
                questions.append(content)

    learnings = learnings[:num_learnings]
    questions = questions[:num_learnings]

    # ------------------------------------------------------------------
    # Phase 3: 用 embedding 相似度将 learning 匹配到源文档 URL
    #   - 不依赖 LLM 做 URL 关联，用代码精确计算
    #   - 每条 learning 匹配到与之最相似的 source content
    # ------------------------------------------------------------------
    citations = {}
    if sources and learnings:
        try:
            embeddings = InternalEmbeddings(
                api_url=os.getenv("INTERNAL_EMBEDDING_URL"),
                app_id=os.getenv("INTERNAL_EMBEDDING_APP_ID"),
                model=os.getenv("INTERNAL_EMBEDDING_MODEL"),
            )
            # 截取 source content 前 500 字符以提高效率和聚焦度
            source_texts = [s['content'][:500] for s in sources]

            # 在线程中执行同步 embedding 调用，避免阻塞事件循环
            learning_vecs = await asyncio.to_thread(
                embeddings.embed_documents, learnings
            )
            source_vecs = await asyncio.to_thread(
                embeddings.embed_documents, source_texts
            )

            for i, learning in enumerate(learnings):
                if not learning_vecs[i]:
                    continue
                best_sim = 0.0
                best_url = ''
                for j, source in enumerate(sources):
                    if not source_vecs[j]:
                        continue
                    sim = _cosine_similarity(learning_vecs[i], source_vecs[j])
                    if sim > best_sim:
                        best_sim = sim
                        best_url = source['url']
                if best_url and best_sim > 0.35:
                    citations[learning] = best_url
                    logger.debug(f"Citation matched (sim={best_sim:.3f}): "
                                 f"'{learning[:40]}...' → {best_url}")
                else:
                    logger.debug(f"No citation match for: '{learning[:40]}...' "
                                 f"(best_sim={best_sim:.3f})")
        except Exception as e:
            logger.warning(f"Embedding citation matching failed, "
                           f"learnings will have no citations: {e}")

    return {
        'learnings': learnings,
        'followUpQuestions': questions,
        'citations': citations
    }


DeepResearchSkill.process_research_results = _patched_process_research_results
logger.info("Patched DeepResearchSkill.process_research_results: "
            "LLM提取learning + embedding匹配citation")

# ======================================================================
# Patch 3: 报告生成 prompt —— 禁止编造 URL，禁止添加免责声明
#
# 原始 prompt 用 "MUST" 强制要求每处都加 URL 引用，当 context 中 URL
# 不足时，LLM 会编造 URL 并可能添加"以上链接为示例"的免责声明。
# ======================================================================
from datetime import date, datetime, timezone  # noqa: E402
from gpt_researcher.utils.enum import ReportSource  # noqa: E402

_original_generate_report_prompt = PromptFamily.generate_report_prompt
_original_generate_deep_research_prompt = PromptFamily.generate_deep_research_prompt

# 通用的引用约束指令（中英双语，确保 LLM 理解）
_CITATION_CONSTRAINT = """
CRITICAL CITATION RULES (引用规则 - 必须严格遵守):
- ONLY use URLs that appear in the provided context/information above. 只使用上文中已有的真实 URL。
- If a piece of information has no corresponding URL in the context, cite it WITHOUT a URL — just describe the source in text. 如果某条信息没有对应 URL，则不加链接，用文字描述来源即可。
- NEVER fabricate, guess, or generate any URL. 绝对不要编造、猜测或生成任何 URL。
- NEVER add disclaimers like "以上链接为示例" or "需替换为真实链接". 绝对不要添加类似"链接为示例"的免责声明。
- It is better to have NO URL than a fake URL. 没有链接好过假链接。
"""


@staticmethod
def _patched_generate_report_prompt(
    question, context, report_source, report_format="apa",
    total_words=1000, tone=None, language="english",
):
    base = _original_generate_report_prompt(
        question, context, report_source, report_format,
        total_words, tone, language,
    )
    return base + _CITATION_CONSTRAINT


@staticmethod
def _patched_generate_deep_research_prompt(
    question, context, report_source, report_format="apa",
    tone=None, total_words=2000, language="english",
):
    base = _original_generate_deep_research_prompt(
        question, context, report_source, report_format,
        tone, total_words, language,
    )
    return base + _CITATION_CONSTRAINT


PromptFamily.generate_report_prompt = _patched_generate_report_prompt
PromptFamily.generate_deep_research_prompt = _patched_generate_deep_research_prompt
logger.info("Patched PromptFamily report prompts: 禁止编造 URL 和免责声明")


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
    # Step 2: 执行研究
    # （Embedding 已通过 Memory 类级别 patch 自动使用 InternalEmbeddings，
    #   对主 researcher 和 deep_research 创建的子 researcher 均生效）
    # ------------------------------------------------------------------
    logger.info(f"Starting research: '{query}' (type={report_type})")
    await researcher.conduct_research()

    # ------------------------------------------------------------------
    # Step 3: 生成报告
    # ------------------------------------------------------------------
    logger.info("Generating report...")
    report = await researcher.write_report()

    # ------------------------------------------------------------------
    # Step 4: 输出统计
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
