"""search 工具：基于 Tavily 的联网搜索。"""

import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from tavily import TavilyClient

from ..errors import ToolError
from ..tools import Tool

load_dotenv()

# 与 agent.OBSERVATION_LIMIT 对齐：工具层先截断，避免观察被 agent 硬切丢信息
_RESULT_LIMIT = 2000
_TAVILY_MAX_RESULTS = 5

def make_search_tool(client: Optional[TavilyClient]=None) -> Tool:
    """构造基于 Tavily 的 search 工具。"""
    if client is None:
        key = os.getenv('TAVILY_API_KEY', '')
        # key 缺失时不构造 client，调用时给出明确提示
        client = TavilyClient(api_key=key) if key else None
    return Tool(
        name='search',
        description='联网搜索网页，返回相关网页的标题、链接与内容摘要',
        parameters={
            'type': 'object',
            'properties': {'query': {'type': 'string', 'description': '搜索关键词或问题'}},
            'required': ['query']
        },
        func=lambda query: _search(query, client)
    )

def _search(query: str, client: Optional[TavilyClient]) -> str:
    """调用 Tavily 搜索并拼装摘要文本。"""
    if client is None:
        raise ToolError('未配置 TAVILY_API_KEY，无法联网搜索（可在 https://tavily.com 免费申请）')
    started = datetime.now()
    logger.debug(f'Tavily 搜索: query={query!r} search_depth=basic max_results={_TAVILY_MAX_RESULTS}')
    response = client.search(query, search_depth='basic', max_results=_TAVILY_MAX_RESULTS)
    results = response.get('results') or []
    logger.debug(f'Tavily 返回 {len(results)} 条结果，耗时 {(datetime.now() - started).total_seconds():.2f}s')
    if not results:
        return f'未找到与「{query}」相关的结果'
    parts = [f'{item["title"]}\n{item["url"]}\n{item["content"]}' for item in results]
    text = '\n\n'.join(parts)
    # 搜索结果常含低信息密度长文（README 表格等），超限截断保护上下文
    if len(text) > _RESULT_LIMIT:
        text = text[:_RESULT_LIMIT] + '…'
    return text
