from typing import Any

from pytest import MonkeyPatch, raises

from runtime._tools.search import make_search_tool
from runtime.errors import ToolError

class FakeTavilyClient:
    """记录调用并返回预设响应的 Tavily 替身。"""

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.last_query = ''
        self.last_kwargs: dict[str, Any] = {}

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.last_query = query
        self.last_kwargs = kwargs
        return self._response

class TestMakeSearchTool:
    def test_returns_titles_urls_contents(self):
        fake = FakeTavilyClient({'results': [
            {'title': '甲', 'url': 'https://a', 'content': '内容甲'},
            {'title': '乙', 'url': 'https://b', 'content': '内容乙'}
        ]})
        tool = make_search_tool(fake)
        output = tool.func('测试词')
        assert '甲' in output and 'https://a' in output and '内容甲' in output
        assert '乙' in output and 'https://b' in output and '内容乙' in output
        # 参数原样传给 Tavily
        assert fake.last_query == '测试词'
        assert fake.last_kwargs['search_depth'] == 'basic'
        assert fake.last_kwargs['max_results'] == 5

    def test_no_results_returns_hint(self):
        tool = make_search_tool(FakeTavilyClient({'results': []}))
        assert '未找到' in tool.func('空结果')

    def test_missing_key_raises(self, monkeypatch: MonkeyPatch):
        monkeypatch.delenv('TAVILY_API_KEY', raising=False)
        tool = make_search_tool()
        with raises(ToolError, match='TAVILY_API_KEY'):
            tool.func('x')

    def test_long_output_truncated(self):
        long_content = '很' * 3000
        fake = FakeTavilyClient(
            {'results': [{'title': 't', 'url': 'u', 'content': long_content}]}
        )
        tool = make_search_tool(fake)
        output = tool.func('x')
        assert len(output) <= 2001
        assert output.endswith('…')
