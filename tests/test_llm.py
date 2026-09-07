import os
from json import loads
from types import SimpleNamespace
from typing import Any, Callable

from dotenv import load_dotenv

from httpx import (
    MockTransport, Client,
    Request, Response, ConnectError
)
from pytest import MonkeyPatch, raises

from runtime.context import Message
from runtime.errors import LLMError
from runtime.llm import OpenAIClient
from runtime.testing import ScriptedLLM

load_dotenv()

test_model = os.getenv('LLM_MODEL', 'test-model')
test_api_key = os.getenv('LLM_API_KEY', 'sk-test')

def make_client(handler: Callable[[Request], Response]) -> OpenAIClient:
    """用 MockTransport 构造离线 OpenAIClient。"""
    transport = MockTransport(handler)
    return OpenAIClient(http_client=Client(transport=transport))

def ok_response(content: str='x') -> dict[str, Any]:
    """构造 openai SDK 能解析的完整 chat.completion 响应体。"""
    return {
        'id': 'chatcmpl-test',
        'object': 'chat.completion',
        'created': 0,
        'model': test_model,
        'choices': [{
            'index': 0,
            'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': content}
        }]
    }

class TestOpenAIClient:
    def test_success_returns_content(self):
        def handler(request: Request) -> Response:
            return Response(200, json=ok_response('你好'))

        client = make_client(handler)
        assert client.complete([Message('user', 'hi')]) == '你好'

    def test_request_body_shape(self):
        captured = {}

        def handler(request: Request) -> Response:
            captured['body'] = request.content.decode('utf-8')
            captured['auth'] = request.headers.get('Authorization')
            return Response(200, json=ok_response())

        client = make_client(handler)
        client.complete([Message('system', 's'), Message('user', 'u'), Message('assistant', 'a')])
        body = loads(captured['body'])
        assert body['model'] == test_model
        assert body['messages'] == [
            {'role': 'system', 'content': 's'},
            {'role': 'user', 'content': 'u'},
            {'role': 'assistant', 'content': 'a'}
        ]
        assert captured['auth'] == f'Bearer {test_api_key}'

    def test_empty_content_raises(self):
        def handler(request: Request) -> Response:
            return Response(200, json=ok_response(''))

        client = make_client(handler)
        with raises(LLMError, match='空输出'):
            client.complete([Message('user', 'hi')])

    def test_extract_content_falls_back_to_reasoning(self):
        # 推理模型 content 为空、内容在 reasoning_content 时兜底提取
        message = SimpleNamespace(content='', reasoning_content='思考后的结论')
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        assert OpenAIClient._extract_content(response) == '思考后的结论'

    def test_extract_content_prefers_content_over_reasoning(self):
        message = SimpleNamespace(content='正文', reasoning_content='思考过程')
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        assert OpenAIClient._extract_content(response) == '正文'

    def test_no_choices_raises(self):
        def handler(request: Request) -> Response:
            body = {**ok_response(), 'choices': []}
            return Response(200, json=body)

        client = make_client(handler)
        with raises(LLMError):
            client.complete([Message('user', 'hi')])

    def test_http_500_raises_with_status(self):
        def handler(request: Request) -> Response:
            return Response(500, text='internal error')

        client = make_client(handler)
        with raises(LLMError, match='HTTP 500'):
            client.complete([Message('user', 'hi')])

    def test_connect_error_raises(self):
        def handler(request: Request) -> Response:
            raise ConnectError('连接被拒绝', request=request)

        client = make_client(handler)
        with raises(LLMError, match='调用失败'):
            client.complete([Message('user', 'hi')])

    def test_invalid_json_response_raises(self):
        def handler(request: Request) -> Response:
            return Response(200, text='not-json')

        client = make_client(handler)
        with raises(LLMError, match='合法 JSON'):
            client.complete([Message('user', 'hi')])

    def test_missing_config_raises(self, monkeypatch: MonkeyPatch):
        # .env 里的 LLM_BASE_URL/LLM_MODEL 会兜底构造参数，先摘掉保证测试环境无关
        monkeypatch.delenv('LLM_BASE_URL', raising=False)
        monkeypatch.delenv('LLM_MODEL', raising=False)
        with raises(ValueError, match='base_url'):
            OpenAIClient(base_url='', model='m')
        with raises(ValueError, match='model'):
            OpenAIClient(base_url='http://x', model='')

class TestScriptedLLM:
    def test_responses_in_order(self):
        llm = ScriptedLLM(['第一次', '第二次'])
        assert llm.complete([Message('user', 'q1')]) == '第一次'
        assert llm.complete([Message('user', 'q2')]) == '第二次'

    def test_exhausted_raises(self):
        llm = ScriptedLLM(['唯一一条'])
        llm.complete([Message('user', 'q1')])
        with raises(LLMError, match='耗尽'):
            llm.complete([Message('user', 'q2')])

    def test_calls_snapshot(self):
        llm = ScriptedLLM(['a', 'b'])
        llm.complete([Message('user', 'q1')])
        llm.complete([Message('user', 'q2')])
        calls = llm.calls
        assert len(calls) == 2
        assert calls[0][0].content == 'q1'
        assert calls[1][0].content == 'q2'
        # 快照是副本，修改不影响内部记录
        calls[0][0].content = '被篡改'
        assert llm.calls[0][0].content == 'q1'

    def test_append_adds_response(self):
        llm = ScriptedLLM()
        llm.append('新输出')
        assert llm.complete([Message('user', 'x')]) == '新输出'
