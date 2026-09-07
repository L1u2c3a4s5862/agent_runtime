"""Agent 主循环的端到端测试：全部用 ScriptedLLM 脚本驱动，零 API 依赖。"""

from pytest import raises

from runtime.agent import Agent
from runtime._tools import make_calculator_tool
from runtime.errors import LLMError, MaxIterationsExceeded, ParseError
from runtime.session import Session, SessionManager
from runtime.testing import ScriptedLLM
from runtime.tools import Tool, ToolRegistry

def join(contents: list[str]) -> str:
    """把消息内容拼成一段文本，便于子串断言。"""
    return ' '.join(contents)

# 测试内嵌的离线假数据：Agent 循环测试关注的是编排逻辑，不是真实工具
_FAKE_PAGES = {
    '北京': '北京是中华人民共和国的首都，位于华北平原北部。',
    'Python': 'Python 是一种解释型、面向对象的高级编程语言。'
}

_FAKE_WEATHER = {
    ('北京', '2026-09-06'): '晴，19~27℃',
    ('北京', '2026-09-07'): '多云转晴，20~28℃'
}

def _fake_search(query: str) -> str:
    """在假页面字典中做子串匹配。"""
    for title, content in _FAKE_PAGES.items():
        if query.lower() in content.lower():
            return f'{title}: {content}'
    return f'未找到与「{query}」相关的条目'

def _fake_weather(city: str, date: str) -> str:
    """查询假天气表。"""
    return _FAKE_WEATHER.get((city, date), f'未收录 {city} 在 {date} 的天气数据')

def make_fake_registry() -> ToolRegistry:
    """组装假数据工具注册表（calculator 真实现 + 离线假 search/weather）。"""
    registry = ToolRegistry()
    registry.register(make_calculator_tool())
    registry.register(Tool(
        name='search',
        description='按关键词检索本地知识库条目',
        parameters={
            'type': 'object',
            'properties': {'query': {'type': 'string', 'description': '检索关键词'}},
            'required': ['query']
        },
        func=_fake_search
    ))
    # weather 的 date 保持必填：test_invalid_args_self_corrects 依赖"缺参数"错误反馈
    registry.register(Tool(
        name='weather',
        description='查询指定城市在指定日期的天气',
        parameters={
            'type': 'object',
            'properties': {
                'city': {'type': 'string', 'description': '城市名，如 北京'},
                'date': {'type': 'string', 'description': '日期，格式 YYYY-MM-DD'}
            },
            'required': ['city', 'date']
        },
        func=_fake_weather
    ))
    return registry

def make_agent(responses: list[str], **kwargs) -> Agent:
    """用脚本构造 Agent，默认挂假数据工具（零 API 依赖）。"""
    # setdefault：需要自定义注册表的测试（如 bomb）可显式传 tools 覆盖
    kwargs.setdefault('tools', make_fake_registry())
    return Agent(ScriptedLLM(responses), **kwargs)

def make_bomb_registry() -> ToolRegistry:
    """构造含一个必抛异常工具的注册表。"""
    def boom():
        raise ValueError('内部爆炸')

    registry = ToolRegistry()
    registry.register(Tool(
        name='bomb',
        description='测试用爆炸工具',
        parameters={'type': 'object', 'properties': {}},
        func=boom
    ))
    return registry

class TestBasicLoop:
    """单回合基本流程：工具调用、直接答复、原始文本入历史。"""

    def test_single_tool_round(self):
        # 第一条脚本输出调工具，第二条给最终答案
        agent = make_agent([
            'Thought: 算一下\nAction: calculator\nAction Input: {"expression": "1+2"}',
            'Thought: 有答案了\nFinal Answer: 结果是 3'
        ])
        session = Session()
        answer = agent.run(session, '帮我算 1+2')
        assert answer == '结果是 3'
        # 第二次 LLM 请求带上了工具观察
        second_call = agent.llm.calls[1]
        assert any(
            m.content.startswith('Observation:')
            and '3' in m.content for m in second_call
        )
        # 历史形状：user → assistant → user(观察) → assistant(final)
        assert [m.role for m in session.history.messages] == ['user', 'assistant', 'user', 'assistant']

    def test_final_direct_without_tool(self):
        # 模型一轮直接给出答案：跳过工具步骤
        agent = make_agent(['Final Answer: 你好！'])
        session = Session()
        assert agent.run(session, '你好') == '你好！'
        assert len(session.history.messages) == 2

    def test_observation_fed_back_with_original_react_text(self):
        react = 'Thought: 查询\nAction: search\nAction Input: {"query": "Python"}'
        agent = make_agent([react, 'Final Answer: Python 是一种语言。'])
        session = Session()
        agent.run(session, '什么是 Python')
        history = session.history.messages
        assert history[1].content == react  # assistant 原始 ReAct 文本无损入历史

class TestFollowUps:
    """多轮回合：追问依赖历史与观察。"""

    def test_pure_conversation_followup(self):
        agent = make_agent([
            'Final Answer: 北京今天晴，19~27℃。',
            'Final Answer: 明天多云，适合出门散步。'
        ])
        session = Session()
        agent.run(session, '北京今天天气怎么样？')
        agent.run(session, '然后呢？')
        # 第二问的请求带上了第一问的问答历史
        second_call = agent.llm.calls[1]
        text = join(m.content for m in second_call)
        assert '北京今天天气怎么样' in text
        assert '北京今天晴' in text

    def test_tool_followup_uses_history(self):
        agent = make_agent([
            'Thought: 查今天\nAction: weather\nAction Input: {"city": "北京", "date": "2026-09-06"}',
            'Final Answer: 北京 2026-09-06 晴，19~27℃。',
            'Thought: 查明天\nAction: weather\nAction Input: {"city": "北京", "date": "2026-09-07"}',
            'Final Answer: 2026-09-07 多云转晴，20~28℃。'
        ])
        session = Session()
        agent.run(session, '北京 2026-09-06 天气怎么样？')
        answer = agent.run(session, '那明天呢？')
        assert '2026-09-07' in answer
        # 第二问的首次请求含首轮观察与城市（凭历史而非新输入取参）
        third_call = agent.llm.calls[2]
        text = join(m.content for m in third_call)
        assert 'Observation' in text
        assert '北京' in text
        assert '2026-09-06' in text

class TestSessionIsolation:
    """多 Session 并存时历史互不串扰。"""

    def test_two_sessions_do_not_leak(self):
        llm = ScriptedLLM([
            'Final Answer: A 的答案 1',
            'Final Answer: B 的答案 1',
            'Final Answer: A 的答案 2'
        ])
        # 注意：这里不带 tools，纯对话不触发工具，无需假数据注册表
        agent = Agent(llm)
        sm = SessionManager()
        session_a = sm.create(label='A')
        session_b = sm.create(label='B')
        agent.run(session_a, 'A 第一问')
        agent.run(session_b, 'B 第一问')
        answer = agent.run(session_a, 'A 第二问')
        assert answer == 'A 的答案 2'
        # A 的第二次请求含 A 首问，不含 B 任何消息
        text = join(m.content for m in llm.calls[2])
        assert 'A 第一问' in text
        assert 'B 第一问' not in text
        assert 'B 的答案' not in text

    def test_switch_restores_history(self):
        llm = ScriptedLLM([
            'Final Answer: 天气答案',
            'Final Answer: 计算答案',
            'Final Answer: 追问答案'
        ])
        agent = Agent(llm)
        sm = SessionManager()
        session_a = sm.create(label='窗口一')
        session_b = sm.create(label='窗口二')
        agent.run(session_a, '查天气')
        agent.run(session_b, '算 2*3')
        sm.switch(session_a.id)
        agent.run(session_a, '然后呢？')
        # 切回 A 后第三次请求含 A 首问历史
        text = join(m.content for m in llm.calls[2])
        assert '查天气' in text
        assert '算 2*3' not in text

class TestContextCompression:
    """端到端验证滑动窗口裁剪。"""

    def test_oldest_turn_dropped_end_to_end(self):
        agent = make_agent([f'Final Answer: 答案{i}' for i in range(4)])
        session = Session(max_turns=2)
        for index in range(4):
            agent.run(session, f'第{index}问')
        # 第 4 次请求：首问已滑出窗口，末问仍在
        text = join(m.content for m in agent.llm.calls[3])
        assert '第0问' not in text
        assert '第3问' in text
        assert session.history.trim_stats is not None
        assert session.history.trim_stats.dropped_turns >= 1

    def test_history_kept_intact_after_trim(self):
        agent = make_agent([f'Final Answer: 答案{i}' for i in range(4)])
        session = Session(max_turns=2)
        for index in range(4):
            agent.run(session, f'第{index}问')
        # 完整历史仍在（trim 只是请求视图），每回合 2 条
        assert len(session.history.messages) == 8

class TestErrorHandling:
    """异常路径：错误反馈喂回、熔断、回合原子性。"""

    def test_unknown_tool_self_corrects(self):
        agent = make_agent([
            'Thought: 试试\nAction: nosuch\nAction Input: {}',
            'Thought: 换正确的\nAction: calculator\nAction Input: {"expression": "1+1"}',
            'Final Answer: 答案是 2'
        ])
        session = Session()
        answer = agent.run(session, '算一下')
        assert answer == '答案是 2'
        # 第二次请求含未知工具错误反馈
        text = join(m.content for m in agent.llm.calls[1])
        assert '未知工具' in text

    def test_invalid_args_self_corrects(self):
        # 第一次调用缺 date 参数，被校验拒绝后模型补参数重试
        agent = make_agent([
            'Thought: 查\nAction: weather\nAction Input: {"city": "北京"}',
            'Thought: 补参数\nAction: weather\nAction Input: {"city": "北京", "date": "2026-09-06"}',
            'Final Answer: 查到了'
        ])
        session = Session()
        answer = agent.run(session, '查天气')
        assert answer == '查到了'
        text = join(m.content for m in agent.llm.calls[1])
        assert '缺少必需参数' in text

    def test_tool_exception_fed_back_then_final(self):
        llm = ScriptedLLM([
            'Thought: 引爆\nAction: bomb\nAction Input: {}',
            'Final Answer: 已处理异常，抱歉。'
        ])
        agent = Agent(llm, tools=make_bomb_registry())
        session = Session()
        answer = agent.run(session, '引爆一下')
        assert answer == '已处理异常，抱歉。'
        # 工具异常转成 Error 观察喂回，模型给出兜底答复
        text = join(m.content for m in llm.calls[1])
        assert '执行异常' in text

    def test_parse_failures_exhausted_raises(self):
        # 3 条全部无法解析：默认 parse_retries=2，连续失败超限抛 ParseError
        agent = make_agent(['胡言乱语没有格式'] * 3)
        session = Session()
        with raises(ParseError, match='无法解析'):
            agent.run(session, 'hi')
        assert session.history.messages == []  # 回合原子性：历史未污染

    def test_max_iterations_exceeded(self):
        # max_steps=3 但脚本给 8 轮工具调用：步数耗尽抛熔断异常
        agent = make_agent(
            ['Thought: 再来\nAction: calculator\nAction Input: {"expression": "1+1"}'] * 8, max_steps=3
        )
        session = Session()
        with raises(MaxIterationsExceeded, match='最大步数'):
            agent.run(session, '死循环')
        assert session.history.messages == []

    def test_llm_exhausted_raises(self):
        # 空脚本：第一次 LLM 调用就耗尽
        agent = make_agent([])
        session = Session()
        with raises(LLMError, match='耗尽'):
            agent.run(session, 'hi')
        assert session.history.messages == []

    def test_parse_error_feedback_guides_retry(self):
        # 第一次输出乱码，重试请求带协议反馈后模型改正
        agent = make_agent([
            '乱七八糟',
            'Final Answer: 这次对了'
        ])
        session = Session()
        answer = agent.run(session, '再来一次')
        assert answer == '这次对了'
        # 重试请求中带上了错误反馈
        text = join(m.content for m in agent.llm.calls[1])
        assert '协议' in text

class TestTrace:
    """轨迹事件：成功与失败的记录。"""

    def test_happy_path_trace_events(self):
        agent = make_agent([
            'Thought: 算\nAction: calculator\nAction Input: {"expression": "2*21"}',
            'Final Answer: 答案是 42'
        ])
        session = Session()
        agent.run(session, '2*21 是多少')
        # 轨迹顺序：llm → tool → llm（两次模型调用夹一次工具调用）
        kinds = [event.kind for event in session.traces]
        assert kinds == ['llm', 'tool', 'llm']
        tool_event = session.traces[1]
        assert tool_event.ok is True
        assert '2*21' in tool_event.payload
        assert '42' in tool_event.payload

    def test_failed_tool_trace_marked(self):
        agent = Agent(ScriptedLLM([
            'Action: bomb\nAction Input: {}',
            'Final Answer: 兜底'
        ]), tools=make_bomb_registry())
        session = Session()
        agent.run(session, '引爆')
        failed = [
            event for event in session.traces if
            event.kind == 'tool' and not event.ok
        ]
        assert len(failed) == 1
        assert 'ValueError' in failed[0].error
