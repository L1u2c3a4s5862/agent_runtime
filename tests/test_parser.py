"""协议解析器测试：Final Answer / ToolTurn 解析、容错与冲突处理。"""

from pytest import raises

from runtime.errors import ParseError
from runtime.parser import FinalTurn, ToolTurn, parse_turn

class TestFinalAnswer:
    """Final Answer 解析：带/不带 Thought、多行答复。"""

    def test_final_with_thought(self):
        text = 'Thought: 已有答案\nFinal Answer: 答案是 42'
        turn = parse_turn(text)
        assert turn == FinalTurn(answer='答案是 42')

    def test_final_without_thought(self):
        # Thought 非必需：直接 Final Answer 也应解析成功
        turn = parse_turn('Final Answer: 直接回答')
        assert turn == FinalTurn(answer='直接回答')

    def test_final_multiline_answer(self):
        # Final Answer 之后的续行并入答案
        text = 'Thought: 总结一下\nFinal Answer: 第一点\n第二点'
        turn = parse_turn(text)
        assert turn == FinalTurn(answer='第一点\n第二点')

class TestToolCall:
    """ToolTurn 解析：JSON 参数的各种形态。"""

    def test_action_with_single_line_json(self):
        text = 'Thought: 算一下\nAction: calculator\nAction Input: {"expression": "1+2"}'
        turn = parse_turn(text)
        assert turn == ToolTurn(
            thought='算一下', action='calculator',
            action_input={'expression': '1+2'}
        )

    def test_action_with_multiline_indented_json(self):
        # 多行缩进 JSON：Action Input 吞掉其后所有行
        text = (
            'Thought: 查天气\n'
            'Action: weather\n'
            'Action Input: {\n'
            '  "city": "北京",\n'
            '  "date": "2026-09-06"\n'
            '}'
        )
        turn = parse_turn(text)
        assert turn == ToolTurn(
            thought='查天气', action='weather',
            action_input={'city': '北京', 'date': '2026-09-06'}
        )

    def test_empty_action_input_becomes_empty_dict(self):
        # 模型偶尔输出裸 Action Input 标签：参数视为空对象
        text = 'Thought: 试试\nAction: search\nAction Input:'
        turn = parse_turn(text)
        assert turn.action_input == {}

    def test_json_wrapped_in_code_fence(self):
        # 唯一容错：剥掉 ```json ... ``` 围栏
        text = 'Thought: 试试\nAction: calculator\nAction Input: ```json\n{"expression": "2**10"}\n```'
        turn = parse_turn(text)
        assert turn.action_input == {'expression': '2**10'}

    def test_action_name_quotes_stripped(self):
        # 模型偶尔给工具名加引号，解析时剥掉
        text = 'Action: "calculator"\nAction Input: {"expression": "1+1"}'
        turn = parse_turn(text)
        assert turn.action == 'calculator'

    def test_multiline_json_containing_tag_like_lines_is_safe(self):
        # JSON 值里含"Final Answer: "字样不得被误判为协议标签
        text = (
            'Action: search\n'
            'Action Input: {\n'
            '  "query": "x",\n'
            '  "note": "Final Answer: 不是真的结束"\n'
            '}'
        )
        turn = parse_turn(text)
        assert turn.action_input == {'query': 'x', 'note': 'Final Answer: 不是真的结束'}

class TestConflictsAndErrors:
    """冲突优先级与各类解析错误。"""

    def test_action_and_final_conflict_prefers_final(self):
        # 同时出现 Action 和 Final Answer：以 Final Answer 为准
        text = 'Action: calculator\nAction Input: {"expression": "1+1"}\nFinal Answer: 不用算了，答案是 2'
        turn = parse_turn(text)
        assert isinstance(turn, FinalTurn)
        assert turn.answer == '不用算了，答案是 2'

    def test_no_tags_raises(self):
        with raises(ParseError, match='协议标记'):
            parse_turn('你好，今天天气不错。')

    def test_plain_text_without_tags_raises(self):
        with raises(ParseError):
            parse_turn('随便说点什么，没有任何格式')

    def test_action_without_action_input_raises(self):
        # 有 Action 缺 Action Input：报缺参数的明确错误供模型纠错
        with raises(ParseError, match='缺少 Action Input'):
            parse_turn('Thought: 要调用工具\nAction: calculator')

    def test_action_input_without_action_raises(self):
        # 只有 Action Input 没有 Action：无法确定工具名
        with raises(ParseError):
            parse_turn('Action Input: {"expression": "1+1"}')

    def test_broken_json_raises_with_excerpt(self):
        # 坏 JSON：错误信息带原文摘录，模型能看到自己写错了什么
        with raises(ParseError, match='不是合法 JSON'):
            parse_turn('Action: calculator\nAction Input: {"expression": "1+"')

    def test_non_object_json_raises(self):
        # 数组等非对象 JSON：工具参数必须是键值对
        with raises(ParseError, match='JSON 对象'):
            parse_turn('Action: calculator\nAction Input: [1, 2, 3]')
