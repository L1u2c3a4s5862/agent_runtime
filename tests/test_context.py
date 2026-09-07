from pytest import raises

from runtime.context import ContextManager, Message

def build_turn(question: str, answer: str, observation: str='') -> list[Message]:
    """构造一个完整回合：user 提问 + assistant 答复（可带 Observation）。"""
    turn = [Message('user', question), Message('assistant', answer)]
    if observation:
        turn.append(Message('user', f'Observation: {observation}'))
    return turn

class TestAppendAndView:
    def test_messages_returns_copy(self):
        ctx = ContextManager()
        ctx.append(Message('user', '你好'))
        view = ctx.messages
        view.clear()
        assert len(ctx.messages) == 1

    def test_trim_within_limit_returns_original(self):
        ctx = ContextManager(max_turns=5, max_chars=1000)
        ctx.append(Message('user', '问题一'))
        ctx.append(Message('assistant', '答案一'))
        trimmed = ctx.trim()
        assert [m.content for m in trimmed] == ['问题一', '答案一']
        assert ctx.trim_stats is None

class TestTurnWindow:
    def test_oldest_turn_dropped_when_over_max_turns(self):
        ctx = ContextManager(max_turns=2)
        for index in range(4):
            for message in build_turn(f'问题{index}', f'答案{index}'):
                ctx.append(message)
        trimmed = ctx.trim()
        contents = [m.content for m in trimmed]
        assert '问题0' not in contents and '答案0' not in contents
        assert '问题1' not in contents
        assert contents[0] == '问题2'
        assert contents[-1] == '答案3'
        assert ctx.trim_stats is not None
        assert ctx.trim_stats.dropped_turns == 2

    def test_observation_does_not_start_new_turn(self):
        ctx = ContextManager(max_turns=2)
        ctx.append(Message('user', '问题0'))
        ctx.append(Message('assistant', 'Action: calculator\nAction Input: {"expression": "1+1"}'))
        ctx.append(Message('user', 'Observation: 2'))
        ctx.append(Message('assistant', 'Final Answer: 2'))
        for message in build_turn('问题1', '答案1'):
            ctx.append(message)
        trimmed = ctx.trim()
        contents = [m.content for m in trimmed]
        assert '问题0' in contents  # 只算 2 个回合，未触发丢弃
        assert trimmed[0].content == '问题0'

    def test_history_not_mutated_by_trim(self):
        ctx = ContextManager(max_turns=1)
        for message in build_turn('问题0', '答案0'):
            ctx.append(message)
        for message in build_turn('问题1', '答案1'):
            ctx.append(message)
        ctx.trim()
        assert len(ctx.messages) == 4  # 完整历史仍在，trim 只是视图


class TestCharBudget:
    def test_whole_turn_dropped_when_over_max_chars(self):
        ctx = ContextManager(max_turns=10, max_chars=30)
        ctx.append(Message('user', '很长的问题零' * 5))
        ctx.append(Message('assistant', '很长的答案零' * 5))
        for message in build_turn('问一', '答一'):
            ctx.append(message)
        trimmed = ctx.trim()
        contents = [m.content for m in trimmed]
        assert contents == ['问一', '答一']
        assert ctx.trim_stats is not None
        assert ctx.trim_stats.dropped_turns == 1

    def test_hard_truncate_when_single_turn_exceeds_budget(self):
        ctx = ContextManager(max_turns=10, max_chars=50)
        long_text = '超长文本' * 100  # 400 字符，单回合自身就超预算
        ctx.append(Message('user', long_text))
        ctx.append(Message('assistant', '答'))
        trimmed = ctx.trim()
        assert '…[已截断' in trimmed[0].content
        assert ctx.trim_stats is not None
        assert ctx.trim_stats.truncated_messages == 1
        assert ctx.trim_stats.dropped_chars > 0

    def test_invalid_limits_raise(self):
        with raises(ValueError):
            ContextManager(max_turns=0)
        with raises(ValueError):
            ContextManager(max_chars=0)
