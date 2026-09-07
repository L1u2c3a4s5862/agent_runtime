"""Message 模型与会话上下文管理（Memory）：回合切分、滑动窗口裁剪。"""

from dataclasses import dataclass
from typing import Literal

Role = Literal['system', 'user', 'assistant']

OBSERVATION_PREFIX = 'Observation: '
ERROR_PREFIX = 'Error: '
# 硬截断时每条最老消息保留的头部字符数
HARD_KEEP = 300

@dataclass
class Message:
    """一条对话消息。"""
    role: Role
    content: str

    def to_openai_dict(self) -> dict[str, str]:
        """转为 OpenAI 兼容消息字典。"""
        return {'role': self.role, 'content': self.content}

@dataclass(frozen=True)
class TrimStats:
    """最近一次 trim 的裁剪统计。"""
    dropped_turns: int
    truncated_messages: int
    dropped_chars: int

class ContextManager:
    """
    多轮对话历史与长度限制。

    回合边界约定：不以 Observation:/Error: 前缀开头的 user 消息是新回合的起点，
    其后到下一回合起点前的消息（含 assistant 原文与观察）都属于该回合。
    """

    def __init__(self, max_turns: int = 6, max_chars: int = 8000) -> None:
        # 上限必须为正，否则任何历史都会被立刻裁空
        if max_turns <= 0:
            raise ValueError(f'max_turns 必须为正数: {max_turns}')
        if max_chars <= 0:
            raise ValueError(f'max_chars 必须为正数: {max_chars}')
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._messages: list[Message] = []
        self._last_trim: TrimStats | None = None

    @property
    def messages(self) -> list[Message]:
        """已提交历史。"""
        # 返回副本：调用方改列表不会污染内部状态
        return list(self._messages)

    @property
    def trim_stats(self) -> TrimStats | None:
        """最近一次 trim 的统计。"""
        return self._last_trim

    def append(self, message: Message) -> None:
        """追加一条消息。"""
        self._messages.append(message)

    def trim(self) -> list[Message]:
        """返回裁剪后的历史视图，不改动已提交历史。"""
        dropped_turns = 0
        truncated = 0
        dropped_chars = 0
        # 第一层：整回合丢弃最老端，直到回合数达标
        turns = self._split_turns(self._messages)
        while len(turns) > self.max_turns:
            dropped_turns += 1
            dropped_chars += self._turn_chars(turns.pop(0))
        messages = self._flatten(turns)
        # 第二层：字符数仍超标则继续整回合丢，但至少保留最近一个回合（保底当前对话）
        while self._total_chars(messages) > self.max_chars and len(turns) > 1:
            dropped_turns += 1
            dropped_chars += self._turn_chars(turns.pop(0))
            messages = self._flatten(turns)
        # 第三层：只剩一个回合仍超长，才对最老消息逐条硬截断
        if self._total_chars(messages) > self.max_chars:
            lost, truncated = self._hard_truncate(messages)
            dropped_chars += lost
        self._last_trim = TrimStats(dropped_turns, truncated, dropped_chars)\
            if (dropped_turns or truncated) else None
        return messages

    def _hard_truncate(self, messages: list[Message]) -> tuple[int, int]:
        """逐条截断最老消息（保留头部 HARD_KEEP 字符）直到总长达标。"""
        dropped = 0
        truncated = 0
        for index, message in enumerate(messages):
            if self._total_chars(messages) <= self.max_chars:
                break
            excess = len(message.content) - HARD_KEEP
            if excess <= 0:
                continue
            kept = message.content[:HARD_KEEP]
            # 截断处打标记：模型能看到信息被裁掉了，而不是默默消失
            messages[index] = Message(message.role, f'{kept}…[已截断 {excess} 字]')
            dropped += excess
            truncated += 1
        return dropped, truncated

    @staticmethod
    def _is_turn_start(message: Message) -> bool:
        """判断是否新回合起点：真实的 user 提问，而非 Observation/Error 观察。"""
        return message.role == 'user' and not (
            message.content.startswith(OBSERVATION_PREFIX)
            or message.content.startswith(ERROR_PREFIX)
        )

    def _split_turns(self, messages: list[Message]) -> list[list[Message]]:
        """按回合起点切分消息。"""
        turns: list[list[Message]] = []
        for message in messages:
            if self._is_turn_start(message):
                turns.append([message])
            elif turns:
                turns[-1].append(message)
            else:
                # 历史以观察开头（异常残留）：单开一个回合兜底，不丢消息
                turns.append([message])
        return turns

    @staticmethod
    def _turn_chars(turn: list[Message]) -> int:
        return sum(len(message.content) for message in turn)

    @staticmethod
    def _flatten(turns: list[list[Message]]) -> list[Message]:
        return [message for turn in turns for message in turn]

    @staticmethod
    def _total_chars(messages: list[Message]) -> int:
        return sum(len(message.content) for message in messages)
