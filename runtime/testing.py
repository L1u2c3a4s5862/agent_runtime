"""测试替身：脚本化 Fake LLM，按预设顺序返回回复，供离线测试与演示。"""

from copy import deepcopy
from typing import Optional

from .context import Message
from .errors import LLMError
from .llm import LLMClient

class ScriptedLLM(LLMClient):
    """脚本化 Fake LLM：按预设顺序返回回复，耗尽抛 LLMError。"""

    def __init__(self, responses: Optional[list[str]]=None) -> None:
        self._responses: list[str] = list(responses) if responses else []
        self._calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        """弹出下一条预设回复；耗尽抛 LLMError。"""
        # 记录时深拷贝：外部后续改 messages 不影响已存快照
        self._calls.append(deepcopy(messages))
        if not self._responses:
            raise LLMError('脚本回复已耗尽')
        return self._responses.pop(0)

    def append(self, text: str) -> None:
        """追加一条预设回复。"""
        self._responses.append(text)

    @property
    def calls(self) -> list[list[Message]]:
        """历次 complete 调用的消息快照（深拷贝，防外部篡改）。"""
        return deepcopy(self._calls)
