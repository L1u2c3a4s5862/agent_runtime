import re
from json import JSONDecodeError, loads
from dataclasses import dataclass
from typing import Union

from loguru import logger

from .errors import ParseError
from .logging import setup_log_file

setup_log_file('agent.log')

_TAG_RE = re.compile(r'^\s*(Thought|Action|Action Input|Final Answer)\s*:')

@dataclass(frozen=True)
class FinalTurn:
    """LLM 的最终答复。"""
    answer: str

@dataclass(frozen=True)
class ToolTurn:
    """LLM 请求调用工具。"""
    thought: str
    action: str
    action_input: dict[str, object]

Turn = Union[FinalTurn, ToolTurn]

def _parse_sections(text: str) -> dict[str, str]:
    """按标签把文本切成各段，Action Input 吞掉其后所有剩余行。"""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = _TAG_RE.match(line)
        # Action Input 吞尾：非本标签的行一律归它（多行 JSON 安全），其他标签则切走
        if current == 'Action Input' and (match is None or match.group(1) == 'Action Input'):
            sections[current].append(line)
            continue
        if match is None:
            if current is not None:
                sections[current].append(line)
            continue
        tag = match.group(1)
        sections.setdefault(tag, [])
        rest = line[match.end():].strip()
        if rest:
            sections[tag].append(rest)
        current = tag
    return {key: '\n'.join(value).strip() for key, value in sections.items()}

def _parse_json(raw: str) -> dict[str, object]:
    """解析 Action Input 的 JSON 对象，剥离 code fence 作为唯一容错。"""
    stripped = raw.strip()
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', stripped, re.S)
    if fence is not None:
        stripped = fence.group(1).strip()
    try:
        value = loads(stripped)
    except JSONDecodeError as exc:
        hint = stripped[:80].replace('\n', ' ')
        raise ParseError(f'Action Input 不是合法 JSON（{exc.msg}），原文摘录: {hint}。请重新输出完整协议格式') from exc
    if not isinstance(value, dict):
        raise ParseError(f'Action Input 必须是 JSON 对象，实际为 {type(value).__name__}')
    return value

def _build_tool_turn(sections: dict[str, str]) -> ToolTurn:
    """由已切好的段构造 ToolTurn，Action Input 为空时参数视为 {}。"""
    action = sections['Action'].strip().splitlines()[0].strip('"\'')
    raw = sections['Action Input']
    args: dict[str, object] = {} if raw == '' else _parse_json(raw)
    return ToolTurn(thought=sections.get('Thought', ''), action=action, action_input=args)

def parse_turn(text: str) -> Turn:
    """解析单条 ReAct 输出为 FinalTurn 或 ToolTurn。"""
    sections = _parse_sections(text)
    if not sections:
        raise ParseError('输出中没有识别到任何协议标记（Thought/Action/Final Answer），请按协议格式重新输出')
    if 'Final Answer' in sections:
        if 'Action' in sections:
            logger.warning('输出同时包含 Action 与 Final Answer，按 Final Answer 处理')
        return FinalTurn(answer=sections['Final Answer'])
    if 'Action' in sections:
        if 'Action Input' not in sections:
            raise ParseError('输出中有 Action 但缺少 Action Input，请补充「Action Input: <JSON 参数>」后重新输出')
        return _build_tool_turn(sections)
    raise ParseError('输出中没有 Action 或 Final Answer，无法确定下一步，请按协议格式重新输出')
