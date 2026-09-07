from json import dumps
from datetime import date, datetime
from typing import Optional

from loguru import logger

from .context import ERROR_PREFIX, OBSERVATION_PREFIX, Message
from .errors import (
    LLMError, MaxIterationsExceeded,
    ParseError, ToolNotFoundError, ToolValidationError
)
from ._tools import make_default_registry
from .llm import LLMClient
from .logging import setup_log_file
from .parser import FinalTurn, ToolTurn, parse_turn
from .session import Session
from .tools import Tool, ToolRegistry, validate_arguments
from .trace import record_event

setup_log_file('agent.log')

OBSERVATION_LIMIT = 2000
TRACE_PAYLOAD_LIMIT = 500

_SYSTEM_TEMPLATE = """\
你是一个通过调用工具逐步解决用户问题的智能 Agent。当前日期：{today}。

# 可用工具
{tool_descriptors}

你只能使用上面列出的工具，不得虚构不存在的工具或参数。

# 工作方式
你按"思考 → 调用工具 → 观察结果 → 再思考"的循环工作：
- 每轮只输出以下两种格式之一，不得混用，不得输出格式以外的任何内容。
- 输出 Action 后，系统会执行该工具，并以 "Observation: 结果" 的形式把返回内容交给你；你据此继续下一轮，直到可以给出最终答案。
- 不得编造工具返回结果；工具结果不足以回答时，继续调用工具，不要猜测。

# 输出格式

## 格式一：需要调用工具时
Thought: 一句话说明当前判断和调用该工具的目的
Action: 工具名（必须严格来自上方"可用工具"列表）
Action Input: {{"参数名": "参数值"}}

要求：
- 每轮只调用一个工具。
- Action Input 必须是一个合法的 JSON 对象，且该行只包含此 JSON 对象，无其他字符。
- 参数键必须严格来自对应工具的说明，不得自创。

## 格式二：已有足够信息回答时
Thought: 一句话说明已掌握足够信息
Final Answer: 用与用户相同的语言，给出准确、完整的最终答复，直接回答问题，不复述思考过程。\
"""

def build_system_prompt(tool_descriptors: str, today: str) -> str:
    """
    拼装 system prompt（身份 + 日期 + 协议 + 工具清单）。

    Args:
        `tool_descriptors`: ToolRegistry.to_prompt_descriptors() 的输出。
        `today`: 今天的日期（ISO 格式），注入后「明天/后天」类追问对模型可计算。

    Returns:
        system prompt 全文。
    """
    return _SYSTEM_TEMPLATE.format(today=today, tool_descriptors=tool_descriptors)

class Agent:
    """ReAct 循环执行器。实例不含会话状态，多 Session 隔离靠 run 显式传 Session。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: Optional[ToolRegistry]=None,
        max_steps: int=8,
        parse_retries: int=2
    ):
        if max_steps <= 0:
            raise ValueError(f'max_steps 必须为正数: {max_steps}')
        if parse_retries < 0:
            raise ValueError(f'parse_retries 不能为负: {parse_retries}')
        self.llm = llm
        self.tools = tools if tools is not None else make_default_registry()
        self.max_steps = max_steps
        self.parse_retries = parse_retries

    @property
    def _llm_name(self) -> str:
        return getattr(self.llm, 'model', type(self.llm).__name__)

    def run(self, session: Session, user_input: str) -> str:
        """
        在指定 session 内跑完一个用户回合。

        Args:
            `session`: 目标会话（历史与轨迹都写入该会话）。
            `user_input`: 用户本轮输入。

        Returns:
            最终答案文本。
        """
        started = datetime.now()
        logger.debug(f'[session={session.id}] 用户输入: {user_input!r}')
        buffer = [Message('user', user_input)]
        steps = 0
        parse_failures = 0
        while steps < self.max_steps:
            steps += 1
            text = self._call_llm(session, buffer)
            turn = self._parse_with_retry(buffer, text, parse_failures)
            if turn is None:
                parse_failures += 1
                continue
            buffer.append(Message('assistant', text))
            if isinstance(turn, FinalTurn):
                self._commit(session, buffer)
                logger.debug(f'[session={session.id}] Agent 回复: {turn.answer!r}')
                logger.info(
                    f'[session={session.id}] 回合结束，共 {steps} 步，'
                    f'耗时 {(datetime.now() - started).total_seconds():.2f}s'
                )
                return turn.answer
            self._run_tool_step(session, buffer, turn)
        logger.error(f'[session={session.id}] 超过最大步数 {self.max_steps}，回合中止')
        raise MaxIterationsExceeded(f'超过最大步数 {self.max_steps}，仍未产出最终答案')

    def _parse_with_retry(self, buffer: list[Message], text: str, parse_failures: int) -> Optional[FinalTurn | ToolTurn]:
        """解析输出；失败且未超重试上限时把错误反馈压入 buffer。"""
        try:
            return parse_turn(text)
        except ParseError as e:
            logger.debug(f'LLM 输出解析失败（第 {parse_failures + 1} 次）: {e}')
            if parse_failures >= self.parse_retries:
                raise ParseError(f'连续 {parse_failures + 1} 次输出无法解析: {e}') from e
            buffer.append(Message('user', f'{ERROR_PREFIX}{e}'))
            return None

    def _call_llm(self, session: Session, buffer: list[Message]) -> str:
        """拼请求（system + 裁剪历史 + 回合 buffer）并调用 LLM。"""
        prompt = build_system_prompt(self.tools.to_prompt_descriptors(), date.today().isoformat())
        messages = [Message('system', prompt)] + session.history.trim() + buffer
        started = datetime.now()
        try:
            text = self.llm.complete(messages)
        except LLMError as e:
            record_event(
                session, 'llm', self._llm_name, str(e),
                started_at=started, ok=False, error=str(e)
            )
            logger.error(
                f'[session={session.id}] LLM 调用失败: 模型={self._llm_name!r} '
                f'消息数={len(messages)} 耗时={(datetime.now() - started).total_seconds():.2f}s: {e}'
            )
            raise
        logger.debug(
            f'[session={session.id}] LLM 调用: 模型={self._llm_name!r} 消息数={len(messages)} '
            f'耗时={(datetime.now() - started).total_seconds():.2f}s 返回 {len(text)} 字'
        )
        record_event(session, 'llm', self._llm_name, text, started_at=started)
        return text

    def _resolve_tool(self, buffer: list[Message], turn: ToolTurn) -> Optional[tuple[Tool, dict[str, object]]]:
        """解析工具与参数；失败时把错误反馈压入回合 buffer。"""
        try:
            tool = self.tools.get(turn.action)
        except ToolNotFoundError as exc:
            available = ', '.join(item.name for item in self.tools.list())
            logger.debug(f'未知工具被调用: {turn.action}（可用: {available}）')
            buffer.append(Message('user', f'{ERROR_PREFIX}{exc}。可用工具: {available}'))
            return None
        try:
            args = validate_arguments(tool.parameters, turn.action_input)
        except ToolValidationError as exc:
            logger.debug(f'工具 {tool.name} 参数非法: {exc}')
            buffer.append(Message('user', f'{ERROR_PREFIX}{exc}。请修正参数后重新调用'))
            return None
        return tool, args

    def _run_tool_step(self, session: Session, buffer: list[Message], turn: ToolTurn):
        """执行一次工具调用，把观察或错误反馈压入回合 buffer。"""
        resolved = self._resolve_tool(buffer, turn)
        if resolved is None:
            return
        tool, args = resolved
        started = datetime.now()
        try:
            result = tool.func(**args)
        except Exception as e:
            record_event(
                session, 'tool', tool.name, dumps(args, ensure_ascii=False),
                started_at=started, ok=False, error=f'{type(e).__name__}: {e}'
            )
            logger.error(
                f'[session={session.id}] 工具 {tool.name} 执行异常: '
                f'参数={dumps(args, ensure_ascii=False)} {type(e).__name__}: {e}'
            )
            buffer.append(Message('user', f'{ERROR_PREFIX}工具 {tool.name} 执行异常: {e}'))
            return
        payload = dumps(args, ensure_ascii=False) + ' -> ' + self._format_result(result)[:TRACE_PAYLOAD_LIMIT]
        record_event(session, 'tool', tool.name, payload, started_at=started)
        buffer.append(Message('user', f'{OBSERVATION_PREFIX}{self._format_result(result)}'))
        logger.info(
            f'[session={session.id}] 工具 {tool.name} 执行成功: '
            f'参数={dumps(args, ensure_ascii=False)} 结果={self._format_result(result)[:200]!r} '
            f'耗时={(datetime.now() - started).total_seconds():.2f}s'
        )

    @staticmethod
    def _format_result(result: object) -> str:
        """工具结果转文本，单条观察上限 OBSERVATION_LIMIT 字符。"""
        if isinstance(result, str):
            text = result
        elif isinstance(result, (dict, list)):
            text = dumps(result, ensure_ascii=False)
        else:
            text = str(result)
        if len(text) > OBSERVATION_LIMIT:
            text = text[:OBSERVATION_LIMIT] + '…[观察已截断]'
        return text

    @staticmethod
    def _commit(session: Session, buffer: list[Message]):
        """整回合一次性提交进历史（回合原子性的唯一入口）。"""
        for message in buffer:
            session.history.append(message)
