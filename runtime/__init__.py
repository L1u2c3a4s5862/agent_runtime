"""agent-runtime 运行时包的公共入口，统一导出对外可用的核心组件。"""

from .agent import Agent, build_system_prompt
from .context import ContextManager, Message
from .errors import (
    AgentRuntimeError,
    LLMError,
    MaxIterationsExceeded,
    ParseError,
    SessionNotFoundError,
    ToolError,
    ToolNotFoundError,
    ToolValidationError
)
from ._tools import make_default_registry
from .llm import LLMClient, OpenAIClient
from .parser import FinalTurn, ToolTurn, parse_turn
from .session import Session, SessionManager
from .testing import ScriptedLLM
from .tools import Tool, ToolRegistry, validate_arguments
from .trace import TraceEvent, render_trace

__all__ = [
    'Agent',
    'AgentRuntimeError',
    'ContextManager',
    'FinalTurn',
    'LLMClient',
    'LLMError',
    'MaxIterationsExceeded',
    'Message',
    'OpenAIClient',
    'ParseError',
    'ScriptedLLM',
    'Session',
    'SessionManager',
    'SessionNotFoundError',
    'Tool',
    'ToolError',
    'ToolNotFoundError',
    'ToolRegistry',
    'ToolTurn',
    'ToolValidationError',
    'TraceEvent',
    'make_default_registry',
    'build_system_prompt',
    'parse_turn',
    'render_trace',
    'validate_arguments'
]
