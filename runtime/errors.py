"""异常体系：全部自定义异常的根类型与分层定义。"""

class AgentRuntimeError(Exception):
    """所有自定义异常的根类型。"""

class LLMError(AgentRuntimeError):
    """LLM API 调用失败：网络/HTTP 错误、空输出、脚本耗尽。"""

class ParseError(AgentRuntimeError):
    """
    LLM 输出文本无法解析。

    message 会被原样反馈给 LLM 用于纠错，措辞需面向模型可理解。
    """

class ToolError(AgentRuntimeError):
    """工具层异常的根类型。"""

class ToolNotFoundError(ToolError):
    """LLM 调用了未注册的工具。"""

class ToolValidationError(ToolError):
    """工具参数不满足 JSON Schema 约束。"""

class MaxIterationsExceeded(AgentRuntimeError):
    """超过 max_steps 仍未产出最终答案，循环熔断。"""

class SessionNotFoundError(AgentRuntimeError):
    """SessionManager 中找不到指定 session。"""
