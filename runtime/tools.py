"""工具系统：Tool 数据模型、参数校验与注册表。"""

import json
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from .errors import ToolNotFoundError, ToolValidationError
from .logging import setup_log_file

setup_log_file('agent.log')

# JSON Schema 的 type 到 Python 类型检查的映射，只支持常用标量与容器
_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    'string': lambda value: isinstance(value, str),
    # bool 是 int 的子类，必须先排除，否则 True 会被当成合法 integer
    'integer': lambda value: isinstance(value, int) and not isinstance(value, bool),
    'number': lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    'boolean': lambda value: isinstance(value, bool),
    'array': lambda value: isinstance(value, list),
    'object': lambda value: isinstance(value, dict)
}

@dataclass(frozen=True)
class Tool:
    """一个可被 Agent 调用的工具。"""
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., object]

def _check_required(required: list[str], args: dict[str, Any]) -> list[str]:
    """校验必填字段。"""
    return [f'缺少必需参数 {key}' for key in required if key not in args]

def _check_typed(properties: dict[str, Any], args: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    """按 type + enum 过滤参数，未知键剔除。"""
    cleaned: dict[str, Any] = {}
    for key, value in args.items():
        spec = properties.get(key)
        # 未知键静默丢弃：模型偶尔多传参数，直接忽略比报错更宽容
        if spec is None:
            continue
        spec_type = spec.get('type')
        if spec_type and not _TYPE_CHECKS.get(spec_type, lambda _: True)(value):
            errors.append(f'参数 {key} 应为 {spec_type} 类型，实际为 {type(value).__name__}')
            continue
        enum = spec.get('enum')
        if enum is not None and value not in enum:
            errors.append(f'参数 {key} 取值必须在 {enum} 中')
            continue
        cleaned[key] = value
    return cleaned

def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """按 required + type + enum 校验工具参数。"""
    properties = schema.get('properties', {})
    errors = _check_required(schema.get('required', []), args)
    cleaned = _check_typed(properties, args, errors)
    if errors:
        raise ToolValidationError('；'.join(errors))
    return cleaned

class ToolRegistry:
    """工具注册表：注册、查找与 system prompt 渲染。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具，同名覆盖并记 warning。"""
        if tool.name in self._tools:
            logger.warning(f'工具 {tool.name} 已存在，将被覆盖')
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """按名取工具。"""
        if name not in self._tools:
            raise ToolNotFoundError(f'未知工具: {name}')
        return self._tools[name]

    def unregister(self, name: str) -> None:
        """移除工具。"""
        if name not in self._tools:
            raise ToolNotFoundError(f'未知工具: {name}')
        del self._tools[name]

    def list(self) -> list[Tool]:
        """按注册顺序返回全部工具。"""
        return list(self._tools.values())

    def to_prompt_descriptors(self) -> str:
        """渲染工具清单，供拼入 system prompt。"""
        lines = []
        for tool in self.list():
            # 每个参数只保留描述或类型，避免把整份 JSON Schema 塞进 prompt
            param_desc = {
                key: spec.get('description') or spec.get('type', '任意')
                for key, spec in tool.parameters.get('properties', {}).items()
            }
            params = json.dumps(param_desc, ensure_ascii=False)
            lines.append(f'- {tool.name}: {tool.description}。参数: {params}')
        return '\n'.join(lines)
