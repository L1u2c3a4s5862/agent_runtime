from ast import (
    Add, UAdd, Sub, USub, Mult,
    Div, FloorDiv, Pow, Mod,
    AST, Expression, Constant,
    BinOp, UnaryOp, Name,
    parse, unparse
)
from operator import (
    add, sub, mul, truediv,
    floordiv, pow, mod, pos, neg
)
from math import pi, e

from ..errors import ToolValidationError
from ..tools import Tool

_BIN_OPS = {
    Add: add,
    Sub: sub,
    Mult: mul,
    Div: truediv,
    FloorDiv: floordiv,
    Pow: pow,
    Mod: mod
}
_UNARY_OPS = {UAdd: pos, USub: neg}
_CONSTANTS = {'pi': pi, 'e': e}
_MAX_POW_EXPONENT = 1000

def _eval_safe(node: AST) -> object:
    """AST 白名单求值：仅常量、四则运算、幂、取模、一元正负与 pi/e。"""
    if isinstance(node, Expression):
        return _eval_safe(node.body)
    if isinstance(node, Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, BinOp) and type(node.op) in _BIN_OPS:
        if type(node.op) is Pow:
            _check_pow(node)
        op = _BIN_OPS[type(node.op)]
        return op(_eval_safe(node.left), _eval_safe(node.right))
    if isinstance(node, UnaryOp) and type(node.op) in _UNARY_OPS:
        op = _UNARY_OPS[type(node.op)]
        return op(_eval_safe(node.operand))
    if isinstance(node, Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise ToolValidationError(f'表达式包含不支持的语法: {unparse(node)}')

def _check_pow(node: BinOp):
    """限制幂指数大小，防止指数塔把求值拖死。"""
    exponent = node.right
    if isinstance(exponent, Constant) and isinstance(exponent.value, (int, float)):
        if abs(exponent.value) > _MAX_POW_EXPONENT:
            raise ToolValidationError(f'幂指数过大（上限 {_MAX_POW_EXPONENT}）')
    if isinstance(exponent, UnaryOp):
        inner = exponent.operand
        if isinstance(inner, Constant) and isinstance(inner.value, (int, float)):
            if abs(inner.value) > _MAX_POW_EXPONENT:
                raise ToolValidationError(f'幂指数过大（上限 {_MAX_POW_EXPONENT}）')

def _calculate(expression: str) -> object:
    """安全求值数学表达式。"""
    if not expression.strip():
        raise ToolValidationError('表达式不能为空')
    try:
        tree = parse(expression, mode='eval')
    except SyntaxError as exc:
        raise ToolValidationError(f'表达式语法错误: {exc.msg}') from exc
    try:
        return _eval_safe(tree)
    except ZeroDivisionError as exc:
        raise ToolValidationError('除数为零') from exc

def make_calculator_tool() -> Tool:
    """构造 calculator 工具。"""
    return Tool(
        name='calculator',
        description='安全求值数学表达式，支持四则运算、幂、取模与常量 pi/e',
        parameters={
            'type': 'object',
            'properties': {'expression': {'type': 'string', 'description': '数学表达式，如 (2+3)*4'}},
            'required': ['expression']
        },
        func=_calculate
    )
