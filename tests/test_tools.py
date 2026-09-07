"""工具系统测试：calculator 安全求值、参数校验、注册表与默认注册表。"""

from pytest import mark, raises

from runtime.errors import ToolNotFoundError, ToolValidationError
from runtime._tools import make_calculator_tool, make_default_registry
from runtime.tools import ToolRegistry, validate_arguments

class TestCalculator:
    @mark.parametrize(
        ('expression', 'expected'),
        [
            ('2+3*4', 14),
            ('(2+3)*4', 20),
            ('2**10', 1024),
            ('10/4', 2.5),
            ('7//2', 3),
            ('10%3', 1),
            ('-3+5', 2),
            ('round(pi*2, 2)', None)  # 调用函数不允许，单独测异常
        ]
    )
    def test_evaluate(self, expression: str, expected: object):
        # 一组合法表达式按预期求值；函数调用单独走异常分支
        tool = make_calculator_tool()
        if expected is None:
            with raises(ToolValidationError):
                tool.func(expression)
        else:
            assert tool.func(expression) == expected

    def test_constant_pi(self):
        tool = make_calculator_tool()
        result = tool.func('pi * 2')
        assert abs(result - 6.283185307179586) < 1e-9

    def test_unsafe_call_raises(self):
        # 白名单外的一切语法都被拦截，包括 import/函数调用
        tool = make_calculator_tool()
        with raises(ToolValidationError, match='不支持的语法'):
            tool.func('__import__("os").system("dir")')

    def test_attribute_access_raises(self):
        tool = make_calculator_tool()
        with raises(ToolValidationError):
            tool.func('().__class__')

    def test_division_by_zero_raises(self):
        tool = make_calculator_tool()
        with raises(ToolValidationError, match='除数为零'):
            tool.func('1/0')

    def test_syntax_error_raises(self):
        tool = make_calculator_tool()
        with raises(ToolValidationError, match='语法错误'):
            tool.func('2+')

    def test_huge_pow_raises(self):
        # 指数塔防护：超出上限的幂指数直接拒绝
        tool = make_calculator_tool()
        with raises(ToolValidationError, match='指数过大'):
            tool.func('2**99999')

    def test_empty_expression_raises(self):
        tool = make_calculator_tool()
        with raises(ToolValidationError):
            tool.func('   ')

class TestValidateArguments:
    def _schema(self) -> dict[str, object]:
        """标准校验 schema：必填 city/level，unit 有枚举限制。"""
        return {
            'type': 'object',
            'properties': {
                'city': {'type': 'string'},
                'level': {'type': 'integer'},
                'unit': {'type': 'string', 'enum': ['c', 'f']}
            },
            'required': ['city', 'level']
        }

    def test_missing_required_raises(self):
        with raises(ToolValidationError, match='缺少必需参数 city'):
            validate_arguments(self._schema(), {'level': 3})

    def test_wrong_type_raises(self):
        with raises(ToolValidationError, match='level 应为 integer'):
            validate_arguments(self._schema(), {'city': '北京', 'level': '三'})

    def test_bool_is_not_integer(self):
        # isinstance(True, int) 为真，必须先于 int 排除
        with raises(ToolValidationError, match='integer'):
            validate_arguments(self._schema(), {'city': '北京', 'level': True})

    def test_enum_out_of_range_raises(self):
        with raises(ToolValidationError, match='unit'):
            validate_arguments(self._schema(), {'city': '北京', 'level': 1, 'unit': 'k'})

    def test_unknown_keys_dropped(self):
        # 模型多传的参数静默剔除，不影响合法参数
        cleaned = validate_arguments(self._schema(), {'city': '北京', 'level': 1, 'hack': 'x'})
        assert cleaned == {'city': '北京', 'level': 1}

    def test_valid_args_pass(self):
        cleaned = validate_arguments(self._schema(), {'city': '北京', 'level': 2, 'unit': 'c'})
        assert cleaned == {'city': '北京', 'level': 2, 'unit': 'c'}

class TestRegistry:
    def test_register_get_list(self):
        registry = ToolRegistry()
        tool = make_calculator_tool()
        registry.register(tool)
        assert registry.get('calculator') is tool
        assert registry.list() == [tool]

    def test_get_unknown_raises(self):
        registry = ToolRegistry()
        with raises(ToolNotFoundError, match='未知工具'):
            registry.get('nope')

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(make_calculator_tool())
        registry.unregister('calculator')
        with raises(ToolNotFoundError):
            registry.get('calculator')

    def test_duplicate_register_overwrites(self):
        # 同名注册：后注册的覆盖先注册的
        registry = ToolRegistry()
        registry.register(make_calculator_tool())
        replacement = make_calculator_tool()
        registry.register(replacement)
        assert registry.get('calculator') is replacement

    def test_default_registry_has_three_tools(self):
        # 默认注册表：calculator + search（Tavily）+ weather（QWeather）
        registry = make_default_registry()
        names = [tool.name for tool in registry.list()]
        assert names == ['calculator', 'search', 'weather']

    def test_prompt_descriptors_contains_tool_names(self):
        # system prompt 工具清单渲染：工具名与参数描述都在
        registry = make_default_registry()
        text = registry.to_prompt_descriptors()
        assert 'calculator' in text
        assert 'weather' in text
        assert 'expression' in text
