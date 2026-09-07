"""内置工具包：calculator / search（Tavily）/ weather（QWeather）与默认注册表。"""

from .calculator import make_calculator_tool
from .search import make_search_tool
from .weather import make_weather_tool

from ..tools import ToolRegistry

def make_default_registry() -> ToolRegistry:
    """组装默认工具注册表（Tavily 联网搜索 + QWeather 天气 + 计算器）。"""
    registry = ToolRegistry()
    registry.register(make_calculator_tool())
    registry.register(make_search_tool())
    registry.register(make_weather_tool())
    return registry

__all__ = ['make_default_registry']
