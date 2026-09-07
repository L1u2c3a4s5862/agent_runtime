"""pytest 共享配置：会话相关 fixture 与测试日志分流。"""

from collections.abc import Callable

from loguru import logger
from pytest import Session as PytestSession, fixture

from runtime.logging import setup_log_file
from runtime.session import Session
from runtime.testing import ScriptedLLM

@fixture
def new_session() -> Session:
    """独立的新会话。"""
    return Session()

@fixture
def make_llm() -> Callable[[list[str]], ScriptedLLM]:
    """构造脚本 LLM 的工厂。"""
    return ScriptedLLM

def pytest_collection_finish(session: PytestSession) -> None:
    """测试执行期间把 loguru 日志改写到 logs/test.log，与正式运行的 logs/agent.log 分离。"""
    # 时机关键：collection 完成意味着测试模块已 import（各模块顶层已注册 agent.log handler），
    # 此时 remove 换 handler，测试期间日志全部进 test.log，产品代码零改动
    logger.remove()
    setup_log_file('test.log')
