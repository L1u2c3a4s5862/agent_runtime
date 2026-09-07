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
    logger.remove()
    setup_log_file('test.log')
