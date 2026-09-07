from argparse import ArgumentParser
from rich_argparse import RawDescriptionRichHelpFormatter
from pathlib import Path

from rich.console import Console
from loguru import logger

from runtime.agent import Agent
from runtime.llm import OpenAIClient
from runtime.logging import setup_log_file
from runtime.session import Session, load_session, save_session

SESSIONS_DIR = Path('sessions')

console = Console()

logger.remove()
setup_log_file('agent.log')

def _agent_loop(agent: Agent, session: Session):
    """多轮对话主循环；每回合成功后落盘，异常不中断会话。"""
    while True:
        try:
            user_input = console.input('[bold magenta]你[/]: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ('/exit', '/quit'):
            break
        if not user_input:
            continue
        try:
            answer = agent.run(session, user_input)
            save_session(session, SESSIONS_DIR)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception(f'本轮对话失败: {e}')
            console.print(f'[bold red][ERROR][/] {e}')
            continue
        console.print(f'[bold cyan]Agent[/]: {answer}')

def main():
    """入口：--session 恢复历史会话，缺省新建会话。"""
    parser = ArgumentParser(description='AI Agent 多轮对话', formatter_class=RawDescriptionRichHelpFormatter)
    parser.add_argument('-s', '--session', help='继续指定会话（session id，存于 sessions/ 目录）')
    args = parser.parse_args()
    if args.session:
        session = load_session(args.session, SESSIONS_DIR)
        console.print(f'[dim]已恢复会话 {session.id}[/]')
    else:
        session = Session()
        console.print(f'[dim]新建会话 {session.id}（下次用 --session {session.id} 继续）[/]')
    _agent_loop(Agent(OpenAIClient()), session)

if __name__ == '__main__':
    main()
