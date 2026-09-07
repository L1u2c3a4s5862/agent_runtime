"""日志配置：把 loguru 日志统一写入项目根的 logs/ 目录。"""

from pathlib import Path

from loguru import logger

# 以本文件位置反推项目根：无论从哪个 cwd 启动，日志都落在同一处
LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'

def setup_log_file(filename: str) -> None:
    """把日志写入项目根 logs/<filename>，目录不存在时自动创建。"""
    # loguru 不会自动建目录，先 mkdir 再 add，否则首次运行直接失败
    LOG_DIR.mkdir(exist_ok=True)
    logger.add(LOG_DIR / filename, encoding='utf-8')
