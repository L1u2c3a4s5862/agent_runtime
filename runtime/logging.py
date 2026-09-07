from pathlib import Path

from loguru import logger

LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'

def setup_log_file(filename: str) -> None:
    """把日志写入项目根 logs/<filename>，目录不存在时自动创建。"""
    LOG_DIR.mkdir(exist_ok=True)
    logger.add(LOG_DIR / filename, encoding='utf-8')
