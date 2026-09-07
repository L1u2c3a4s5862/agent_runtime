"""轨迹记录：每次 LLM / 工具调用挂载一个 TraceEvent 到 session，可 rich 渲染。"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

from rich.console import Console
from rich.table import Table

# 仅类型标注用，避免运行时循环导入
if TYPE_CHECKING:
    from .session import Session

TraceKind = Literal['llm', 'tool']

@dataclass
class TraceEvent:
    """一次 LLM 或工具调用的轨迹事件。"""
    kind: TraceKind
    name: str
    payload: str
    started_at: datetime
    duration_ms: float | None = None
    ok: bool = True
    error: str = ''
    finished_at: datetime = field(default_factory=datetime.now)

def record_event(
    session: 'Session',
    kind: TraceKind,
    name: str,
    payload: str,
    started_at: Optional[datetime]=None,
    ok: bool=True,
    error: str=''
) -> TraceEvent:
    """
    构造并挂载一个轨迹事件到 session。

    Args:
        `session`: 目标会话。
        `kind`: 事件类型（llm/tool）。
        `name`: 模型名或工具名。
        `payload`: LLM 完整输出，或工具参数 + 结果摘要。
        `started_at`: 开始时间；传入时自动计算耗时。
        `ok`: 是否成功。
        `error`: 失败原因。

    Returns:
        已挂载的事件对象。
    """
    begin = started_at or datetime.now()
    event = TraceEvent(
        kind=kind, name=name, payload=payload,
        started_at=begin, ok=ok, error=error
    )
    # finished_at 由 default_factory 生成，与 begin 相减即真实耗时
    if started_at is not None:
        event.duration_ms = (event.finished_at - begin).total_seconds() * 1000
    session.traces.append(event)
    return event

def render_trace(events: list[TraceEvent]) -> None:
    """用 rich 表格打印轨迹。"""
    table = Table(title='Trace 轨迹')
    table.add_column('序号', justify='right')
    table.add_column('类型')
    table.add_column('名称')
    table.add_column('耗时(ms)', justify='right')
    table.add_column('状态')
    table.add_column('内容摘要')
    for index, event in enumerate(events, start=1):
        status = 'OK' if event.ok else f'ERROR: {event.error}'
        duration = f'{event.duration_ms:.1f}' if event.duration_ms is not None else '-'
        # 摘要截 60 字并把换行压成空格，保证表格单行可读
        summary = event.payload[:60].replace('\n', ' ') + ('…' if len(event.payload) > 60 else '')
        table.add_row(str(index), event.kind, event.name, duration, status, summary)
    Console().print(table)
