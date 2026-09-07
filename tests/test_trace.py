from datetime import datetime

from runtime.session import Session
from runtime.trace import TraceEvent, record_event, render_trace

def test_record_llm_event_fields():
    session = Session()
    start = datetime.now()
    event = record_event(
        session, 'llm', 'test-model',
        '模型输出文本', started_at=start
    )
    assert event.kind == 'llm'
    assert event.name == 'test-model'
    assert event.payload == '模型输出文本'
    assert event.ok is True
    assert event.error == ''
    assert event.duration_ms is not None and event.duration_ms >= 0
    assert session.traces == [event]

def test_record_tool_event():
    session = Session()
    event = record_event(
        session, 'tool', 'calculator',
        '{"expression": "1+1"} -> 2'
    )
    assert event.kind == 'tool'
    assert event.ok is True
    assert event.duration_ms is None  # 未传 started_at 则不计算耗时

def test_record_failed_event():
    session = Session()
    event = record_event(
        session, 'tool', 'search',
        'args', ok=False, error='索引爆炸'
    )
    assert event.ok is False
    assert event.error == '索引爆炸'

def test_render_trace_does_not_raise():
    events = [
        TraceEvent(
            kind='llm', name='m', payload='输出',
            started_at=datetime.now(), duration_ms=1.5
        ),
        TraceEvent(
            kind='tool', name='t', payload='参数',
            started_at=datetime.now(), ok=False, error='挂了'
        )
    ]
    render_trace(events)  # 冒烟：能渲染不抛异常即可
