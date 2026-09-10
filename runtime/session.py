"""会话模型：多 Session 隔离、切换与 JSON 持久化。"""

from json import dumps, loads
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .context import ContextManager, Message
from .errors import SessionNotFoundError
from .trace import TraceEvent

@dataclass
class Session:
    """一个对话会话：全部可变状态（历史、轨迹）收敛于此，即隔离的实体。"""
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=datetime.now)
    label: str = ''
    max_turns: int = 6
    max_chars: int = 8000
    history: ContextManager = field(init=False)
    traces: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # 历史管理器依赖 max_turns/max_chars，必须在字段就绪后初始化
        self.history = ContextManager(self.max_turns, self.max_chars)

    def to_dict(self) -> dict[str, object]:
        """序列化会话核心状态；traces 为诊断数据，不持久化。"""
        return {
            'id': self.id,
            'label': self.label,
            'created_at': self.created_at.isoformat(),
            'max_turns': self.max_turns,
            'max_chars': self.max_chars,
            'metadata': self.metadata,
            'messages': [
                {'role': message.role, 'content': message.content}
                for message in self.history.messages
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> 'Session':
        """从 to_dict 的输出重建会话。"""
        session = cls(
            id=str(data['id']),
            label=str(data.get('label', '')),
            created_at=datetime.fromisoformat(str(data['created_at'])),
            max_turns=int(data.get('max_turns', 6)),
            max_chars=int(data.get('max_chars', 8000))
        )
        session.metadata = dict(data.get('metadata', {}))
        for item in data.get('messages', []):
            message = dict(item)
            session.history.append(Message(str(message['role']), str(message['content'])))
        return session

def save_session(session: Session, directory: Path):
    """把会话写入 directory/<id>.json。"""
    # 目录由调用方指定（app 用 sessions/，测试用 tmp_path）
    directory.mkdir(exist_ok=True)
    # indent 便于人工查看；ensure_ascii=False 保留中文原文
    payload = dumps(session.to_dict(), ensure_ascii=False, indent=2)
    (directory / f'{session.id}.json').write_text(payload, encoding='utf-8')

def load_session(session_id: str, directory: Path) -> Session:
    """从 directory/<id>.json 恢复会话；文件不存在抛 SessionNotFoundError。"""
    path = directory / f'{session_id}.json'
    if not path.is_file():
        # 报错带路径提示：用户直接看 sessions/ 目录文件名就能找到正确 id
        raise SessionNotFoundError(f'会话不存在: {session_id}（{directory} 下无 {path.name}）')
    return Session.from_dict(loads(path.read_text(encoding='utf-8')))

class SessionManager:
    """会话注册表：创建、切换、删除与查询。"""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._active_id: Optional[str] = None

    def create(self, label: str='', max_turns: int=6, max_chars: int=8000) -> Session:
        """新建会话并设为活跃。"""
        session = Session(label=label, max_turns=max_turns, max_chars=max_chars)
        self._sessions[session.id] = session
        self._active_id = session.id
        return session

    def get(self, session_id: str) -> Session:
        """按 id 取会话。"""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f'会话不存在: {session_id}')
        return self._sessions[session_id]

    def switch(self, session_id: str) -> Session:
        """切换到指定会话并返回。"""
        session = self.get(session_id)
        self._active_id = session_id
        return session

    def delete(self, session_id: str):
        """删除会话；若删除的是活跃会话则清空活跃标记。"""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f'会话不存在: {session_id}')
        del self._sessions[session_id]
        if self._active_id == session_id:
            self._active_id = None

    @property
    def active_session(self) -> Optional[Session]:
        """当前活跃会话。"""
        return self._sessions.get(self._active_id) if self._active_id else None

    @property
    def sessions(self) -> list[Session]:
        """全部会话，按创建顺序。"""
        return list(self._sessions.values())
