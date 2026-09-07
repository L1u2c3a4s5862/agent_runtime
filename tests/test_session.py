from pytest import raises

from runtime.context import Message
from runtime.errors import SessionNotFoundError
from runtime.session import (
    Session, SessionManager,
    load_session, save_session
)

class TestCreate:
    def test_create_generates_unique_ids(self):
        sm = SessionManager()
        first = sm.create()
        second = sm.create()
        assert first.id != second.id

    def test_create_sets_active(self):
        sm = SessionManager()
        session = sm.create(label='天气')
        assert sm.active_session is session
        assert sm.sessions == [session]

    def test_histories_are_independent(self):
        sm = SessionManager()
        first = sm.create()
        second = sm.create()
        from runtime.context import Message

        first.history.append(Message('user', 'A 的消息'))
        assert len(first.history.messages) == 1
        assert len(second.history.messages) == 0

class TestGetSwitchDelete:
    def test_get_hit_and_miss(self):
        sm = SessionManager()
        session = sm.create()
        assert sm.get(session.id) is session
        with raises(SessionNotFoundError, match='会话不存在'):
            sm.get('nope')

    def test_switch_restores_session(self):
        sm = SessionManager()
        first = sm.create(label='窗口一')
        second = sm.create(label='窗口二')
        assert sm.switch(first.id) is first
        assert sm.active_session is first
        assert sm.switch(second.id) is second
        assert sm.active_session is second

    def test_switch_unknown_raises(self):
        sm = SessionManager()
        with raises(SessionNotFoundError):
            sm.switch('nope')

    def test_delete_active_clears_pointer(self):
        sm = SessionManager()
        session = sm.create()
        sm.delete(session.id)
        assert sm.active_session is None
        with raises(SessionNotFoundError):
            sm.get(session.id)

    def test_delete_inactive_keeps_active(self):
        sm = SessionManager()
        first = sm.create()
        second = sm.create()
        sm.delete(first.id)
        assert sm.active_session is second

class TestPersistence:
    def test_roundtrip_restores_history_and_params(self):
        session = Session(label='天气', max_turns=4, max_chars=5000)
        session.metadata['主题'] = '北京'
        session.history.append(Message('user', '北京天气怎么样'))
        session.history.append(Message('assistant', 'Final Answer: 晴'))

        restored = Session.from_dict(session.to_dict())
        assert restored.id == session.id
        assert restored.label == '天气'
        assert restored.max_turns == 4
        assert restored.max_chars == 5000
        assert restored.metadata == {'主题': '北京'}
        assert restored.history.messages == session.history.messages
        assert restored.created_at == session.created_at

    def test_save_and_load_file(self, tmp_path):
        session = Session(label='文件往返')
        session.history.append(Message('user', '第一条'))
        save_session(session, tmp_path)

        restored = load_session(session.id, tmp_path)
        assert restored.id == session.id
        assert restored.label == '文件往返'
        assert [m.content for m in restored.history.messages] == ['第一条']

    def test_load_unknown_raises(self, tmp_path):
        with raises(SessionNotFoundError, match='会话不存在'):
            load_session('nope', tmp_path)

    def test_restored_session_continues_dialogue(self, tmp_path):
        session = Session()
        session.history.append(Message('user', '第一问'))
        session.history.append(Message('assistant', 'Final Answer: 答一'))
        save_session(session, tmp_path)

        restored = load_session(session.id, tmp_path)
        restored.history.append(Message('user', '第二问'))
        save_session(restored, tmp_path)

        again = load_session(session.id, tmp_path)
        assert [m.content for m in again.history.messages] == ['第一问', 'Final Answer: 答一', '第二问']
