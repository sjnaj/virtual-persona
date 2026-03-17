"""
聊天上下文管理 —— 区分私聊和群聊，管理每个窗口的对话流
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChatWindow:
    """一个聊天窗口的状态"""
    chat_id: int
    chat_type: str                     # "private" / "group" / "supergroup"
    
    # 群聊专属
    group_title: str = ""
    group_members: list = field(default_factory=list)  # 已知成员 user_id 列表
    
    # 对话流
    message_history: list = field(default_factory=list)  # 最近N条消息
    last_activity: Optional[datetime] = None
    my_last_message_time: Optional[datetime] = None
    
    # 群聊行为控制
    consecutive_silence: int = 0       # 连续没参与的消息数
    last_mentioned: bool = False       # 上一条消息是否@我
    active_thread_participants: list = field(default_factory=list)
    
    # 会话状态
    is_active_conversation: bool = False  # 当前是否在活跃对话中
    conversation_start: Optional[datetime] = None


class ChatContextManager:
    MAX_HISTORY = 30       # 每个窗口保留最近30条
    CONVERSATION_TIMEOUT = timedelta(minutes=30)  # 30分钟无互动视为会话结束

    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        self.windows: Dict[int, ChatWindow] = {}

    def get_or_create(self, chat_id: int, chat_type: str = "private",
                      group_title: str = "") -> ChatWindow:
        if chat_id not in self.windows:
            self.windows[chat_id] = ChatWindow(
                chat_id=chat_id,
                chat_type=chat_type,
                group_title=group_title,
            )
        w = self.windows[chat_id]
        if group_title:
            w.group_title = group_title
        return w

    def add_message(self, chat_id: int, user_id: int, user_name: str,
                    text: str, is_me: bool = False, mentioned_me: bool = False):
        window = self.windows.get(chat_id)
        if not window:
            return

        now = datetime.now()
        msg = {
            "user_id": user_id,
            "user_name": user_name if not is_me else self.persona_name,
            "text": text,
            "time": now.isoformat(),
            "is_me": is_me,
            "mentioned_me": mentioned_me,
        }
        window.message_history.append(msg)
        if len(window.message_history) > self.MAX_HISTORY:
            window.message_history = window.message_history[-self.MAX_HISTORY:]

        window.last_activity = now

        if is_me:
            window.my_last_message_time = now
            window.consecutive_silence = 0
        else:
            window.consecutive_silence += 1
            window.last_mentioned = mentioned_me

        # 更新群成员列表
        if window.chat_type in ("group", "supergroup") and not is_me:
            if user_id not in window.group_members:
                window.group_members.append(user_id)

        # 会话活跃状态
        if not is_me:
            window.is_active_conversation = True
            if not window.conversation_start:
                window.conversation_start = now

    def get_recent_context(self, chat_id: int, limit: int = 15) -> str:
        """获取最近的对话上下文（用于prompt）"""
        window = self.windows.get(chat_id)
        if not window:
            return "（新对话）"

        msgs = window.message_history[-limit:]
        lines = []
        for m in msgs:
            name = m["user_name"]
            prefix = "你" if m["is_me"] else name
            lines.append(f"{prefix}: {m['text']}")
        return "\n".join(lines) if lines else "（新对话）"

    def check_conversation_ended(self, chat_id: int) -> bool:
        """检查某个窗口的会话是否已超时"""
        window = self.windows.get(chat_id)
        if not window or not window.is_active_conversation:
            return True
        if window.last_activity:
            if datetime.now() - window.last_activity > self.CONVERSATION_TIMEOUT:
                window.is_active_conversation = False
                window.conversation_start = None
                return True
        return False