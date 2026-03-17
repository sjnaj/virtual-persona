"""
异步事件总线 —— Agent 间解耦通信
"""
import asyncio
import logging
from typing import Callable, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Event:
    name: str
    data: dict
    timestamp: datetime = field(default_factory=datetime.now)


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_log: List[Event] = []

    def subscribe(self, event_name: str, callback: Callable):
        self._subscribers.setdefault(event_name, []).append(callback)

    async def emit(self, event_name: str, data: dict = None):
        event = Event(name=event_name, data=data or {})
        self._event_log.append(event)

        # 只保留最近500条事件日志
        if len(self._event_log) > 500:
            self._event_log = self._event_log[-500:]

        callbacks = self._subscribers.get(event_name, [])
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as e:
                logger.error(f"Event handler error [{event_name}]: {e}")

    def recent_events(self, event_prefix: str = "", limit: int = 20) -> List[Event]:
        filtered = [
            e for e in self._event_log
            if e.name.startswith(event_prefix)
        ]
        return filtered[-limit:]