# src/core/event_engine.py
from typing import Dict, Any, Callable, List
import threading
from queue import Queue, Empty
import time
from datetime import datetime
from core.base_agent import BaseAgent
from utils.logger import get_logger

class Event:
    """事件类"""
    def __init__(self, event_type: str, data: Any = None, source: str = None):
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now()
        self.event_id = f"{event_type}_{int(self.timestamp.timestamp()*1000)}"

class EventEngine(BaseAgent):
    """事件引擎，负责事件分发和处理"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EventEngine", config)
        self.event_queue = Queue()
        self.handlers: Dict[str, List[Callable]] = {}
        self.workers = []
        self.max_workers = config.get("max_workers", 5)
        
    def initialize(self) -> bool:
        """初始化事件引擎"""
        self.is_running = True
        # 启动工作线程
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._process_events, daemon=True)
            worker.start()
            self.workers.append(worker)
        self.logger.info(f"事件引擎初始化完成，启动 {self.max_workers} 个工作线程")
        return True
        
    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        with self._lock:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            
    def unregister_handler(self, event_type: str, handler: Callable):
        """注销事件处理器"""
        with self._lock:
            if event_type in self.handlers and handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)
                
    def put_event(self, event: Event):
        """放入事件"""
        if self.is_running:
            self.event_queue.put(event)
            
    def _process_events(self):
        """处理事件的工作线程函数"""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1)
                self._dispatch_event(event)
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"处理事件时发生错误: {e}")
                
    def _dispatch_event(self, event: Event):
        """分发事件到对应的处理器"""
        handlers = self.handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"事件处理器执行错误: {e}")
                
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行事件处理"""
        event = Event(data.get("event_type", "unknown"), data.get("data"))
        self.put_event(event)
        return {"status": "event_queued", "event_id": event.event_id}
        
    def shutdown(self):
        """关闭事件引擎"""
        self.is_running = False
        for worker in self.workers:
            worker.join(timeout=5)
        self.logger.info("事件引擎已关闭")