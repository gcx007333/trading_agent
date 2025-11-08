# src/core/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import threading
from datetime import datetime
from utils.logger import get_logger

class BaseAgent(ABC):
    """Agent基类，定义所有Agent的通用接口"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = get_logger(name)
        self.is_running = False
        self._lock = threading.RLock()
        
    @abstractmethod
    def initialize(self) -> bool:
        """初始化Agent"""
        pass
        
    @abstractmethod
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行主要逻辑"""
        pass
        
    @abstractmethod
    def shutdown(self):
        """关闭Agent"""
        pass
        
    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            "name": self.name,
            "is_running": self.is_running,
            "timestamp": datetime.now().isoformat()
        }