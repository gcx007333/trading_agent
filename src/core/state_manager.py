# src/core/state_manager.py
from typing import Dict, Any, Optional
import threading
from datetime import datetime
import json
from typing import List
from .base_agent import BaseAgent

class StateManager(BaseAgent):
    """状态管理器，负责维护系统状态"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("StateManager", config)
        self.states: Dict[str, Any] = {}
        self.state_history: Dict[str, List[Dict]] = {}
        self.max_history_length = config.get("max_history_length", 1000)
        
    def initialize(self) -> bool:
        """初始化状态管理器"""
        # 初始化基础状态
        self.states = {
            "system": {
                "status": "initialized",
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            },
            "trading": {
                "market_status": "closed",
                "current_position": {},
                "cash_balance": 0,
                "total_assets": 0
            },
            "risk": {
                "current_risk_level": "low",
                "max_drawdown": 0,
                "volatility": 0
            }
        }
        self.is_running = True
        return True
        
    def set_state(self, domain: str, key: str, value: Any, save_history: bool = True):
        """设置状态"""
        with self._lock:
            if domain not in self.states:
                self.states[domain] = {}
                
            old_value = self.states[domain].get(key)
            self.states[domain][key] = value
            self.states[domain]["last_update"] = datetime.now().isoformat()
            
            # 保存历史记录
            if save_history:
                self._save_state_history(domain, key, value, old_value)
                
    def get_state(self, domain: str, key: str = None, default: Any = None) -> Any:
        """获取状态"""
        with self._lock:
            if domain not in self.states:
                return default
            if key is None:
                return self.states[domain]
            return self.states[domain].get(key, default)
            
    def _save_state_history(self, domain: str, key: str, new_value: Any, old_value: Any):
        """保存状态历史"""
        history_key = f"{domain}.{key}"
        if history_key not in self.state_history:
            self.state_history[history_key] = []
            
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "old_value": old_value,
            "new_value": new_value
        }
        
        self.state_history[history_key].append(history_entry)
        
        # 限制历史记录长度
        if len(self.state_history[history_key]) > self.max_history_length:
            self.state_history[history_key] = self.state_history[history_key][-self.max_history_length:]
            
    def get_state_history(self, domain: str, key: str, limit: int = 100) -> List[Dict]:
        """获取状态历史"""
        history_key = f"{domain}.{key}"
        history = self.state_history.get(history_key, [])
        return history[-limit:] if limit else history
        
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行状态操作"""
        operation = data.get("operation")
        domain = data.get("domain")
        key = data.get("key")
        value = data.get("value")
        
        if operation == "set":
            self.set_state(domain, key, value)
            return {"status": "success"}
        elif operation == "get":
            result = self.get_state(domain, key)
            return {"status": "success", "value": result}
        else:
            return {"status": "error", "message": f"未知操作: {operation}"}
            
    def shutdown(self):
        """关闭状态管理器"""
        self.is_running = False
        # 保存最终状态
        self._save_final_state()
        
    def _save_final_state(self):
        """保存最终状态"""
        try:
            with open("final_state.json", "w", encoding="utf-8") as f:
                json.dump(self.states, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存最终状态失败: {e}")