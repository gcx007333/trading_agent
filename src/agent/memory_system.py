# src/agent/memory_system.py
import os
from typing import Dict, Any, List, Optional
import sqlite3
import json
from datetime import datetime, timedelta
from core.base_agent import BaseAgent

class MemorySystem(BaseAgent):
    """记忆系统，负责存储和检索交易经验"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MemorySystem", config)
        self.db_path = config.get("db_path", "data/database/memory.db")
        self.connection = None
        self.max_memories = config.get("max_memories", 10000)
        
    def initialize(self) -> bool:
        """初始化记忆系统 - 自动创建数据库"""
        try:
            # 确保数据库目录存在
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  # 如果路径包含目录
                os.makedirs(db_dir, exist_ok=True)
                self.logger.info(f"确保数据库目录存在: {db_dir}")
            
            # 连接数据库（如果不存在会自动创建）
            self.connection = sqlite3.connect(self.db_path)
            self._create_tables()
            self.is_running = True
            self.logger.info(f"记忆系统初始化完成，数据库: {self.db_path}")
            return True
        except Exception as e:
            self.logger.error(f"记忆系统初始化失败: {e}")
            return False
            
    def _create_tables(self):
        """创建数据库表"""
        cursor = self.connection.cursor()
        
        # 交易记忆表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_type ON trading_memories(symbol, memory_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON trading_memories(importance)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON trading_memories(created_at)')
        
        self.connection.commit()
        
    def store_memory(self, symbol: str, memory_type: str, content: Dict[str, Any], 
                    importance: float = 0.5) -> bool:
        """存储记忆"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO trading_memories (symbol, memory_type, content, metadata, importance)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, memory_type, json.dumps(content), 
                 json.dumps({"created": datetime.now().isoformat()}), importance))
            
            self.connection.commit()
            
            # 清理旧记忆
            self._cleanup_old_memories()
            
            return True
        except Exception as e:
            self.logger.error(f"存储记忆失败: {e}")
            return False
            
    def retrieve_memories(self, symbol: str, memory_type: str = None, 
                         limit: int = 10, min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """检索记忆"""
        try:
            cursor = self.connection.cursor()
            
            if memory_type:
                cursor.execute('''
                    SELECT * FROM trading_memories 
                    WHERE symbol = ? AND memory_type = ? AND importance >= ?
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT ?
                ''', (symbol, memory_type, min_importance, limit))
            else:
                cursor.execute('''
                    SELECT * FROM trading_memories 
                    WHERE symbol = ? AND importance >= ?
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT ?
                ''', (symbol, min_importance, limit))
                
            rows = cursor.fetchall()
            memories = []
            
            for row in rows:
                memory = {
                    'id': row[0],
                    'symbol': row[1],
                    'memory_type': row[2],
                    'content': json.loads(row[3]),
                    'metadata': json.loads(row[4]),
                    'importance': row[5],
                    'created_at': row[6],
                    'accessed_count': row[7]
                }
                memories.append(memory)
                
            # 更新访问记录
            self._update_access_count([row[0] for row in rows])
            
            return memories
        except Exception as e:
            self.logger.error(f"检索记忆失败: {e}")
            return []
            
    def _update_access_count(self, memory_ids: List[int]):
        """更新访问计数"""
        try:
            placeholders = ','.join('?' * len(memory_ids))
            cursor = self.connection.cursor()
            cursor.execute(f'''
                UPDATE trading_memories 
                SET accessed_count = accessed_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
            ''', memory_ids)
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"更新访问计数失败: {e}")
            
    def _cleanup_old_memories(self):
        """清理旧记忆"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT COUNT(*) FROM trading_memories')
            count = cursor.fetchone()[0]
            
            if count > self.max_memories:
                # 删除最不重要的记忆
                cursor.execute('''
                    DELETE FROM trading_memories 
                    WHERE id IN (
                        SELECT id FROM trading_memories 
                        ORDER BY importance ASC, last_accessed ASC 
                        LIMIT ?
                    )
                ''', (count - self.max_memories,))
                self.connection.commit()
        except Exception as e:
            self.logger.error(f"清理记忆失败: {e}")
            
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行记忆操作"""
        operation = data.get("operation")
        
        if operation == "store":
            return {
                "status": "success" if self.store_memory(**data.get("params", {})) else "error"
            }
        elif operation == "retrieve":
            memories = self.retrieve_memories(**data.get("params", {}))
            return {"status": "success", "memories": memories}
        else:
            return {"status": "error", "message": f"未知操作: {operation}"}
            
    def shutdown(self):
        """关闭记忆系统"""
        self.is_running = False
        if self.connection:
            self.connection.close()