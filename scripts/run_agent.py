# scripts/run_agent.py
#!/usr/bin/env python3
"""
启动交易Agent的主脚本
"""

import sys
import os
import signal
import time
from pathlib import Path
from typing import Optional

from pyparsing import Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.agent.trading_agent import TradingAgent
from src.utils.logger import initialize_logging

class AgentRunner:
    """Agent运行器"""
    
    def __init__(self, config_path: str = "config"):
        self.config_path = config_path
        self.agent = None
        self.is_running = False
        
    def initialize(self) -> bool:
        """初始化Agent运行器"""
        try:
            # 加载配置
            config_loader = ConfigLoader(self.config_path)
            config = config_loader.load_all_configs()
            
            # 设置日志
            initialize_logging()
            
            # 创建交易Agent
            self.agent = TradingAgent(config.get("agent", {}))
            
            # 初始化Agent
            if not self.agent.initialize():
                print("Agent初始化失败")
                return False
                
            # 设置信号处理
            self._setup_signal_handlers()
            
            return True
            
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
            
    def _setup_signal_handlers(self):
        """设置信号处理"""
        def signal_handler(sig, frame):
            print("\n收到关闭信号，正在关闭Agent...")
            self.shutdown()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def run(self):
        """运行Agent"""
        if not self.agent:
            print("Agent未初始化")
            return
            
        self.is_running = True
        
        try:
            # 启动交易
            self.agent.start_trading()
            
            print("交易Agent已启动，按Ctrl+C停止...")
            
            # 主循环
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n收到中断信号")
        except Exception as e:
            print(f"运行错误: {e}")
        finally:
            self.shutdown()
            
    def shutdown(self):
        """关闭Agent"""
        self.is_running = False
        if self.agent:
            self.agent.shutdown()
        print("Agent已关闭")

def main():
    """主函数"""
    runner = AgentRunner()
    
    if not runner.initialize():
        sys.exit(1)
        
    runner.run()

if __name__ == "__main__":
    main()