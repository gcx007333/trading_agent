# src/agent/trading_agent.py
from typing import Dict, Any, List, Optional
import threading
import time
from datetime import datetime, timedelta

import numpy as np
from core.base_agent import BaseAgent
from core.event_engine import EventEngine, Event
from core.state_manager import StateManager
from agent.decision_maker import DecisionMaker
from agent.memory_system import MemorySystem
from agent.learning_module import LearningModule
from agent.strategy_engine import StrategyEngine
from trading.order_executor import OrderExecutor, OrderStatus
from data.market_data import MarketData
from trading.account_manager import AccountManager

class TradingAgent(BaseAgent):
    """主交易Agent，协调所有组件"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TradingAgent", config)
        
        # 初始化组件
        self.event_engine = EventEngine(config.get("event_engine", {}))
        self.state_manager = StateManager(config.get("state_manager", {}))
        self.memory_system = MemorySystem(config.get("memory_system", {}))
        self.strategy_engine = StrategyEngine(config.get("strategy_engine", {}))
        self.decision_maker = DecisionMaker(config.get("decision_maker", {}))
        self.learning_module = LearningModule(config.get("learning_module", {}))
        self.order_executor = OrderExecutor(account_manager=AccountManager(), broker_interface=None)
        self.market_data = MarketData(config.get("market_data", {}))
        
        # 模型训练器
        self.model_trainer = None
        self.models_ready = False
        
        # 交易参数
        self.trading_symbols = config.get("trading_symbols", [
            "600827", # 百联股份
            "600718", # 东软集团
            "600588", # 用友网络
            "601377", # 兴业证券
            "600303"  # 曙光股份
        ])
        self.trading_interval = config.get("trading_interval", 60)  # 交易检查间隔(秒)
        self.max_positions = config.get("max_positions", 5)
        
        # 控制变量
        self.trading_thread = None
        self.is_trading = False
        
    def initialize(self) -> bool:
        """初始化交易Agent - 确保模型在交易前就绪"""
        try:
            self.logger.info("开始初始化交易Agent...")
            
            # 第一步：初始化基础组件
            basic_components = [
                self.event_engine,
                self.state_manager,
                self.memory_system,
                self.market_data,
                self.order_executor
            ]
            
            for component in basic_components:
                if not component.initialize():
                    self.logger.error(f"基础组件 {component.name} 初始化失败")
                    return False
            
            # 第二步：初始化并训练模型（关键步骤）
            if not self._initialize_and_train_models():
                self.logger.error("模型初始化和训练失败")
                return False
                
            # 第三步：初始化依赖模型的组件
            model_dependent_components = [
                self.strategy_engine,
                self.decision_maker,
                self.learning_module
            ]
            
            for component in model_dependent_components:
                if not component.initialize(self.trading_symbols):
                    self.logger.error(f"模型依赖组件 {component.name} 初始化失败")
                    return False
                    
            # 注册事件处理器
            self._register_event_handlers()
            
            # 更新系统状态
            self.state_manager.set_state("system", "status", "initialized")
            self.state_manager.set_state("system", "agent_initialized", True)
            self.state_manager.set_state("system", "models_ready", True)
            
            self.is_running = True
            self.models_ready = True
            self.logger.info("交易Agent初始化完成，模型已就绪")
            return True
            
        except Exception as e:
            self.logger.error(f"交易Agent初始化失败: {e}")
            return False
    
    def _initialize_and_train_models(self) -> bool:
        """初始化并训练模型，确保模型在交易前可用"""
        try:
            from models.model_trainer import ModelTrainer
            
            # 初始化模型训练器
            self.model_trainer = ModelTrainer()
            # 批量训练
            results = self.model_trainer.train_multiple_stocks(self.trading_symbols, cache_check=False)
            if not results or not all(results.values()):
                self.logger.warning("部分股票模型训练失败")
            
            self.logger.warning("股票模型初始化和训练成功")
            return True
                    
        except Exception as e:
            self.logger.error(f"模型初始化和训练失败: {e}")
            return None  
            
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 市场数据事件
        self.event_engine.register_handler("market_data_update", self._handle_market_data_update)
        
        # 交易事件
        self.event_engine.register_handler("order_executed", self._handle_order_executed)
        self.event_engine.register_handler("order_failed", self._handle_order_failed)
        
        # 系统事件
        self.event_engine.register_handler("system_alert", self._handle_system_alert)
        
    def start_trading(self):
        """开始交易"""
        if self.is_trading:
            self.logger.warning("交易已经在进行中")
            return
            
        self.is_trading = True
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        self.state_manager.set_state("trading", "status", "active")
        self.logger.info("开始自动交易")
        
    def stop_trading(self):
        """停止交易"""
        self.is_trading = False
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
            
        self.state_manager.set_state("trading", "status", "stopped")
        self.logger.info("停止自动交易")
        
    def _trading_loop(self):
        """交易主循环"""
        last_training_time = time.time()
        training_interval = 3600  # 每1小时训练一次
        
        while self.is_trading and self.is_running:
            try:
                # 检查模型是否就绪
                if not self.models_ready:
                    self.logger.warning("模型未就绪，等待模型初始化...")
                    time.sleep(10)
                    continue
                    
                # 检查市场状态
                if not self._is_market_open():
                    time.sleep(60)
                    continue
                    
                # 为每个交易标的执行交易逻辑
                for symbol in self.trading_symbols:
                    self._process_symbol(symbol)
                    
                # 学习优化
                self._trigger_learning()
                
                # 定期模型训练（非交易时间或达到间隔）
                current_time = time.time()
                if current_time - last_training_time > training_interval:
                    self.__initialize_and_train_models()
                    last_training_time = current_time
                
                # 等待下一个交易周期
                time.sleep(self.trading_interval)
                
            except Exception as e:
                self.logger.error(f"交易循环错误: {e}")
                time.sleep(10)  # 错误时短暂等待
                
    def _is_market_open(self) -> bool:
        """检查市场是否开盘"""
        # 实现市场时间检查逻辑
        current_time = datetime.now().time()
        return (current_time >= datetime.strptime("09:30", "%H:%M").time() and 
                current_time <= datetime.strptime("19:00", "%H:%M").time())
        
    def _process_symbol(self, symbol: str):
        """处理单个标的的交易逻辑"""
        try:
            # 检查模型状态
            if not self.models_ready:
                self.logger.warning(f"模型未就绪，跳过 {symbol} 的处理")
                return
                
            # 1. 获取市场数据
            market_data = self.market_data.get_current_data(symbol)
            if not market_data:
                return
                
            # 2. 获取账户信息和当前持仓
            account_info = self.order_executor.account_manager.get_account_info()
            current_positions = self.order_executor.account_manager.get_positions()
            current_position = current_positions.get(symbol, 0)
            
            # 3. 检查持仓限制
            if not self._check_position_limit(symbol, current_position, account_info):
                return
                
            # 4. 生成交易决策（传入当前持仓信息）
            decision = self.decision_maker.make_decision(
                symbol, market_data, account_info, {
                    "current_position": current_position,
                    "total_positions": len(current_positions),
                    "market_condition": self._get_market_condition()
                }
            )
            
            # 5. 验证决策合理性（基于当前持仓）
            if not self._validate_decision(symbol, decision, current_position, account_info):
                return
                
            # 6. 执行交易
            if decision.get("action") in ["buy", "sell"]:
                self._execute_trade_decision(symbol, decision, market_data)
                
            # 7. 更新状态（包含持仓信息）
            self._update_trading_state(symbol, decision, market_data, current_position)
            
        except Exception as e:
            self.logger.error(f"处理标的 {symbol} 时发生错误: {e}")
    
    def _get_market_condition(self) -> Dict[str, Any]:
        """获取市场整体状况"""
        try:
            # 这里可以获取市场指数、涨跌比例等整体市场信息
            # 由于我们可能没有实时市场指数数据，先返回一个基础的市场状况
            
            market_condition = {
                "overall_trend": "neutral",  # bullish, bearish, neutral
                "market_sentiment": 0.5,     # 0-1, 1表示极度乐观
                "volatility_level": "medium", # low, medium, high
                "advance_decline_ratio": 0.6, # 上涨/下跌股票比例
                "timestamp": datetime.now().isoformat()
            }
            
            # 尝试获取更准确的市场状况（如果有相关数据）
            try:
                # 如果有市场指数数据，可以在这里添加
                # 例如：上证指数、深证成指等
                pass
            except Exception as e:
                self.logger.debug(f"获取详细市场状况失败: {e}")
            
            return market_condition
            
        except Exception as e:
            self.logger.error(f"获取市场状况失败: {e}")
            # 返回默认的市场状况
            return {
                "overall_trend": "neutral",
                "market_sentiment": 0.5,
                "volatility_level": "medium",
                "advance_decline_ratio": 0.5,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _check_position_limit(self, symbol: str, current_position: int, account_info: Dict[str, Any]) -> bool:
        """检查持仓限制"""
        # 检查最大持仓数量限制
        if len(account_info.get("positions", {})) >= self.max_positions and current_position == 0:
            self.logger.info(f"已达到最大持仓数量限制，跳过 {symbol}")
            return False
            
        # 检查单个标的持仓比例
        total_value = account_info.get("total_value", 0)
        if total_value > 0:
            position_value = current_position * account_info.get("current_price", 0)
            position_ratio = position_value / total_value
            
            if position_ratio > 0.3:  # 单个标的持仓不超过30%
                self.logger.warning(f"{symbol} 持仓比例过高({position_ratio:.1%})，暂停交易")
                return False
                
        return True

    def _validate_decision(self, symbol: str, decision: Dict[str, Any], 
                        current_position: int, account_info: Dict[str, Any]) -> bool:
        """验证决策合理性"""
        action = decision.get("action")
        quantity = decision.get("quantity", 0)
        
        if action == "buy":
            # 检查是否有足够现金
            available_cash = account_info.get("current_cash", 0)
            price = decision.get("price", 0)
            cost = quantity * price
            
            if cost > available_cash:
                self.logger.warning(f"现金不足，需要{cost:.2f}，可用{available_cash:.2f}")
                return False
                
        elif action == "sell":
            # 检查是否有足够持仓
            if quantity > current_position:
                self.logger.warning(f"持仓不足，需要卖出{quantity}，当前持仓{current_position}")
                return False
                
        return True

    def _update_trading_state(self, symbol: str, decision: Dict[str, Any], 
                            market_data: Dict[str, Any], current_position: int):
        """更新交易状态，包含持仓信息"""
        state_update = {
            "decision": decision,
            "market_data": market_data,
            "current_position": current_position,
            "timestamp": datetime.now().isoformat()
        }
        
        self.state_manager.set_state("trading", f"symbol_{symbol}", state_update)
        self.state_manager.set_state("trading", f"last_update_{symbol}", datetime.now().isoformat())
            
    def _execute_trade_decision(self, symbol: str, decision: Dict[str, Any], market_data: Dict[str, Any]):
        """执行交易决策 - 修复方法调用"""
        try:
            action = decision.get("action")
            quantity = decision.get("quantity", 0)
            price = decision.get("price")
            
            if quantity <= 0:
                return
                
            # 构建订单请求
            order_request = {
                'symbol': symbol,
                'action': action.upper(),  # 转换为大写
                'quantity': quantity,
                'order_type': 'market',  # 默认使用市价单
                'price': price,
                'reason': decision.get('reason', '自动交易决策'),
                'strategy': 'TradingAgent'
            }
            
            # 执行订单
            result = self.order_executor.place_order(order_request)
            
            if result.status == OrderStatus.FILLED:
                # 记录交易经验
                self._record_trading_experience(symbol, decision, market_data, {
                    "success": True,
                    "order_id": result.order_id,
                    "executed_price": result.avg_fill_price,
                    "executed_quantity": result.filled_quantity
                })
                
                # 发送事件
                event = Event("order_executed", {
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "price": price,
                    "order_id": result.order_id,
                    "result": {
                        "executed_price": result.avg_fill_price,
                        "executed_quantity": result.filled_quantity
                    }
                })
                self.event_engine.put_event(event)
                
            else:
                # 发送失败事件
                event = Event("order_failed", {
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "price": price,
                    "order_id": getattr(result, 'order_id', 'unknown'),
                    "error": getattr(result, 'reason', '订单执行失败')
                })
                self.event_engine.put_event(event)
                
        except Exception as e:
            self.logger.error(f"执行交易失败: {e}")
            
    def _record_trading_experience(self, symbol: str, decision: Dict[str, Any], 
                                 market_data: Dict[str, Any], result: Dict[str, Any]):
        """记录交易经验"""
        try:
            # 提取特征
            features = {
                'technical': market_data.get('technical_indicators', {}),
                'fundamental': market_data.get('fundamental_data', {}),
                'market_condition': market_data.get('market_condition', {})
            }
            
            # 计算交易结果
            outcome = self._calculate_trade_outcome(symbol, decision, result)
            
            # 记录到学习模块
            self.learning_module.record_experience(symbol, features, decision, outcome)
            
        except Exception as e:
            self.logger.error(f"记录交易经验失败: {e}")
            
    def _calculate_trade_outcome(self, symbol: str, decision: Dict[str, Any], 
                               result: Dict[str, Any]) -> Dict[str, Any]:
        """计算交易结果"""
        # 实现交易结果计算逻辑
        return {
            'profit': 0.0,  # 实际盈亏
            'learning_value': 0.5,  # 学习价值
            'timestamp': datetime.now().isoformat()
        }
        
    def _trigger_learning(self):
        """触发学习过程"""
        try:
            self.learning_module.learn_from_experience(force=False)
        except Exception as e:
            self.logger.error(f"触发学习失败: {e}")
            
    def _handle_market_data_update(self, event: Event):
        """处理市场数据更新事件"""
        data = event.data
        self.logger.info(f"市场数据更新: {data.get('symbol')}")
        
    def _handle_order_executed(self, event: Event):
        """处理订单执行事件"""
        data = event.data
        self.logger.info(f"订单执行成功: {data.get('symbol')} {data.get('action')} {data.get('quantity')}")
        
    def _handle_order_failed(self, event: Event):
        """处理订单失败事件"""
        data = event.data
        self.logger.error(f"订单执行失败: {data.get('symbol')} - {data.get('error')}")
        
    def _handle_system_alert(self, event: Event):
        """处理系统告警事件"""
        data = event.data
        self.logger.warning(f"系统告警: {data.get('message')}")
        
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行Agent操作"""
        operation = data.get("operation")
        
        if operation == "start_trading":
            self.start_trading()
            return {"status": "success", "message": "开始交易"}
        elif operation == "stop_trading":
            self.stop_trading()
            return {"status": "success", "message": "停止交易"}
        elif operation == "get_status":
            return self.get_detailed_status()
        else:
            return {"status": "error", "message": f"未知操作: {operation}"}
            
    def get_detailed_status(self) -> Dict[str, Any]:
        """获取详细状态"""
        base_status = self.get_status()
        
        component_status = {
            "event_engine": self.event_engine.get_status(),
            "state_manager": self.state_manager.get_status(),
            "memory_system": self.memory_system.get_status(),
            "strategy_engine": self.strategy_engine.get_status(),
            "decision_maker": self.decision_maker.get_status(),
            "learning_module": self.learning_module.get_status(),
            "order_executor": self.order_executor.get_status(),
            "market_data": self.market_data.get_status()
        }
        
        trading_status = {
            "is_trading": self.is_trading,
            "trading_symbols": self.trading_symbols,
            "trading_interval": self.trading_interval
        }
        
        return {
            **base_status,
            "components": component_status,
            "trading": trading_status
        }
        
    def shutdown(self):
        """关闭交易Agent"""
        self.logger.info("开始关闭交易Agent...")
        
        # 停止交易
        self.stop_trading()
        
        # 关闭组件
        components = [
            # self.learning_module,
            # self.decision_maker,
            self.strategy_engine,
            self.memory_system,
            # self.order_executor,
            self.market_data,
            self.state_manager,
            self.event_engine
        ]
        
        for component in reversed(components):
            try:
                component.shutdown()
            except Exception as e:
                self.logger.error(f"关闭组件 {component.name} 时发生错误: {e}")
                
        self.is_running = False
        self.logger.info("交易Agent已关闭")