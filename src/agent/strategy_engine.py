# src/agent/strategy_engine.py
from typing import Dict, Any, List, Optional
import yaml
import importlib.util
from pathlib import Path
from core.base_agent import BaseAgent

class Strategy:
    """策略基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_active = True
        
    def initialize(self):
        """初始化策略"""
        pass
        
    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成交易信号"""
        return {"action": "hold", "confidence": 0.0}
        
    def update_parameters(self, new_params: Dict[str, Any]):
        """更新策略参数"""
        self.config.update(new_params)
        
    def get_status(self) -> Dict[str, Any]:
        """获取策略状态"""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "parameters": self.config
        }

class MomentumStrategy(Strategy):
    """动量策略"""
    
    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        technicals = data.get('technical_indicators', {})
        
        # 简单的动量策略逻辑
        rsi = technicals.get('rsi', 50)
        macd = technicals.get('macd', 0)
        macd_signal = technicals.get('macd_signal', 0)
        
        if rsi < 30 and macd > macd_signal:
            return {"action": "buy", "confidence": 0.7, "reason": "超卖反弹"}
        elif rsi > 70 and macd < macd_signal:
            return {"action": "sell", "confidence": 0.7, "reason": "超买回调"}
        else:
            return {"action": "hold", "confidence": 0.3}

class MeanReversionStrategy(Strategy):
    """均值回归策略"""
    
    def generate_signal(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        technicals = data.get('technical_indicators', {})
        
        # 简单的均值回归逻辑
        price = data.get('current_price', 0)
        bollinger_upper = technicals.get('bollinger_upper', price)
        bollinger_lower = technicals.get('bollinger_lower', price)
        bollinger_middle = technicals.get('bollinger_middle', price)
        
        if price < bollinger_lower:
            return {"action": "buy", "confidence": 0.6, "reason": "价格低于布林带下轨"}
        elif price > bollinger_upper:
            return {"action": "sell", "confidence": 0.6, "reason": "价格高于布林带上轨"}
        else:
            return {"action": "hold", "confidence": 0.4}

class StrategyEngine(BaseAgent):
    """策略引擎，管理多种交易策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("StrategyEngine", config)
        self.strategies: Dict[str, Strategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_configs = {}
        
    def initialize(self,symbols:list=None) -> bool:
        """初始化策略引擎"""
        try:
            # 加载策略配置
            self._load_strategy_configs()
            
            # 初始化内置策略
            self._initialize_builtin_strategies()
            
            # 加载自定义策略
            self._load_custom_strategies()
            
            # 设置策略权重
            self._setup_strategy_weights()
            
            self.is_running = True
            self.logger.info(f"策略引擎初始化完成，加载 {len(self.strategies)} 个策略")
            return True
        except Exception as e:
            self.logger.error(f"策略引擎初始化失败: {e}")
            return False
            
    def _load_strategy_configs(self):
        """加载策略配置"""
        config_path = self.config.get("strategy_config_path", "config/strategies.yaml")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.strategy_configs = yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.warning(f"加载策略配置失败: {e}，使用默认配置")
            self.strategy_configs = {}
            
    def _initialize_builtin_strategies(self):
        """初始化内置策略"""
        builtin_strategies = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy
        }
        
        for strategy_name, strategy_class in builtin_strategies.items():
            if strategy_name in self.strategy_configs:
                config = self.strategy_configs[strategy_name]
                if config.get('enabled', True):
                    strategy = strategy_class(strategy_name, config)
                    strategy.initialize()
                    self.strategies[strategy_name] = strategy
                    self.logger.info(f"初始化内置策略: {strategy_name}")
                    
    def _load_custom_strategies(self):
        """加载自定义策略"""
        custom_strategies_path = self.config.get("custom_strategies_path", "strategies")
        strategies_dir = Path(custom_strategies_path)
        
        if not strategies_dir.exists():
            return
            
        for strategy_file in strategies_dir.glob("*.py"):
            try:
                strategy_name = strategy_file.stem
                spec = importlib.util.spec_from_file_location(strategy_name, strategy_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 假设每个文件有一个名为CustomStrategy的类
                if hasattr(module, 'CustomStrategy'):
                    strategy_class = getattr(module, 'CustomStrategy')
                    config = self.strategy_configs.get(strategy_name, {})
                    
                    if config.get('enabled', True):
                        strategy = strategy_class(strategy_name, config)
                        strategy.initialize()
                        self.strategies[strategy_name] = strategy
                        self.logger.info(f"加载自定义策略: {strategy_name}")
                        
            except Exception as e:
                self.logger.error(f"加载自定义策略 {strategy_file} 失败: {e}")
                
    def _setup_strategy_weights(self):
        """设置策略权重"""
        total_weight = 0
        for strategy_name, strategy in self.strategies.items():
            weight = self.strategy_configs.get(strategy_name, {}).get('weight', 1.0)
            self.strategy_weights[strategy_name] = weight
            total_weight += weight
            
        # 归一化权重
        if total_weight > 0:
            for strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] /= total_weight
                
    def generate_combined_signal(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合交易信号"""
        strategy_signals = []
        
        for strategy_name, strategy in self.strategies.items():
            if strategy.is_active:
                signal = strategy.generate_signal(symbol, data)
                signal['strategy'] = strategy_name
                signal['weight'] = self.strategy_weights.get(strategy_name, 0)
                strategy_signals.append(signal)
                
        if not strategy_signals:
            return {"action": "hold", "confidence": 0.0, "reason": "无有效策略"}
            
        # 加权综合信号
        return self._aggregate_signals(strategy_signals)
        
    def _aggregate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多个策略的信号"""
        action_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        
        for signal in signals:
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0.0)
            weight = signal.get('weight', 0.0)
            
            action_scores[action] += confidence * weight
            
        # 选择得分最高的动作
        best_action = max(action_scores, key=action_scores.get)
        best_score = action_scores[best_action]
        
        return {
            "action": best_action,
            "confidence": best_score,
            "component_signals": signals,
            "reason": f"综合策略信号，{best_action}得分: {best_score:.2f}"
        }
        
    def add_strategy(self, strategy_name: str, strategy: Strategy, weight: float = 1.0):
        """添加策略"""
        with self._lock:
            self.strategies[strategy_name] = strategy
            self.strategy_weights[strategy_name] = weight
            self._normalize_weights()
            
    def remove_strategy(self, strategy_name: str):
        """移除策略"""
        with self._lock:
            if strategy_name in self.strategies:
                del self.strategies[strategy_name]
                del self.strategy_weights[strategy_name]
                self._normalize_weights()
                
    def update_strategy_weight(self, strategy_name: str, new_weight: float):
        """更新策略权重"""
        with self._lock:
            if strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] = new_weight
                self._normalize_weights()
                
    def _normalize_weights(self):
        """归一化权重"""
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy_name in self.strategy_weights:
                self.strategy_weights[strategy_name] /= total_weight
                
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行策略操作"""
        symbol = data.get("symbol")
        market_data = data.get("market_data", {})
        
        signal = self.generate_combined_signal(symbol, market_data)
        return {"status": "success", "signal": signal}
        
    def shutdown(self):
        """关闭策略引擎"""
        self.is_running = False
        for strategy in self.strategies.values():
            if hasattr(strategy, 'shutdown'):
                strategy.shutdown()