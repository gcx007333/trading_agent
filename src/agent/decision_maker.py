# src/agent/decision_maker.py
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
from core.base_agent import BaseAgent
from models.model_predictor import ModelPredictor
from trading.risk_manager import RiskManager

class DecisionMaker(BaseAgent):
    """决策器，负责生成交易决策"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DecisionMaker", config)
        self.model_predictor = None
        self.risk_manager = None
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.max_position_size = config.get("max_position_size", 0.1)  # 最大仓位比例
        
    def initialize(self, symbols:list=None) -> bool:
        """初始化决策器"""
        try:
            # 初始化模型预测器
            self.model_predictor = ModelPredictor("config")
            self.model_predictor.initialize(symbols)
            
            # 初始化风险管理器
            self.risk_manager = RiskManager(None, None)
            # self.risk_manager.initialize()
            
            self.is_running = True
            self.logger.info("决策器初始化完成")
            return True
        except Exception as e:
            self.logger.error(f"决策器初始化失败: {e}")
            return False
            
    def make_decision(self, symbol: str, market_data: Dict[str, Any], 
                    account_info: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成交易决策"""
        try:
            # 检查模型是否可用
            if not self._is_model_ready():
                return self._make_fallback_decision(symbol, market_data, account_info)
            
            # 1. 获取模型预测
            prediction = self.model_predictor.predict(symbol, market_data)
            
            # 2. 风险评估
            risk_assessment = self.risk_manager.assess_trade_risk(
                symbol, market_data, prediction, account_info
            )
            
            # 3. 生成决策
            decision = self._generate_trading_decision(
                symbol, prediction, risk_assessment, account_info, context
            )
            
            # 4. 记录决策日志
            self._log_decision(symbol, decision, prediction, risk_assessment)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"生成决策失败: {e}")
            return self._make_fallback_decision(symbol, market_data, account_info)

    def _is_model_ready(self) -> bool:
        """检查模型是否就绪"""
        if self.model_predictor is None:
            return False
        return self.model_predictor.is_ready()

    def _make_fallback_decision(self, symbol: str, market_data: Dict[str, Any], 
                            account_info: Dict[str, Any]) -> Dict[str, Any]:
        """备用决策策略（当模型不可用时使用）"""
        try:
            current_position = account_info.get("positions", {}).get(symbol, 0)
            
            # 简单的备用策略：基于价格变化和持仓情况
            price_change = market_data.get('change_pct', 0)
            current_price = market_data.get('current', 0)
            
            # 如果价格下跌超过3%且有持仓，考虑卖出
            if price_change < -3 and current_position > 0:
                sell_quantity = min(current_position, 100)  # 卖出100股或全部
                return {
                    "symbol": symbol,
                    "action": "sell",
                    "quantity": sell_quantity,
                    "price": current_price,
                    "confidence": 0.3,
                    "reason": "备用策略: 价格大幅下跌",
                    "current_position": current_position,
                    "timestamp": datetime.now().isoformat()
                }
            # 如果价格上涨超过2%且无持仓，考虑买入
            elif price_change > 2 and current_position == 0:
                available_cash = account_info.get("current_cash", 0)
                if available_cash > current_price * 100:  # 确保有足够现金买100股
                    return {
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": 100,
                        "price": current_price,
                        "confidence": 0.3,
                        "reason": "备用策略: 价格大幅上涨",
                        "current_position": current_position,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # 默认持有
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence": 0.1,
                "reason": "备用策略: 无明确信号",
                "current_position": current_position,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"备用策略失败: {e}")
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence": 0.0,
                "reason": f"决策错误: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            
    def _generate_trading_decision(self, symbol: str, prediction: Dict[str, Any],
                                risk_assessment: Dict[str, Any], account_info: Dict[str, Any],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """生成具体的交易决策"""
        
        # 获取当前持仓和账户信息
        current_position = account_info.get("positions", {}).get(symbol, 0)
        total_value = account_info.get("total_value", 0)
        available_cash = account_info.get("current_cash", 0)
        
        # 如果风险过高，保持观望
        if risk_assessment.get("risk_level") == "high":
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence": 0.0,
                "reason": "风险过高",
                "current_position": current_position,
                "timestamp": datetime.now().isoformat()
            }
            
        # 获取预测置信度
        confidence = prediction.get("confidence", 0.0)
        predicted_direction = prediction.get("direction", "hold")
        
        # 如果置信度低于阈值，保持观望
        if confidence < self.confidence_threshold:
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence": confidence,
                "reason": f"置信度过低: {confidence:.2f}",
                "current_position": current_position,
                "timestamp": datetime.now().isoformat()
            }
            
        # 基于当前持仓的决策逻辑
        position_value = current_position * prediction.get("expected_price", 0)
        position_ratio = position_value / total_value if total_value > 0 else 0
        
        # 决策逻辑优化
        if predicted_direction == "buy":
            # 买入条件：当前持仓不足或需要加仓
            if current_position <= 0 or position_ratio < 0.05:  # 持仓比例小于5%
                quantity = self._calculate_buy_quantity(symbol, prediction, risk_assessment, account_info)
                if quantity > 0:
                    return {
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": quantity,
                        "price": prediction.get("expected_price"),
                        "confidence": confidence,
                        "reason": f"模型预测上涨，当前持仓{current_position}股",
                        "current_position": current_position,
                        "target_position": current_position + quantity,
                        "timestamp": datetime.now().isoformat()
                    }
                    
        elif predicted_direction == "sell":
            # 卖出条件：当前有持仓且需要减仓
            if current_position > 0:
                quantity = self._calculate_sell_quantity(symbol, prediction, risk_assessment, account_info)
                if quantity > 0:
                    return {
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": quantity,
                        "price": prediction.get("expected_price"),
                        "confidence": confidence,
                        "reason": f"模型预测下跌，当前持仓{current_position}股",
                        "current_position": current_position,
                        "target_position": current_position - quantity,
                        "timestamp": datetime.now().isoformat()
                    }
        
        # 持仓管理：如果持仓过重且预测中性，考虑减仓
        if position_ratio > 0.2 and predicted_direction == "hold" and confidence > 0.6:
            quantity = min(current_position, int(current_position * 0.3))  # 减仓30%
            return {
                "symbol": symbol,
                "action": "sell",
                "quantity": quantity,
                "price": prediction.get("expected_price"),
                "confidence": confidence,
                "reason": f"持仓过重({position_ratio:.1%})，主动减仓",
                "current_position": current_position,
                "target_position": current_position - quantity,
                "timestamp": datetime.now().isoformat()
            }
            
        return {
            "symbol": symbol,
            "action": "hold",
            "confidence": confidence,
            "reason": f"保持持仓{current_position}股",
            "current_position": current_position,
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_buy_quantity(self, symbol: str, prediction: Dict[str, Any],
                            risk_assessment: Dict[str, Any], account_info: Dict[str, Any]) -> int:
        """计算买入数量，考虑当前持仓"""
        current_position = account_info.get("positions", {}).get(symbol, 0)
        available_cash = account_info.get("current_cash", 0)
        current_price = prediction.get("expected_price", 0)
        
        if current_price <= 0:
            return 0
            
        # 基于风险调整的仓位计算
        risk_score = risk_assessment.get("risk_score", 1.0)
        confidence = prediction.get("confidence", 0.0)
        
        # 基础仓位比例
        base_position_ratio = self.max_position_size * confidence
        
        # 风险调整
        risk_adjusted_ratio = base_position_ratio / max(risk_score, 0.1)
        
        # 最终仓位比例
        position_ratio = min(risk_adjusted_ratio, self.max_position_size)
        
        # 计算目标市值
        total_value = account_info.get("total_value", 0)
        target_value = total_value * position_ratio
        
        # 计算需要买入的金额（考虑已有持仓）
        current_value = current_position * current_price
        buy_value = max(0, target_value - current_value)
        
        # 考虑可用资金限制
        buy_value = min(buy_value, available_cash * 0.95)  # 留5%现金
        
        # 计算股数
        quantity = int(buy_value / current_price)
        
        # 确保至少100股（A股最小交易单位）
        return max(quantity // 100 * 100, 0)

    def _calculate_sell_quantity(self, symbol: str, prediction: Dict[str, Any],
                            risk_assessment: Dict[str, Any], account_info: Dict[str, Any]) -> int:
        """计算卖出数量，基于当前持仓"""
        current_position = account_info.get("positions", {}).get(symbol, 0)
        confidence = prediction.get("confidence", 0.0)
        
        if current_position <= 0:
            return 0
            
        # 基于置信度的卖出比例
        if confidence > 0.8:
            sell_ratio = 1.0  # 高置信度时全卖
        elif confidence > 0.6:
            sell_ratio = 0.5  # 中等置信度时卖一半
        else:
            sell_ratio = 0.3  # 低置信度时卖30%
        
        # 计算卖出数量
        quantity = int(current_position * sell_ratio)
        
        # 确保是100股的整数倍
        quantity = quantity // 100 * 100
        
        # 确保至少卖出100股，且不超过当前持仓
        return max(100, min(quantity, current_position))
            
    def _calculate_position_size(self, symbol: str, prediction: Dict[str, Any],
                               risk_assessment: Dict[str, Any], account_info: Dict[str, Any]) -> int:
        """计算仓位大小"""
        available_cash = account_info.get("current_cash", 0)
        current_price = prediction.get("expected_price", 0)
        
        if current_price <= 0:
            return 0
            
        # 基于风险调整的仓位计算
        risk_score = risk_assessment.get("risk_score", 1.0)
        confidence = prediction.get("confidence", 0.0)
        
        # 基础仓位比例
        base_position_ratio = self.max_position_size * confidence
        
        # 风险调整
        risk_adjusted_ratio = base_position_ratio / max(risk_score, 0.1)
        
        # 最终仓位比例
        position_ratio = min(risk_adjusted_ratio, self.max_position_size)
        
        # 计算股数
        position_value = available_cash * position_ratio
        quantity = int(position_value / current_price)
        
        # 确保至少1股
        return max(quantity, 1) if quantity > 0 else 0
        
    def _log_decision(self, symbol: str, decision: Dict[str, Any], 
                     prediction: Dict[str, Any], risk_assessment: Dict[str, Any]):
        """记录决策日志"""
        log_entry = {
            "symbol": symbol,
            "decision": decision,
            "prediction": prediction,
            "risk_assessment": risk_assessment,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.info(f"交易决策: {log_entry}")
        
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行决策生成"""
        symbol = data.get("symbol")
        market_data = data.get("market_data", {})
        account_info = data.get("account_info", {})
        context = data.get("context", {})
        
        decision = self.make_decision(symbol, market_data, account_info, context)
        return {"status": "success", "decision": decision}
        
    def shutdown(self):
        """关闭决策器"""
        self.is_running = False
        if self.model_predictor:
            self.model_predictor.shutdown()
        if self.risk_manager:
            self.risk_manager.shutdown()