# src/trading/risk_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum

from utils.config_loader import get_system_config
from utils.helpers import timer, singleton, safe_divide
from utils.performance import RiskMetrics
from trading.account_manager import AccountManager
from trading.position_manager import PositionManager

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskAssessment:
    """风险评估结果"""
    overall_risk: RiskLevel
    risk_score: float  # 0-10分数
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RiskLimit:
    """风险限制"""
    name: str
    current_value: float
    limit_value: float
    unit: str = "percent"
    is_violated: bool = False

@dataclass
class RiskEvent:
    """风险事件"""
    event_id: str
    event_type: str
    severity: RiskLevel
    description: str
    trigger_value: float
    limit_value: float
    timestamp: datetime
    symbol: str = ""
    account_id: str = ""

@singleton
class RiskManager:
    """
    风险管理器
    监控和控制交易风险
    """
    
    def __init__(self, account_manager: AccountManager, position_manager: PositionManager):
        self.account_manager = account_manager
        self.position_manager = position_manager
        self.config = get_system_config()
        self.risk_events: List[RiskEvent] = []
        self.risk_history: List[RiskAssessment] = []
        
        logger.info("风险管理器初始化完成")
    
    def assess_portfolio_risk(self, account_id: str = None) -> RiskAssessment:
        """评估投资组合风险"""
        account_id = account_id or self.account_manager.active_account_id
        
        try:
            # 获取投资组合分析
            portfolio_analysis = self.position_manager.analyze_portfolio(account_id)
            account_info = self.account_manager.get_account_info(account_id)
            
            # 检查各项风险限制
            risk_limits = self._check_all_risk_limits(account_id, portfolio_analysis, account_info)
            violations = [limit for limit in risk_limits if limit.is_violated]
            
            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(account_id, portfolio_analysis)
            
            # 计算总体风险分数
            risk_score = self._calculate_overall_risk_score(violations, risk_metrics)
            
            # 确定风险等级
            overall_risk = self._determine_risk_level(risk_score)
            
            # 生成建议
            recommendations = self._generate_risk_recommendations(violations, risk_metrics, overall_risk)
            
            assessment = RiskAssessment(
                overall_risk=overall_risk,
                risk_score=risk_score,
                violations=violations,
                recommendations=recommendations,
                metrics=risk_metrics
            )
            
            # 记录风险评估
            self.risk_history.append(assessment)
            
            # 记录风险事件（如果有违规）
            if violations:
                self._record_risk_events(violations, account_id)
            
            logger.info(f"风险评估完成: {overall_risk.value} (分数: {risk_score:.2f})")
            
            return assessment
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            # 返回一个高风险评估作为安全措施
            return RiskAssessment(
                overall_risk=RiskLevel.HIGH,
                risk_score=8.0,
                violations=[{"error": str(e)}],
                recommendations=["风险评估失败，建议暂停交易"],
                metrics={}
            )
    
    def _check_all_risk_limits(self, account_id: str, portfolio_analysis: Any, account_info: Dict) -> List[RiskLimit]:
        """检查所有风险限制"""
        risk_limits = []
        
        # 1. 最大回撤限制
        current_drawdown = self._calculate_current_drawdown(account_id)
        max_drawdown_limit = self.config.risk.max_drawdown
        risk_limits.append(RiskLimit(
            name="最大回撤",
            current_value=current_drawdown,
            limit_value=max_drawdown_limit,
            is_violated=current_drawdown > max_drawdown_limit
        ))
        
        # 2. 单票仓位限制
        max_position_size = self.config.risk.max_position_size
        positions = self.position_manager.account_manager.get_positions(account_id)
        for symbol, position in positions.items():
            position_weight = position.market_value / account_info['total_value']
            risk_limits.append(RiskLimit(
                name=f"单票仓位({symbol})",
                current_value=position_weight,
                limit_value=max_position_size,
                is_violated=position_weight > max_position_size
            ))
        
        # 3. 行业集中度限制
        max_sector_exposure = 0.3  # 最大行业暴露30%
        for sector, exposure in portfolio_analysis.sector_exposure.items():
            risk_limits.append(RiskLimit(
                name=f"行业集中度({sector})",
                current_value=exposure,
                limit_value=max_sector_exposure,
                is_violated=exposure > max_sector_exposure
            ))
        
        # 4. 组合集中度限制
        concentration_limit = 0.6  # 前5大持仓不超过60%
        risk_limits.append(RiskLimit(
            name="组合集中度",
            current_value=portfolio_analysis.concentration_ratio,
            limit_value=concentration_limit,
            is_violated=portfolio_analysis.concentration_ratio > concentration_limit
        ))
        
        # 5. 现金比例限制
        min_cash_ratio = 0.1  # 最少保持10%现金
        risk_limits.append(RiskLimit(
            name="现金比例",
            current_value=portfolio_analysis.cash_weight,
            limit_value=min_cash_ratio,
            is_violated=portfolio_analysis.cash_weight < min_cash_ratio
        ))
        
        # 6. 单日亏损限制
        daily_loss = self._calculate_daily_pnl(account_id)
        daily_loss_limit = self.config.risk.max_daily_loss
        risk_limits.append(RiskLimit(
            name="单日亏损",
            current_value=abs(min(daily_loss, 0)),
            limit_value=daily_loss_limit,
            is_violated=daily_loss < -daily_loss_limit
        ))
        
        return risk_limits
    
    def _calculate_risk_metrics(self, account_id: str, portfolio_analysis: Any) -> Dict[str, float]:
        """计算风险指标"""
        # 获取历史收益率数据（这里需要从数据库或缓存中获取）
        historical_returns = self._get_historical_returns(account_id)
        
        metrics = {}
        
        if len(historical_returns) >= 2:
            returns_series = pd.Series(historical_returns)
            
            # 计算传统风险指标
            metrics['volatility'] = returns_series.std() * np.sqrt(252)  # 年化波动率
            metrics['var_95'] = RiskMetrics.calculate_var(historical_returns, 0.95)
            metrics['cvar_95'] = RiskMetrics.calculate_cvar(historical_returns, 0.95)
            metrics['max_drawdown'] = self._calculate_max_drawdown(account_id)
            metrics['sharpe_ratio'] = safe_divide(returns_series.mean() * 252, metrics['volatility'])
            
            # 计算下行风险
            downside_returns = [r for r in historical_returns if r < 0]
            metrics['downside_volatility'] = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
            metrics['sortino_ratio'] = safe_divide(returns_series.mean() * 252, metrics['downside_volatility'])
        
        # 添加组合结构指标
        metrics['cash_ratio'] = portfolio_analysis.cash_weight
        metrics['concentration'] = portfolio_analysis.concentration_ratio
        metrics['avg_holding_period'] = portfolio_analysis.avg_holding_period
        
        return metrics
    
    def _calculate_overall_risk_score(self, violations: List[RiskLimit], metrics: Dict[str, float]) -> float:
        """计算总体风险分数"""
        base_score = 3.0  # 基础分数
        
        # 违规处罚
        violation_penalty = len(violations) * 1.0
        
        # 波动率贡献
        volatility_score = min(metrics.get('volatility', 0) * 10, 3.0)
        
        # 回撤贡献
        drawdown_score = min(metrics.get('max_drawdown', 0) * 20, 3.0)
        
        # 集中度贡献
        concentration_score = min(metrics.get('concentration', 0) * 5, 2.0)
        
        total_score = base_score + violation_penalty + volatility_score + drawdown_score + concentration_score
        
        return min(total_score, 10.0)  # 限制在0-10之间
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """确定风险等级"""
        if risk_score <= 3:
            return RiskLevel.LOW
        elif risk_score <= 6:
            return RiskLevel.MEDIUM
        elif risk_score <= 8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def _generate_risk_recommendations(self, violations: List[RiskLimit], 
                                    metrics: Dict[str, float], 
                                    risk_level: RiskLevel) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        # 基于违规的建议
        for violation in violations:
            if violation.name == "最大回撤":
                recommendations.append(f"回撤过大({violation.current_value:.1%})，建议减仓或设置止损")
            elif "单票仓位" in violation.name:
                recommendations.append(f"{violation.name}超限，建议减仓")
            elif "行业集中度" in violation.name:
                recommendations.append(f"{violation.name}过高，建议分散投资")
            elif violation.name == "现金比例":
                recommendations.append("现金比例不足，建议保留更多现金")
            elif violation.name == "单日亏损":
                recommendations.append("单日亏损超限，建议暂停交易")
        
        # 基于风险指标的建议
        if metrics.get('volatility', 0) > 0.25:
            recommendations.append("组合波动率过高，建议降低风险暴露")
        
        if metrics.get('concentration', 0) > 0.7:
            recommendations.append("组合过于集中，建议分散持仓")
        
        if risk_level == RiskLevel.EXTREME:
            recommendations.append("风险等级极高，强烈建议立即减仓")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("风险等级高，建议降低仓位")
        
        if not recommendations:
            recommendations.append("风险可控，继续保持")
        
        return recommendations
    
    def _record_risk_events(self, violations: List[RiskLimit], account_id: str):
        """记录风险事件"""
        from utils.helpers import generate_id
        
        for violation in violations:
            if violation.is_violated:
                risk_event = RiskEvent(
                    event_id=generate_id("RISK"),
                    event_type="LIMIT_VIOLATION",
                    severity=RiskLevel.HIGH,
                    description=f"{violation.name}超限: {violation.current_value:.1%} > {violation.limit_value:.1%}",
                    trigger_value=violation.current_value,
                    limit_value=violation.limit_value,
                    timestamp=datetime.now(),
                    account_id=account_id
                )
                self.risk_events.append(risk_event)
                logger.warning(f"风险事件记录: {risk_event.description}")
    
    def validate_trade(self, symbol: str, action: str, quantity: int, 
                      price: float, account_id: str = None) -> Tuple[bool, List[str]]:
        """验证交易是否符合风控要求"""
        account_id = account_id or self.account_manager.active_account_id
        issues = []
        
        try:
            account_info = self.account_manager.get_account_info(account_id)
            positions = self.account_manager.get_positions(account_id)
            
            # 1. 检查单票仓位限制
            if action == "BUY":
                trade_value = quantity * price
                new_position_value = trade_value
                if symbol in positions:
                    current_position = positions[symbol]
                    new_position_value += current_position.shares * current_position.current_price
                
                position_weight = new_position_value / account_info['total_value']
                max_position_size = self.config.risk.max_position_size
                
                if position_weight > max_position_size:
                    issues.append(f"买入后{symbol}仓位将达到{position_weight:.1%}，超过限制{max_position_size:.1%}")
            
            # 2. 检查现金是否足够（对于买入）
            if action == "BUY":
                required_cash = quantity * price * (1 + self.config.trading.commission_rate)
                if required_cash > account_info['current_cash']:
                    issues.append(f"资金不足: 需要{required_cash:.2f}，可用{account_info['current_cash']:.2f}")
            
            # 3. 检查持仓是否足够（对于卖出）
            if action == "SELL":
                if symbol not in positions or positions[symbol].shares < quantity:
                    issues.append(f"持仓不足: 需要卖出{quantity}股，当前持仓{positions[symbol].shares if symbol in positions else 0}股")
            
            # 4. 检查当前风险等级
            risk_assessment = self.assess_portfolio_risk(account_id)
            if risk_assessment.overall_risk in [RiskLevel.HIGH, RiskLevel.EXTREME]:
                issues.append(f"当前风险等级{risk_assessment.overall_risk.value}，建议暂停交易")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"交易验证失败: {e}")
            return False, [f"验证失败: {str(e)}"]
    
    def get_position_sizing(self, symbol: str, confidence: float, 
                          stop_loss_pct: float, account_id: str = None) -> int:
        """基于风险计算合适的仓位大小"""
        account_id = account_id or self.account_manager.active_account_id
        account_info = self.account_manager.get_account_info(account_id)
        
        # 获取当前价格
        current_price = self._get_current_price(symbol)
        if current_price is None:
            logger.warning(f"无法获取{symbol}价格，使用默认仓位")
            return 100  # 默认100股
        
        # 基于凯利公式的简化仓位计算
        risk_per_trade = 0.02  # 每笔交易风险2%
        risk_amount = account_info['total_value'] * risk_per_trade
        
        # 调整基于置信度
        confidence_multiplier = min(confidence / 0.7, 1.0)  # 置信度调整
        
        # 计算每股风险
        risk_per_share = current_price * stop_loss_pct
        
        # 计算股数
        shares = int(risk_amount * confidence_multiplier / risk_per_share)
        
        # 确保最小和最大限制
        min_shares = 100
        max_shares = int(account_info['total_value'] * 0.1 / current_price)  # 不超过总资产的10%
        
        shares = max(min_shares, min(shares, max_shares))
        
        logger.debug(f"仓位计算: {symbol} 置信度{confidence:.2f} -> {shares}股")
        
        return shares
    
    def _calculate_current_drawdown(self, account_id: str) -> float:
        """计算当前回撤"""
        # 这里应该从账户历史中计算
        # 目前返回一个模拟值
        return 0.05  # 默认5%回撤
    
    def _calculate_max_drawdown(self, account_id: str) -> float:
        """计算最大回撤"""
        # 这里应该从账户历史中计算
        return 0.08  # 默认8%最大回撤
    
    def _calculate_daily_pnl(self, account_id: str) -> float:
        """计算当日盈亏"""
        # 这里应该从当日交易记录中计算
        return 0.0  # 默认盈亏为0
    
    def _get_historical_returns(self, account_id: str) -> List[float]:
        """获取历史收益率数据"""
        # 这里应该从数据库获取账户历史收益率
        # 目前返回模拟数据
        return [0.001, -0.002, 0.003, -0.001, 0.002, 0.001, -0.003, 0.002, 0.001, -0.001]
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        # 这里应该调用市场数据接口
        # 目前返回模拟价格
        price_map = {
            '000001': 15.0,
            '000002': 25.0,
            '000858': 180.0,
            '600036': 35.0,
            '600519': 1600.0
        }
        return price_map.get(symbol, 10.0)
    
    def check_position_risk(self, symbol: str, current_position: int, 
                       account_info: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """检查持仓风险"""
        total_value = account_info.get("total_value", 0)
        position_value = current_position * market_data.get("current", 0)
        
        risk_assessment = {
            "position_ratio": position_value / total_value if total_value > 0 else 0,
            "concentration_risk": False,
            "liquidity_risk": False,
            "recommended_action": "hold"
        }
        
        # 集中度风险
        if risk_assessment["position_ratio"] > 0.2:
            risk_assessment["concentration_risk"] = True
            risk_assessment["recommended_action"] = "reduce"
            
        # 流动性风险（假设检查）
        if market_data.get("volume", 0) < 1000000:  # 成交量小于100万股
            risk_assessment["liquidity_risk"] = True
            risk_assessment["recommended_action"] = "reduce"
            
        return risk_assessment
    
    def get_risk_report(self, account_id: str = None) -> Dict[str, Any]:
        """生成风险报告"""
        risk_assessment = self.assess_portfolio_risk(account_id)
        account_info = self.account_manager.get_account_info(account_id)
        
        report = {
            'account_id': account_id or self.account_manager.active_account_id,
            'assessment_time': risk_assessment.timestamp.isoformat(),
            'overall_risk': risk_assessment.overall_risk.value,
            'risk_score': risk_assessment.risk_score,
            'account_value': account_info['total_value'],
            'violations': [{
                'name': v.name,
                'current_value': v.current_value,
                'limit_value': v.limit_value,
                'unit': v.unit
            } for v in risk_assessment.violations],
            'risk_metrics': risk_assessment.metrics,
            'recommendations': risk_assessment.recommendations,
            'recent_events': [{
                'event_type': e.event_type,
                'description': e.description,
                'timestamp': e.timestamp.isoformat(),
                'severity': e.severity.value
            } for e in self.risk_events[-10:]]  # 最近10个事件
        }
        
        return report
    
    """
    风险管理器 - 添加assess_trade_risk方法以兼容DecisionMaker
    """
    
    # 现有的初始化和其他方法保持不变...
    
    def assess_trade_risk(self, symbol: str, market_data: Dict[str, Any], 
                        prediction: Dict[str, Any], account_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估交易风险 - 供DecisionMaker调用
        接口兼容Agent架构
        """
        try:
            # 使用现有的风险评估逻辑
            risk_score = 0.0
            risk_reasons = []
            
            # 1. 检查市场波动风险
            volatility_risk = self._assess_volatility_risk_simple(market_data)
            if volatility_risk["high_risk"]:
                risk_score += 0.3
                risk_reasons.append(volatility_risk["reason"])
            
            # 2. 检查仓位集中风险
            concentration_risk = self._assess_concentration_risk_simple(symbol, account_info, market_data)
            if concentration_risk["high_risk"]:
                risk_score += 0.3
                risk_reasons.append(concentration_risk["reason"])
            
            # 3. 检查预测置信度风险
            confidence_risk = self._assess_confidence_risk_simple(prediction)
            if confidence_risk["high_risk"]:
                risk_score += 0.2
                risk_reasons.append(confidence_risk["reason"])
            
            # 4. 检查回撤风险
            drawdown_risk = self._assess_drawdown_risk_simple(account_info)
            if drawdown_risk["high_risk"]:
                risk_score += 0.2
                risk_reasons.append(drawdown_risk["reason"])
            
            # 确定风险等级
            risk_level = self._determine_risk_level_simple(risk_score)
            
            assessment = {
                "symbol": symbol,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "reasons": risk_reasons,
                "can_trade": risk_level != "high",
                "suggested_position_ratio": self._calculate_suggested_position_ratio_simple(risk_score),
                "timestamp": datetime.now().isoformat()
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"交易风险评估失败 {symbol}: {e}")
            return self._get_default_risk_assessment_simple(symbol)

    def _assess_volatility_risk_simple(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """简化版波动率风险评估"""
        try:
            current_price = market_data.get('current_price', 0)
            historical_data = market_data.get('historical_data')
            
            if historical_data is not None and len(historical_data) > 0:
                # 计算近期波动率
                recent_prices = historical_data['Close'].tail(10)
                price_std = recent_prices.std()
                avg_price = recent_prices.mean()
                
                if avg_price > 0:
                    volatility = price_std / avg_price
                    volatility_limit = getattr(self.config.risk, 'volatility_limit', 0.02)
                    if volatility > volatility_limit:
                        return {
                            "high_risk": True,
                            "reason": f"波动率过高: {volatility:.2%}"
                        }
            
            return {"high_risk": False, "reason": "波动率正常"}
            
        except Exception as e:
            logger.warning(f"波动率风险评估失败: {e}")
            return {"high_risk": False, "reason": "波动率评估异常"}

    def _assess_concentration_risk_simple(self, symbol: str, account_info: Dict[str, Any], 
                                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """简化版集中度风险评估"""
        try:
            positions = account_info.get("positions", {})
            total_assets = account_info.get("total_assets", 1)
            
            # 检查该股票是否已经重仓
            current_position = positions.get(symbol, 0)
            current_price = market_data.get('current_price', 10.0)  # 从market_data获取当前价格
            
            position_value = current_position * current_price
            position_ratio = position_value / total_assets if total_assets > 0 else 0
            
            max_position_size = getattr(self.config.risk, 'max_position_size', 0.1)
            if position_ratio > max_position_size:
                return {
                    "high_risk": True,
                    "reason": f"仓位过重: {position_ratio:.2%}"
                }
            
            return {"high_risk": False, "reason": "仓位正常"}
            
        except Exception as e:
            logger.warning(f"集中度风险评估失败: {e}")
            return {"high_risk": False, "reason": "集中度评估异常"}

    def _assess_confidence_risk_simple(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """简化版置信度风险评估"""
        try:
            confidence = prediction.get("confidence", 0.0)
            
            if confidence < 0.6:  # 置信度阈值
                return {
                    "high_risk": True,
                    "reason": f"预测置信度过低: {confidence:.2f}"
                }
            
            return {"high_risk": False, "reason": "置信度正常"}
            
        except Exception as e:
            logger.warning(f"置信度风险评估失败: {e}")
            return {"high_risk": False, "reason": "置信度评估异常"}

    def _assess_drawdown_risk_simple(self, account_info: Dict[str, Any]) -> Dict[str, Any]:
        """简化版回撤风险评估"""
        try:
            total_profit = account_info.get("total_profit", 0)
            peak_assets = account_info.get("peak_assets", account_info.get("total_assets", 1))
            current_assets = account_info.get("total_assets", 1)
            
            if peak_assets > 0:
                current_drawdown = (peak_assets - current_assets) / peak_assets
                max_drawdown = getattr(self.config.risk, 'max_drawdown', 0.05)
                
                if current_drawdown > max_drawdown:
                    return {
                        "high_risk": True,
                        "reason": f"回撤过大: {current_drawdown:.2%}"
                    }
            
            return {"high_risk": False, "reason": "回撤正常"}
            
        except Exception as e:
            logger.warning(f"回撤风险评估失败: {e}")
            return {"high_risk": False, "reason": "回撤评估异常"}

    def _determine_risk_level_simple(self, risk_score: float) -> str:
        """简化版风险等级确定"""
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"

    def _calculate_suggested_position_ratio_simple(self, risk_score: float) -> float:
        """简化版建议仓位比例计算"""
        max_position_size = getattr(self.config.risk, 'max_position_size', 0.1)
        base_ratio = max_position_size
        risk_adjusted_ratio = base_ratio * (1 - risk_score)
        return max(risk_adjusted_ratio, 0.01)  # 至少1%

    def _get_default_risk_assessment_simple(self, symbol: str) -> Dict[str, Any]:
        """简化版默认风险评估"""
        return {
            "symbol": symbol,
            "risk_score": 1.0,  # 最高风险
            "risk_level": "high",
            "reasons": ["风险评估异常"],
            "can_trade": False,
            "suggested_position_ratio": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    
class StopLossManager:
    """止损管理器"""
    
    def __init__(self, risk_manager: RiskManager, order_executor):
        self.risk_manager = risk_manager
        self.order_executor = order_executor
        self.config = get_system_config()
        self.active_stops: Dict[str, Dict] = {}
    
    def set_stop_loss(self, symbol: str, stop_price: float, 
                     account_id: str = None, reason: str = "") -> str:
        """设置止损单"""
        account_id = account_id or self.risk_manager.account_manager.active_account_id
        positions = self.risk_manager.account_manager.get_positions(account_id)
        
        if symbol not in positions:
            logger.warning(f"无法设置止损，没有{symbol}持仓")
            return ""
        
        position = positions[symbol]
        current_price = position.current_price
        
        # 验证止损价格合理性
        if stop_price >= current_price:
            logger.warning(f"止损价格{stop_price}高于当前价格{current_price}")
            return ""
        
        # 创建止损单
        stop_order = {
            'symbol': symbol,
            'stop_price': stop_price,
            'position_shares': position.shares,
            'account_id': account_id,
            'set_time': datetime.now(),
            'reason': reason,
            'original_cost': position.avg_cost
        }
        
        stop_id = f"STOP_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.active_stops[stop_id] = stop_order
        
        logger.info(f"设置止损: {symbol} @ {stop_price}, 理由: {reason}")
        
        return stop_id
    
    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[str]:
        """检查并触发止损"""
        triggered_stops = []
        
        for stop_id, stop_order in list(self.active_stops.items()):
            symbol = stop_order['symbol']
            
            if symbol in current_prices and current_prices[symbol] <= stop_order['stop_price']:
                # 触发止损
                if self._execute_stop_loss(stop_id, stop_order, current_prices[symbol]):
                    triggered_stops.append(stop_id)
                    logger.info(f"止损触发: {symbol} @ {current_prices[symbol]}")
        
        return triggered_stops
    
    def _execute_stop_loss(self, stop_id: str, stop_order: Dict, current_price: float) -> bool:
        """执行止损"""
        try:
            # 创建卖出订单
            order_request = {
                'account_id': stop_order['account_id'],
                'symbol': stop_order['symbol'],
                'action': 'SELL',
                'quantity': stop_order['position_shares'],
                'order_type': 'market',
                'reason': f"止损触发: {stop_order['reason']}",
                'strategy': 'stop_loss'
            }
            
            order = self.order_executor.place_order(order_request)
            
            if order.status.value in ['filled', 'partial']:
                # 移除止损单
                del self.active_stops[stop_id]
                return True
            else:
                logger.error(f"止损单执行失败: {order.reason}")
                return False
                
        except Exception as e:
            logger.error(f"执行止损失败: {e}")
            return False
    
    def get_active_stops(self, account_id: str = None) -> Dict[str, Dict]:
        """获取活跃止损单"""
        if account_id:
            return {stop_id: stop_order for stop_id, stop_order in self.active_stops.items()
                   if stop_order['account_id'] == account_id}
        else:
            return self.active_stops.copy()
    
    def remove_stop_loss(self, stop_id: str) -> bool:
        """移除止损单"""
        if stop_id in self.active_stops:
            del self.active_stops[stop_id]
            logger.info(f"移除止损单: {stop_id}")
            return True
        else:
            logger.warning(f"止损单不存在: {stop_id}")
            return False