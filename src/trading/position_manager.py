# src/trading/position_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

from utils.config_loader import get_system_config
from utils.helpers import timer, singleton
from trading.account_manager import AccountManager, Position

logger = logging.getLogger(__name__)

@dataclass
class PositionAnalysis:
    """持仓分析结果"""
    symbol: str
    current_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    holding_days: int
    weight: float
    sector: str = ""
    risk_score: float = 0.0

@dataclass
class PortfolioAnalysis:
    """投资组合分析"""
    total_value: float
    cash_weight: float
    stock_weight: float
    sector_exposure: Dict[str, float]
    concentration_ratio: float  # 前5大持仓集中度
    avg_holding_period: float
    total_unrealized_pnl: float
    daily_pnl: float

@singleton
class PositionManager:
    """
    持仓管理器
    负责持仓监控、分析和风险管理
    """
    
    def __init__(self, account_manager: AccountManager):
        self.account_manager = account_manager
        self.config = get_system_config()
        self.position_history: Dict[str, List] = {}
        self.analysis_cache = {}
        
        logger.info("持仓管理器初始化完成")
    
    def update_position_prices(self, price_updates: Dict[str, float], account_id: str = None):
        """更新持仓价格"""
        account_id = account_id or self.account_manager.active_account_id
        positions = self.account_manager.get_positions(account_id)
        
        for symbol, price in price_updates.items():
            if symbol in positions:
                self.account_manager.update_position_price(symbol, price, account_id)
        
        logger.debug(f"更新了 {len(price_updates)} 个持仓价格")
    
    def analyze_position(self, symbol: str, account_id: str = None) -> PositionAnalysis:
        """分析单个持仓"""
        account_id = account_id or self.account_manager.active_account_id
        positions = self.account_manager.get_positions(account_id)
        
        if symbol not in positions:
            logger.warning(f"持仓不存在: {symbol}")
            return None
        
        position = positions[symbol]
        account_info = self.account_manager.get_account_info(account_id)
        
        # 计算持仓权重
        weight = position.market_value / account_info['total_value'] if account_info['total_value'] > 0 else 0
        
        # 计算持有天数（简化实现）
        holding_days = self._calculate_holding_days(symbol, account_id)
        
        analysis = PositionAnalysis(
            symbol=symbol,
            current_value=position.market_value,
            cost_basis=position.shares * position.avg_cost,
            unrealized_pnl=position.unrealized_pnl,
            unrealized_pnl_pct=position.unrealized_pnl_pct,
            holding_days=holding_days,
            weight=weight,
            sector=self._get_stock_sector(symbol),
            risk_score=self._calculate_position_risk(symbol, position)
        )
        
        return analysis
    
    def analyze_portfolio(self, account_id: str = None) -> PortfolioAnalysis:
        """分析整个投资组合"""
        account_id = account_id or self.account_manager.active_account_id
        account_info = self.account_manager.get_account_info(account_id)
        positions = self.account_manager.get_positions(account_id)
        
        total_value = account_info['total_value']
        cash = account_info['current_cash']
        
        # 计算权重
        cash_weight = cash / total_value if total_value > 0 else 1.0
        stock_weight = 1.0 - cash_weight
        
        # 计算行业暴露
        sector_exposure = {}
        for symbol, position in positions.items():
            sector = self._get_stock_sector(symbol)
            sector_weight = position.market_value / total_value
            sector_exposure[sector] = sector_exposure.get(sector, 0) + sector_weight
        
        # 计算集中度
        position_values = [pos.market_value for pos in positions.values()]
        position_values.sort(reverse=True)
        top5_value = sum(position_values[:5])
        concentration_ratio = top5_value / total_value if total_value > 0 else 0
        
        # 计算平均持有期
        holding_periods = [self._calculate_holding_days(symbol, account_id) 
                          for symbol in positions.keys()]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # 计算总盈亏
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        
        analysis = PortfolioAnalysis(
            total_value=total_value,
            cash_weight=cash_weight,
            stock_weight=stock_weight,
            sector_exposure=sector_exposure,
            concentration_ratio=concentration_ratio,
            avg_holding_period=avg_holding_period,
            total_unrealized_pnl=total_unrealized_pnl,
            daily_pnl=0.0  # 需要额外计算
        )
        
        return analysis
    
    def _calculate_holding_days(self, symbol: str, account_id: str) -> int:
        """计算持有天数（简化实现）"""
        # 这里应该从交易历史中计算实际持有天数
        # 目前返回一个固定值
        return 30  # 默认30天
    
    def _get_stock_sector(self, symbol: str) -> str:
        """获取股票行业（简化实现）"""
        # 这里应该从数据层获取股票行业信息
        # 目前基于股票代码简单分类
        sector_map = {
            '000001': '银行',
            '000002': '房地产', 
            '000858': '食品饮料',
            '600036': '银行',
            '600519': '食品饮料',
            '601318': '保险'
        }
        return sector_map.get(symbol, '其他')
    
    def _calculate_position_risk(self, symbol: str, position: Position) -> float:
        """计算持仓风险评分"""
        # 基于波动率、仓位权重等的简单风险评分
        volatility = self._get_stock_volatility(symbol)
        risk_score = volatility * position.unrealized_pnl_pct * 10
        
        return min(max(risk_score, 0), 10)  # 限制在0-10之间
    
    def _get_stock_volatility(self, symbol: str) -> float:
        """获取股票波动率（简化实现）"""
        # 这里应该从数据层获取历史波动率
        volatility_map = {
            '000001': 0.02,
            '000002': 0.03,
            '000858': 0.025,
            '600036': 0.018,
            '600519': 0.022,
            '601318': 0.028
        }
        return volatility_map.get(symbol, 0.02)
    
    def get_risk_assessment(self, account_id: str = None) -> Dict[str, Any]:
        """获取风险评估"""
        portfolio_analysis = self.analyze_portfolio(account_id)
        positions_analysis = {}
        
        account_id = account_id or self.account_manager.active_account_id
        positions = self.account_manager.get_positions(account_id)
        
        for symbol in positions.keys():
            positions_analysis[symbol] = self.analyze_position(symbol, account_id)
        
        # 计算总体风险指标
        total_risk_score = np.mean([pa.risk_score for pa in positions_analysis.values()]) if positions_analysis else 0
        max_single_risk = max([pa.risk_score for pa in positions_analysis.values()]) if positions_analysis else 0
        
        risk_assessment = {
            'overall_risk_score': total_risk_score,
            'max_position_risk': max_single_risk,
            'sector_concentration': portfolio_analysis.concentration_ratio,
            'cash_ratio': portfolio_analysis.cash_weight,
            'high_risk_positions': [
                symbol for symbol, analysis in positions_analysis.items() 
                if analysis.risk_score > 7
            ],
            'recommendations': self._generate_risk_recommendations(portfolio_analysis, positions_analysis)
        }
        
        return risk_assessment
    
    def _generate_risk_recommendations(self, portfolio: PortfolioAnalysis, 
                                    positions: Dict[str, PositionAnalysis]) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        # 检查现金比例
        if portfolio.cash_weight < 0.1:
            recommendations.append("现金比例过低，建议保持10%以上现金")
        
        # 检查集中度
        if portfolio.concentration_ratio > 0.6:
            recommendations.append("持仓过于集中，建议分散投资")
        
        # 检查高风险持仓
        high_risk_count = len([p for p in positions.values() if p.risk_score > 7])
        if high_risk_count > 3:
            recommendations.append(f"高风险持仓过多({high_risk_count}个)，建议减少")
        
        # 检查行业集中度
        for sector, exposure in portfolio.sector_exposure.items():
            if exposure > 0.3:
                recommendations.append(f"{sector}行业暴露过高({exposure:.1%})")
        
        if not recommendations:
            recommendations.append("投资组合风险可控")
        
        return recommendations
    
    def suggest_rebalancing(self, target_weights: Dict[str, float], 
                          account_id: str = None) -> List[Dict[str, Any]]:
        """生成再平衡建议"""
        account_id = account_id or self.account_manager.active_account_id
        current_positions = self.account_manager.get_positions(account_id)
        account_info = self.account_manager.get_account_info(account_id)
        
        total_value = account_info['total_value']
        rebalancing_actions = []
        
        # 计算当前权重
        current_weights = {}
        for symbol, position in current_positions.items():
            current_weights[symbol] = position.market_value / total_value
        
        # 生成调整建议
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # 权重差异超过1%才调整
                target_value = total_value * target_weight
                current_value = total_value * current_weight
                adjustment_value = target_value - current_value
                
                if adjustment_value > 0:
                    # 需要买入
                    action = "BUY"
                    shares = int(adjustment_value / current_positions[symbol].current_price) if symbol in current_positions else int(adjustment_value / 10)  # 假设价格10元
                else:
                    # 需要卖出
                    action = "SELL" 
                    shares = int(abs(adjustment_value) / current_positions[symbol].current_price)
                
                rebalancing_actions.append({
                    'symbol': symbol,
                    'action': action,
                    'shares': shares,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'adjustment_value': abs(adjustment_value)
                })
        
        return rebalancing_actions
    
    def calculate_tax_implications(self, account_id: str = None) -> Dict[str, Any]:
        """计算税务影响（针对卖出决策）"""
        account_id = account_id or self.account_manager.active_account_id
        positions = self.account_manager.get_positions(account_id)
        
        tax_analysis = {}
        
        for symbol, position in positions.items():
            holding_days = self._calculate_holding_days(symbol, account_id)
            
            # 简单税务计算（基于中国A股规则）
            if holding_days <= 30:
                tax_rate = 0.2  # 短期持有税率
            else:
                tax_rate = 0.1  # 长期持有税率
            
            potential_tax = position.unrealized_pnl * tax_rate if position.unrealized_pnl > 0 else 0
            
            tax_analysis[symbol] = {
                'holding_days': holding_days,
                'tax_rate': tax_rate,
                'potential_tax': potential_tax,
                'after_tax_pnl': position.unrealized_pnl - potential_tax
            }
        
        return tax_analysis
    
    def monitor_position_limits(self, account_id: str = None) -> List[Dict[str, Any]]:
        """监控持仓限制"""
        account_id = account_id or self.account_manager.active_account_id
        portfolio_analysis = self.analyze_portfolio(account_id)
        positions_analysis = {}
        
        positions = self.account_manager.get_positions(account_id)
        for symbol in positions.keys():
            positions_analysis[symbol] = self.analyze_position(symbol, account_id)
        
        violations = []
        
        # 检查单票仓位限制
        max_position_size = self.config.risk.max_position_size
        for symbol, analysis in positions_analysis.items():
            if analysis.weight > max_position_size:
                violations.append({
                    'type': 'POSITION_SIZE',
                    'symbol': symbol,
                    'current_weight': analysis.weight,
                    'limit': max_position_size,
                    'message': f'{symbol}仓位({analysis.weight:.1%})超过限制({max_position_size:.1%})'
                })
        
        # 检查行业暴露限制
        max_sector_exposure = 0.3  # 最大行业暴露30%
        for sector, exposure in portfolio_analysis.sector_exposure.items():
            if exposure > max_sector_exposure:
                violations.append({
                    'type': 'SECTOR_EXPOSURE',
                    'sector': sector,
                    'current_exposure': exposure,
                    'limit': max_sector_exposure,
                    'message': f'{sector}行业暴露({exposure:.1%})超过限制({max_sector_exposure:.1%})'
                })
        
        return violations

class PositionOptimizer:
    """
    持仓优化器
    基于现代投资组合理论优化持仓
    """
    
    def __init__(self, position_manager: PositionManager):
        self.position_manager = position_manager
    
    def optimize_portfolio(self, target_return: float = 0.1, 
                         risk_free_rate: float = 0.02) -> Dict[str, float]:
        """优化投资组合（马科维茨模型简化版）"""
        # 这里应该实现完整的马科维茨投资组合优化
        # 目前返回一个简单的等权重组合
        
        account_id = self.position_manager.account_manager.active_account_id
        positions = self.position_manager.account_manager.get_positions(account_id)
        
        if not positions:
            return {}
        
        # 简单等权重分配
        symbols = list(positions.keys())
        equal_weight = 1.0 / len(symbols)
        
        optimized_weights = {symbol: equal_weight for symbol in symbols}
        
        logger.info(f"投资组合优化完成: {len(symbols)} 只股票，等权重分配")
        
        return optimized_weights
    
    def calculate_efficient_frontier(self, target_returns: List[float]) -> Dict[str, Any]:
        """计算有效前沿"""
        # 这里应该实现有效前沿计算
        # 目前返回模拟数据
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            portfolio = {
                'target_return': target_return,
                'expected_risk': target_return * 0.3,  # 模拟风险
                'weights': {},  # 实际应该计算最优权重
                'sharpe_ratio': (target_return - 0.02) / (target_return * 0.3)  # 模拟夏普比率
            }
            efficient_portfolios.append(portfolio)
        
        return {
            'efficient_portfolios': efficient_portfolios,
            'max_sharpe_portfolio': max(efficient_portfolios, key=lambda x: x['sharpe_ratio']),
            'min_variance_portfolio': min(efficient_portfolios, key=lambda x: x['expected_risk'])
        }