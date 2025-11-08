# src/trading/portfolio.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import json

from utils.config_loader import get_system_config
from utils.helpers import timer, singleton, generate_id
from utils.performance import PerformanceCalculator, BenchmarkComparator
from trading.account_manager import AccountManager
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)

@dataclass
class PortfolioConfig:
    """投资组合配置"""
    name: str
    strategy: str
    target_weights: Dict[str, float]
    rebalance_frequency: str  # daily, weekly, monthly
    max_tracking_error: float = 0.05
    enabled: bool = True

@dataclass
class RebalanceAction:
    """再平衡操作"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    current_shares: int
    target_shares: int
    shares_diff: int
    current_weight: float
    target_weight: float
    current_value: float
    target_value: float

@dataclass
class PortfolioSnapshot:
    """投资组合快照"""
    portfolio_id: str
    timestamp: datetime
    total_value: float
    cash_weight: float
    stock_weight: float
    sector_allocation: Dict[str, float]
    performance_metrics: Dict[str, float]
    positions: Dict[str, Any]

@singleton
class PortfolioManager:
    """
    投资组合管理器
    管理多策略投资组合，执行再平衡和绩效监控
    """
    
    def __init__(self, account_manager: AccountManager, 
                 position_manager: PositionManager,
                 risk_manager: RiskManager):
        self.account_manager = account_manager
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.config = get_system_config()
        
        self.portfolios: Dict[str, PortfolioConfig] = {}
        self.performance_calculator = PerformanceCalculator()
        self.snapshots: List[PortfolioSnapshot] = []
        self.rebalance_history: List[Dict] = []
        
        self._initialize_default_portfolio()
        logger.info("投资组合管理器初始化完成")
    
    def _initialize_default_portfolio(self):
        """初始化默认投资组合"""
        default_portfolio = PortfolioConfig(
            name="默认组合",
            strategy="均衡配置",
            target_weights={
                '000001': 0.2,  # 平安银行
                '000002': 0.15, # 万科A
                '000858': 0.15, # 五粮液
                '600036': 0.2,  # 招商银行
                '600519': 0.1,  # 贵州茅台
                'CASH': 0.2     # 现金
            },
            rebalance_frequency="monthly"
        )
        
        self.create_portfolio("default", default_portfolio)
    
    def create_portfolio(self, portfolio_id: str, portfolio_config: PortfolioConfig) -> str:
        """创建投资组合"""
        self.portfolios[portfolio_id] = portfolio_config
        logger.info(f"创建投资组合: {portfolio_id} - {portfolio_config.name}")
        return portfolio_id
    
    def delete_portfolio(self, portfolio_id: str) -> bool:
        """删除投资组合"""
        if portfolio_id in self.portfolios:
            del self.portfolios[portfolio_id]
            logger.info(f"删除投资组合: {portfolio_id}")
            return True
        return False
    
    def rebalance_portfolio(self, portfolio_id: str, account_id: str = None) -> List[RebalanceAction]:
        """执行投资组合再平衡"""
        if portfolio_id not in self.portfolios:
            logger.error(f"投资组合不存在: {portfolio_id}")
            return []
        
        portfolio_config = self.portfolios[portfolio_id]
        account_id = account_id or self.account_manager.active_account_id
        
        # 检查是否到达再平衡时间
        if not self._should_rebalance(portfolio_id, account_id):
            logger.info(f"投资组合 {portfolio_id} 未到再平衡时间")
            return []
        
        # 获取当前持仓
        current_positions = self.account_manager.get_positions(account_id)
        account_info = self.account_manager.get_account_info(account_id)
        total_value = account_info['total_value']
        
        # 生成再平衡建议
        rebalance_actions = self._calculate_rebalance_actions(
            portfolio_config, current_positions, total_value, account_id
        )
        
        # 执行再平衡操作
        executed_actions = self._execute_rebalance_actions(rebalance_actions, account_id, portfolio_id)
        
        # 记录再平衡历史
        self._record_rebalance(portfolio_id, account_id, executed_actions)
        
        logger.info(f"投资组合再平衡完成: {portfolio_id}, 执行 {len(executed_actions)} 个操作")
        
        return executed_actions
    
    def _should_rebalance(self, portfolio_id: str, account_id: str) -> bool:
        """检查是否应该执行再平衡"""
        portfolio_config = self.portfolios[portfolio_id]
        
        # 检查再平衡频率
        last_rebalance = self._get_last_rebalance_time(portfolio_id, account_id)
        if not last_rebalance:
            return True
        
        current_time = datetime.now()
        time_since_rebalance = current_time - last_rebalance
        
        if portfolio_config.rebalance_frequency == "daily":
            return time_since_rebalance.days >= 1
        elif portfolio_config.rebalance_frequency == "weekly":
            return time_since_rebalance.days >= 7
        elif portfolio_config.rebalance_frequency == "monthly":
            return time_since_rebalance.days >= 30
        else:
            return False
    
    def _get_last_rebalance_time(self, portfolio_id: str, account_id: str) -> Optional[datetime]:
        """获取上次再平衡时间"""
        for record in reversed(self.rebalance_history):
            if (record['portfolio_id'] == portfolio_id and 
                record['account_id'] == account_id):
                return record['timestamp']
        return None
    
    def _calculate_rebalance_actions(self, portfolio_config: PortfolioConfig,
                                  current_positions: Dict[str, Any],
                                  total_value: float,
                                  account_id: str) -> List[RebalanceAction]:
        """计算再平衡操作"""
        actions = []
        cash = self.account_manager.get_account_info(account_id)['current_cash']
        
        # 计算现金权重
        cash_weight = cash / total_value if total_value > 0 else 0
        
        # 处理现金目标
        if 'CASH' in portfolio_config.target_weights:
            target_cash_weight = portfolio_config.target_weights['CASH']
            cash_diff = target_cash_weight - cash_weight
            
            if abs(cash_diff) > 0.01:  # 现金权重差异超过1%
                # 计算需要调整的总金额
                adjustment_amount = cash_diff * total_value
                
                # 如果需要增加现金，卖出股票；如果需要减少现金，买入股票
                if adjustment_amount > 0:
                    # 需要卖出股票来增加现金
                    sell_actions = self._calculate_sell_actions(
                        current_positions, adjustment_amount, portfolio_config.target_weights
                    )
                    actions.extend(sell_actions)
                else:
                    # 需要买入股票来减少现金
                    buy_actions = self._calculate_buy_actions(
                        current_positions, -adjustment_amount, portfolio_config.target_weights, total_value
                    )
                    actions.extend(buy_actions)
        
        # 处理个股权重调整（不考虑现金）
        stock_target_weights = {k: v for k, v in portfolio_config.target_weights.items() if k != 'CASH'}
        total_stock_weight = sum(stock_target_weights.values())
        
        # 标准化权重
        if total_stock_weight > 0:
            stock_target_weights = {k: v/total_stock_weight for k, v in stock_target_weights.items()}
        
        for symbol, target_weight in stock_target_weights.items():
            current_weight = 0
            current_shares = 0
            current_value = 0
            
            if symbol in current_positions:
                position = current_positions[symbol]
                current_value = position.market_value
                current_weight = current_value / total_value
                current_shares = position.shares
            
            # 计算目标价值和股数
            target_value = target_weight * total_value * (1 - portfolio_config.target_weights.get('CASH', 0))
            target_shares = int(target_value / current_positions[symbol].current_price) if symbol in current_positions else 0
            
            # 计算差异
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff) > 0:  # 有调整需要
                action_type = "BUY" if shares_diff > 0 else "SELL"
                
                rebalance_action = RebalanceAction(
                    symbol=symbol,
                    action=action_type,
                    current_shares=current_shares,
                    target_shares=target_shares,
                    shares_diff=abs(shares_diff),
                    current_weight=current_weight,
                    target_weight=target_weight,
                    current_value=current_value,
                    target_value=target_value
                )
                
                actions.append(rebalance_action)
        
        return actions
    
    def _calculate_sell_actions(self, current_positions: Dict[str, Any],
                              sell_amount: float,
                              target_weights: Dict[str, float]) -> List[RebalanceAction]:
        """计算卖出操作"""
        actions = []
        remaining_amount = sell_amount
        
        # 按权重过高程度排序，优先卖出超配最多的股票
        over_weight_stocks = []
        
        for symbol, position in current_positions.items():
            if symbol in target_weights:
                current_weight = position.market_value / sum(p.market_value for p in current_positions.values())
                over_weight = current_weight - target_weights[symbol]
                if over_weight > 0:
                    over_weight_stocks.append((symbol, over_weight, position))
        
        # 按超配程度降序排序
        over_weight_stocks.sort(key=lambda x: x[1], reverse=True)
        
        for symbol, over_weight, position in over_weight_stocks:
            if remaining_amount <= 0:
                break
            
            # 计算可卖出金额（不超过超配部分）
            max_sell_value = over_weight * sum(p.market_value for p in current_positions.values())
            sell_value = min(remaining_amount, max_sell_value)
            sell_shares = int(sell_value / position.current_price)
            
            if sell_shares > 0:
                action = RebalanceAction(
                    symbol=symbol,
                    action="SELL",
                    current_shares=position.shares,
                    target_shares=position.shares - sell_shares,
                    shares_diff=sell_shares,
                    current_weight=position.market_value / sum(p.market_value for p in current_positions.values()),
                    target_weight=target_weights[symbol],
                    current_value=position.market_value,
                    target_value=position.market_value - sell_value
                )
                
                actions.append(action)
                remaining_amount -= sell_value
        
        return actions
    
    def _calculate_buy_actions(self, current_positions: Dict[str, Any],
                             buy_amount: float,
                             target_weights: Dict[str, float],
                             total_value: float) -> List[RebalanceAction]:
        """计算买入操作"""
        actions = []
        remaining_amount = buy_amount
        
        # 按权重不足程度排序，优先买入低配最多的股票
        under_weight_stocks = []
        
        for symbol, target_weight in target_weights.items():
            if symbol != 'CASH':
                current_weight = 0
                if symbol in current_positions:
                    current_weight = current_positions[symbol].market_value / total_value
                
                under_weight = target_weight - current_weight
                if under_weight > 0:
                    under_weight_stocks.append((symbol, under_weight, target_weight))
        
        # 按低配程度降序排序
        under_weight_stocks.sort(key=lambda x: x[1], reverse=True)
        
        for symbol, under_weight, target_weight in under_weight_stocks:
            if remaining_amount <= 0:
                break
            
            # 计算可买入金额（不超过低配部分）
            buy_value = min(remaining_amount, under_weight * total_value)
            
            # 获取当前价格
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue
            
            buy_shares = int(buy_value / current_price)
            
            if buy_shares > 0:
                current_shares = current_positions[symbol].shares if symbol in current_positions else 0
                current_value = current_positions[symbol].market_value if symbol in current_positions else 0
                
                action = RebalanceAction(
                    symbol=symbol,
                    action="BUY",
                    current_shares=current_shares,
                    target_shares=current_shares + buy_shares,
                    shares_diff=buy_shares,
                    current_weight=current_value / total_value,
                    target_weight=target_weight,
                    current_value=current_value,
                    target_value=current_value + buy_value
                )
                
                actions.append(action)
                remaining_amount -= buy_value
        
        return actions
    
    def _execute_rebalance_actions(self, rebalance_actions: List[RebalanceAction],
                                 account_id: str, portfolio_id: str) -> List[RebalanceAction]:
        """执行再平衡操作"""
        from .order_executor import OrderExecutor
        
        executed_actions = []
        order_executor = OrderExecutor(self.account_manager)
        
        for action in rebalance_actions:
            if action.shares_diff == 0:
                continue
            
            # 验证交易风险
            current_price = self._get_current_price(action.symbol)
            is_valid, issues = self.risk_manager.validate_trade(
                action.symbol, action.action, action.shares_diff, current_price, account_id
            )
            
            if not is_valid:
                logger.warning(f"再平衡交易被风控拒绝: {action.symbol} {action.action} {action.shares_diff} - {issues}")
                continue
            
            # 执行交易
            order_request = {
                'account_id': account_id,
                'symbol': action.symbol,
                'action': action.action,
                'quantity': action.shares_diff,
                'order_type': 'market',
                'reason': f"投资组合再平衡: {portfolio_id}",
                'strategy': 'portfolio_rebalance'
            }
            
            order = order_executor.place_order(order_request)
            
            if order.status.value in ['filled', 'partial']:
                executed_actions.append(action)
                logger.info(f"再平衡交易执行: {action.symbol} {action.action} {action.shares_diff}")
            else:
                logger.warning(f"再平衡交易执行失败: {action.symbol} - {order.reason}")
        
        return executed_actions
    
    def _record_rebalance(self, portfolio_id: str, account_id: str, actions: List[RebalanceAction]):
        """记录再平衡操作"""
        rebalance_record = {
            'rebalance_id': generate_id("REB"),
            'portfolio_id': portfolio_id,
            'account_id': account_id,
            'timestamp': datetime.now(),
            'actions_count': len(actions),
            'actions_details': [{
                'symbol': action.symbol,
                'action': action.action,
                'shares': action.shares_diff
            } for action in actions]
        }
        
        self.rebalance_history.append(rebalance_record)
    
    def take_portfolio_snapshot(self, portfolio_id: str, account_id: str = None) -> PortfolioSnapshot:
        """记录投资组合快照"""
        account_id = account_id or self.account_manager.active_account_id
        
        portfolio_analysis = self.position_manager.analyze_portfolio(account_id)
        account_info = self.account_manager.get_account_info(account_id)
        performance_metrics = self.performance_calculator.calculate_all_metrics()
        
        snapshot = PortfolioSnapshot(
            portfolio_id=portfolio_id,
            timestamp=datetime.now(),
            total_value=account_info['total_value'],
            cash_weight=portfolio_analysis.cash_weight,
            stock_weight=portfolio_analysis.stock_weight,
            sector_allocation=portfolio_analysis.sector_exposure,
            performance_metrics=performance_metrics,
            positions=self.account_manager.get_positions(account_id)
        )
        
        self.snapshots.append(snapshot)
        logger.debug(f"投资组合快照记录: {portfolio_id}")
        
        return snapshot
    
    def get_portfolio_performance(self, portfolio_id: str, account_id: str = None) -> Dict[str, Any]:
        """获取投资组合绩效"""
        account_id = account_id or self.account_manager.active_account_id
        
        # 获取基本账户信息
        account_info = self.account_manager.get_account_info(account_id)
        performance_metrics = self.performance_calculator.calculate_all_metrics()
        
        # 获取风险评估
        risk_assessment = self.risk_manager.assess_portfolio_risk(account_id)
        
        # 获取再平衡历史
        rebalance_history = [r for r in self.rebalance_history 
                           if r['portfolio_id'] == portfolio_id and r['account_id'] == account_id]
        
        performance_report = {
            'portfolio_id': portfolio_id,
            'account_id': account_id,
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': {
                'total_value': account_info['total_value'],
                'total_return': performance_metrics.get('total_return', 0),
                'annual_return': performance_metrics.get('annual_return', 0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0)
            },
            'risk_metrics': {
                'risk_level': risk_assessment.overall_risk.value,
                'risk_score': risk_assessment.risk_score,
                'volatility': risk_assessment.metrics.get('volatility', 0),
                'var_95': risk_assessment.metrics.get('var_95', 0)
            },
            'rebalance_info': {
                'total_rebalances': len(rebalance_history),
                'last_rebalance': rebalance_history[-1]['timestamp'] if rebalance_history else None,
                'rebalance_frequency': self.portfolios[portfolio_id].rebalance_frequency
            },
            'current_allocation': self.position_manager.analyze_portfolio(account_id).__dict__
        }
        
        return performance_report
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        # 这里应该调用市场数据接口
        price_map = {
            '000001': 15.0,
            '000002': 25.0, 
            '000858': 180.0,
            '600036': 35.0,
            '600519': 1600.0
        }
        return price_map.get(symbol, 10.0)

class MultiPortfolioOptimizer:
    """多组合优化器"""
    
    def __init__(self, portfolio_manager: PortfolioManager):
        self.portfolio_manager = portfolio_manager
    
    def optimize_across_portfolios(self, target_returns: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """跨组合优化资产配置"""
        # 这里应该实现跨组合优化算法
        # 目前返回简单的等比例分配
        
        optimized_allocations = {}
        total_target_return = sum(target_returns.values())
        
        for portfolio_id, target_return in target_returns.items():
            if portfolio_id in self.portfolio_manager.portfolios:
                # 简单的按目标回报比例分配
                weight = target_return / total_target_return if total_target_return > 0 else 1.0 / len(target_returns)
                
                # 获取当前组合配置
                current_config = self.portfolio_manager.portfolios[portfolio_id]
                optimized_allocations[portfolio_id] = {
                    symbol: weight * current_weight 
                    for symbol, current_weight in current_config.target_weights.items()
                }
        
        return optimized_allocations
    
    def calculate_correlation_matrix(self, portfolio_ids: List[str]) -> pd.DataFrame:
        """计算组合间相关性矩阵"""
        # 这里应该基于历史收益率计算组合间相关性
        # 目前返回模拟数据
        
        n_portfolios = len(portfolio_ids)
        correlation_matrix = np.eye(n_portfolios)  # 初始为单位矩阵
        
        # 添加一些随机相关性（模拟）
        for i in range(n_portfolios):
            for j in range(i+1, n_portfolios):
                correlation = np.random.uniform(-0.3, 0.7)  # 随机相关性
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return pd.DataFrame(correlation_matrix, index=portfolio_ids, columns=portfolio_ids)