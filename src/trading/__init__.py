# src/trading/__init__.py
"""
交易层模块
负责交易执行、风险管理和投资组合管理
"""

from .account_manager import AccountManager, AccountType, MultiAccountManager
from .order_executor import OrderExecutor, OrderType, OrderStatus, SmartOrderRouter
from .position_manager import PositionManager, PositionOptimizer
from .risk_manager import RiskManager, StopLossManager, RiskLevel
from .portfolio import PortfolioManager, MultiPortfolioOptimizer
from .broker_interface import BrokerManager, BrokerFactory, BrokerType

__all__ = [
    # 账户管理
    'AccountManager', 'AccountType', 'MultiAccountManager',
    
    # 订单执行
    'OrderExecutor', 'OrderType', 'OrderStatus', 'SmartOrderRouter',
    
    # 持仓管理
    'PositionManager', 'PositionOptimizer',
    
    # 风险管理
    'RiskManager', 'StopLossManager', 'RiskLevel',
    
    # 投资组合管理
    'PortfolioManager', 'MultiPortfolioOptimizer',
    
    # 券商接口
    'BrokerManager', 'BrokerFactory', 'BrokerType'
]