# __init__.py
# This is the src package initialization file

# src/data/__init__.py
from .data import DataCollector
from .data import DataProcessor
from .data import FeatureEngineer
from .data import MarketData
from .data import DatabaseManager

# src/utils/__init__.py
from .utils import ConfigLoader, get_system_config, get_feature_config, update_system_config
from .utils import setup_logging, get_logger, PerformanceLogger, DebugLogger, initialize_logging, debug_logger
from .utils import PerformanceCalculator, RiskMetrics, BenchmarkComparator
from .utils import TradingVisualizer, RealTimeMonitor
from .utils import (timer, retry, singleton, generate_id, calculate_hash, 
                     safe_divide, normalize_data, format_currency, format_percentage,
                     parse_date, get_trading_days, calculate_position_size,
                     detect_anomalies, DataValidator, MemoryManager, setup_environment)
from .utils import (TradingDataValidator, ModelInputValidator, 
                        TradingDecisionValidator, ConfigurationValidator)

# src/core/__init__.py
from .core import BaseAgent
from .core import EventEngine, Event
from .core import StateManager

# src/agent/__init__.py
from .agent import TradingAgent
from .agent import DecisionMaker
from .agent import LearningModule
from .agent import StrategyEngine
from .agent import MemorySystem
from .agent import InvestmentReportAgent

# src/model/__init__.py
from .models import ModelTrainer
from .models import ModelPredictor

# src/trading/__init__.py
from .trading import AccountManager, AccountType, MultiAccountManager
from .trading import OrderExecutor, OrderType, OrderStatus, SmartOrderRouter
from .trading import PositionManager, PositionOptimizer
from .trading import RiskManager, StopLossManager, RiskLevel
from .trading import PortfolioManager, MultiPortfolioOptimizer
from .trading import BrokerManager, BrokerFactory, BrokerType

__all__ = [

    # src/data/__init__.py
    'DataCollector', 'DataProcessor', 'FeatureEngineer', 'MarketData', 'DatabaseManager',

    # src/utils/__init__.py
    # 配置管理
    'ConfigLoader', 'get_system_config', 'get_feature_config', 'update_system_config',
    
    # 日志系统
    'setup_logging', 'get_logger', 'PerformanceLogger','DebugLogger', 'initialize_logging', 'debug_logger',
    
    # 绩效计算
    'PerformanceCalculator', 'RiskMetrics', 'BenchmarkComparator',
    
    # 可视化
    'TradingVisualizer', 'RealTimeMonitor',
    
    # 辅助函数
    'timer', 'retry', 'singleton', 'generate_id', 'calculate_hash', 'safe_divide',
    'normalize_data', 'format_currency', 'format_percentage', 'parse_date',
    'get_trading_days', 'calculate_position_size', 'detect_anomalies',
    'DataValidator', 'MemoryManager', 'setup_environment',
    
    # 验证器
    'TradingDataValidator', 'ModelInputValidator', 'TradingDecisionValidator', 
    'ConfigurationValidator',

    # src/core/__init__.py
    'BaseAgent', 'EventEngine', 'Event', 'StateManager',

    # src/agent/__init__.py
    'TradingAgent', 'DecisionMaker', 'LearningModule', 'StrategyEngine', 'MemorySystem', 'InvestmentReportAgent',

    # src/model/__init__.py
    'ModelTrainer', 'ModelPredictor',

    # src/trading/__init__.py
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