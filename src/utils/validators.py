# src/utils/validators.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TradingDataValidator:
    """交易数据验证器"""
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证价格数据"""
        issues = []
        
        if data.empty:
            issues.append("价格数据为空")
            return False, issues
        
        # 检查必要列
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"缺少价格列: {missing_columns}")
            return False, issues
        
        # 检查价格合理性
        for col in required_columns:
            if (data[col] <= 0).any():
                issues.append(f"{col}列包含非正值")
        
        # 检查价格关系: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        if not (data['High'] >= data['Low']).all():
            issues.append("最高价低于最低价")
        
        if not (data['High'] >= data['Open']).all():
            issues.append("最高价低于开盘价")
        
        if not (data['High'] >= data['Close']).all():
            issues.append("最高价低于收盘价")
        
        if not (data['Low'] <= data['Open']).all():
            issues.append("最低价高于开盘价")
        
        if not (data['Low'] <= data['Close']).all():
            issues.append("最低价高于收盘价")
        
        # 检查价格跳空（大幅波动）
        price_changes = data['Close'].pct_change().abs()
        large_moves = price_changes[price_changes > 0.2]  # 20%以上波动
        if not large_moves.empty:
            issues.append(f"检测到大幅价格波动: {len(large_moves)}次")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_volume_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证成交量数据"""
        issues = []
        
        if 'Volume' not in data.columns:
            issues.append("缺少成交量列")
            return False, issues
        
        # 检查成交量非负
        if (data['Volume'] < 0).any():
            issues.append("成交量包含负值")
        
        # 检查零成交量
        zero_volume = (data['Volume'] == 0).sum()
        if zero_volume > 0:
            issues.append(f"存在{zero_volume}条零成交量记录")
        
        # 检查异常成交量（超过平均值的10倍）
        volume_mean = data['Volume'].mean()
        if volume_mean > 0:
            abnormal_volume = (data['Volume'] > volume_mean * 10).sum()
            if abnormal_volume > 0:
                issues.append(f"检测到{abnormal_volume}条异常高成交量记录")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_timestamp_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证时间戳数据"""
        issues = []
        
        if data.index.duplicated().any():
            issues.append("存在重复的时间戳")
        
        # 检查时间连续性
        if len(data) > 1:
            time_diff = data.index.to_series().diff().dropna()
            if (time_diff > pd.Timedelta(days=7)).any():
                issues.append("时间序列存在较大间隔")
            
            # 检查是否包含非交易日
            weekdays = data.index.dayofweek
            if (weekdays >= 5).any():  # 5=周六, 6=周日
                issues.append("数据包含非交易日（周末）")
        
        return len(issues) == 0, issues

class ModelInputValidator:
    """模型输入验证器"""
    
    @staticmethod
    def validate_features(features: pd.DataFrame, expected_features: List[str]) -> Tuple[bool, List[str]]:
        """验证特征数据"""
        issues = []
        
        if features.empty:
            issues.append("特征数据为空")
            return False, issues
        
        # 检查特征列是否匹配
        missing_features = set(expected_features) - set(features.columns)
        if missing_features:
            issues.append(f"缺少特征: {missing_features}")
        
        extra_features = set(features.columns) - set(expected_features)
        if extra_features:
            issues.append(f"多余特征: {extra_features}")
        
        # 检查缺失值
        missing_values = features.isnull().sum().sum()
        if missing_values > 0:
            issues.append(f"特征数据包含{missing_values}个缺失值")
        
        # 检查无穷值
        infinite_values = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        if infinite_values > 0:
            issues.append(f"特征数据包含{infinite_values}个无穷值")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_target(target: pd.Series) -> Tuple[bool, List[str]]:
        """验证目标变量"""
        issues = []
        
        if target.empty:
            issues.append("目标变量为空")
            return False, issues
        
        # 检查缺失值
        if target.isnull().any():
            issues.append("目标变量包含缺失值")
        
        # 检查分类目标的类别
        if target.dtype in ['object', 'category']:
            unique_values = target.unique()
            if len(unique_values) != 2:
                issues.append(f"分类目标应该有2个类别，实际有{len(unique_values)}个")
        
        return len(issues) == 0, issues

class TradingDecisionValidator:
    """交易决策验证器"""
    
    @staticmethod
    def validate_order(order: Dict[str, Any], account_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证订单"""
        issues = []
        
        # 检查必要字段
        required_fields = ['symbol', 'action', 'quantity', 'order_type']
        for field in required_fields:
            if field not in order:
                issues.append(f"订单缺少必要字段: {field}")
        
        if issues:
            return False, issues
        
        # 检查订单类型
        valid_actions = ['BUY', 'SELL', 'HOLD']
        if order['action'] not in valid_actions:
            issues.append(f"无效的操作类型: {order['action']}")
        
        # 检查数量
        if order['quantity'] <= 0:
            issues.append("订单数量必须为正")
        
        # 检查资金（对于买入订单）
        if order['action'] == 'BUY':
            available_cash = account_state.get('available_cash', 0)
            estimated_cost = order['quantity'] * account_state.get('current_price', 0)
            
            if estimated_cost > available_cash:
                issues.append(f"资金不足: 需要{estimated_cost:.2f}, 可用{available_cash:.2f}")
        
        # 检查持仓（对于卖出订单）
        elif order['action'] == 'SELL':
            current_position = account_state.get('positions', {}).get(order['symbol'], 0)
            if order['quantity'] > current_position:
                issues.append(f"持仓不足: 需要卖出{order['quantity']}, 当前持仓{current_position}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_risk_limits(order: Dict[str, Any], portfolio_state: Dict[str, Any], 
                           risk_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证风险限制"""
        issues = []
        
        # 检查单票仓位限制
        symbol = order['symbol']
        current_price = portfolio_state.get('current_prices', {}).get(symbol, 0)
        order_value = order['quantity'] * current_price
        portfolio_value = portfolio_state.get('total_value', 1)
        
        position_size = order_value / portfolio_value
        max_position_size = risk_config.get('max_position_size', 0.2)
        
        if position_size > max_position_size:
            issues.append(f"超过单票仓位限制: {position_size:.1%} > {max_position_size:.1%}")
        
        # 检查整体风险暴露
        current_drawdown = portfolio_state.get('current_drawdown', 0)
        max_drawdown = risk_config.get('max_drawdown', 0.1)
        
        if current_drawdown > max_drawdown:
            issues.append(f"超过最大回撤限制: {current_drawdown:.1%} > {max_drawdown:.1%}")
        
        return len(issues) == 0, issues

class ConfigurationValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_system_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证系统配置"""
        issues = []
        
        # 检查交易配置
        trading_config = config.get('trading', {})
        if trading_config.get('initial_capital', 0) <= 0:
            issues.append("初始资金必须为正")
        
        if not 0 <= trading_config.get('commission_rate', 0) <= 0.01:
            issues.append("佣金率应该在0-1%之间")
        
        if not 0 <= trading_config.get('stamp_tax_rate', 0) <= 0.01:
            issues.append("印花税率应该在0-1%之间")
        
        # 检查风险配置
        risk_config = config.get('risk', {})
        if not 0 <= risk_config.get('stop_loss', 0) <= 0.5:
            issues.append("止损比例应该在0-50%之间")
        
        if not 0 <= risk_config.get('max_position_size', 0) <= 1:
            issues.append("单票最大仓位应该在0-100%之间")
        
        # 检查模型配置
        model_config = config.get('model', {})
        if model_config.get('n_estimators', 0) <= 0:
            issues.append("模型树数量必须为正")
        
        if not 0 < model_config.get('learning_rate', 0) <= 1:
            issues.append("学习率应该在0-1之间")
        
        return len(issues) == 0, issues