# src/trading/account_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from utils.config_loader import get_system_config
from utils.helpers import generate_id, timer, retry, singleton
from utils.validators import TradingDecisionValidator
from utils.logger import get_logger, performance_logger, debug_logger

# 使用项目统一的日志工具
logger = get_logger(__name__)

class AccountType(Enum):
    """账户类型"""
    SIMULATION = "simulation"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"

@dataclass
class AccountConfig:
    """账户配置"""
    account_id: str
    account_type: AccountType
    initial_capital: float
    currency: str = "CNY"
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    transfer_fee_rate: float = 0.00002
    enabled: bool = True

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    shares: int
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Transaction:
    """交易记录"""
    transaction_id: str
    account_id: str
    symbol: str
    action: str  # BUY, SELL
    shares: int
    price: float
    amount: float
    commission: float
    tax: float
    net_amount: float
    timestamp: datetime
    order_id: str
    strategy: str = ""
    reasoning: str = ""

@dataclass
class AccountSnapshot:
    """账户快照"""
    account_id: str
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float
    positions: Dict[str, Position]

@singleton
class AccountManager:
    """
    账户管理器
    管理多个交易账户，支持模拟盘和实盘
    """
    
    def __init__(self, config_path: str = "config/trading_config.yaml"):
        self.config = get_system_config()
        self.accounts: Dict[str, Dict] = {}
        self.transactions: List[Transaction] = []
        self.snapshots: List[AccountSnapshot] = []
        self._initialize_accounts()
        
        logger.info("账户管理器初始化完成", extra={
            'trading_context': {
                'action': 'account_manager_initialized',
                'config_path': config_path
            }
        })
    
    def _initialize_accounts(self):
        """初始化账户"""
        # 创建默认模拟账户
        default_account = AccountConfig(
            account_id="simulation_100k",
            account_type=AccountType.SIMULATION,
            initial_capital=self.config.trading.initial_capital,
            commission_rate=self.config.trading.commission_rate,
            stamp_tax_rate=self.config.trading.stamp_tax_rate
        )
        
        self.create_account(default_account)
        self.active_account_id = default_account.account_id
        
        logger.info("默认账户已创建", extra={
            'trading_context': {
                'action': 'default_account_created',
                'account_id': self.active_account_id,
                'initial_capital': default_account.initial_capital,
                'account_type': default_account.account_type.value
            }
        })
    
    def create_account(self, account_config: AccountConfig) -> str:
        """创建新账户"""
        account_id = account_config.account_id
        
        self.accounts[account_id] = {
            'config': account_config,
            'cash': account_config.initial_capital,
            'positions': {},
            'realized_pnl': 0.0,
            'transaction_history': [],
            'created_time': datetime.now(),
            'last_updated': datetime.now()
        }
        
        # 创建初始快照
        self._take_snapshot(account_id)
        
        logger.info("创建新交易账户", extra={
            'trading_context': {
                'action': 'account_created',
                'account_id': account_id,
                'account_type': account_config.account_type.value,
                'initial_capital': account_config.initial_capital,
                'commission_rate': account_config.commission_rate
            }
        })
        return account_id
    
    def switch_account(self, account_id: str) -> bool:
        """切换当前活跃账户"""
        if account_id in self.accounts:
            self.active_account_id = account_id
            logger.info("切换活跃账户", extra={
                'trading_context': {
                    'action': 'account_switched',
                    'account_id': account_id
                }
            })
            return True
        else:
            logger.error("切换账户失败", extra={
                'trading_context': {
                    'action': 'account_switch_failed',
                    'account_id': account_id,
                    'error': 'account_not_found'
                }
            })
            return False
    
    def get_account_info(self, account_id: str = None) -> Dict[str, Any]:
        """获取账户信息"""
        account_id = account_id or self.active_account_id
        
        if account_id not in self.accounts:
            logger.error("获取账户信息失败", extra={
                'trading_context': {
                    'action': 'get_account_info_failed',
                    'account_id': account_id,
                    'error': 'account_not_found'
                }
            })
            return {}
        
        account = self.accounts[account_id]
        config = account['config']
        
        # 计算当前总价值
        total_value = self.calculate_total_value(account_id)
        
        debug_logger.log_data_processing(account_id, "account_info_retrieved", {
            "total_value": total_value,
            "positions_count": len(account['positions']),
            "transaction_count": len(account['transaction_history'])
        })
        
        return {
            'account_id': account_id,
            'account_type': config.account_type.value,
            'initial_capital': config.initial_capital,
            'current_cash': account['cash'],
            'total_value': total_value,
            'positions_value': total_value - account['cash'],
            'realized_pnl': account['realized_pnl'],
            'unrealized_pnl': total_value - config.initial_capital,
            'positions_count': len(account['positions']),
            'transaction_count': len(account['transaction_history']),
            'created_time': account['created_time'],
            'last_updated': account['last_updated']
        }
    
    def calculate_total_value(self, account_id: str, current_prices: Dict[str, float] = None) -> float:
        """计算账户总价值"""
        if account_id not in self.accounts:
            return 0.0
        
        account = self.accounts[account_id]
        total_value = account['cash']
        
        # 计算持仓价值
        for symbol, position in account['positions'].items():
            if current_prices and symbol in current_prices:
                price = current_prices[symbol]
            else:
                price = position.get('current_price', 0)
            
            shares = position['shares']
            total_value += shares * price
        
        return total_value
    
    def get_positions(self, account_id: str = None) -> Dict[str, Position]:
        """获取持仓信息"""
        account_id = account_id or self.active_account_id
        
        if account_id not in self.accounts:
            logger.warning("获取持仓信息失败", extra={
                'trading_context': {
                    'action': 'get_positions_failed',
                    'account_id': account_id,
                    'error': 'account_not_found'
                }
            })
            return {}
        
        positions = {}
        account_positions = self.accounts[account_id]['positions']
        
        for symbol, pos_data in account_positions.items():
            position = Position(
                symbol=symbol,
                shares=pos_data['shares'],
                avg_cost=pos_data['avg_cost'],
                current_price=pos_data.get('current_price', 0)
            )
            
            # 计算市值和盈亏
            position.market_value = position.shares * position.current_price
            position.unrealized_pnl = position.market_value - (position.shares * position.avg_cost)
            position.unrealized_pnl_pct = position.unrealized_pnl / (position.shares * position.avg_cost) if position.shares * position.avg_cost > 0 else 0
            
            positions[symbol] = position
        
        debug_logger.log_data_processing(account_id, "positions_retrieved", {
            "positions_count": len(positions),
            "total_market_value": sum(p.market_value for p in positions.values())
        })
        
        return positions
    
    def update_position_price(self, symbol: str, current_price: float, account_id: str = None):
        """更新持仓价格"""
        account_id = account_id or self.active_account_id
        
        if account_id not in self.accounts:
            return
        
        if symbol in self.accounts[account_id]['positions']:
            self.accounts[account_id]['positions'][symbol]['current_price'] = current_price
            self.accounts[account_id]['positions'][symbol]['last_updated'] = datetime.now()
            
            debug_logger.log_data_processing(account_id, "position_price_updated", {
                "symbol": symbol,
                "current_price": current_price
            })
    
    def record_transaction(self, transaction: Transaction, account_id: str = None):
        """记录交易"""
        account_id = account_id or self.active_account_id
        
        if account_id not in self.accounts:
            logger.error("记录交易失败", extra={
                'trading_context': {
                    'action': 'record_transaction_failed',
                    'account_id': account_id,
                    'transaction_id': transaction.transaction_id,
                    'error': 'account_not_found'
                }
            })
            return
        
        # 添加到交易历史
        self.accounts[account_id]['transaction_history'].append(transaction)
        self.transactions.append(transaction)
        
        # 更新账户现金
        if transaction.action == "BUY":
            self.accounts[account_id]['cash'] -= transaction.net_amount
        else:  # SELL
            self.accounts[account_id]['cash'] += transaction.net_amount
            # 计算已实现盈亏
            cost = transaction.shares * self._get_position_avg_cost(account_id, transaction.symbol)
            realized_pnl = transaction.net_amount - cost
            self.accounts[account_id]['realized_pnl'] += realized_pnl
        
        # 更新最后更新时间
        self.accounts[account_id]['last_updated'] = datetime.now()
        
        # 记录快照
        self._take_snapshot(account_id)
        
        # 记录性能日志
        performance_logger.log_trade({
            'symbol': transaction.symbol,
            'action': transaction.action,
            'shares': transaction.shares,
            'price': transaction.price,
            'amount': transaction.amount,
            'commission': transaction.commission,
            'pnl': realized_pnl if transaction.action == "SELL" else 0
        })
        
        logger.info("交易记录已保存", extra={
            'trading_context': {
                'action': 'transaction_recorded',
                'account_id': account_id,
                'symbol': transaction.symbol,
                'transaction_type': transaction.action,
                'shares': transaction.shares,
                'price': transaction.price,
                'net_amount': transaction.net_amount
            }
        })
    
    def _get_position_avg_cost(self, account_id: str, symbol: str) -> float:
        """获取持仓平均成本"""
        if (account_id in self.accounts and 
            symbol in self.accounts[account_id]['positions']):
            return self.accounts[account_id]['positions'][symbol]['avg_cost']
        return 0.0
    
    def update_position(self, account_id: str, symbol: str, shares: int, price: float, action: str):
        """更新持仓"""
        if account_id not in self.accounts:
            return
        
        account = self.accounts[account_id]
        
        if symbol not in account['positions']:
            # 新建持仓
            account['positions'][symbol] = {
                'shares': shares,
                'avg_cost': price,
                'current_price': price,
                'last_updated': datetime.now()
            }
            
            logger.info("新建持仓", extra={
                'trading_context': {
                    'action': 'position_created',
                    'account_id': account_id,
                    'symbol': symbol,
                    'shares': shares,
                    'avg_cost': price
                }
            })
        else:
            # 更新现有持仓
            position = account['positions'][symbol]
            old_shares = position['shares']
            old_avg_cost = position['avg_cost']
            
            if action == "BUY":
                new_shares = old_shares + shares
                new_avg_cost = (old_shares * old_avg_cost + shares * price) / new_shares
            else:  # SELL
                new_shares = old_shares - shares
                new_avg_cost = old_avg_cost  # 卖出不影响平均成本
            
            position['shares'] = new_shares
            position['avg_cost'] = new_avg_cost
            position['last_updated'] = datetime.now()
            
            logger.info("更新持仓", extra={
                'trading_context': {
                    'action': 'position_updated',
                    'account_id': account_id,
                    'symbol': symbol,
                    'old_shares': old_shares,
                    'new_shares': new_shares,
                    'operation': action,
                    'shares_change': shares
                }
            })
            
            # 如果持仓为0，删除该持仓记录
            if new_shares == 0:
                del account['positions'][symbol]
                logger.info("清空持仓", extra={
                    'trading_context': {
                        'action': 'position_cleared',
                        'account_id': account_id,
                        'symbol': symbol
                    }
                })
    
    def _take_snapshot(self, account_id: str):
        """记录账户快照"""
        account_info = self.get_account_info(account_id)
        positions = self.get_positions(account_id)
        
        snapshot = AccountSnapshot(
            account_id=account_id,
            timestamp=datetime.now(),
            total_value=account_info['total_value'],
            cash=account_info['current_cash'],
            positions_value=account_info['positions_value'],
            realized_pnl=account_info['realized_pnl'],
            unrealized_pnl=account_info['unrealized_pnl'],
            daily_pnl=0.0,  # 需要额外计算
            positions=positions
        )
        
        self.snapshots.append(snapshot)
        
        debug_logger.log_data_processing(account_id, "account_snapshot_taken", {
            "total_value": snapshot.total_value,
            "cash": snapshot.cash,
            "positions_value": snapshot.positions_value,
            "realized_pnl": snapshot.realized_pnl,
            "unrealized_pnl": snapshot.unrealized_pnl
        })
    
    def get_transaction_history(self, account_id: str = None, 
                              start_date: datetime = None,
                              end_date: datetime = None) -> List[Transaction]:
        """获取交易历史"""
        account_id = account_id or self.active_account_id
        
        if account_id not in self.accounts:
            return []
        
        transactions = self.accounts[account_id]['transaction_history']
        
        if start_date:
            transactions = [t for t in transactions if t.timestamp >= start_date]
        if end_date:
            transactions = [t for t in transactions if t.timestamp <= end_date]
        
        return transactions
    
    def get_performance_metrics(self, account_id: str = None) -> Dict[str, Any]:
        """获取账户绩效指标"""
        account_id = account_id or self.active_account_id
        account_info = self.get_account_info(account_id)
        
        if not account_info:
            return {}
        
        # 计算基本指标
        total_return = (account_info['total_value'] - account_info['initial_capital']) / account_info['initial_capital']
        
        # 计算交易相关指标
        transactions = self.get_transaction_history(account_id)
        winning_trades = [t for t in transactions if t.action == "SELL" and t.net_amount > 0]
        losing_trades = [t for t in transactions if t.action == "SELL" and t.net_amount <= 0]
        
        win_rate = len(winning_trades) / len(transactions) if transactions else 0
        avg_win = np.mean([t.net_amount for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.net_amount for t in losing_trades]) if losing_trades else 0
        
        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'current_value': account_info['total_value'],
            'cash_balance': account_info['current_cash'],
            'win_rate': win_rate,
            'total_trades': len(transactions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'max_single_win': max([t.net_amount for t in winning_trades]) if winning_trades else 0,
            'max_single_loss': min([t.net_amount for t in losing_trades]) if losing_trades else 0
        }
        
        # 记录性能指标
        performance_logger.log_performance({
            'account_performance': {
                'account_id': account_id,
                'total_return_pct': metrics['total_return_pct'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades'],
                'current_value': metrics['current_value']
            }
        })
        
        logger.info("账户绩效指标计算完成", extra={
            'trading_context': {
                'action': 'performance_metrics_calculated',
                'account_id': account_id,
                'total_return_pct': metrics['total_return_pct'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades']
            }
        })
        
        return metrics
    
    def export_account_data(self, account_id: str = None, export_path: str = None):
        """导出账户数据"""
        account_id = account_id or self.active_account_id
        
        if account_id not in self.accounts:
            logger.error("导出账户数据失败", extra={
                'trading_context': {
                    'action': 'export_account_data_failed',
                    'account_id': account_id,
                    'error': 'account_not_found'
                }
            })
            return
        
        export_data = {
            'account_info': self.get_account_info(account_id),
            'positions': {symbol: pos.__dict__ for symbol, pos in self.get_positions(account_id).items()},
            'transactions': [t.__dict__ for t in self.get_transaction_history(account_id)],
            'performance': self.get_performance_metrics(account_id),
            'export_time': datetime.now().isoformat()
        }
        
        if export_path:
            try:
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                logger.info("账户数据导出成功", extra={
                    'trading_context': {
                        'action': 'account_data_exported',
                        'account_id': account_id,
                        'export_path': export_path,
                        'data_size': len(json.dumps(export_data))
                    }
                })
            except Exception as e:
                logger.error("账户数据导出失败", extra={
                    'trading_context': {
                        'action': 'account_data_export_failed',
                        'account_id': account_id,
                        'export_path': export_path,
                        'error_message': str(e)
                    }
                })
        
        return export_data

class MultiAccountManager:
    """
    多账户管理器
    用于管理策略在不同账户上的表现
    """
    
    def __init__(self, account_manager: AccountManager):
        self.account_manager = account_manager
        self.strategy_accounts: Dict[str, List[str]] = {}  # 策略到账户的映射
        
        logger.info("多账户管理器初始化完成", extra={
            'trading_context': {
                'action': 'multi_account_manager_initialized'
            }
        })
    
    def allocate_strategy_to_account(self, strategy_id: str, account_id: str, allocation_ratio: float = 1.0):
        """为策略分配账户"""
        if strategy_id not in self.strategy_accounts:
            self.strategy_accounts[strategy_id] = []
        
        self.strategy_accounts[strategy_id].append({
            'account_id': account_id,
            'allocation_ratio': allocation_ratio
        })
        
        logger.info("策略分配到账户", extra={
            'trading_context': {
                'action': 'strategy_allocated_to_account',
                'strategy_id': strategy_id,
                'account_id': account_id,
                'allocation_ratio': allocation_ratio
            }
        })
    
    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """获取策略绩效"""
        if strategy_id not in self.strategy_accounts:
            logger.warning("获取策略绩效失败", extra={
                'trading_context': {
                    'action': 'get_strategy_performance_failed',
                    'strategy_id': strategy_id,
                    'error': 'strategy_not_found'
                }
            })
            return {}
        
        total_performance = {
            'total_value': 0,
            'total_return': 0,
            'win_rate': 0,
            'total_trades': 0
        }
        
        strategy_accounts = self.strategy_accounts[strategy_id]
        
        for allocation in strategy_accounts:
            account_id = allocation['account_id']
            ratio = allocation['allocation_ratio']
            
            account_perf = self.account_manager.get_performance_metrics(account_id)
            
            # 加权汇总
            total_performance['total_value'] += account_perf.get('current_value', 0) * ratio
            total_performance['total_return'] += account_perf.get('total_return', 0) * ratio
            total_performance['win_rate'] += account_perf.get('win_rate', 0) * ratio
            total_performance['total_trades'] += account_perf.get('total_trades', 0)
        
        logger.info("策略绩效计算完成", extra={
            'trading_context': {
                'action': 'strategy_performance_calculated',
                'strategy_id': strategy_id,
                'total_value': total_performance['total_value'],
                'total_return': total_performance['total_return'],
                'win_rate': total_performance['win_rate']
            }
        })
        
        return total_performance