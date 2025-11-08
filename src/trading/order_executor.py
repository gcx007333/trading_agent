# src/trading/order_executor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass, field

from utils.config_loader import get_system_config
from utils.helpers import generate_id, timer, retry
from utils.validators import TradingDecisionValidator
from trading.account_manager import AccountManager, Transaction

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"      # 市价单
    LIMIT = "limit"        # 限价单
    STOP = "stop"          # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"    # 待处理
    PARTIAL = "partial"    # 部分成交
    FILLED = "filled"      # 完全成交
    CANCELLED = "cancelled" # 已取消
    REJECTED = "rejected"  # 已拒绝

@dataclass
class Order:
    """订单信息"""
    order_id: str
    account_id: str
    symbol: str
    order_type: OrderType
    action: str  # BUY, SELL
    quantity: int
    price: Optional[float] = None  # 对于限价单
    stop_price: Optional[float] = None  # 对于止损单
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    reason: str = ""
    strategy: str = ""

@dataclass
class ExecutionResult:
    """执行结果"""
    order_id: str
    symbol: str
    action: str
    executed_quantity: int
    executed_price: float
    commission: float
    tax: float
    net_amount: float
    status: OrderStatus
    timestamp: datetime
    message: str = ""

class OrderExecutor:
    """
    订单执行引擎
    处理各种类型的订单执行
    """
    
    def __init__(self, account_manager: AccountManager, broker_interface=None):
        self.account_manager = account_manager
        self.broker_interface = broker_interface
        self.config = get_system_config()
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.execution_history: List[ExecutionResult] = []
        
        logger.info("订单执行引擎初始化完成")

    def initialize(self) -> bool:
        """初始化订单执行器"""
        try:
            # 初始化券商API
            # self._initialize_broker_api()
            
            # 获取账户信息
            # self.account_info = self._get_account_info()
            
            # 获取持仓信息
            # self.positions = self._get_positions()
            
            self.is_running = True
            logger.info("订单执行器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"订单执行器初始化失败: {e}")
            return False
    
    def place_order(self, order_request: Dict[str, Any]) -> Order:
        """下达订单"""
        try:
            # 验证订单请求
            is_valid, issues = TradingDecisionValidator.validate_order(
                order_request, 
                self._get_account_state(order_request.get('account_id'))
            )
            
            if not is_valid:
                logger.error(f"订单验证失败: {issues}")
                return self._create_rejected_order(order_request, issues)
            
            # 创建订单
            order = self._create_order(order_request)
            self.pending_orders[order.order_id] = order
            self.order_history.append(order)
            
            logger.info(f"下达订单: {order.symbol} {order.action} {order.quantity} {order.order_type.value}")
            
            # 根据订单类型执行
            if order.order_type == OrderType.MARKET:
                return self._execute_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                return self._place_limit_order(order)
            elif order.order_type == OrderType.STOP:
                return self._place_stop_order(order)
            else:
                return self._reject_order(order, f"不支持的订单类型: {order.order_type}")
                
        except Exception as e:
            logger.error(f"下达订单失败: {e}")
            return self._create_rejected_order(order_request, str(e))
    
    def _create_order(self, order_request: Dict[str, Any]) -> Order:
        """创建订单对象"""
        return Order(
            order_id=generate_id("ORD"),
            account_id=order_request.get('account_id', self.account_manager.active_account_id),
            symbol=order_request['symbol'],
            order_type=OrderType(order_request.get('order_type', 'market')),
            action=order_request['action'].upper(),
            quantity=order_request['quantity'],
            price=order_request.get('price'),
            stop_price=order_request.get('stop_price'),
            reason=order_request.get('reason', ''),
            strategy=order_request.get('strategy', '')
        )
    
    def _create_rejected_order(self, order_request: Dict[str, Any], reason: str) -> Order:
        """创建被拒绝的订单"""
        order = self._create_order(order_request)
        order.status = OrderStatus.REJECTED
        order.reason = reason
        self.order_history.append(order)
        return order
    
    @retry(max_retries=3, delay=1.0)
    def _execute_market_order(self, order: Order) -> Order:
        """执行市价单"""
        try:
            # 获取当前市场价格
            current_price = self._get_current_price(order.symbol)
            if current_price is None:
                return self._reject_order(order, "无法获取当前价格")
            
            # 检查资金或持仓
            validation_result = self._validate_order_execution(order, current_price)
            if not validation_result['is_valid']:
                return self._reject_order(order, validation_result['reason'])
            
            # 执行订单
            execution_result = self._execute_trade(order, current_price, order.quantity)
            
            # 更新订单状态
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = current_price
            order.updated_time = datetime.now()
            
            # 从待处理订单中移除
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            
            logger.info(f"市价单执行完成: {order.symbol} {order.action} {order.quantity} @ {current_price}")
            
            return order
            
        except Exception as e:
            logger.error(f"执行市价单失败: {e}")
            return self._reject_order(order, f"执行失败: {str(e)}")
    
    def _place_limit_order(self, order: Order) -> Order:
        """放置限价单（等待条件触发）"""
        if order.price is None:
            return self._reject_order(order, "限价单必须指定价格")
        
        # 检查当前价格是否已经满足条件
        current_price = self._get_current_price(order.symbol)
        if current_price is not None:
            if (order.action == "BUY" and current_price <= order.price) or \
               (order.action == "SELL" and current_price >= order.price):
                # 立即执行
                return self._execute_market_order(order)
        
        # 添加到待处理订单
        self.pending_orders[order.order_id] = order
        logger.info(f"限价单已挂单: {order.symbol} {order.action} {order.quantity} @ {order.price}")
        
        return order
    
    def _place_stop_order(self, order: Order) -> Order:
        """放置止损单"""
        if order.stop_price is None:
            return self._reject_order(order, "止损单必须指定止损价格")
        
        # 添加到待处理订单
        self.pending_orders[order.order_id] = order
        logger.info(f"止损单已挂单: {order.symbol} {order.action} {order.quantity} @ {order.stop_price}")
        
        return order
    
    def _reject_order(self, order: Order, reason: str) -> Order:
        """拒绝订单"""
        order.status = OrderStatus.REJECTED
        order.reason = reason
        order.updated_time = datetime.now()
        
        if order.order_id in self.pending_orders:
            del self.pending_orders[order.order_id]
        
        logger.warning(f"订单被拒绝: {order.order_id} - {reason}")
        return order
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        # 这里应该调用市场数据接口
        # 目前返回一个模拟价格
        try:
            if self.broker_interface:
                return self.broker_interface.get_current_price(symbol)
            else:
                # 模拟价格 - 实际应该从数据层获取
                return 10.0  # 默认价格
        except Exception as e:
            logger.error(f"获取价格失败 {symbol}: {e}")
            return None
    
    def _get_account_state(self, account_id: str = None) -> Dict[str, Any]:
        """获取账户状态"""
        account_info = self.account_manager.get_account_info(account_id)
        positions = self.account_manager.get_positions(account_id)
        
        return {
            'available_cash': account_info.get('current_cash', 0),
            'total_value': account_info.get('total_value', 0),
            'positions': positions,
            'current_prices': {sym: pos.current_price for sym, pos in positions.items()}
        }
    
    def _validate_order_execution(self, order: Order, current_price: float) -> Dict[str, Any]:
        """验证订单执行"""
        account_state = self._get_account_state(order.account_id)
        
        if order.action == "BUY":
            # 检查资金是否足够
            total_cost = order.quantity * current_price
            commission = total_cost * self.config.trading.commission_rate
            transfer_fee = total_cost * self.config.trading.transfer_fee_rate
            total_required = total_cost + commission + transfer_fee
            
            if total_required > account_state['available_cash']:
                return {
                    'is_valid': False,
                    'reason': f"资金不足: 需要 {total_required:.2f}, 可用 {account_state['available_cash']:.2f}"
                }
        
        elif order.action == "SELL":
            # 检查持仓是否足够
            current_position = account_state['positions'].get(order.symbol, None)
            if not current_position or current_position.shares < order.quantity:
                return {
                    'is_valid': False,
                    'reason': f"持仓不足: 需要卖出 {order.quantity}, 当前持仓 {current_position.shares if current_position else 0}"
                }
        
        return {'is_valid': True, 'reason': ''}
    
    def _execute_trade(self, order: Order, execution_price: float, executed_quantity: int) -> ExecutionResult:
        """执行交易"""
        # 计算交易金额和费用
        trade_amount = executed_quantity * execution_price
        commission = trade_amount * self.config.trading.commission_rate
        transfer_fee = trade_amount * self.config.trading.transfer_fee_rate
        
        # 卖出时计算印花税
        stamp_tax = 0.0
        if order.action == "SELL":
            stamp_tax = trade_amount * self.config.trading.stamp_tax_rate
        
        # 计算净金额
        if order.action == "BUY":
            net_amount = trade_amount + commission + transfer_fee
        else:  # SELL
            net_amount = trade_amount - commission - stamp_tax - transfer_fee
        
        # 创建执行结果
        execution_result = ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            action=order.action,
            executed_quantity=executed_quantity,
            executed_price=execution_price,
            commission=commission,
            tax=stamp_tax,
            net_amount=net_amount,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(),
            message="订单执行完成"
        )
        
        # 记录交易
        transaction = Transaction(
            transaction_id=generate_id("TRX"),
            account_id=order.account_id,
            symbol=order.symbol,
            action=order.action,
            shares=executed_quantity,
            price=execution_price,
            amount=trade_amount,
            commission=commission,
            tax=stamp_tax,
            net_amount=net_amount,
            timestamp=datetime.now(),
            order_id=order.order_id,
            strategy=order.strategy,
            reasoning=order.reason
        )
        
        # 更新账户
        self.account_manager.record_transaction(transaction, order.account_id)
        self.account_manager.update_position(
            order.account_id, order.symbol, executed_quantity, execution_price, order.action
        )
        
        # 记录执行历史
        self.execution_history.append(execution_result)
        
        logger.info(f"交易执行: {order.symbol} {order.action} {executed_quantity} @ {execution_price}")
        
        return execution_result
    
    def check_pending_orders(self, current_prices: Dict[str, float]) -> List[ExecutionResult]:
        """检查并执行待处理订单"""
        executed_orders = []
        
        for order_id, order in list(self.pending_orders.items()):
            if order.symbol not in current_prices:
                continue
            
            current_price = current_prices[order.symbol]
            should_execute = False
            
            # 检查订单条件
            if order.order_type == OrderType.LIMIT:
                if (order.action == "BUY" and current_price <= order.price) or \
                   (order.action == "SELL" and current_price >= order.price):
                    should_execute = True
            
            elif order.order_type == OrderType.STOP:
                if (order.action == "BUY" and current_price >= order.stop_price) or \
                   (order.action == "SELL" and current_price <= order.stop_price):
                    should_execute = True
            
            if should_execute:
                # 执行订单
                execution_result = self._execute_trade(order, current_price, order.quantity)
                executed_orders.append(execution_result)
                
                # 更新订单状态
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.avg_fill_price = current_price
                order.updated_time = datetime.now()
                
                # 从待处理订单中移除
                del self.pending_orders[order_id]
                
                logger.info(f"条件单触发执行: {order.symbol} {order.action} {order.quantity}")
        
        return executed_orders
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            order.updated_time = datetime.now()
            
            del self.pending_orders[order_id]
            
            logger.info(f"订单已取消: {order_id}")
            return True
        else:
            logger.warning(f"订单不存在或无法取消: {order_id}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        # 首先检查待处理订单
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        
        # 然后检查历史订单
        for order in self.order_history:
            if order.order_id == order_id:
                return order
        
        return None
    
    def get_pending_orders(self, account_id: str = None) -> List[Order]:
        """获取待处理订单"""
        if account_id:
            return [order for order in self.pending_orders.values() 
                   if order.account_id == account_id]
        else:
            return list(self.pending_orders.values())
    
    def get_execution_history(self, symbol: str = None, 
                            start_date: datetime = None,
                            end_date: datetime = None) -> List[ExecutionResult]:
        """获取执行历史"""
        history = self.execution_history
        
        if symbol:
            history = [h for h in history if h.symbol == symbol]
        if start_date:
            history = [h for h in history if h.timestamp >= start_date]
        if end_date:
            history = [h for h in history if h.timestamp <= end_date]
        
        return history

class SmartOrderRouter:
    """
    智能订单路由器
    优化订单执行策略，减少市场冲击
    """
    
    def __init__(self, order_executor: OrderExecutor):
        self.order_executor = order_executor
        self.config = get_system_config()
    
    def execute_smart_order(self, order_request: Dict[str, Any]) -> List[ExecutionResult]:
        """执行智能订单（可能分拆为多个子订单）"""
        large_order_threshold = 10000  # 大单阈值（股）
        order_quantity = order_request['quantity']
        
        if order_quantity <= large_order_threshold:
            # 小单直接执行
            order = self.order_executor.place_order(order_request)
            if order.status == OrderStatus.FILLED:
                execution = self.order_executor.get_execution_history(order_id=order.order_id)
                return execution if execution else []
            return []
        else:
            # 大单分拆执行
            return self._execute_large_order(order_request)
    
    def _execute_large_order(self, order_request: Dict[str, Any]) -> List[ExecutionResult]:
        """执行大单（分拆策略）"""
        total_quantity = order_request['quantity']
        symbol = order_request['symbol']
        action = order_request['action']
        
        # 计算分拆数量（基于历史成交量）
        chunk_sizes = self._calculate_chunk_sizes(total_quantity, symbol)
        all_executions = []
        
        logger.info(f"大单分拆执行: {symbol} {action} {total_quantity} -> {len(chunk_sizes)} 个子单")
        
        for i, chunk_size in enumerate(chunk_sizes):
            # 创建子订单
            sub_order_request = order_request.copy()
            sub_order_request['quantity'] = chunk_size
            sub_order_request['reason'] = f"{order_request.get('reason', '')} [子单 {i+1}/{len(chunk_sizes)}]"
            
            # 执行子订单
            order = self.order_executor.place_order(sub_order_request)
            
            if order.status == OrderStatus.FILLED:
                execution = self.order_executor.get_execution_history(order_id=order.order_id)
                if execution:
                    all_executions.extend(execution)
            
            # 添加延迟以减少市场冲击
            if i < len(chunk_sizes) - 1:  # 不是最后一个订单
                import time
                time.sleep(2)  # 2秒延迟
        
        logger.info(f"大单执行完成: {len(all_executions)} 个子单")
        return all_executions
    
    def _calculate_chunk_sizes(self, total_quantity: int, symbol: str) -> List[int]:
        """计算分拆大小"""
        # 基于平均成交量的简单分拆策略
        avg_volume = self._get_average_volume(symbol)
        max_chunk_size = max(1000, int(avg_volume * 0.1))  # 不超过日均成交量的10%
        
        chunks = []
        remaining = total_quantity
        
        while remaining > 0:
            chunk_size = min(remaining, max_chunk_size)
            chunks.append(chunk_size)
            remaining -= chunk_size
        
        return chunks
    
    def _get_average_volume(self, symbol: str) -> float:
        """获取平均成交量"""
        # 这里应该从数据层获取历史成交量数据
        # 目前返回一个默认值
        return 1000000  # 默认100万股