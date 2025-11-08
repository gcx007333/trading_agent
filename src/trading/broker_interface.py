# src/trading/broker_interface.py
import requests
import json
import time
import hashlib
import hmac
import base64
import numpy as np  # 添加这行
import random  # 添加这行，作为备选方案
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from utils.config_loader import get_system_config
from utils.helpers import timer, retry, generate_id

logger = logging.getLogger(__name__)

class BrokerType(Enum):
    """券商类型"""
    SIMULATION = "simulation"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"

@dataclass
class BrokerAccount:
    """券商账户信息"""
    broker_name: str
    account_id: str
    account_type: BrokerType
    balance: float
    available_cash: float
    positions: Dict[str, Any]
    last_updated: datetime

@dataclass
class BrokerOrder:
    """券商订单"""
    broker_order_id: str
    symbol: str
    action: str
    quantity: int
    price: float
    status: str
    filled_quantity: int
    avg_fill_price: float
    create_time: datetime
    update_time: datetime

class BaseBrokerInterface:
    """券商接口基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.broker_name = config.get('broker_name', 'unknown')
        self.is_connected = False
    
    def connect(self) -> bool:
        """连接券商"""
        raise NotImplementedError
    
    def disconnect(self) -> bool:
        """断开连接"""
        raise NotImplementedError
    
    def get_account_info(self) -> BrokerAccount:
        """获取账户信息"""
        raise NotImplementedError
    
    def place_order(self, order_request: Dict[str, Any]) -> BrokerOrder:
        """下达订单"""
        raise NotImplementedError
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        raise NotImplementedError
    
    def get_order_status(self, order_id: str) -> BrokerOrder:
        """获取订单状态"""
        raise NotImplementedError
    
    def get_positions(self) -> Dict[str, Any]:
        """获取持仓"""
        raise NotImplementedError
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        raise NotImplementedError

class SimulationBroker(BaseBrokerInterface):
    """模拟券商接口"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.orders: Dict[str, BrokerOrder] = {}
        self.positions: Dict[str, Any] = {}
        self.cash_balance = config.get('initial_capital', 100000.0)
        self.commission_rate = config.get('commission_rate', 0.0003)
        self.stamp_tax_rate = config.get('stamp_tax_rate', 0.001)
    
    def connect(self) -> bool:
        """连接模拟券商"""
        self.is_connected = True
        logger.info(f"连接到模拟券商: {self.broker_name}")
        return True
    
    def disconnect(self) -> bool:
        """断开连接"""
        self.is_connected = False
        logger.info("断开模拟券商连接")
        return True
    
    def get_account_info(self) -> BrokerAccount:
        """获取模拟账户信息"""
        total_value = self.cash_balance + sum(pos['market_value'] for pos in self.positions.values())
        
        return BrokerAccount(
            broker_name=self.broker_name,
            account_id="simulation_account",
            account_type=BrokerType.SIMULATION,
            balance=total_value,
            available_cash=self.cash_balance,
            positions=self.positions.copy(),
            last_updated=datetime.now()
        )
    
    @retry(max_retries=3, delay=0.5)
    def place_order(self, order_request: Dict[str, Any]) -> BrokerOrder:
        """下达模拟订单"""
        symbol = order_request['symbol']
        action = order_request['action']
        quantity = order_request['quantity']
        order_type = order_request.get('order_type', 'market')
        
        # 获取当前价格
        current_price = self.get_current_price(symbol)
        if current_price is None:
            raise Exception(f"无法获取{symbol}价格")
        
        # 创建订单
        broker_order = BrokerOrder(
            broker_order_id=generate_id("BROKER"),
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=current_price,
            status="pending",
            filled_quantity=0,
            avg_fill_price=0.0,
            create_time=datetime.now(),
            update_time=datetime.now()
        )
        
        # 模拟订单执行
        if order_type == 'market':
            # 市价单立即执行
            self._execute_market_order(broker_order, current_price)
        else:
            # 其他订单类型添加到待处理列表
            self.orders[broker_order.broker_order_id] = broker_order
        
        logger.info(f"模拟订单下达: {symbol} {action} {quantity} @ {current_price}")
        
        return broker_order
    
    def _execute_market_order(self, order: BrokerOrder, execution_price: float):
        """执行市价单"""
        # 计算交易金额和费用
        trade_amount = order.quantity * execution_price
        commission = trade_amount * self.commission_rate
        
        if order.action == "BUY":
            # 检查资金是否足够
            total_cost = trade_amount + commission
            if total_cost > self.cash_balance:
                order.status = "rejected"
                order.update_time = datetime.now()
                return
            
            # 更新现金
            self.cash_balance -= total_cost
            
            # 更新持仓
            if order.symbol not in self.positions:
                self.positions[order.symbol] = {
                    'shares': order.quantity,
                    'avg_cost': execution_price,
                    'market_value': order.quantity * execution_price
                }
            else:
                position = self.positions[order.symbol]
                total_shares = position['shares'] + order.quantity
                total_cost_basis = position['shares'] * position['avg_cost'] + order.quantity * execution_price
                new_avg_cost = total_cost_basis / total_shares
                
                position['shares'] = total_shares
                position['avg_cost'] = new_avg_cost
                position['market_value'] = total_shares * execution_price
        
        else:  # SELL
            # 检查持仓是否足够
            if order.symbol not in self.positions or self.positions[order.symbol]['shares'] < order.quantity:
                order.status = "rejected"
                order.update_time = datetime.now()
                return
            
            # 计算印花税
            stamp_tax = trade_amount * self.stamp_tax_rate
            net_proceeds = trade_amount - commission - stamp_tax
            
            # 更新现金
            self.cash_balance += net_proceeds
            
            # 更新持仓
            position = self.positions[order.symbol]
            position['shares'] -= order.quantity
            position['market_value'] = position['shares'] * execution_price
            
            # 如果持仓为0，删除该持仓
            if position['shares'] == 0:
                del self.positions[order.symbol]
        
        # 更新订单状态
        order.status = "filled"
        order.filled_quantity = order.quantity
        order.avg_fill_price = execution_price
        order.update_time = datetime.now()
    
    def cancel_order(self, order_id: str) -> bool:
        """取消模拟订单"""
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = "cancelled"
            order.update_time = datetime.now()
            logger.info(f"模拟订单取消: {order_id}")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> BrokerOrder:
        """获取模拟订单状态"""
        if order_id in self.orders:
            return self.orders[order_id]
        else:
            raise Exception(f"订单不存在: {order_id}")
    
    def get_positions(self) -> Dict[str, Any]:
        """获取模拟持仓"""
        return self.positions.copy()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取模拟当前价格"""
        # 模拟价格数据
        price_map = {
            '000001': 15.0 + np.random.normal(0, 0.1),
            '000002': 25.0 + np.random.normal(0, 0.2),
            '000858': 180.0 + np.random.normal(0, 1.0),
            '600036': 35.0 + np.random.normal(0, 0.15),
            '600519': 1600.0 + np.random.normal(0, 5.0)
        }
        return price_map.get(symbol, 10.0 + np.random.normal(0, 0.1))

class FutuBroker(BaseBrokerInterface):
    """富途券商接口"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.base_url = config.get('base_url', 'https://api.futunn.com')
        self.session = requests.Session()
    
    def connect(self) -> bool:
        """连接富途券商"""
        try:
            # 这里应该实现富途API的实际连接逻辑
            # 目前只是模拟连接
            self.is_connected = True
            logger.info(f"连接到富途券商: {self.broker_name}")
            return True
        except Exception as e:
            logger.error(f"连接富途券商失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """断开连接"""
        self.is_connected = False
        self.session.close()
        logger.info("断开富途券商连接")
        return True
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """生成API签名"""
        # 富途API签名逻辑
        param_str = '&'.join([f'{k}={v}' for k, v in sorted(params.items())])
        signature = hmac.new(
            self.secret_key.encode(),
            param_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    @retry(max_retries=3, delay=1.0)
    def get_account_info(self) -> BrokerAccount:
        """获取富途账户信息"""
        try:
            # 这里应该调用富途API获取真实账户信息
            # 目前返回模拟数据
            
            return BrokerAccount(
                broker_name=self.broker_name,
                account_id="futu_account_123",
                account_type=BrokerType.LIVE_TRADING,
                balance=150000.0,
                available_cash=50000.0,
                positions={
                    '000001': {'shares': 1000, 'avg_cost': 14.5, 'market_value': 15000.0},
                    '00700': {'shares': 100, 'avg_cost': 350.0, 'market_value': 35000.0}
                },
                last_updated=datetime.now()
            )
        except Exception as e:
            logger.error(f"获取富途账户信息失败: {e}")
            raise
    
    @retry(max_retries=3, delay=1.0)
    def place_order(self, order_request: Dict[str, Any]) -> BrokerOrder:
        """下达富途订单"""
        try:
            # 这里应该调用富途API下达真实订单
            # 目前返回模拟订单
            
            symbol = order_request['symbol']
            action = order_request['action']
            quantity = order_request['quantity']
            
            # 模拟订单执行
            current_price = self.get_current_price(symbol)
            
            broker_order = BrokerOrder(
                broker_order_id=generate_id("FUTU"),
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=current_price,
                status="filled",  # 模拟立即成交
                filled_quantity=quantity,
                avg_fill_price=current_price,
                create_time=datetime.now(),
                update_time=datetime.now()
            )
            
            logger.info(f"富途订单下达: {symbol} {action} {quantity} @ {current_price}")
            
            return broker_order
            
        except Exception as e:
            logger.error(f"下达富途订单失败: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """取消富途订单"""
        try:
            # 这里应该调用富途API取消真实订单
            logger.info(f"富途订单取消: {order_id}")
            return True
        except Exception as e:
            logger.error(f"取消富途订单失败: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> BrokerOrder:
        """获取富途订单状态"""
        try:
            # 这里应该调用富途API获取真实订单状态
            # 目前返回模拟数据
            
            return BrokerOrder(
                broker_order_id=order_id,
                symbol="000001",
                action="BUY",
                quantity=1000,
                price=15.0,
                status="filled",
                filled_quantity=1000,
                avg_fill_price=15.0,
                create_time=datetime.now(),
                update_time=datetime.now()
            )
        except Exception as e:
            logger.error(f"获取富途订单状态失败: {e}")
            raise
    
    def get_positions(self) -> Dict[str, Any]:
        """获取富途持仓"""
        try:
            # 这里应该调用富途API获取真实持仓
            # 目前返回模拟数据
            
            return {
                '000001': {'shares': 1000, 'avg_cost': 14.5, 'market_value': 15000.0},
                '00700': {'shares': 100, 'avg_cost': 350.0, 'market_value': 35000.0}
            }
        except Exception as e:
            logger.error(f"获取富途持仓失败: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取富途当前价格"""
        try:
            # 这里应该调用富途API获取真实价格
            # 目前返回模拟数据
            
            price_map = {
                '000001': 15.0,
                '000002': 25.0,
                '000858': 180.0,
                '600036': 35.0,
                '600519': 1600.0,
                '00700': 350.0
            }
            return price_map.get(symbol)
        except Exception as e:
            logger.error(f"获取富途价格失败: {e}")
            return None

class BrokerFactory:
    """券商工厂"""
    
    @staticmethod
    def create_broker(broker_config: Dict[str, Any]) -> BaseBrokerInterface:
        """创建券商接口实例"""
        broker_type = broker_config.get('broker_type', 'simulation')
        
        if broker_type == 'simulation':
            return SimulationBroker(broker_config)
        elif broker_type == 'futu':
            return FutuBroker(broker_config)
        else:
            raise ValueError(f"不支持的券商类型: {broker_type}")

class BrokerManager:
    """券商管理器"""
    
    def __init__(self):
        self.brokers: Dict[str, BaseBrokerInterface] = {}
        self.active_broker: Optional[BaseBrokerInterface] = None
    
    def add_broker(self, broker_id: str, broker_config: Dict[str, Any]) -> bool:
        """添加券商"""
        try:
            broker = BrokerFactory.create_broker(broker_config)
            self.brokers[broker_id] = broker
            logger.info(f"添加券商: {broker_id}")
            return True
        except Exception as e:
            logger.error(f"添加券商失败: {e}")
            return False
    
    def connect_broker(self, broker_id: str) -> bool:
        """连接券商"""
        if broker_id not in self.brokers:
            logger.error(f"券商不存在: {broker_id}")
            return False
        
        broker = self.brokers[broker_id]
        if broker.connect():
            self.active_broker = broker
            logger.info(f"设置活跃券商: {broker_id}")
            return True
        else:
            logger.error(f"连接券商失败: {broker_id}")
            return False
    
    def get_active_broker(self) -> Optional[BaseBrokerInterface]:
        """获取活跃券商"""
        return self.active_broker
    
    def disconnect_all(self):
        """断开所有券商连接"""
        for broker_id, broker in self.brokers.items():
            broker.disconnect()
        self.active_broker = None
        logger.info("断开所有券商连接")