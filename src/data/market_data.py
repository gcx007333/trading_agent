import pandas as pd
import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketData:
    """
    市场数据接口
    提供实时数据订阅和推送功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_collector = None  # 稍后初始化
        self.subscribers = {}
        self.is_running = False
        self.thread = None
        self.cache = {}
        self.last_update = {}
    
    def initialize(self) -> bool:
        """初始化市场数据服务"""
        try:
            # 延迟导入，避免循环依赖
            from .data_collector import DataCollector
            self.data_collector = DataCollector()
            
            # 启动数据服务
            self.start()
            
            logger.info("市场数据服务初始化完成")
            return True
        except Exception as e:
            logger.error(f"市场数据服务初始化失败: {e}")
            return False
    
    def _load_config(self, config_path):
        """加载市场数据配置"""
        return {
            "polling_interval": 60,  # 轮询间隔秒数
            "cache_ttl": 300,       # 缓存有效期
            "real_time_sources": ["akshare"],
            "fallback_to_history": True
        }
    
    def subscribe(self, symbols: List[str], callback: Callable, 
                 data_types: List[str] = ["quote"]):
        """
        订阅实时数据
        """
        for symbol in symbols:
            if symbol not in self.subscribers:
                self.subscribers[symbol] = []
            
            self.subscribers[symbol].append({
                'callback': callback,
                'data_types': data_types,
                'last_data': None
            })
        
        logger.info(f"订阅 {len(symbols)} 只股票的实时数据")
    
    def unsubscribe(self, symbol: str, callback: Callable = None):
        """
        取消订阅
        """
        if symbol in self.subscribers:
            if callback is None:
                del self.subscribers[symbol]
            else:
                self.subscribers[symbol] = [
                    sub for sub in self.subscribers[symbol] 
                    if sub['callback'] != callback
                ]
                
                if not self.subscribers[symbol]:
                    del self.subscribers[symbol]
    
    def start(self):
        """启动市场数据服务"""
        if self.is_running:
            logger.warning("市场数据服务已经在运行")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._data_loop, daemon=True)
        self.thread.start()
        logger.info("市场数据服务已启动")
    
    def stop(self):
        """停止市场数据服务"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("市场数据服务已停止")
    
    def _data_loop(self):
        """数据循环"""
        while self.is_running:
            try:
                self._update_all_subscriptions()
                time.sleep(self.config.get("polling_interval", 60))
            except Exception as e:
                logger.error(f"市场数据循环出错: {e}")
                time.sleep(10)  # 出错后等待10秒再继续
    
    def _update_all_subscriptions(self):
        """更新所有订阅"""
        for symbol in list(self.subscribers.keys()):
            self._update_symbol_data(symbol)
    
    def _update_symbol_data(self, symbol: str):
        """更新单个符号的数据"""
        try:
            # 检查缓存
            current_time = datetime.now()
            if (symbol in self.cache and 
                symbol in self.last_update and
                (current_time - self.last_update[symbol]).total_seconds() < self.config.get("cache_ttl", 300)):
                # 使用缓存数据
                data = self.cache[symbol]
            else:
                # 获取新数据
                data = self._get_real_time_data(symbol)
                if data is not None:
                    self.cache[symbol] = data
                    self.last_update[symbol] = current_time
            
            # 通知订阅者
            if data is not None and symbol in self.subscribers:
                for subscriber in self.subscribers[symbol]:
                    self._notify_subscriber(subscriber, symbol, data)
                    
        except Exception as e:
            logger.error(f"更新 {symbol} 数据失败: {e}")
    
    def _get_real_time_data(self, symbol: str) -> Optional[Dict]:
        """获取实时数据"""
        if self.data_collector is None:
            logger.error("DataCollector未初始化")
            return None
            
        # 尝试实时数据源
        for source_name in self.config.get("real_time_sources", ["akshare"]):
            try:
                if source_name == "akshare":
                    data = self.data_collector.get_current_price(symbol)
                    if data is not None:
                        return data
            except Exception as e:
                logger.warning(f"实时数据源 {source_name} 失败: {e}")
        
        # 备用方案：使用最新历史数据
        if self.config.get("fallback_to_history", True):
            try:
                recent_data = self.data_collector.download_recent_data(symbol, days=1)
                if recent_data is not None and not recent_data.empty:
                    latest = recent_data.iloc[-1]
                    return {
                        'symbol': symbol,
                        'open': latest.get('Open', 0),
                        'close': latest.get('Close', 0),
                        'high': latest.get('High', 0),
                        'low': latest.get('Low', 0),
                        'volume': latest.get('Volume', 0),
                        'amount': latest.get('Amount', 0),
                        'source': 'historical_fallback'
                    }
            except Exception as e:
                logger.warning(f"历史数据回退失败: {e}")
        
        return None
    
    def _notify_subscriber(self, subscriber: Dict, symbol: str, data: Dict):
        """通知订阅者"""
        try:
            subscriber['callback'](symbol, data)
        except Exception as e:
            logger.error(f"通知订阅者失败: {e}")
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        批量获取报价
        """
        quotes = {}
        
        for symbol in symbols:
            quote = self._get_real_time_data(symbol)
            if quote is not None:
                quotes[symbol] = quote
        
        return quotes
    
    def get_market_status(self) -> Dict:
        """
        获取市场状态
        """
        # 这里可以添加市场开市状态、交易时间等逻辑
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # 简单的A股交易时间判断
        is_trading_hours = (
            (current_time.weekday() < 5) and  # 周一到周五
            ((9 <= current_hour < 12) or (13 <= current_hour < 15))  # 交易时间段
        )
        
        return {
            'is_trading': is_trading_hours,
            'current_time': current_time.isoformat(),
            'subscribed_symbols': list(self.subscribers.keys()),
            'cache_size': len(self.cache)
        }

    def get_current_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取当前数据 - 供TradingAgent调用
        """
        try:
            # 获取实时数据
            realtime_data = self._get_real_time_data(symbol)
            if realtime_data is None:
                return None
            
            # 获取历史数据用于技术指标计算
            historical_data = self.data_collector.download_recent_data(symbol, days=60)
            
            # 组合数据
            market_data = {
                'symbol': symbol,
                'current_price': realtime_data.get('current', 0),
                'open': realtime_data.get('open', 0),
                'high': realtime_data.get('high', 0),
                'low': realtime_data.get('low', 0),
                'volume': realtime_data.get('volume', 0),
                'amount': realtime_data.get('amount', 0),
                'timestamp': datetime.now().isoformat(),
                'historical_data': historical_data
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"获取当前数据失败 {symbol}: {e}")
            return None
        
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d"):
        """获取历史数据"""
        try:
            logger.info(f"获取 {symbol} 历史数据: {start_date} 到 {end_date}")
            
            # 使用 data_collector 获取历史数据
            historical_data = self.data_collector.get_stock_data(
                symbol, start_date, end_date, adjust_type="qfq"
            )
            
            if historical_data is None or historical_data.empty:
                logger.warning(f"获取到空的 {symbol} 历史数据")
                return None
                
            # 确保数据包含必要的列
            required_columns = ['Close', 'Volume']
            for col in required_columns:
                if col not in historical_data.columns:
                    logger.error(f"历史数据缺少列: {col}")
                    return None
            
            logger.info(f"成功获取 {symbol} 历史数据，共 {len(historical_data)} 条")
            return historical_data
            
        except Exception as e:
            logger.error(f"获取 {symbol} 历史数据失败: {e}")
            return None    
        
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            "is_running": self.is_running,
            "subscribed_symbols_count": len(self.subscribers),
            "cache_size": len(self.cache),
            "market_status": self.get_market_status()
        }
    
    def shutdown(self):
        """关闭市场数据服务"""
        self.stop()