# src/utils/helpers.py
import os
import sys
import time
import random
import string
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def timer(func: Callable) -> Callable:
    """函数执行时间计时器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"函数 {func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result
    return wrapper

def retry(max_retries: int = 3, delay: float = 1.0, 
          exceptions: tuple = (Exception,), backoff_factor: float = 2.0):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1} 次失败: {e}. "
                            f"{current_delay} 秒后重试..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败"
                        )
            
            raise last_exception
        return wrapper
    return decorator

def singleton(cls):
    """单例装饰器"""
    instances = {}
    
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

def generate_id(prefix: str = "", length: int = 8) -> str:
    """生成唯一ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    
    if prefix:
        return f"{prefix}_{timestamp}_{random_str}"
    else:
        return f"{timestamp}_{random_str}"

def calculate_hash(data: Any) -> str:
    """计算数据的哈希值"""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    elif isinstance(data, pd.DataFrame):
        data_str = data.to_json()
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    if denominator == 0:
        return default
    return numerator / denominator

def normalize_data(data: pd.Series, method: str = "minmax") -> pd.Series:
    """数据归一化"""
    if method == "minmax":
        if data.max() == data.min():
            return pd.Series([0.5] * len(data), index=data.index)
        return (data - data.min()) / (data.max() - data.min())
    elif method == "zscore":
        if data.std() == 0:
            return pd.Series([0] * len(data), index=data.index)
        return (data - data.mean()) / data.std()
    else:
        raise ValueError(f"不支持的归一化方法: {method}")

def format_currency(amount: float, currency: str = "CNY") -> str:
    """格式化货币金额"""
    if currency == "CNY":
        return f"¥{amount:,.2f}"
    elif currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """格式化百分比"""
    return f"{value:.{decimals}%}"

def parse_date(date_str: str, formats: List[str] = None) -> Optional[datetime]:
    """解析日期字符串"""
    if formats is None:
        formats = ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%d/%m/%Y"]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"无法解析日期字符串: {date_str}")
    return None

def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """获取交易日期（简化版本，实际应该从数据源获取）"""
    # 这里应该调用数据源获取真实的交易日期
    # 目前返回所有工作日作为简化实现
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_days = [day for day in all_days if day.weekday() < 5]  # 周一到周五
    return trading_days

def calculate_position_size(account_value: float, risk_per_trade: float, 
                          stop_loss_pct: float, price: float) -> int:
    """计算仓位大小"""
    risk_amount = account_value * risk_per_trade
    risk_per_share = price * stop_loss_pct
    shares = int(risk_amount / risk_per_share)
    return max(1, shares)  # 至少1股

def detect_anomalies(data: pd.Series, method: str = "iqr", threshold: float = 3.0) -> pd.Series:
    """检测异常值"""
    if method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    else:
        raise ValueError(f"不支持的异常检测方法: {method}")

class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_stock_data(data: pd.DataFrame) -> Dict[str, Any]:
        """验证股票数据"""
        issues = []
        
        if data.empty:
            issues.append("数据为空")
            return {"is_valid": False, "issues": issues}
        
        # 检查必要列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"缺少必要列: {missing_columns}")
        
        # 检查数据范围
        if 'Close' in data.columns:
            if (data['Close'] <= 0).any():
                issues.append("存在非正收盘价")
        
        if 'Volume' in data.columns:
            if (data['Volume'] < 0).any():
                issues.append("存在负成交量")
        
        # 检查数据连续性
        if len(data) > 1:
            date_diff = data.index.to_series().diff().dt.days
            if (date_diff > 7).any():  # 假设最大间隔7天
                issues.append("数据存在较大时间间隔")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "record_count": len(data),
            "date_range": {
                "start": data.index.min(),
                "end": data.index.max()
            }
        }
    
    @staticmethod
    def validate_trade_data(trade_data: Dict) -> Dict[str, Any]:
        """验证交易数据"""
        issues = []
        
        required_fields = ['symbol', 'action', 'shares', 'price']
        missing_fields = [field for field in required_fields if field not in trade_data]
        if missing_fields:
            issues.append(f"缺少必要字段: {missing_fields}")
        
        if 'shares' in trade_data and trade_data['shares'] <= 0:
            issues.append("股数必须为正")
        
        if 'price' in trade_data and trade_data['price'] <= 0:
            issues.append("价格必须为正")
        
        if 'action' in trade_data and trade_data['action'] not in ['BUY', 'SELL']:
            issues.append("操作类型必须是 BUY 或 SELL")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }

class MemoryManager:
    """内存管理器"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # 常驻内存
                "vms_mb": memory_info.vms / 1024 / 1024,  # 虚拟内存
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not installed"}
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type == 'object':
                # 尝试转换为category类型
                if df_optimized[col].nunique() / len(df_optimized[col]) < 0.5:
                    df_optimized[col] = df_optimized[col].astype('category')
            
            elif col_type in ['int64', 'int32']:
                # 向下转换整数类型
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
            
            elif col_type in ['float64', 'float32']:
                # 向下转换浮点类型
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_optimized[col] = df_optimized[col].astype(np.float32)
        
        return df_optimized

def setup_environment():
    """设置运行环境"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # 设置matplotlib中文字体
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        pass
    
    logger.info("环境设置完成")