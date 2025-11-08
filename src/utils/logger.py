# src/utils/logger.py
import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .config_loader import get_system_config

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器 - 修复重复显示问题"""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        # 默认格式，不强制包含trading_context
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录 - 修复重复显示问题"""
        # 处理trading_context，将其合并到消息中（如果不重复）
        if hasattr(record, 'trading_context') and record.trading_context:
            try:
                original_message = record.getMessage()
                
                if isinstance(record.trading_context, dict):
                    # 提取关键信息，避免重复
                    context_parts = []
                    
                    # 检查原始消息中是否已经包含这些信息
                    message_lower = original_message.lower()
                    
                    for key, value in record.trading_context.items():
                        # 如果消息中已经包含这个值，就不重复添加
                        value_str = str(value).lower()
                        if value_str not in message_lower:
                            context_parts.append(f"{key}={value}")
                    
                    if context_parts:
                        context_str = " | " + " | ".join(context_parts)
                        record.msg = original_message + context_str
            except Exception:
                # 如果处理失败，保持原样
                pass
        
        # 处理异常信息
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)
        else:
            record.exc_text = ""
        
        return super().format(record)

class TradingFilter(logging.Filter):
    """交易专用日志过滤器"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤日志记录"""
        # 添加交易相关上下文
        if not hasattr(record, 'trading_context'):
            record.trading_context = {}
        
        # 可以在这里添加交易特定的过滤逻辑
        return True

def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    设置日志系统
    """
    if config is None:
        config = get_system_config().logging
    
    # 创建日志目录
    log_path = Path(config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = StructuredFormatter(config.format)
    
    # 文件处理器 - 按大小轮转
    file_handler = logging.handlers.RotatingFileHandler(
        filename=config.file_path,
        maxBytes=config.max_file_size * 1024 * 1024,  # MB to bytes
        backupCount=config.backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(TradingFilter())
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(TradingFilter())
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logging.info("日志系统初始化完成")

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    获取指定名称的日志记录器
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger

class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, name: str = "performance"):
        self.logger = get_logger(name)
        self.metrics = {}
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """记录交易"""
        self.logger.info(
            "交易执行",
            extra={
                'trading_context': {
                    'symbol': trade_data.get('symbol'),
                    'action': trade_data.get('action'),
                    'shares': trade_data.get('shares'),
                    'price': trade_data.get('price'),
                    'pnl': trade_data.get('pnl', 0)
                }
            }
        )
    
    def log_prediction(self, prediction_data: Dict[str, Any]):
        """记录预测 - 修改为接受字典参数"""
        self.logger.info(
            "模型预测",
            extra={
                'trading_context': {
                    'symbol': prediction_data.get('symbol'),
                    'prediction': prediction_data.get('prediction'),
                    'confidence': prediction_data.get('confidence'),
                    'up_probability': prediction_data.get('up_probability'),
                    'duration_seconds': prediction_data.get('duration_seconds'),
                    'model_accuracy': prediction_data.get('model_accuracy')
                }
            }
        )
    
    def log_performance(self, metrics: Dict[str, Any]):
        """记录绩效指标"""
        self.metrics.update(metrics)
        self.logger.info(
            "绩效更新",
            extra={
                'trading_context': {
                    'metrics': metrics
                }
            }
        )
    
    def log_risk_event(self, event_type: str, details: Dict[str, Any]):
        """记录风险事件"""
        self.logger.warning(
            f"风险事件: {event_type}",
            extra={
                'trading_context': {
                    'risk_event': event_type,
                    'details': details
                }
            }
        )
    
    def log_model_training(self, symbol: str, metrics: Dict[str, Any]):
        """记录模型训练"""
        self.logger.info(
            "模型训练",
            extra={
                'trading_context': {
                    'symbol': symbol,
                    'training_metrics': metrics
                }
            }
        )

class DebugLogger:
    """调试日志记录器"""
    
    def __init__(self, name: str = "debug"):
        self.logger = get_logger(name, "DEBUG")
    
    def log_data_processing(self, symbol: str, step: str, details: Dict[str, Any]):
        """记录数据处理过程"""
        self.logger.debug(
            f"数据处理: {symbol} - {step}",
            extra={
                'trading_context': {
                    'symbol': symbol,
                    'processing_step': step,
                    'details': details
                }
            }
        )
    
    def log_feature_engineering(self, symbol: str, feature_count: int, details: Dict[str, Any]):
        """记录特征工程"""
        self.logger.debug(
            f"特征工程: {symbol}",
            extra={
                'trading_context': {
                    'symbol': symbol,
                    'feature_count': feature_count,
                    'feature_details': details
                }
            }
        )
    
    def log_model_training(self, symbol: str, metrics: Dict[str, Any]):
        """记录模型训练"""
        self.logger.debug(
            f"模型训练: {symbol}",
            extra={
                'trading_context': {
                    'symbol': symbol,
                    'training_metrics': metrics
                }
            }
        )

# 创建全局日志记录器实例
performance_logger = PerformanceLogger()
debug_logger = DebugLogger()

def initialize_logging():
    """初始化日志系统"""
    setup_logging()
    return performance_logger, debug_logger