# src/data/data_processor.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    数据处理器
    负责数据清洗、预处理和质量检查
    """
    
    def __init__(self, config_path="config/data_processing.yaml"):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path):
        """加载数据处理配置"""
        return {
            "cleaning": {
                "handle_missing": "drop",  # drop, fill, interpolate
                "fill_method": "ffill",    # 填充方法
                "remove_duplicates": True,
                "outlier_detection": True,
                "outlier_threshold": 3.0   # 标准差阈值
            },
            "validation": {
                "check_negative_prices": True,
                "check_volume_zero": True,
                "check_price_spikes": True
            }
        }
    
    def clean_data(self, data: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """
        数据清洗
        """
        if data is None or data.empty:
            return data
            
        df = data.copy()
        original_len = len(df)
        
        # 1. 处理缺失值
        df = self._handle_missing_values(df)
        
        # 2. 去除重复值
        if self.config["cleaning"]["remove_duplicates"]:
            df = df[~df.index.duplicated(keep='first')]
        
        # 3. 检测和处理异常值
        if self.config["cleaning"]["outlier_detection"]:
            df = self._handle_outliers(df)
        
        # 4. 数据验证
        validation_results = self._validate_data(df)
        if not validation_results["is_valid"]:
            logger.warning(f"数据验证失败 {symbol}: {validation_results['issues']}")
        
        cleaned_len = len(df)
        if original_len != cleaned_len:
            logger.info(f"数据清洗: {symbol} {original_len} -> {cleaned_len} 条记录")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        handle_method = self.config["cleaning"]["handle_missing"]
        
        if handle_method == "drop":
            # 只删除全部为NaN的行
            df = df.dropna(how='all')
            # 对于部分缺失，使用前向填充
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif handle_method == "fill":
            fill_method = self.config["cleaning"]["fill_method"]
            if fill_method == "ffill":
                df = df.fillna(method='ffill').fillna(method='bfill')
            elif fill_method == "interpolate":
                df = df.interpolate(method='linear')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['Open', 'High', 'Low', 'Close']:
                # 价格数据异常值处理
                df = self._handle_price_outliers(df, column)
            elif column == 'Volume':
                # 成交量异常值处理
                df = self._handle_volume_outliers(df, column)
        
        return df
    
    def _handle_price_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """处理价格异常值"""
        threshold = self.config["cleaning"]["outlier_threshold"]
        
        # 计算价格变化率
        price_pct_change = df[column].pct_change().abs()
        
        # 标记异常值（价格单日变化超过阈值标准差）
        outlier_mask = price_pct_change > price_pct_change.std() * threshold
        
        if outlier_mask.any():
            logger.debug(f"检测到 {column} 异常值: {outlier_mask.sum()} 个")
            # 使用前后值插值替换异常值
            df.loc[outlier_mask, column] = np.nan
            df[column] = df[column].interpolate(method='linear')
        
        return df
    
    def _handle_volume_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """处理成交量异常值"""
        threshold = self.config["cleaning"]["outlier_threshold"]
        
        # 计算成交量对数变化
        log_volume = np.log1p(df[column])
        volume_zscore = (log_volume - log_volume.mean()) / log_volume.std()
        
        # 标记异常值
        outlier_mask = volume_zscore.abs() > threshold
        
        if outlier_mask.any():
            logger.debug(f"检测到 {column} 异常值: {outlier_mask.sum()} 个")
            # 使用移动平均替换异常值
            df.loc[outlier_mask, column] = np.nan
            df[column] = df[column].fillna(df[column].rolling(5, min_periods=1).mean())
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> Dict:
        """数据验证"""
        issues = []
        
        # 检查负价格
        if self.config["validation"]["check_negative_prices"]:
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in df.columns and (df[col] <= 0).any():
                    issues.append(f"负价格出现在 {col}")
        
        # 检查零成交量
        if self.config["validation"]["check_volume_zero"]:
            if 'Volume' in df.columns and (df['Volume'] == 0).any():
                issues.append("存在零成交量")
        
        # 检查价格 spikes
        if self.config["validation"]["check_price_spikes"]:
            if all(col in df.columns for col in ['High', 'Low']):
                price_range = (df['High'] - df['Low']) / df['Low']
                spike_threshold = 0.2  # 20% 价格波动
                if (price_range > spike_threshold).any():
                    issues.append("检测到异常价格波动")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "records_count": len(df),
            "date_range": {
                "start": df.index.min() if not df.empty else None,
                "end": df.index.max() if not df.empty else None
            }
        }
    
    def calculate_returns(self, df: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
        """
        计算收益率
        """
        if price_column not in df.columns:
            logger.error(f"价格列不存在: {price_column}")
            return df
        
        df = df.copy()
        
        # 简单收益率
        df['return'] = df[price_column].pct_change()
        
        # 对数收益率
        df['log_return'] = np.log(df[price_column] / df[price_column].shift(1))
        
        # 累计收益率
        df['cumulative_return'] = (1 + df['return']).cumprod() - 1
        
        return df
    
    def resample_data(self, df: pd.DataFrame, freq: str = 'W') -> pd.DataFrame:
        """
        数据重采样
        """
        if df.empty:
            return df
        
        # 周线数据
        if freq == 'W':
            resampled = df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'Amount': 'sum'
            })
        # 月线数据
        elif freq == 'M':
            resampled = df.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'Amount': 'sum'
            })
        else:
            logger.warning(f"不支持的采样频率: {freq}")
            return df
        
        return resampled.dropna()
    
    def align_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        对齐多个股票的数据时间轴
        """
        if not data_dict:
            return {}
        
        # 找到共同的时间范围
        common_index = None
        for symbol, df in data_dict.items():
            if df is not None and not df.empty:
                if common_index is None:
                    common_index = df.index
                else:
                    common_index = common_index.intersection(df.index)
        
        if common_index is None or len(common_index) == 0:
            logger.warning("没有共同的时间范围")
            return data_dict
        
        # 对齐所有数据
        aligned_data = {}
        for symbol, df in data_dict.items():
            if df is not None and not df.empty:
                aligned_data[symbol] = df.reindex(common_index)
        
        logger.info(f"数据对齐完成: {len(common_index)} 个共同时间点")
        return aligned_data