# src/data/feature_engineer.py
import pandas as pd
import numpy as np
import talib
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from utils.logger import get_logger, debug_logger

# 使用项目统一的日志工具
logger = get_logger(__name__)

class FeatureEngineer:
    """
    特征工程器
    负责创建技术指标和特征
    """
    
    def __init__(self, config_path="config/feature_config.yaml"):
        self.config = self._load_config(config_path)
        self.feature_groups = self._initialize_feature_groups()
        # 新增：初始化数据收集器用于获取板块和大盘数据
        from data.data_collector import DataCollector
        self.data_collector = DataCollector()
        
        logger.info("特征工程器初始化完成 - 增强版（包含板块大盘特征）", extra={
            'trading_context': {
                'action': 'feature_engineer_enhanced_initialized',
                'feature_groups': list(self.feature_groups.keys()),
                'new_features': ['market_relative', 'sector_relative', 'capital_flow']
            }
        })
    
    def _load_config(self, config_path):
        """加载特征工程配置 - 增强版"""
        base_config = {
            "target_type": "close_close",
            "technical_indicators": {
                "trend": True,
                "momentum": True, 
                "volatility": True,
                "volume": True,
                "cycle": False
            },
            "feature_groups": {
                "price_features": True,
                "volume_features": True, 
                "technical_features": True,
                "statistical_features": True,
                "time_features": True,
                # 新增特征组
                "market_relative_features": True,
                "sector_relative_features": True,
                "capital_flow_features": True
            },
            "lag_periods": [1, 2, 3, 5, 10, 20],
            "rolling_windows": [5, 10, 20, 60],
            "create_interactions": True,
            # 新增配置
            "market_index": "000300",  # 沪深300
            "sector_classification": "sw",  # 申万行业分类
            "relative_strength_windows": [1, 3, 5, 10, 20]  # 相对强弱计算窗口
        }
        return base_config
    
    def _initialize_feature_groups(self):
        """初始化特征组 - 增强版"""
        groups = {}
        
        # 原有特征组
        if self.config["feature_groups"]["price_features"]:
            groups["price"] = self._create_price_features
        if self.config["feature_groups"]["volume_features"]:
            groups["volume"] = self._create_volume_features
        if self.config["feature_groups"]["technical_features"]:
            groups["technical"] = self._create_technical_features
        if self.config["feature_groups"]["statistical_features"]:
            groups["statistical"] = self._create_statistical_features
        if self.config["feature_groups"]["time_features"]:
            groups["time"] = self._create_time_features
        
        # 新增特征组
        if self.config["feature_groups"]["market_relative_features"]:
            groups["market_relative"] = self._create_market_relative_features
        if self.config["feature_groups"]["sector_relative_features"]:
            groups["sector_relative"] = self._create_sector_relative_features
        if self.config["feature_groups"]["capital_flow_features"]:
            groups["capital_flow"] = self._create_capital_flow_features
            
        return groups

    # === 新增：大盘相对强弱特征 ===
    def _create_market_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建大盘相对强弱特征（优化版本）"""
        try:
            symbol = "000300"
            debug_logger.log_feature_engineering("market_features", "creating", {
                "market_index": symbol
            })
            
            # 获取大盘数据
            start_date = df.index.min().strftime("%Y%m%d")
            end_date = df.index.max().strftime("%Y%m%d")
            market_data = self.data_collector.get_index_data(symbol, start_date, end_date)
            
            if market_data is None or market_data.empty:
                logger.warning("无法获取大盘数据，跳过大盘特征", extra={
                    'trading_context': {
                        'warning': 'market_data_unavailable',
                        'market_index': symbol
                    }
                })
                return df
            
            # 确保日期对齐
            market_data = market_data.reindex(df.index)
            market_close = market_data['Close']
            
            # 计算收益率
            market_returns = market_close.pct_change()
            stock_returns = df['Close'].pct_change()
            
            # 使用字典收集所有新特征
            new_features = {}
            
            for window in self.config["relative_strength_windows"]:
                stock_momentum = stock_returns.rolling(window).mean()
                market_momentum = market_returns.rolling(window).mean()
                
                # 收集特征
                new_features[f'vs_market_strength_{window}d'] = stock_momentum - market_momentum
                new_features[f'vs_market_ratio_{window}d'] = (1 + stock_momentum) / (1 + market_momentum) - 1
                
                # 相对强弱方向一致性
                stock_direction = stock_returns.rolling(window).apply(lambda x: (x > 0).mean())
                market_direction = market_returns.rolling(window).apply(lambda x: (x > 0).mean())
                new_features[f'vs_market_direction_{window}d'] = stock_direction - market_direction
            
            # 大盘技术指标
            new_features['market_trend_5d'] = market_returns.rolling(5).mean()
            new_features['market_trend_20d'] = market_returns.rolling(20).mean()
            new_features['market_volatility_20d'] = market_returns.rolling(20).std()
            
            # 大盘状态特征
            new_features['market_bullish_5d'] = (market_returns.rolling(5).mean() > 0).astype(int)
            new_features['market_bullish_20d'] = (market_returns.rolling(20).mean() > 0).astype(int)
            
            # 一次性创建所有新列
            if new_features:
                features_df = pd.DataFrame(new_features, index=df.index)
                df = pd.concat([df, features_df], axis=1)
            
            debug_logger.log_feature_engineering("market_features", "created", {
                "features_added": len(new_features)
            })
            
            return df.copy()  # 使用copy()消除碎片化
            
        except Exception as e:
            logger.error("创建大盘特征失败", extra={
                'trading_context': {
                    'error': 'market_features_failed',
                    'error_message': str(e)
                }
            })
            return df

    # === 新增：板块相对强弱特征 ===
    def _create_sector_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建板块相对强弱特征（优化版本）"""
        try:
            debug_logger.log_feature_engineering("sector_features", "creating", {})
            
            # 简化的板块相对强弱计算
            stock_returns = df['Close'].pct_change()
            
            # 模拟板块数据
            sector_returns = stock_returns.rolling(10).mean() + np.random.normal(0, 0.001, len(stock_returns))
            
            # 使用字典收集所有新特征，避免逐个添加列
            new_features = {}
            
            for window in self.config["relative_strength_windows"]:
                # 个股相对板块强弱
                stock_momentum = stock_returns.rolling(window).mean()
                sector_momentum = sector_returns.rolling(window).mean()
                
                # 收集特征到字典中
                new_features[f'vs_sector_strength_{window}d'] = stock_momentum - sector_momentum
                new_features[f'vs_sector_ratio_{window}d'] = (1 + stock_momentum) / (1 + sector_momentum) - 1
                
                # 板块内相对排名（优化版本）
                rolling_returns = stock_returns.rolling(window).mean()
                # 确保有足够的数据计算排名
                if len(rolling_returns.dropna()) >= window:
                    new_features[f'sector_rank_{window}d'] = rolling_returns.rank(pct=True)
            
            # 板块动量特征
            new_features['sector_momentum_5d'] = sector_returns.rolling(5).mean()
            new_features['sector_momentum_10d'] = sector_returns.rolling(10).mean()
            
            # 一次性创建所有新列
            if new_features:
                features_df = pd.DataFrame(new_features, index=df.index)
                df = pd.concat([df, features_df], axis=1)
            
            debug_logger.log_feature_engineering("sector_features", "created", {
                "features_added": len(new_features)
            })
            
            return df.copy()  # 使用copy()消除碎片化
            
        except Exception as e:
            logger.error("创建板块特征失败", extra={
                'trading_context': {
                    'error': 'sector_features_failed',
                    'error_message': str(e)
                }
            })
            return df

    # === 新增：资金流向特征 ===
    def _create_capital_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建资金流向特征"""
        try:
            debug_logger.log_feature_engineering("capital_flow_features", "creating", {})
            
            # 使用成交量和价格变化模拟资金流向
            close = df['Close']
            volume = df['Volume']
            returns = close.pct_change()
            
            # 价格-成交量协同指标
            df['price_volume_sync'] = returns * np.log1p(volume)
            
            # 资金流向强度
            df['capital_flow_1d'] = returns * volume
            df['capital_flow_5d'] = returns.rolling(5).mean() * volume.rolling(5).mean()
            df['capital_flow_20d'] = returns.rolling(20).mean() * volume.rolling(20).mean()
            
            # 量价背离检测
            price_trend = returns.rolling(5).mean()
            volume_trend = volume.pct_change().rolling(5).mean()
            df['price_volume_divergence'] = price_trend - volume_trend
            
            # 主力资金行为指标（简化版）
            large_volume_threshold = volume.rolling(20).quantile(0.8)
            df['large_volume_ratio'] = (volume > large_volume_threshold).astype(int)
            df['large_volume_impact'] = df['large_volume_ratio'] * returns.abs()
            
            # 资金流向一致性
            df['flow_consistency_5d'] = (returns.rolling(5).apply(lambda x: (x > 0).mean()) * 
                                       volume.rolling(5).apply(lambda x: (x > x.median()).mean()))
            
            debug_logger.log_feature_engineering("capital_flow_features", "created", {
                "features_added": len([col for col in df.columns if 'flow' in col or 'volume' in col or 'capital' in col])
            })
            
            return df
            
        except Exception as e:
            logger.error("创建资金流向特征失败", extra={
                'trading_context': {
                    'error': 'capital_flow_features_failed',
                    'error_message': str(e)
                }
            })
            return df

    # 修改现有的create_features方法，确保新特征被调用    
    def create_features(self, data: pd.DataFrame, target_type: str = "close_close", 
                       for_prediction: bool = False) -> Optional[pd.DataFrame]:
        """
        创建特征 - 主入口方法（优化版本，避免碎片化警告）
        """
        return self.calculate_features_optimized(data, target_type, for_prediction)
    
    def calculate_features_optimized(self, data: pd.DataFrame, target_type: str = "close_close", 
                                   for_prediction: bool = False) -> Optional[pd.DataFrame]:
        """
        优化版本的特征计算 - 避免DataFrame碎片化
        """
        if data is None or data.empty:
            logger.error("输入数据为空", extra={
                'trading_context': {'error': 'empty_input_data'}
            })
            return None
        
        df = data.copy()
        original_columns = set(df.columns)
        start_time = datetime.now()
        
        logger.info("开始优化版特征工程", extra={
            'trading_context': {
                'action': 'optimized_feature_engineering_start',
                'data_points': len(df),
                'target_type': target_type,
                'for_prediction': for_prediction
            }
        })
        
        try:
            # 1. 按顺序创建所有特征组
            for group_name, feature_func in self.feature_groups.items():
                group_start = datetime.now()
                df = feature_func(df)
                group_duration = (datetime.now() - group_start).total_seconds()
                
                new_cols = set(df.columns) - original_columns
                debug_logger.log_feature_engineering("feature_group", group_name, {
                    "duration_seconds": group_duration,
                    "features_added": len(new_cols)
                })
                original_columns = set(df.columns)
            
            # 2. 创建滞后特征
            lag_start = datetime.now()
            df = self._create_lag_features(df)
            lag_features_count = len([col for col in df.columns if '_lag_' in col])
            debug_logger.log_feature_engineering("lag_features", "created", {
                "duration_seconds": (datetime.now() - lag_start).total_seconds(),
                "lag_features_count": lag_features_count
            })
            
            # 3. 创建交互特征
            if self.config["create_interactions"]:
                interaction_start = datetime.now()
                df = self._create_interaction_features(df)
                interaction_features_count = len([col for col in df.columns if 'interaction' in col or 'cross' in col])
                debug_logger.log_feature_engineering("interaction_features", "created", {
                    "duration_seconds": (datetime.now() - interaction_start).total_seconds(),
                    "interaction_features_count": interaction_features_count
                })
            
            # 4. 创建目标变量（如果不是预测模式）
            if not for_prediction:
                target_start = datetime.now()
                df = self._create_target_variable(df, target_type)
                debug_logger.log_feature_engineering("target_variable", "created", {
                    "duration_seconds": (datetime.now() - target_start).total_seconds(),
                    "target_type": target_type
                })
            
            # 5. 清理数据
            clean_start = datetime.now()
            df = self._clean_features(df)
            debug_logger.log_feature_engineering("data_cleaning", "completed", {
                "duration_seconds": (datetime.now() - clean_start).total_seconds(),
                "final_data_points": len(df)
            })
            
            # 统计结果
            new_features = set(df.columns) - original_columns
            enhanced_features = [f for f in new_features if any(x in f for x in ['market', 'sector', 'flow', 'capital'])]
            total_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info("优化版特征工程完成", extra={
                'trading_context': {
                    'action': 'optimized_feature_engineering_complete',
                    'new_features_created': len(new_features),
                    'enhanced_features_count': len(enhanced_features),
                    'total_features': len(df.columns),
                    'final_data_points': len(df),
                    'duration_seconds': total_duration,
                    'enhanced_feature_types': {
                        'market_relative': len([f for f in enhanced_features if 'market' in f]),
                        'sector_relative': len([f for f in enhanced_features if 'sector' in f]),
                        'capital_flow': len([f for f in enhanced_features if 'flow' in f or 'capital' in f])
                    }
                }
            })
            
            # 返回去碎片化的DataFrame
            return df.copy()
            
        except Exception as e:
            logger.error("优化版特征工程失败", extra={
                'trading_context': {
                    'error': 'optimized_feature_engineering_failed',
                    'error_message': str(e),
                    'duration_seconds': (datetime.now() - start_time).total_seconds()
                }
            })
            return None
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建价格相关特征"""
        if 'Close' not in df.columns:
            logger.warning("缺少Close列，无法创建价格特征", extra={
                'trading_context': {
                    'warning': 'missing_close_column'
                }
            })
            return df
        
        close = df['Close']
        open_price = df.get('Open', close)
        high = df.get('High', close)
        low = df.get('Low', close)
        
        debug_logger.log_feature_engineering("price_features", "creating", {
            "has_open": 'Open' in df.columns,
            "has_high": 'High' in df.columns,
            "has_low": 'Low' in df.columns
        })
        
        # 基础价格特征
        df['price_change'] = close.pct_change()
        df['price_change_abs'] = close.diff().abs()
        df['open_close_ratio'] = close / open_price
        df['high_low_ratio'] = high / low
        df['body_ratio'] = (close - open_price) / (high - low).replace(0, 1e-10)
        
        # 价格位置特征
        df['price_position'] = (close - low) / (high - low).replace(0, 1e-10)
        
        # 移动平均特征
        ma_features = []
        for window in self.config["rolling_windows"]:
            df[f'sma_{window}'] = close.rolling(window, min_periods=1).mean()
            df[f'ema_{window}'] = close.ewm(span=window, min_periods=1).mean()
            df[f'price_sma_ratio_{window}'] = close / df[f'sma_{window}'].replace(0, 1e-10)
            ma_features.extend([f'sma_{window}', f'ema_{window}', f'price_sma_ratio_{window}'])
        
        debug_logger.log_feature_engineering("price_features", "created", {
            "moving_average_features": len(ma_features),
            "basic_price_features": 5  # price_change, price_change_abs, open_close_ratio, high_low_ratio, body_ratio
        })
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建成交量相关特征"""
        if 'Volume' not in df.columns:
            logger.warning("缺少Volume列，无法创建成交量特征", extra={
                'trading_context': {
                    'warning': 'missing_volume_column'
                }
            })
            return df
        
        volume = df['Volume']
        close = df.get('Close', 1)
        
        debug_logger.log_feature_engineering("volume_features", "creating", {})
        
        # 基础成交量特征
        df['volume_change'] = volume.pct_change()
        df['volume_price_trend'] = volume * close
        
        # 成交量移动平均
        volume_ma_features = []
        for window in self.config["rolling_windows"]:
            df[f'volume_sma_{window}'] = volume.rolling(window, min_periods=1).mean()
            df[f'volume_ratio_{window}'] = volume / df[f'volume_sma_{window}'].replace(0, 1e-10)
            volume_ma_features.extend([f'volume_sma_{window}', f'volume_ratio_{window}'])
        
        # OBV能量潮
        df['obv'] = self._calculate_obv(close, volume)
        
        # 价量相关性
        volume_corr_features = []
        for window in [5, 10, 20]:
            df[f'price_volume_corr_{window}'] = close.rolling(window, min_periods=2).corr(volume)
            volume_corr_features.append(f'price_volume_corr_{window}')
        
        debug_logger.log_feature_engineering("volume_features", "created", {
            "volume_ma_features": len(volume_ma_features),
            "volume_correlation_features": len(volume_corr_features),
            "obv_created": 'obv' in df.columns
        })
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标特征"""
        if 'Close' not in df.columns:
            logger.warning("缺少Close列，无法创建技术指标", extra={
                'trading_context': {
                    'warning': 'missing_close_technical'
                }
            })
            return df
        
        debug_logger.log_feature_engineering("technical_features", "creating", {
            "technical_indicators_enabled": self.config["technical_indicators"]
        })
        
        # 确保数据是float64类型
        close = df['Close'].astype(np.float64).values
        high = df.get('High', df['Close']).astype(np.float64).values
        low = df.get('Low', df['Close']).astype(np.float64).values
        volume = df.get('Volume', pd.Series(1, index=df.index)).astype(np.float64).values
        
        technical_features_count = 0
        fallback_used = False
        
        try:
            # 趋势指标
            if self.config["technical_indicators"]["trend"]:
                df['sma_5'] = talib.SMA(close, timeperiod=5)
                df['sma_20'] = talib.SMA(close, timeperiod=20)
                df['ema_12'] = talib.EMA(close, timeperiod=12)
                df['ema_26'] = talib.EMA(close, timeperiod=26)
                
                # MACD
                macd, macdsignal, macdhist = talib.MACD(close)
                df['macd'] = macd
                df['macd_signal'] = macdsignal
                df['macd_hist'] = macdhist
                
                technical_features_count += 7  # sma_5, sma_20, ema_12, ema_26, macd, macd_signal, macd_hist
            
            # 动量指标
            if self.config["technical_indicators"]["momentum"]:
                df['rsi'] = talib.RSI(close)
                stoch_k, stoch_d = talib.STOCH(high, low, close)
                df['stoch_k'] = stoch_k
                df['stoch_d'] = stoch_d
                df['williams_r'] = talib.WILLR(high, low, close)
                df['cci'] = talib.CCI(high, low, close)
                
                technical_features_count += 5  # rsi, stoch_k, stoch_d, williams_r, cci
            
            # 波动率指标
            if self.config["technical_indicators"]["volatility"]:
                df['atr'] = talib.ATR(high, low, close)
                df['natr'] = talib.NATR(high, low, close)
                boll_upper, boll_middle, boll_lower = talib.BBANDS(close)
                df['boll_upper'] = boll_upper
                df['boll_middle'] = boll_middle
                df['boll_lower'] = boll_lower
                
                # 布林带位置
                boll_diff = df['boll_upper'] - df['boll_lower']
                df['boll_position'] = (close - df['boll_lower']) / boll_diff.replace(0, 1e-10)
                
                technical_features_count += 6  # atr, natr, boll_upper, boll_middle, boll_lower, boll_position
            
            # 成交量指标
            if self.config["technical_indicators"]["volume"]:
                df['ad'] = talib.AD(high, low, close, volume)
                df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
                
                technical_features_count += 2  # ad, adosc
            
        except Exception as e:
            logger.warning("技术指标计算失败，使用备用方法", extra={
                'trading_context': {
                    'warning': 'technical_indicators_fallback',
                    'error_message': str(e)
                }
            })
            # 如果TA-Lib计算失败，使用备用方法计算基础指标
            df = self._fallback_technical_indicators(df)
            fallback_used = True
            technical_features_count = 7  # 基础指标数量
        
        debug_logger.log_feature_engineering("technical_features", "created", {
            "technical_features_count": technical_features_count,
            "fallback_used": fallback_used,
            "indicators_created": {
                'trend': self.config["technical_indicators"]["trend"],
                'momentum': self.config["technical_indicators"]["momentum"],
                'volatility': self.config["technical_indicators"]["volatility"],
                'volume': self.config["technical_indicators"]["volume"]
            }
        })
        
        return df
    
    def _fallback_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """备用技术指标计算方法（当TA-Lib失败时使用）"""
        close = df['Close']
        high = df.get('High', close)
        low = df.get('Low', close)
        
        debug_logger.log_feature_engineering("technical_fallback", "using", {})
        
        # 基础移动平均
        df['sma_5'] = close.rolling(5, min_periods=1).mean()
        df['sma_20'] = close.rolling(20, min_periods=1).mean()
        df['ema_12'] = close.ewm(span=12, min_periods=1).mean()
        df['ema_26'] = close.ewm(span=26, min_periods=1).mean()
        
        # 简易MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 简易RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建统计特征"""
        if 'Close' not in df.columns:
            logger.warning("缺少Close列，无法创建统计特征", extra={
                'trading_context': {
                    'warning': 'missing_close_statistical'
                }
            })
            return df
        
        close = df['Close']
        returns = close.pct_change()
        
        debug_logger.log_feature_engineering("statistical_features", "creating", {})
        
        statistical_features_count = 0
        
        # 波动率特征
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = returns.rolling(window, min_periods=2).std()
            # 使用自定义函数计算偏度和峰度
            df[f'return_skew_{window}'] = returns.rolling(window, min_periods=3).apply(
                lambda x: x.skew() if len(x) >= 3 else np.nan, raw=False
            )
            df[f'return_kurtosis_{window}'] = returns.rolling(window, min_periods=4).apply(
                lambda x: x.kurtosis() if len(x) >= 4 else np.nan, raw=False
            )
            statistical_features_count += 3
        
        # 滚动统计量
        for window in [10, 20]:
            rolling_std = close.rolling(window, min_periods=2).std().replace(0, 1e-10)
            df[f'zscore_{window}'] = (close - close.rolling(window, min_periods=1).mean()) / rolling_std
            statistical_features_count += 1
        
        # 分位数特征
        for window in [20, 60]:
            rolling_q = close.rolling(window, min_periods=5)
            df[f'price_quantile_25_{window}'] = rolling_q.quantile(0.25)
            df[f'price_quantile_75_{window}'] = rolling_q.quantile(0.75)
            statistical_features_count += 2
        
        debug_logger.log_feature_engineering("statistical_features", "created", {
            "statistical_features_count": statistical_features_count,
            "feature_types": {
                'volatility': 3,  # 3 windows
                'skewness': 3,    # 3 windows  
                'kurtosis': 3,    # 3 windows
                'zscore': 2,      # 2 windows
                'quantiles': 4    # 2 windows * 2 quantiles
            }
        })
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间特征"""
        if df.index is None or len(df.index) == 0:
            logger.warning("数据索引为空，无法创建时间特征", extra={
                'trading_context': {
                    'warning': 'empty_index_time_features'
                }
            })
            return df
            
        # 确保索引是DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("数据索引不是DatetimeIndex，无法创建时间特征", extra={
                'trading_context': {
                    'warning': 'non_datetime_index',
                    'index_type': type(df.index).__name__
                }
            })
            return df
        
        debug_logger.log_feature_engineering("time_features", "creating", {
            "index_type": "DatetimeIndex",
            "date_range": f"{df.index.min()} to {df.index.max()}" if len(df.index) > 0 else "empty"
        })
            
        # 日期特征
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # 周特征
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        
        # 季节特征
        df['season'] = (df.index.month % 12 + 3) // 3
        
        debug_logger.log_feature_engineering("time_features", "created", {
            "time_features_count": 8,
            "feature_types": {
                'day_features': 2,
                'month_features': 3,
                'week_features': 1,
                'season_features': 1,
                'special_dates': 2
            }
        })
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建滞后特征"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        debug_logger.log_feature_engineering("lag_features", "creating", {
            "numeric_columns_count": len(numeric_columns),
            "lag_periods": self.config["lag_periods"],
            "expected_features": len(numeric_columns) * len(self.config["lag_periods"])
        })
        
        lag_data = {}
        
        for column in numeric_columns:
            for lag in self.config["lag_periods"]:
                lag_data[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
        # 一次性添加所有滞后特征
        if lag_data:
            lag_df = pd.DataFrame(lag_data, index=df.index)
            df = pd.concat([df, lag_df], axis=1)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        interaction_features = []
        
        debug_logger.log_feature_engineering("interaction_features", "creating", {
            "create_interactions": self.config["create_interactions"]
        })
        
        # 价格-成交量交互
        if all(col in df.columns for col in ['Close', 'Volume']):
            df['price_volume_interaction'] = df['Close'] * np.log1p(df['Volume'])
            interaction_features.append('price_volume_interaction')
        
        # 技术指标交互
        if all(col in df.columns for col in ['rsi', 'macd']):
            df['rsi_macd_interaction'] = df['rsi'] * df['macd']
            interaction_features.append('rsi_macd_interaction')
        
        # 移动平均交叉
        if all(col in df.columns for col in ['sma_5', 'sma_20']):
            df['sma_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
            df['sma_cross_ratio'] = df['sma_5'] / df['sma_20'].replace(0, 1e-10)
            interaction_features.extend(['sma_cross', 'sma_cross_ratio'])
        
        debug_logger.log_feature_engineering("interaction_features", "created", {
            "interaction_features_count": len(interaction_features),
            "interaction_types": interaction_features
        })
        
        return df
    
    def _create_target_variable(self, df: pd.DataFrame, target_type: str) -> pd.DataFrame:
        """创建目标变量"""
        try:
            debug_logger.log_feature_engineering("target_variable", "creating", {
                "target_type": target_type
            })
            
            if target_type == "open_close":
                if 'Open' in df.columns and 'Close' in df.columns:
                    # 明天收盘相对于开盘的涨跌
                    df['tomorrow_open'] = df['Open'].shift(-1)
                    df['tomorrow_close'] = df['Close'].shift(-1)
                    df['target'] = (df['tomorrow_close'] > df['tomorrow_open']).astype(int)
                    
                    # 删除临时列
                    df = df.drop(['tomorrow_open', 'tomorrow_close'], axis=1)
                    
                    debug_logger.log_feature_engineering("target_variable", "created", {
                        "target_type": "open_close",
                        "target_values_distribution": df['target'].value_counts().to_dict()
                    })
                else:
                    logger.warning("缺少Open或Close列，无法创建open_close目标变量", extra={
                        'trading_context': {
                            'warning': 'missing_columns_for_target',
                            'required_columns': ['Open', 'Close'],
                            'available_columns': list(df.columns)
                        }
                    })
            
            elif target_type == "close_close":
                if 'Close' in df.columns:
                    # 明天收盘相对于今天收盘的涨跌
                    df['tomorrow_close'] = df['Close'].shift(-1)
                    df['target'] = (df['tomorrow_close'] > df['Close']).astype(int)
                    
                    # 删除临时列
                    df = df.drop(['tomorrow_close'], axis=1)
                    
                    debug_logger.log_feature_engineering("target_variable", "created", {
                        "target_type": "close_close",
                        "target_values_distribution": df['target'].value_counts().to_dict()
                    })
                else:
                    logger.warning("缺少Close列，无法创建close_close目标变量", extra={
                        'trading_context': {
                            'warning': 'missing_columns_for_target',
                            'required_columns': ['Close'],
                            'available_columns': list(df.columns)
                        }
                    })
            
            else:
                logger.warning(f"不支持的目标变量类型: {target_type}", extra={
                    'trading_context': {
                        'warning': 'unsupported_target_type',
                        'target_type': target_type,
                        'supported_types': ['open_close', 'close_close']
                    }
                })
            
        except Exception as e:
            logger.error("创建目标变量失败", extra={
                'trading_context': {
                    'error': 'target_variable_creation_failed',
                    'error_message': str(e),
                    'target_type': target_type
                }
            })
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理特征数据"""
        try:
            original_shape = df.shape
            original_columns = set(df.columns)
            
            debug_logger.log_feature_engineering("data_cleaning", "starting", {
                "original_shape": original_shape,
                "original_columns": len(original_columns)
            })
            
            # 删除全为NaN的列
            df = df.dropna(axis=1, how='all')
            after_nan_clean = df.shape
            
            # 删除方差为0的常数列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
            if constant_cols:
                df = df.drop(columns=constant_cols)
                logger.info("删除常数列", extra={
                    'trading_context': {
                        'action': 'constant_columns_removed',
                        'constant_columns_count': len(constant_cols),
                        'constant_columns': constant_cols
                    }
                })
            
            after_constant_clean = df.shape
            
            # 修复：使用新的ffill()和bfill()方法替代fillna(method=...)
            df = df.ffill().bfill()
            
            # 删除仍然有NaN的行
            df = df.dropna()
            final_shape = df.shape
            
            debug_logger.log_feature_engineering("data_cleaning", "completed", {
                "rows_removed": original_shape[0] - final_shape[0],
                "columns_removed": original_shape[1] - final_shape[1],
                "nan_columns_removed": original_shape[1] - after_nan_clean[1],
                "constant_columns_removed": after_nan_clean[1] - after_constant_clean[1],
                "final_shape": final_shape
            })
            
            return df
            
        except Exception as e:
            logger.error("清理特征数据失败", extra={
                'trading_context': {
                    'error': 'feature_cleaning_failed',
                    'error_message': str(e)
                }
            })
            return df
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算OBV能量潮"""
        try:
            debug_logger.log_feature_engineering("obv_calculation", "starting", {
                "data_points": len(close)
            })
            
            obv = pd.Series(0, index=close.index)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            debug_logger.log_feature_engineering("obv_calculation", "completed", {
                "obv_range": f"{obv.min():.2f} to {obv.max():.2f}"
            })
            
            return obv
        except Exception as e:
            logger.error("计算OBV失败", extra={
                'trading_context': {
                    'error': 'obv_calculation_failed',
                    'error_message': str(e)
                }
            })
            return pd.Series(0, index=close.index)
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """获取特征重要性"""
        try:
            debug_logger.log_feature_engineering("feature_importance", "calculating", {
                "model_type": type(model).__name__,
                "features_count": len(feature_names)
            })
            
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                top_features = importance_df.head(10).to_dict('records')
                debug_logger.log_feature_engineering("feature_importance", "calculated", {
                    "top_features": top_features,
                    "total_features": len(importance_df)
                })
                
                return importance_df
            else:
                logger.warning("模型不支持特征重要性计算", extra={
                    'trading_context': {
                        'warning': 'feature_importance_not_supported',
                        'model_type': type(model).__name__
                    }
                })
                return pd.DataFrame()
        except Exception as e:
            logger.error("获取特征重要性失败", extra={
                'trading_context': {
                    'error': 'feature_importance_failed',
                    'error_message': str(e)
                }
            })
            return pd.DataFrame()