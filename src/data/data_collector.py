# src/data/data_collector.py
import pandas as pd
import numpy as np
import akshare as ak
import tushare as ts
import yfinance as yf
import requests
import json
import os
import time
import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path

from utils.logger import get_logger, debug_logger

# 使用项目统一的日志工具
logger = get_logger(__name__)

class DataCollector:
    """
    统一数据收集器
    支持多个数据源，提供统一的数据获取接口
    """
    
    def __init__(self, config_path="config/data_sources.yaml"):
        self.config = self._load_config(config_path)
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_data_sources()
        
        logger.info("数据收集器初始化完成", extra={
            'trading_context': {
                'action': 'data_collector_initialized',
                'sources': list(self.sources.keys()),
                'cache_enabled': self.config["cache"]["enabled"]
            }
        })
        
    def _load_config(self, config_path):
        """加载数据源配置"""
        default_config = {
            "data_sources": {
                "akshare": {
                    "enabled": True,
                    "priority": 1,
                    "rate_limit": 1.0  # 请求间隔秒数
                },
                "tushare": {
                    "enabled": False,
                    "priority": 2,
                    "token": None
                },
                "yfinance": {
                    "enabled": True,
                    "priority": 3,
                    "rate_limit": 0.5
                }
            },
            "cache": {
                "enabled": True,
                "ttl_minutes": 60  # 缓存有效期
            },
            "retry": {
                "max_retries": 3,
                "backoff_factor": 1.0
            }
        }
        
        # 这里可以添加从文件加载配置的逻辑
        return default_config
    
    def _initialize_data_sources(self):
        """初始化数据源"""
        self.sources = {}
        
        if self.config["data_sources"]["akshare"]["enabled"]:
            self.sources["akshare"] = AkshareDataSource(self.config["data_sources"]["akshare"])
        
        if self.config["data_sources"]["tushare"]["enabled"]:
            self.sources["tushare"] = TushareDataSource(self.config["data_sources"]["tushare"])
            
        if self.config["data_sources"]["yfinance"]["enabled"]:
            self.sources["yfinance"] = YFinanceDataSource(self.config["data_sources"]["yfinance"])
        
        debug_logger.log_data_processing("data_collector", "sources_initialized", {
            "available_sources": list(self.sources.keys())
        })
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, 
                      adjust_type: str = "qfq", source: str = "auto", cache_check : bool = True) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据
        """
        debug_logger.log_data_processing(symbol, "data_fetch_start", {
            "start_date": start_date,
            "end_date": end_date,
            "adjust_type": adjust_type,
            "source": source
        })
        
        # 检查缓存
        cache_key = f"stock_{symbol}_{start_date}_{end_date}_{adjust_type}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None and cache_check is True:
            debug_logger.log_data_processing(symbol, "cache_hit", {
                "cache_key": cache_key,
                "data_points": len(cached_data)
            })
            return cached_data
        
        debug_logger.log_data_processing(symbol, "cache_miss", {"cache_key": cache_key})
        
        # 确定数据源
        data_source = self._select_data_source(symbol, source)
        if data_source is None:
            logger.error(f"没有可用的数据源获取 {symbol} 数据", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'no_available_data_source',
                    'requested_source': source,
                    'available_sources': list(self.sources.keys())
                }
            })
            return None
        
        try:
            data = data_source.get_stock_data(symbol, start_date, end_date, adjust_type)
            if data is not None and not data.empty:
                # 缓存数据
                self._save_to_cache(cache_key, data)
                logger.info(f"成功获取 {symbol} 数据", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'action': 'data_fetch_success',
                        'data_points': len(data),
                        'data_source': data_source.__class__.__name__,
                        'date_range': f"{start_date} to {end_date}"
                    }
                })
                return data
            else:
                logger.warning(f"数据源返回空数据: {symbol}", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'warning': 'empty_data_from_source',
                        'data_source': data_source.__class__.__name__
                    }
                })
                return None
                
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'data_fetch_failed',
                    'error_message': str(e),
                    'data_source': data_source.__class__.__name__
                }
            })
            return None
    
    def download_recent_data(self, symbol: str, days: int = 120, 
                           adjust_type: str = "qfq") -> Optional[pd.DataFrame]:
        """
        下载近期数据（用于预测）
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        debug_logger.log_data_processing(symbol, "recent_data_request", {
            "days": days,
            "adjust_type": adjust_type,
            "date_range": f"{start_date} to {end_date}"
        })
        
        return self.get_stock_data(symbol, start_date, end_date, adjust_type)
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        获取当前价格信息
        """
        debug_logger.log_data_processing(symbol, "current_price_request", {})
        
        # 优先使用akshare获取实时数据
        if "akshare" in self.sources:
            try:
                realtime_data = self.sources["akshare"].get_realtime_quote(symbol)
                if realtime_data:
                    logger.debug(f"成功获取 {symbol} 实时价格", extra={
                        'trading_context': {
                            'symbol': symbol,
                            'action': 'realtime_price_success',
                            'current_price': realtime_data.get('current'),
                            'data_source': 'akshare'
                        }
                    })
                    return realtime_data
                else:
                    logger.warning(f"akshare实时数据为空: {symbol}", extra={
                        'trading_context': {
                            'symbol': symbol,
                            'warning': 'empty_realtime_data',
                            'data_source': 'akshare'
                        }
                    })
            except Exception as e:
                logger.warning(f"akshare实时数据获取失败", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'warning': 'realtime_data_failed',
                        'error_message': str(e),
                        'data_source': 'akshare'
                    }
                })
        
        # 备用方案：从最新历史数据获取
        debug_logger.log_data_processing(symbol, "fallback_to_historical", {})
        recent_data = self.download_recent_data(symbol, days=1)
        if recent_data is not None and not recent_data.empty:
            latest = recent_data.iloc[-1]
            price_data = {
                'symbol': symbol,
                'open': latest.get('Open', 0),
                'close': latest.get('Close', 0),
                'high': latest.get('High', 0),
                'low': latest.get('Low', 0),
                'volume': latest.get('Volume', 0),
                'amount': latest.get('Amount', 0)
            }
            logger.debug(f"使用历史数据作为当前价格: {symbol}", extra={
                'trading_context': {
                    'symbol': symbol,
                    'action': 'historical_price_used',
                    'price_data': price_data
                }
            })
            return price_data
        
        logger.warning(f"无法获取 {symbol} 的当前价格", extra={
            'trading_context': {
                'symbol': symbol,
                'warning': 'no_price_data_available'
            }
        })
        return None
    
    def get_multiple_stocks_data(self, symbols: List[str], start_date: str, 
                               end_date: str, adjust_type: str = "qfq") -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票数据
        """
        logger.info(f"开始批量获取 {len(symbols)} 只股票数据", extra={
            'trading_context': {
                'action': 'batch_data_fetch_start',
                'symbol_count': len(symbols),
                'date_range': f"{start_date} to {end_date}",
                'adjust_type': adjust_type
            }
        })
        
        results = {}
        start_time = datetime.now()
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, start_date, end_date, adjust_type)
            if data is not None:
                results[symbol] = data
            else:
                logger.warning(f"跳过 {symbol}，数据获取失败", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'warning': 'symbol_skipped_data_fetch_failed'
                    }
                })
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"批量获取完成", extra={
            'trading_context': {
                'action': 'batch_data_fetch_complete',
                'successful_fetches': len(results),
                'total_symbols': len(symbols),
                'success_rate': len(results) / len(symbols) if symbols else 0,
                'duration_seconds': duration
            }
        })
        
        return results
    
    def get_multiple_indices_data(self, index_codes: List[str], start_date: str, 
                            end_date: str) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个指数数据
        """
        logger.info(f"开始批量获取 {len(index_codes)} 个指数数据", extra={
            'trading_context': {
                'action': 'batch_index_fetch_start',
                'index_count': len(index_codes),
                'date_range': f"{start_date} to {end_date}"
            }
        })
        
        results = {}
        start_time = datetime.now()
        
        for index_code in index_codes:
            data = self.get_index_data(index_code, start_date, end_date)
            if data is not None:
                results[index_code] = data
            else:
                logger.warning(f"跳过指数 {index_code}，数据获取失败", extra={
                    'trading_context': {
                        'index_code': index_code,
                        'warning': 'index_skipped_data_fetch_failed'
                    }
                })
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"批量指数获取完成", extra={
            'trading_context': {
                'action': 'batch_index_fetch_complete',
                'successful_fetches': len(results),
                'total_indices': len(index_codes),
                'success_rate': len(results) / len(index_codes) if index_codes else 0,
                'duration_seconds': duration
            }
        })
        
        return results
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str, 
                    source: str = "auto", cache_check: bool = True) -> Optional[pd.DataFrame]:
        """
        获取指数数据（带缓存版本）
        """
        debug_logger.log_data_processing(index_code, "index_data_request", {
            "index_code": index_code,
            "start_date": start_date,
            "end_date": end_date,
            "source": source
        })
        
        # 检查缓存
        cache_key = f"index_{index_code}_{start_date}_{end_date}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None and cache_check is True:
            debug_logger.log_data_processing(index_code, "cache_hit", {
                "cache_key": cache_key,
                "data_points": len(cached_data)
            })
            return cached_data
        
        debug_logger.log_data_processing(index_code, "cache_miss", {"cache_key": cache_key})
        
        # 确定数据源
        data_source = None
        if source == "auto" and "akshare" in self.sources:
            data_source = self.sources["akshare"]
        elif source in self.sources:
            data_source = self.sources[source]
        
        if data_source is None:
            logger.error(f"没有可用的数据源获取指数 {index_code} 数据", extra={
                'trading_context': {
                    'index_code': index_code,
                    'error': 'no_available_data_source',
                    'requested_source': source,
                    'available_sources': list(self.sources.keys())
                }
            })
            return None
        
        try:
            # 调用数据源的指数数据获取方法
            if hasattr(data_source, 'get_index_data'):
                data = data_source.get_index_data(index_code, start_date, end_date)
            elif hasattr(data_source, 'get_index_data_tencent'):
                data = data_source.get_index_data_tencent(index_code, start_date, end_date)
            else:
                logger.error(f"数据源不支持指数数据获取", extra={
                    'trading_context': {
                        'index_code': index_code,
                        'error': 'index_data_not_supported',
                        'data_source': data_source.__class__.__name__
                    }
                })
                return None
            
            if data is not None and not data.empty:
                # 缓存数据
                self._save_to_cache(cache_key, data)
                logger.info(f"成功获取指数 {index_code} 数据", extra={
                    'trading_context': {
                        'index_code': index_code,
                        'action': 'index_data_success',
                        'data_points': len(data),
                        'data_source': data_source.__class__.__name__,
                        'date_range': f"{start_date} to {end_date}"
                    }
                })
                return data
            else:
                logger.warning(f"数据源返回空指数数据: {index_code}", extra={
                    'trading_context': {
                        'index_code': index_code,
                        'warning': 'empty_index_data_from_source',
                        'data_source': data_source.__class__.__name__
                    }
                })
                return None
                
        except Exception as e:
            logger.error(f"获取指数 {index_code} 数据失败", extra={
                'trading_context': {
                    'index_code': index_code,
                    'error': 'index_data_fetch_failed',
                    'error_message': str(e),
                    'data_source': data_source.__class__.__name__
                }
            })
            return None
    
    def get_fundamental_data(self, symbol: str, report_date: str = None) -> Optional[Dict]:
        """
        获取基本面数据
        """
        debug_logger.log_data_processing(symbol, "fundamental_data_request", {
            "symbol": symbol,
            "report_date": report_date
        })
        
        if "akshare" in self.sources:
            data = self.sources["akshare"].get_fundamental_data(symbol, report_date)
            if data is not None:
                logger.debug(f"成功获取基本面数据: {symbol}", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'action': 'fundamental_data_success',
                        'data_items': len(data)
                    }
                })
            return data
        
        logger.warning(f"无法获取基本面数据，akshare数据源不可用", extra={
            'trading_context': {
                'symbol': symbol,
                'warning': 'fundamental_data_source_unavailable'
            }
        })
        return None
    
    def _select_data_source(self, symbol: str, source: str) -> Optional[object]:
        """
        选择数据源
        """
        if source != "auto":
            selected = self.sources.get(source)
            if selected:
                debug_logger.log_data_processing(symbol, "source_selected", {
                    "source": source,
                    "selection_type": "manual"
                })
                return selected
            else:
                logger.warning(f"指定的数据源不可用: {source}", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'warning': 'specified_source_unavailable',
                        'requested_source': source,
                        'available_sources': list(self.sources.keys())
                    }
                })
                return None
        
        # 自动选择数据源
        for source_name in sorted(self.sources.keys(), 
                                key=lambda x: self.config["data_sources"][x]["priority"]):
            if self.sources[source_name].supports_symbol(symbol):
                debug_logger.log_data_processing(symbol, "source_selected", {
                    "source": source_name,
                    "selection_type": "auto",
                    "priority": self.config["data_sources"][source_name]["priority"]
                })
                return self.sources[source_name]
        
        logger.warning(f"没有找到支持 {symbol} 的数据源", extra={
            'trading_context': {
                'symbol': symbol,
                'warning': 'no_supported_source',
                'available_sources': list(self.sources.keys())
            }
        })
        return None
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        if not self.config["cache"]["enabled"]:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            # 检查缓存是否过期
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            cache_age = (datetime.now() - file_time).total_seconds()
            ttl_seconds = self.config["cache"]["ttl_minutes"] * 60
            
            if cache_age < ttl_seconds:
                try:
                    data = pd.read_parquet(cache_file)
                    debug_logger.log_data_processing("cache", "cache_hit_details", {
                        "cache_key": cache_key,
                        "cache_age_seconds": cache_age,
                        "data_points": len(data)
                    })
                    return data
                except Exception as e:
                    logger.warning(f"读取缓存失败", extra={
                        'trading_context': {
                            'error': 'cache_read_failed',
                            'cache_key': cache_key,
                            'error_message': str(e)
                        }
                    })
            else:
                debug_logger.log_data_processing("cache", "cache_expired", {
                    "cache_key": cache_key,
                    "cache_age_seconds": cache_age,
                    "ttl_seconds": ttl_seconds
                })
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """保存数据到缓存"""
        if not self.config["cache"]["enabled"]:
            return
            
        try:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            data.to_parquet(cache_file)
            debug_logger.log_data_processing("cache", "cache_saved", {
                "cache_key": cache_key,
                "data_points": len(data),
                "cache_file": str(cache_file)
            })
        except Exception as e:
            logger.warning(f"保存缓存失败", extra={
                'trading_context': {
                    'error': 'cache_save_failed',
                    'cache_key': cache_key,
                    'error_message': str(e)
                }
            })

# 数据源实现类
class AkshareDataSource:
    """akshare数据源实现"""
    
    def __init__(self, config):
        self.config = config
        self.last_request_time = 0
        self.logger = get_logger(f"{__name__}.akshare")
        
    def supports_symbol(self, symbol: str) -> bool:
        """检查是否支持该股票代码"""
        return True  # akshare支持所有A股
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, 
                      adjust_type: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        self._rate_limit()
        
        debug_logger.log_data_processing(symbol, "akshare_request", {
            "start_date": start_date,
            "end_date": end_date,
            "adjust_type": adjust_type
        })
        
        try:
            # 调用akshare接口
            df = ak.stock_zh_a_hist(
                symbol=symbol, 
                period="daily", 
                start_date=start_date, 
                end_date=end_date, 
                adjust=adjust_type
            )
            
            if df.empty:
                self.logger.warning(f"akshare返回空数据", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'warning': 'akshare_empty_response'
                    }
                })
                return None
            
            # 标准化列名
            df.rename(columns={
                '日期': 'Date',
                '开盘': 'Open',
                '收盘': 'Close', 
                '最高': 'High',
                '最低': 'Low',
                '成交量': 'Volume',
                '成交额': 'Amount',
                '振幅': 'Amplitude',
                '涨跌幅': 'ChangePct',
                '涨跌额': 'Change',
                '换手率': 'Turnover'
            }, inplace=True)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            debug_logger.log_data_processing(symbol, "akshare_success", {
                "data_points": len(df),
                "columns": list(df.columns)
            })
            
            return df
            
        except Exception as e:
            self.logger.error(f"akshare获取数据失败", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'akshare_fetch_failed',
                    'error_message': str(e)
                }
            })
            return None
    
    def get_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """获取实时行情"""
        self._rate_limit()
        
        # debug_logger.log_data_processing(symbol, "akshare_realtime_request", {})
        debug_logger.log_data_processing(symbol, "tc_realtime_request", {})

        try:
            # 获取实时数据
            # df = ak.stock_zh_a_spot_em()
            # stock_data = df[df['代码'] == symbol]

            # 确定市场前缀
            if symbol.startswith('6'):
                market_symbol = f"sh{symbol}"
            else:
                market_symbol = f"sz{symbol}"
            
            url = f"http://qt.gtimg.cn/q={market_symbol}"

            session = requests.Session()
            session.trust_env = False  # 确保不使用代理
            
            response = session.get(url, timeout=10)

            # if not stock_data.empty:
            #     quote_data = {
            #         'symbol': symbol,
            #        'name': stock_data.iloc[0]['名称'],
            #        'current': stock_data.iloc[0]['最新价'],
            #        'change': stock_data.iloc[0]['涨跌额'],
            #        'change_pct': stock_data.iloc[0]['涨跌幅'],
            #        'volume': stock_data.iloc[0]['成交量'],
            #        'amount': stock_data.iloc[0]['成交额']
            #    }
            #    debug_logger.log_data_processing(symbol, "akshare_realtime_success", {
            #        "quote_data": quote_data
            #    })
            #    return quote_data

            if response.status_code == 200:
                # 解析数据
                match = re.search(r'="([^"]+)"', response.text)
                if match:
                    data = match.group(1).split('~')
                    
                    if len(data) >= 40:
                        current_price = float(data[3])
                        prev_close = float(data[4])
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100
                        
                        return {
                            'symbol': symbol,
                            'name': data[1],
                            'current': current_price,
                            'open': float(data[5]),
                            'change': change,
                            'change_pct': change_pct,
                            'volume': int(data[6]) * 100,  # 手转股
                            'amount': float(data[37]) * 10000 if data[37] else 0,  # 万转元
                            'source': '腾讯财经'
                        }
            
            self.logger.warning(f"未找到实时行情数据", extra={
                'trading_context': {
                    'symbol': symbol,
                    'warning': 'tc_realtime_not_found'
                }
            })
            return None
            
        except Exception as e:
            self.logger.error(f"获取实时行情失败", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'tc_realtime_failed',
                    'error_message': str(e)
                }
            })
            return None
    
    def get_index_data_tencent(self, index_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取指数数据（腾讯财经版）- 即将弃用，请使用 get_index_data"""
        import warnings
        warnings.warn("get_index_data_tencent 即将弃用，请使用 get_index_data 方法", DeprecationWarning)
        return self.get_index_data(index_code, start_date, end_date)
    
    def get_fundamental_data(self, symbol: str, report_date: str = None) -> Optional[Dict]:
        """获取基本面数据"""
        self._rate_limit()
        
        debug_logger.log_data_processing(symbol, "akshare_fundamental_request", {
            "report_date": report_date
        })
        
        try:
            # 获取股票基本信息
            info_df = ak.stock_individual_info_em(symbol=symbol)
            if info_df.empty:
                self.logger.warning(f"akshare返回空基本面数据", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'warning': 'akshare_fundamental_empty'
                    }
                })
                return None
                
            info_dict = {}
            for _, row in info_df.iterrows():
                info_dict[row['item']] = row['value']
            
            debug_logger.log_data_processing(symbol, "akshare_fundamental_success", {
                "data_items": len(info_dict)
            })
            return info_dict
            
        except Exception as e:
            self.logger.error(f"获取基本面数据失败", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'akshare_fundamental_failed',
                    'error_message': str(e)
                }
            })
            return None
    
    def _rate_limit(self):
        """速率限制"""
        current_time = datetime.now().timestamp()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.config.get("rate_limit", 1.0):
            sleep_time = self.config["rate_limit"] - time_since_last
            debug_logger.log_data_processing("akshare", "rate_limit", {
                "sleep_time": sleep_time
            })
            time.sleep(sleep_time)
        self.last_request_time = datetime.now().timestamp()

    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取指数数据（优化版本）"""
        self._rate_limit()
        
        debug_logger.log_data_processing(index_code, "index_data_request", {
            "start_date": start_date,
            "end_date": end_date
        })
        
        try:
            # 腾讯财经指数代码映射
            index_mapping = {
                '000001': 'sh000001',  # 上证指数
                '399001': 'sz399001',  # 深证成指
                '000300': 'sh000300',  # 沪深300
                '000016': 'sh000016',  # 上证50
                '399006': 'sz399006',  # 创业板指
                '000905': 'sh000905',  # 中证500
                '399005': 'sz399005',  # 中小板指
                '000688': 'sh000688',  # 科创50
            }
            
            tencent_code = index_mapping.get(index_code, 
                f'sh{index_code}' if index_code.startswith('000') else f'sz{index_code}')
            
            # 方法1: 尝试使用腾讯接口
            try:
                df = ak.stock_zh_index_daily_tx(symbol=tencent_code)
                if not df.empty:
                    return self._process_index_data(df, start_date, end_date, "腾讯财经")
            except Exception as e:
                self.logger.debug(f"腾讯指数接口失败，尝试备用接口: {e}")
            
            # 方法2: 备用接口 - 使用通用指数接口
            try:
                df = ak.index_zh_a_hist(symbol=index_code, period="daily", 
                                    start_date=start_date, end_date=end_date)
                if not df.empty:
                    return self._process_index_data(df, start_date, end_date, "akshare通用")
            except Exception as e:
                self.logger.debug(f"通用指数接口失败: {e}")
            
            # 方法3: 最后尝试 - 使用股票接口（某些指数可以作为股票获取）
            try:
                df = ak.stock_zh_a_hist(symbol=tencent_code, period="daily", 
                                    start_date=start_date, end_date=end_date)
                if not df.empty:
                    return self._process_index_data(df, start_date, end_date, "股票接口")
            except Exception as e:
                self.logger.debug(f"股票接口失败: {e}")
            
            self.logger.warning(f"所有指数数据接口都失败了", extra={
                'trading_context': {
                    'index_code': index_code,
                    'tencent_code': tencent_code,
                    'warning': 'all_index_interfaces_failed'
                }
            })
            return None
            
        except Exception as e:
            self.logger.error(f"获取指数数据失败", extra={
                'trading_context': {
                    'index_code': index_code,
                    'error': 'index_data_fetch_failed',
                    'error_message': str(e)
                }
            })
            return None

    def _process_index_data(self, df: pd.DataFrame, start_date: str, end_date: str, source: str) -> pd.DataFrame:
        """处理指数数据（统一格式）"""
        if df.empty:
            return df
        
        # 标准化列名
        column_mapping = {
            # 腾讯财经列名
            'date': 'Date', 'open': 'Open', 'close': 'Close', 
            'high': 'High', 'low': 'Low', 'volume': 'Volume', 'amount': 'Amount',
            # akshare通用列名  
            '日期': 'Date', '开盘': 'Open', '收盘': 'Close', '最高': 'High',
            '最低': 'Low', '成交量': 'Volume', '成交额': 'Amount'
        }
        
        # 只重命名实际存在的列
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df.rename(columns=existing_columns, inplace=True)
        
        # 确保Date列存在并设置为索引
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # 筛选日期范围
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        # 确保数值列的数据类型
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        if existing_numeric:
            df[existing_numeric] = df[existing_numeric].apply(pd.to_numeric, errors='coerce')
        
        debug_logger.log_data_processing("index", "data_processed", {
            "source": source,
            "data_points": len(df),
            "columns": list(df.columns)
        })
        
        return df

class TushareDataSource:
    """Tushare数据源实现"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(f"{__name__}.tushare")
        if config.get("token"):
            ts.set_token(config["token"])
        self.pro = ts.pro_api()
    
    def supports_symbol(self, symbol: str) -> bool:
        return True
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, 
                      adjust_type: str) -> Optional[pd.DataFrame]:
        # Tushare实现代码
        self.logger.info(f"Tushare数据获取: {symbol}", extra={
            'trading_context': {
                'symbol': symbol,
                'action': 'tushare_data_request',
                'start_date': start_date,
                'end_date': end_date
            }
        })
        # 这里添加实际的Tushare实现代码
        pass

class YFinanceDataSource:
    """YFinance数据源实现（用于港股、美股）"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(f"{__name__}.yfinance")
    
    def supports_symbol(self, symbol: str) -> bool:
        # 主要支持港股和美股
        return symbol.endswith('.HK') or not symbol.isdigit()
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, 
                      adjust_type: str) -> Optional[pd.DataFrame]:
        # YFinance实现代码
        self.logger.info(f"YFinance数据获取: {symbol}", extra={
            'trading_context': {
                'symbol': symbol,
                'action': 'yfinance_data_request',
                'start_date': start_date,
                'end_date': end_date
            }
        })
        # 这里添加实际的YFinance实现代码
        pass