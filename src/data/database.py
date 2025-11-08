# src/data/database.py
import pandas as pd
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    数据库管理器
    负责数据的持久化存储和查询
    """
    
    def __init__(self, db_path: str = "data/database/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        try:
            # 股票数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    adjust_type TEXT,
                    source TEXT,
                    created_time TEXT,
                    PRIMARY KEY (symbol, date, adjust_type)
                )
            ''')
            
            # 特征数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feature_data (
                    symbol TEXT,
                    date TEXT,
                    feature_name TEXT,
                    feature_value REAL,
                    created_time TEXT,
                    PRIMARY KEY (symbol, date, feature_name)
                )
            ''')
            
            # 预测结果表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    symbol TEXT,
                    prediction_date TEXT,
                    prediction_type TEXT,
                    prediction_value REAL,
                    confidence REAL,
                    features_used TEXT,
                    model_version TEXT,
                    created_time TEXT,
                    PRIMARY KEY (symbol, prediction_date, prediction_type)
                )
            ''')
            
            # 交易记录表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    trade_date TEXT,
                    action TEXT,
                    shares INTEGER,
                    price REAL,
                    amount REAL,
                    commission REAL,
                    tax REAL,
                    account_id TEXT,
                    strategy TEXT,
                    reasoning TEXT,
                    created_time TEXT
                )
            ''')
            
            conn.commit()
            logger.info("数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
        finally:
            conn.close()
    
    def _get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def save_stock_data(self, symbol: str, data: pd.DataFrame, 
                       adjust_type: str = "qfq", source: str = "akshare"):
        """
        保存股票数据到数据库
        """
        if data is None or data.empty:
            return False
        
        try:
            conn = self._get_connection()
            
            # 准备数据
            records = []
            for date, row in data.iterrows():
                record = (
                    symbol,
                    date.strftime('%Y-%m-%d'),
                    row.get('Open'),
                    row.get('High'),
                    row.get('Low'),
                    row.get('Close'),
                    row.get('Volume'),
                    row.get('Amount'),
                    adjust_type,
                    source,
                    datetime.now().isoformat()
                )
                records.append(record)
            
            # 使用UPSERT避免重复
            conn.executemany('''
                INSERT OR REPLACE INTO stock_data 
                (symbol, date, open, high, low, close, volume, amount, adjust_type, source, created_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"保存 {symbol} 数据到数据库: {len(records)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"保存股票数据失败 {symbol}: {e}")
            return False
    
    def load_stock_data(self, symbol: str, start_date: str, end_date: str,
                       adjust_type: str = "qfq") -> Optional[pd.DataFrame]:
        """
        从数据库加载股票数据
        """
        try:
            conn = self._get_connection()
            
            query = '''
                SELECT date, open, high, low, close, volume, amount
                FROM stock_data
                WHERE symbol = ? AND date BETWEEN ? AND ? AND adjust_type = ?
                ORDER BY date
            '''
            
            df = pd.read_sql_query(query, conn, 
                                 params=[symbol, start_date, end_date, adjust_type],
                                 parse_dates=['date'])
            
            conn.close()
            
            if not df.empty:
                df.set_index('date', inplace=True)
                logger.debug(f"从数据库加载 {symbol} 数据: {len(df)} 条记录")
                return df
            else:
                logger.debug(f"数据库中没有 {symbol} 的数据")
                return None
                
        except Exception as e:
            logger.error(f"加载股票数据失败 {symbol}: {e}")
            return None
    
    def save_features(self, symbol: str, feature_data: pd.DataFrame):
        """
        保存特征数据
        """
        if feature_data is None or feature_data.empty:
            return False
        
        try:
            conn = self._get_connection()
            
            records = []
            for date, row in feature_data.iterrows():
                for feature_name, feature_value in row.items():
                    if pd.notna(feature_value):
                        record = (
                            symbol,
                            date.strftime('%Y-%m-%d'),
                            feature_name,
                            float(feature_value),
                            datetime.now().isoformat()
                        )
                        records.append(record)
            
            conn.executemany('''
                INSERT OR REPLACE INTO feature_data 
                (symbol, date, feature_name, feature_value, created_time)
                VALUES (?, ?, ?, ?, ?)
            ''', records)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"保存 {symbol} 特征数据: {len(records)} 条特征值")
            return True
            
        except Exception as e:
            logger.error(f"保存特征数据失败 {symbol}: {e}")
            return False
    
    def save_prediction(self, prediction_data: Dict):
        """
        保存预测结果
        """
        try:
            conn = self._get_connection()
            
            conn.execute('''
                INSERT OR REPLACE INTO predictions 
                (symbol, prediction_date, prediction_type, prediction_value, 
                 confidence, features_used, model_version, created_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data.get('symbol'),
                prediction_data.get('prediction_date'),
                prediction_data.get('prediction_type', 'open_close'),
                prediction_data.get('prediction_value'),
                prediction_data.get('confidence'),
                json.dumps(prediction_data.get('features_used', [])),
                prediction_data.get('model_version', '1.0'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"保存预测结果: {prediction_data.get('symbol')}")
            return True
            
        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")
            return False
    
    def save_trade(self, trade_data: Dict):
        """
        保存交易记录
        """
        try:
            conn = self._get_connection()
            
            conn.execute('''
                INSERT INTO trades 
                (trade_id, symbol, trade_date, action, shares, price, amount,
                 commission, tax, account_id, strategy, reasoning, created_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('trade_id'),
                trade_data.get('symbol'),
                trade_data.get('trade_date'),
                trade_data.get('action'),
                trade_data.get('shares'),
                trade_data.get('price'),
                trade_data.get('amount'),
                trade_data.get('commission'),
                trade_data.get('tax'),
                trade_data.get('account_id'),
                trade_data.get('strategy'),
                trade_data.get('reasoning'),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"保存交易记录: {trade_data.get('trade_id')}")
            return True
            
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
            return False
    
    def get_trading_history(self, symbol: str = None, start_date: str = None,
                           end_date: str = None) -> pd.DataFrame:
        """
        获取交易历史
        """
        try:
            conn = self._get_connection()
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                query += " AND trade_date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND trade_date <= ?"
                params.append(end_date)
            
            query += " ORDER BY trade_date"
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['trade_date', 'created_time'])
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"获取交易历史失败: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        获取绩效指标
        """
        try:
            conn = self._get_connection()
            
            # 计算总交易次数
            total_trades = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE trade_date BETWEEN ? AND ?",
                [start_date, end_date]
            ).fetchone()[0]
            
            # 计算盈利交易次数
            profitable_trades = conn.execute('''
                SELECT COUNT(*) FROM trades 
                WHERE trade_date BETWEEN ? AND ? AND action = 'SELL'
                AND (amount - (shares * price) - commission - tax) > 0
            ''', [start_date, end_date]).fetchone()[0]
            
            # 计算总盈亏
            pnl_result = conn.execute('''
                SELECT 
                    SUM(CASE WHEN action = 'SELL' THEN 
                        (amount - (shares * price) - commission - tax) 
                    ELSE 0 END) as total_pnl
                FROM trades 
                WHERE trade_date BETWEEN ? AND ?
            ''', [start_date, end_date]).fetchone()
            
            total_pnl = pnl_result[0] if pnl_result[0] is not None else 0
            
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            conn.close()
            
            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'period': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            logger.error(f"获取绩效指标失败: {e}")
            return {}