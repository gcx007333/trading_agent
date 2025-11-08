# src/utils/performance.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PerformanceCalculator:
    """
    绩效计算器
    计算各种投资绩效指标
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.daily_returns = []
        self.daily_values = []
        self.dates = []
        self.trades = []
    
    def add_daily_value(self, date: datetime, portfolio_value: float):
        """添加每日组合价值"""
        self.dates.append(date)
        self.daily_values.append(portfolio_value)
        
        # 计算日收益率
        if len(self.daily_values) > 1:
            daily_return = (portfolio_value - self.daily_values[-2]) / self.daily_values[-2]
            self.daily_returns.append(daily_return)
        elif len(self.daily_values) == 1:
            self.daily_returns.append(0.0)
    
    def add_trade(self, trade_data: Dict):
        """添加交易记录"""
        self.trades.append(trade_data)
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """计算所有绩效指标"""
        if len(self.daily_returns) < 2:
            return {}
        
        returns_series = pd.Series(self.daily_returns[1:])  # 跳过第一个0
        
        metrics = {
            'total_return': self._calculate_total_return(),
            'annual_return': self._calculate_annual_return(returns_series),
            'annual_volatility': self._calculate_annual_volatility(returns_series),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns_series),
            'max_drawdown': self._calculate_max_drawdown(),
            'calmar_ratio': self._calculate_calmar_ratio(returns_series),
            'sortino_ratio': self._calculate_sortino_ratio(returns_series),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'avg_win': self._calculate_avg_win(),
            'avg_loss': self._calculate_avg_loss(),
            'max_win': self._calculate_max_win(),
            'max_loss': self._calculate_max_loss()
        }
        
        return metrics
    
    def _calculate_total_return(self) -> float:
        """计算总收益率"""
        if not self.daily_values:
            return 0.0
        return (self.daily_values[-1] - self.initial_capital) / self.initial_capital
    
    def _calculate_annual_return(self, returns_series: pd.Series) -> float:
        """计算年化收益率"""
        if len(returns_series) < 2:
            return 0.0
        
        total_days = (self.dates[-1] - self.dates[0]).days
        if total_days == 0:
            return 0.0
        
        total_return = self._calculate_total_return()
        annual_return = (1 + total_return) ** (365.25 / total_days) - 1
        return annual_return
    
    def _calculate_annual_volatility(self, returns_series: pd.Series) -> float:
        """计算年化波动率"""
        if len(returns_series) < 2:
            return 0.0
        
        daily_volatility = returns_series.std()
        annual_volatility = daily_volatility * np.sqrt(252)  # 年化
        return annual_volatility
    
    def _calculate_sharpe_ratio(self, returns_series: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        annual_return = self._calculate_annual_return(returns_series)
        annual_volatility = self._calculate_annual_volatility(returns_series)
        
        if annual_volatility == 0:
            return 0.0
        
        excess_return = annual_return - risk_free_rate
        return excess_return / annual_volatility
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.daily_values:
            return 0.0
        
        peak = self.daily_values[0]
        max_drawdown = 0.0
        
        for value in self.daily_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _calculate_calmar_ratio(self, returns_series: pd.Series) -> float:
        """计算Calmar比率"""
        annual_return = self._calculate_annual_return(returns_series)
        max_drawdown = self._calculate_max_drawdown()
        
        if max_drawdown == 0:
            return 0.0
        
        return annual_return / max_drawdown
    
    def _calculate_sortino_ratio(self, returns_series: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算Sortino比率"""
        annual_return = self._calculate_annual_return(returns_series)
        
        # 只计算负收益的标准差
        negative_returns = returns_series[returns_series < 0]
        if len(negative_returns) == 0:
            downside_volatility = 0.0
        else:
            downside_volatility = negative_returns.std() * np.sqrt(252)
        
        if downside_volatility == 0:
            return 0.0
        
        excess_return = annual_return - risk_free_rate
        return excess_return / downside_volatility
    
    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.trades:
            return 0.0
        
        profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        return len(profitable_trades) / len(self.trades)
    
    def _calculate_profit_factor(self) -> float:
        """计算盈利因子"""
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_avg_win(self) -> float:
        """计算平均盈利"""
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        if not winning_trades:
            return 0.0
        return np.mean([t.get('pnl', 0) for t in winning_trades])
    
    def _calculate_avg_loss(self) -> float:
        """计算平均亏损"""
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        if not losing_trades:
            return 0.0
        return np.mean([t.get('pnl', 0) for t in losing_trades])
    
    def _calculate_max_win(self) -> float:
        """计算最大盈利"""
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        if not winning_trades:
            return 0.0
        return max(t.get('pnl', 0) for t in winning_trades)
    
    def _calculate_max_loss(self) -> float:
        """计算最大亏损"""
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        if not losing_trades:
            return 0.0
        return min(t.get('pnl', 0) for t in losing_trades)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成绩效报告"""
        metrics = self.calculate_all_metrics()
        
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_value': self.daily_values[-1] if self.daily_values else self.initial_capital,
                'total_trades': len(self.trades),
                'period_start': self.dates[0] if self.dates else None,
                'period_end': self.dates[-1] if self.dates else None,
                'trading_days': len(self.daily_returns)
            },
            'returns': {
                'total_return': metrics.get('total_return', 0),
                'annual_return': metrics.get('annual_return', 0),
                'best_day': max(self.daily_returns) if self.daily_returns else 0,
                'worst_day': min(self.daily_returns) if self.daily_returns else 0
            },
            'risk': {
                'annual_volatility': metrics.get('annual_volatility', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0)
            },
            'trading': {
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'avg_win': metrics.get('avg_win', 0),
                'avg_loss': metrics.get('avg_loss', 0),
                'max_win': metrics.get('max_win', 0),
                'max_loss': metrics.get('max_loss', 0)
            }
        }
        
        return report

class RiskMetrics:
    """风险指标计算"""
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.95) -> float:
        """计算VaR（风险价值）"""
        if not returns:
            return 0.0
        
        returns_series = pd.Series(returns)
        var = returns_series.quantile(1 - confidence_level)
        return var
    
    @staticmethod
    def calculate_cvar(returns: List[float], confidence_level: float = 0.95) -> float:
        """计算CVaR（条件风险价值）"""
        if not returns:
            return 0.0
        
        returns_series = pd.Series(returns)
        var = RiskMetrics.calculate_var(returns, confidence_level)
        cvar = returns_series[returns_series <= var].mean()
        return cvar
    
    @staticmethod
    def calculate_beta(portfolio_returns: List[float], benchmark_returns: List[float]) -> float:
        """计算Beta系数"""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        portfolio_series = pd.Series(portfolio_returns)
        benchmark_series = pd.Series(benchmark_returns)
        
        covariance = portfolio_series.cov(benchmark_series)
        benchmark_variance = benchmark_series.var()
        
        if benchmark_variance == 0:
            return 0.0
        
        return covariance / benchmark_variance
    
    @staticmethod
    def calculate_tracking_error(portfolio_returns: List[float], benchmark_returns: List[float]) -> float:
        """计算跟踪误差"""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        active_returns = pd.Series(portfolio_returns) - pd.Series(benchmark_returns)
        tracking_error = active_returns.std() * np.sqrt(252)  # 年化
        return tracking_error

class BenchmarkComparator:
    """基准比较器"""
    
    def __init__(self, benchmark_returns: Dict[str, List[float]]):
        self.benchmark_returns = benchmark_returns
    
    def compare_performance(self, portfolio_returns: List[float]) -> Dict[str, Dict[str, float]]:
        """比较组合与基准的表现"""
        comparison = {}
        
        for benchmark_name, benchmark_returns in self.benchmark_returns.items():
            if len(portfolio_returns) != len(benchmark_returns):
                logger.warning(f"组合与基准 {benchmark_name} 数据长度不匹配")
                continue
            
            portfolio_series = pd.Series(portfolio_returns)
            benchmark_series = pd.Series(benchmark_returns)
            
            # 计算超额收益
            excess_returns = portfolio_series - benchmark_series
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            comparison[benchmark_name] = {
                'portfolio_return': portfolio_series.mean() * 252,
                'benchmark_return': benchmark_series.mean() * 252,
                'excess_return': excess_returns.mean() * 252,
                'information_ratio': information_ratio,
                'tracking_error': RiskMetrics.calculate_tracking_error(portfolio_returns, benchmark_returns),
                'beta': RiskMetrics.calculate_beta(portfolio_returns, benchmark_returns)
            }
        
        return comparison