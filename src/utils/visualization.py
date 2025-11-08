# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class TradingVisualizer:
    """
    交易可视化工具
    生成各种交易相关的图表
    """
    
    def __init__(self, style: str = "seaborn"):
        self.style = style
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """设置绘图样式"""
        if self.style == "seaborn":
            # 使用seaborn的现代主题设置
            sns.set_theme(style="whitegrid")  # 或者其他主题如 darkgrid, white, dark, ticks
            sns.set_palette("husl")
        elif self.style == "plotly":
            # Plotly 样式在具体方法中设置
            pass
        else:
            plt.style.use('default')
    
    def plot_portfolio_performance(self, performance_data: Dict[str, List], 
                                 save_path: Optional[str] = None) -> go.Figure:
        """绘制组合绩效图表"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('组合价值曲线', '累计收益率', '每日收益率', '回撤分析'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": True}]]
            )
            
            dates = performance_data['dates']
            portfolio_values = performance_data['portfolio_values']
            benchmark_values = performance_data.get('benchmark_values')
            daily_returns = performance_data['daily_returns']
            
            # 组合价值曲线
            fig.add_trace(
                go.Scatter(x=dates, y=portfolio_values, name='组合价值', line=dict(color='blue')),
                row=1, col=1
            )
            
            if benchmark_values:
                fig.add_trace(
                    go.Scatter(x=dates, y=benchmark_values, name='基准', line=dict(color='red', dash='dash')),
                    row=1, col=1
            )
            
            # 累计收益率
            cumulative_returns = [(v / portfolio_values[0] - 1) * 100 for v in portfolio_values]
            fig.add_trace(
                go.Scatter(x=dates, y=cumulative_returns, name='累计收益率', line=dict(color='green')),
                row=1, col=2
            )
            
            # 每日收益率
            fig.add_trace(
                go.Bar(x=dates[1:], y=daily_returns, name='日收益率', marker_color='lightblue'),
                row=2, col=1
            )
            
            # 回撤分析
            drawdowns = self._calculate_drawdowns(portfolio_values)
            fig.add_trace(
                go.Scatter(x=dates, y=drawdowns, name='回撤', line=dict(color='red'), fill='tozeroy'),
                row=2, col=2
            )
            
            fig.update_layout(
                title='投资组合绩效分析',
                height=800,
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"绩效图表已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制绩效图表失败: {e}")
            return go.Figure()
    
    def plot_trade_analysis(self, trades: List[Dict], save_path: Optional[str] = None) -> go.Figure:
        """绘制交易分析图表"""
        try:
            if not trades:
                logger.warning("没有交易数据可分析")
                return go.Figure()
            
            # 准备交易数据
            trade_df = pd.DataFrame(trades)
            trade_df['trade_date'] = pd.to_datetime(trade_df['trade_date'])
            trade_df['pnl_pct'] = trade_df['pnl'] / (trade_df['shares'] * trade_df['price'])
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('交易盈亏分布', '累计盈亏', '交易时间分布', '个股表现'),
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # 交易盈亏分布
            fig.add_trace(
                go.Histogram(x=trade_df['pnl'], name='盈亏分布', nbinsx=50),
                row=1, col=1
            )
            
            # 累计盈亏
            trade_df = trade_df.sort_values('trade_date')
            trade_df['cumulative_pnl'] = trade_df['pnl'].cumsum()
            fig.add_trace(
                go.Scatter(x=trade_df['trade_date'], y=trade_df['cumulative_pnl'], 
                          name='累计盈亏', line=dict(color='blue')),
                row=1, col=2
            )
            
            # 交易时间分布（按月）
            monthly_trades = trade_df.groupby(trade_df['trade_date'].dt.to_period('M')).size()
            fig.add_trace(
                go.Bar(x=monthly_trades.index.astype(str), y=monthly_trades.values,
                      name='月度交易次数'),
                row=2, col=1
            )
            
            # 个股表现
            stock_performance = trade_df.groupby('symbol')['pnl'].sum().sort_values()
            fig.add_trace(
                go.Bar(x=stock_performance.index, y=stock_performance.values,
                      name='个股盈亏'),
                row=2, col=2
            )
            
            fig.update_layout(
                title='交易分析报告',
                height=800,
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"交易分析图表已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制交易分析图表失败: {e}")
            return go.Figure()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 20, save_path: Optional[str] = None) -> plt.Figure:
        """绘制特征重要性图表"""
        try:
            # 取前top_n个特征
            top_features = feature_importance.head(top_n)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            y_pos = range(len(top_features))
            bars = ax.barh(y_pos, top_features['importance'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('特征重要性')
            ax.set_title(f'Top {top_n} 特征重要性')
            
            # 在条形图右侧添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.4f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"特征重要性图表已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制特征重要性图表失败: {e}")
            return plt.Figure()
    
    def plot_correlation_heatmap(self, data: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """绘制相关性热力图"""
        try:
            # 计算相关性矩阵
            corr_matrix = data.corr()
            
            fig, ax = plt.subplots(figsize=(16, 14))
            
            # 创建热力图
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                       center=0, square=True, linewidths=0.5, ax=ax,
                       cbar_kws={"shrink": 0.8})
            
            ax.set_title('特征相关性热力图')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"相关性热力图已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制相关性热力图失败: {e}")
            return plt.Figure()
    
    def plot_model_performance(self, training_history: Dict, save_path: Optional[str] = None) -> go.Figure:
        """绘制模型性能图表"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('训练准确率', '验证准确率', '训练损失', '特征重要性'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 训练准确率
            if 'train_accuracy' in training_history:
                fig.add_trace(
                    go.Scatter(y=training_history['train_accuracy'], 
                              name='训练准确率', line=dict(color='blue')),
                    row=1, col=1
                )
            
            # 验证准确率
            if 'val_accuracy' in training_history:
                fig.add_trace(
                    go.Scatter(y=training_history['val_accuracy'], 
                              name='验证准确率', line=dict(color='red')),
                    row=1, col=2
                )
            
            # 训练损失
            if 'train_loss' in training_history:
                fig.add_trace(
                    go.Scatter(y=training_history['train_loss'], 
                              name='训练损失', line=dict(color='green')),
                    row=2, col=1
                )
            
            # 特征重要性（如果有）
            if 'feature_importance' in training_history:
                importance_df = training_history['feature_importance']
                top_features = importance_df.head(10)
                
                fig.add_trace(
                    go.Bar(x=top_features['importance'], y=top_features['feature'],
                          name='特征重要性', orientation='h'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='模型训练性能',
                height=600,
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"模型性能图表已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"绘制模型性能图表失败: {e}")
            return go.Figure()
    
    def _calculate_drawdowns(self, portfolio_values: List[float]) -> List[float]:
        """计算回撤序列"""
        peak = portfolio_values[0]
        drawdowns = []
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        return drawdowns

class RealTimeMonitor:
    """实时监控可视化"""
    
    def __init__(self):
        self.figures = {}
    
    def create_dashboard(self, metrics: Dict[str, Any]) -> go.Figure:
        """创建实时监控仪表盘"""
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('组合价值', '今日盈亏', '持仓分布', '风险指标', '交易信号', '市场状态'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "pie"}, {"type": "bar"}],
                       [{"type": "table"}, {"type": "table"}]]
            )
            
            # 组合价值指标
            current_value = metrics.get('current_portfolio_value', 0)
            initial_capital = metrics.get('initial_capital', 100000)
            total_return = (current_value - initial_capital) / initial_capital * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=current_value,
                    number={'prefix': "¥", 'valueformat': ".2f"},
                    delta={'position': "top", 'reference': initial_capital},
                    title={"text": "组合价值"},
                    domain={'row': 0, 'column': 0}
                ),
                row=1, col=1
            )
            
            # 今日盈亏
            today_pnl = metrics.get('today_pnl', 0)
            today_pnl_pct = metrics.get('today_pnl_pct', 0)
            
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=today_pnl,
                    number={'prefix': "¥", 'valueformat': ".2f"},
                    delta={'position': "top", 'reference': 0, 'relative': False},
                    title={"text": "今日盈亏"},
                    domain={'row': 0, 'column': 1}
                ),
                row=1, col=2
            )
            
            # 持仓分布
            if 'positions' in metrics:
                positions = metrics['positions']
                symbols = list(positions.keys())
                values = [pos['market_value'] for pos in positions.values()]
                
                fig.add_trace(
                    go.Pie(labels=symbols, values=values, name="持仓分布"),
                    row=2, col=1
                )
            
            # 风险指标
            risk_metrics = metrics.get('risk_metrics', {})
            risk_values = list(risk_metrics.values())
            risk_names = list(risk_metrics.keys())
            
            fig.add_trace(
                go.Bar(x=risk_names, y=risk_values, name="风险指标"),
                    row=2, col=2
            )
            
            # 交易信号
            signals = metrics.get('signals', [])
            if signals:
                signal_df = pd.DataFrame(signals)
                fig.add_trace(
                    go.Table(
                        header=dict(values=list(signal_df.columns)),
                        cells=dict(values=[signal_df[col] for col in signal_df.columns])
                    ),
                    row=3, col=1
                )
            
            # 市场状态
            market_status = metrics.get('market_status', {})
            if market_status:
                status_df = pd.DataFrame([market_status])
                fig.add_trace(
                    go.Table(
                        header=dict(values=list(status_df.columns)),
                        cells=dict(values=[status_df[col] for col in status_df.columns])
                    ),
                    row=3, col=2
                )
            
            fig.update_layout(
                title='交易系统实时监控',
                height=1000
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"创建监控仪表盘失败: {e}")
            return go.Figure()