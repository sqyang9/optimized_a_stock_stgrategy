"""
可视化模块 V2.0
合并 visualizer.py 和 visualizer_simple.py
新增: 基准对比、水下分析、盈亏分布、持仓时间分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from utils import setup_logger, format_percent, format_number

logger = setup_logger(__name__)


class StrategyVisualizer:
    """策略可视化器 V2.0"""
    
    def __init__(self, output_dir: str = "./output", simple_mode: bool = False):
        """
        Args:
            output_dir: 输出目录
            simple_mode: 简单模式（只生成基础图表）
        """
        self.output_dir = output_dir
        self.simple_mode = simple_mode
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置样式
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'strategy': '#2E86AB',      # 策略线 - 蓝色
            'benchmark': '#A23B72',     # 基准线 - 紫红色
            'positive': '#2ECC71',      # 正收益 - 绿色
            'negative': '#E74C3C',      # 负收益 - 红色
            'drawdown': '#E74C3C',      # 回撤 - 红色
            'underwater': '#3498DB',    # 水下区域 - 蓝色
        }
    
    def generate_report(self, 
                       daily_results: List[Dict],
                       trades: List,
                       initial_capital: float = 1_000_000,
                       benchmark_values: List[float] = None,
                       performance_summary: Dict = None) -> str:
        """生成完整可视化报告
        
        Args:
            daily_results: 每日结果列表
            trades: 交易记录列表
            initial_capital: 初始资金
            benchmark_values: 基准净值列表
            performance_summary: 绩效摘要
        
        Returns:
            HTML报告路径
        """
        logger.info("开始生成可视化报告...")
        
        # 准备数据
        df_daily = pd.DataFrame(daily_results)
        df_trades = self._prepare_trades_df(trades)
        
        # 处理数据格式
        df_daily = self._prepare_daily_data(df_daily)
        
        # 计算指标（如果未提供）
        if performance_summary is None:
            performance_summary = self._calculate_metrics(df_daily, initial_capital)
        
        # 生成图表
        self._plot_equity_with_benchmark(df_daily, benchmark_values, initial_capital)
        self._plot_underwater_chart(df_daily)
        
        if not self.simple_mode:
            self._plot_monthly_returns_heatmap(df_daily)
            self._plot_trade_analysis(df_trades)
            self._plot_risk_metrics(df_daily)
            self._plot_pnl_distribution(df_trades)
            self._plot_holding_analysis(df_trades)
        
        # 生成HTML报告
        html_path = self._generate_html_report(
            df_daily, df_trades, performance_summary, 
            benchmark_values, initial_capital
        )
        
        logger.info(f"可视化报告已生成: {html_path}")
        return html_path
    
    def _prepare_trades_df(self, trades: List) -> pd.DataFrame:
        """准备交易数据DataFrame"""
        if not trades:
            return pd.DataFrame()
        
        # 支持多种输入格式
        if hasattr(trades[0], '__dict__'):
            return pd.DataFrame([t.__dict__ for t in trades])
        elif isinstance(trades[0], dict):
            return pd.DataFrame(trades)
        else:
            return pd.DataFrame()
    
    def _prepare_daily_data(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """准备每日数据"""
        if df_daily.empty:
            return df_daily
        
        # 查找资产列
        asset_col = None
        date_col = None
        
        for col in df_daily.columns:
            if '资产' in col or 'asset' in col.lower() or 'value' in col.lower():
                asset_col = col
            if 'date' in col.lower() or '日期' in col:
                date_col = col
        
        if asset_col and asset_col != '总资产':
            df_daily['总资产'] = df_daily[asset_col]
        
        if date_col and date_col != 'date':
            df_daily['date'] = df_daily[date_col]
        
        # 确保数据类型
        if '总资产' in df_daily.columns:
            if df_daily['总资产'].dtype == 'object':
                df_daily['总资产'] = df_daily['总资产'].astype(str)
                df_daily['总资产'] = df_daily['总资产'].str.replace('万', '')
                df_daily['总资产'] = pd.to_numeric(df_daily['总资产'], errors='coerce') * 10000
        
        if 'date' in df_daily.columns:
            df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        return df_daily
    
    def _calculate_metrics(self, df_daily: pd.DataFrame, 
                          initial_capital: float) -> Dict:
        """计算绩效指标"""
        if df_daily.empty or '总资产' not in df_daily.columns:
            return {}
        
        try:
            final_value = float(df_daily['总资产'].iloc[-1])
            total_return = (final_value - initial_capital) / initial_capital
            
            daily_returns = df_daily['总资产'].pct_change().dropna()
            
            trading_days = len(df_daily)
            annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
            volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
            sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0
            
            # 最大回撤
            cumulative = (1 + daily_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            
            # 最长水下天数
            underwater_periods = []
            current_period = 0
            for dd in drawdown:
                if dd < 0:
                    current_period += 1
                else:
                    if current_period > 0:
                        underwater_periods.append(current_period)
                    current_period = 0
            if current_period > 0:
                underwater_periods.append(current_period)
            max_underwater_days = max(underwater_periods) if underwater_periods else 0
            
            win_days = len(daily_returns[daily_returns > 0])
            win_rate = win_days / len(daily_returns) if len(daily_returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'max_underwater_days': max_underwater_days,
                'win_rate': win_rate,
                'final_value': final_value,
                'trading_days': trading_days
            }
        except Exception as e:
            logger.error(f"计算指标时出错: {e}")
            return {}
    
    def _plot_equity_with_benchmark(self, df_daily: pd.DataFrame,
                                   benchmark_values: List[float],
                                   initial_capital: float):
        """绘制资金曲线（含基准对比）"""
        if df_daily.empty or '总资产' not in df_daily.columns or 'date' not in df_daily.columns:
            logger.warning("数据不完整，跳过资金曲线绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        dates = df_daily['date']
        strategy_values = df_daily['总资产'] / initial_capital  # 归一化
        
        # 策略净值曲线
        ax1.plot(dates, strategy_values, linewidth=2, 
                color=self.colors['strategy'], label='策略净值')
        
        # 基准净值曲线
        if benchmark_values and len(benchmark_values) == len(dates):
            ax1.plot(dates, benchmark_values, linewidth=1.5, 
                    color=self.colors['benchmark'], linestyle='--', 
                    label='沪深300基准', alpha=0.8)
        
        # 初始资金线
        ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='初始净值')
        
        # 添加统计信息
        final_return = (strategy_values.iloc[-1] - 1) * 100
        info_text = f"累计收益: {final_return:+.2f}%"
        if benchmark_values:
            bench_return = (benchmark_values[-1] - 1) * 100
            excess = final_return - bench_return
            info_text += f"\n基准收益: {bench_return:+.2f}%"
            info_text += f"\n超额收益: {excess:+.2f}%"
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_title('策略净值曲线 vs 基准', fontsize=14, fontweight='bold')
        ax1.set_ylabel('净值', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 日收益率
        daily_returns = df_daily['总资产'].pct_change().fillna(0)
        colors = [self.colors['positive'] if x >= 0 else self.colors['negative'] 
                 for x in daily_returns]
        ax2.bar(dates, daily_returns * 100, color=colors, alpha=0.7, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_title('日收益率', fontsize=12, fontweight='bold')
        ax2.set_ylabel('收益率 (%)', fontsize=11)
        ax2.set_xlabel('日期', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'equity_curve.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_underwater_chart(self, df_daily: pd.DataFrame):
        """绘制水下图（回撤分析）"""
        if df_daily.empty or '总资产' not in df_daily.columns:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        dates = df_daily['date']
        daily_returns = df_daily['总资产'].pct_change().fillna(0)
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = ((cumulative - rolling_max) / rolling_max) * 100
        
        # 水下区域填充
        ax.fill_between(dates, drawdown, 0, 
                       where=(drawdown < 0),
                       color=self.colors['drawdown'], 
                       alpha=0.4, label='回撤区域')
        ax.plot(dates, drawdown, color=self.colors['drawdown'], linewidth=1)
        
        # 标记最大回撤点
        min_idx = drawdown.idxmin()
        min_val = drawdown.min()
        min_date = dates.iloc[min_idx] if isinstance(min_idx, int) else dates.loc[min_idx]
        
        ax.scatter([min_date], [min_val], color='darkred', s=100, zorder=5)
        ax.annotate(f'最大回撤: {min_val:.2f}%',
                   xy=(min_date, min_val),
                   xytext=(20, 20), textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
        
        # 计算并显示最长水下期
        underwater_start = None
        max_period = 0
        max_period_start = None
        max_period_end = None
        current_period = 0
        
        for i, (date, dd) in enumerate(zip(dates, drawdown)):
            if dd < 0:
                if underwater_start is None:
                    underwater_start = date
                current_period += 1
            else:
                if current_period > max_period:
                    max_period = current_period
                    max_period_start = underwater_start
                    max_period_end = dates.iloc[i-1] if i > 0 else date
                underwater_start = None
                current_period = 0
        
        if current_period > max_period:
            max_period = current_period
            max_period_start = underwater_start
            max_period_end = dates.iloc[-1]
        
        # 标注最长水下期
        if max_period_start is not None and max_period > 5:
            mid_date = max_period_start + (max_period_end - max_period_start) / 2
            ax.axvspan(max_period_start, max_period_end, alpha=0.2, color='blue')
            ax.text(mid_date, drawdown.min() * 0.5, f'最长水下期\n{max_period}天',
                   ha='center', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_title('水下图（回撤分析）', fontsize=14, fontweight='bold')
        ax.set_ylabel('回撤 (%)', fontsize=11)
        ax.set_xlabel('日期', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=5)  # 留出顶部空间
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'underwater_chart.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_monthly_returns_heatmap(self, df_daily: pd.DataFrame):
        """绘制月度收益热力图"""
        if df_daily.empty or '总资产' not in df_daily.columns:
            return
        
        df = df_daily.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        monthly_returns = df['总资产'].resample('M').last().pct_change().fillna(0)
        monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
        
        monthly_df = monthly_returns.reset_index()
        monthly_df.columns = ['month', 'return']
        monthly_df['year'] = monthly_df['month'].str[:4]
        monthly_df['month_num'] = monthly_df['month'].str[5:7]
        
        pivot_table = monthly_df.pivot(index='year', columns='month_num', values='return') * 100
        
        # 月份标签
        month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                       '7月', '8月', '9月', '10月', '11月', '12月']
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 使用红绿色板
        cmap = sns.diverging_palette(10, 130, as_cmap=True)
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap=cmap,
                   center=0, cbar_kws={'label': '月收益率 (%)'},
                   ax=ax, linewidths=0.5)
        
        ax.set_xticklabels(month_labels[:len(pivot_table.columns)])
        ax.set_title('月度收益热力图', fontsize=14, fontweight='bold')
        ax.set_xlabel('月份', fontsize=11)
        ax.set_ylabel('年份', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'monthly_returns.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pnl_distribution(self, df_trades: pd.DataFrame):
        """绘制盈亏分布图"""
        if df_trades.empty:
            return
        
        sell_trades = df_trades[df_trades['direction'] == 'sell']
        if sell_trades.empty or 'pnl_pct' not in sell_trades.columns:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        pnl_pcts = sell_trades['pnl_pct'] * 100
        
        # 左图：盈亏分布直方图
        bins = np.linspace(pnl_pcts.min(), pnl_pcts.max(), 30)
        colors = [self.colors['positive'] if x >= 0 else self.colors['negative'] 
                 for x in bins[:-1]]
        
        n, bins_out, patches = ax1.hist(pnl_pcts, bins=bins, alpha=0.7, edgecolor='black')
        
        # 根据正负染色
        for i, patch in enumerate(patches):
            if bins_out[i] >= 0:
                patch.set_facecolor(self.colors['positive'])
            else:
                patch.set_facecolor(self.colors['negative'])
        
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax1.axvline(x=pnl_pcts.mean(), color='blue', linestyle='-', 
                   alpha=0.7, label=f'均值: {pnl_pcts.mean():.2f}%')
        
        # 添加统计信息
        win_rate = len(pnl_pcts[pnl_pcts > 0]) / len(pnl_pcts) * 100
        avg_win = pnl_pcts[pnl_pcts > 0].mean() if len(pnl_pcts[pnl_pcts > 0]) > 0 else 0
        avg_loss = pnl_pcts[pnl_pcts <= 0].mean() if len(pnl_pcts[pnl_pcts <= 0]) > 0 else 0
        
        stats_text = f'胜率: {win_rate:.1f}%\n平均盈利: {avg_win:.2f}%\n平均亏损: {avg_loss:.2f}%'
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_title('交易盈亏分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('收益率 (%)', fontsize=11)
        ax1.set_ylabel('交易次数', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 右图：累计盈亏曲线
        cumulative_pnl = pnl_pcts.cumsum()
        trade_nums = range(1, len(cumulative_pnl) + 1)
        
        ax2.plot(trade_nums, cumulative_pnl, linewidth=2, color=self.colors['strategy'])
        ax2.fill_between(trade_nums, cumulative_pnl, 0,
                        where=(cumulative_pnl >= 0), color=self.colors['positive'], alpha=0.3)
        ax2.fill_between(trade_nums, cumulative_pnl, 0,
                        where=(cumulative_pnl < 0), color=self.colors['negative'], alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax2.set_title('累计收益曲线（按交易）', fontsize=12, fontweight='bold')
        ax2.set_xlabel('交易序号', fontsize=11)
        ax2.set_ylabel('累计收益 (%)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pnl_distribution.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_holding_analysis(self, df_trades: pd.DataFrame):
        """绘制持仓时间分析"""
        if df_trades.empty:
            return
        
        sell_trades = df_trades[df_trades['direction'] == 'sell']
        if sell_trades.empty:
            return
        
        # 检查是否有持仓天数数据
        if 'hold_days' not in sell_trades.columns:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        hold_days = sell_trades['hold_days']
        pnl_pcts = sell_trades.get('pnl_pct', pd.Series([0] * len(sell_trades))) * 100
        
        # 左图：持仓天数分布
        ax1.hist(hold_days, bins=20, alpha=0.7, color=self.colors['strategy'], edgecolor='black')
        ax1.axvline(x=hold_days.mean(), color='red', linestyle='--', 
                   label=f'均值: {hold_days.mean():.1f}天')
        ax1.axvline(x=hold_days.median(), color='orange', linestyle='--',
                   label=f'中位数: {hold_days.median():.1f}天')
        
        ax1.set_title('持仓天数分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('持仓天数', fontsize=11)
        ax1.set_ylabel('交易次数', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：持仓时间 vs 盈亏散点图
        colors = [self.colors['positive'] if p > 0 else self.colors['negative'] for p in pnl_pcts]
        ax2.scatter(hold_days, pnl_pcts, c=colors, alpha=0.6, s=50)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 添加趋势线
        if len(hold_days) > 2:
            z = np.polyfit(hold_days, pnl_pcts, 1)
            p = np.poly1d(z)
            ax2.plot(sorted(hold_days), p(sorted(hold_days)), 
                    color='gray', linestyle='--', alpha=0.7, label='趋势线')
        
        ax2.set_title('持仓时间 vs 盈亏', fontsize=12, fontweight='bold')
        ax2.set_xlabel('持仓天数', fontsize=11)
        ax2.set_ylabel('收益率 (%)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [Patch(facecolor=self.colors['positive'], label='盈利'),
                          Patch(facecolor=self.colors['negative'], label='亏损')]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'holding_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_trade_analysis(self, df_trades: pd.DataFrame):
        """绘制交易分析图"""
        if df_trades.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 各股票交易次数
        if 'code' in df_trades.columns:
            trade_counts = df_trades['code'].value_counts().head(10)
            ax1.barh(range(len(trade_counts)), trade_counts.values, 
                    color=self.colors['strategy'], alpha=0.7)
            ax1.set_yticks(range(len(trade_counts)))
            ax1.set_yticklabels(trade_counts.index)
            ax1.set_title('交易最频繁的股票 (Top 10)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('交易次数', fontsize=11)
            ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. 每月交易次数
        if 'date' in df_trades.columns:
            df_trades['date'] = pd.to_datetime(df_trades['date'])
            monthly_trades = df_trades.resample('M', on='date').size()
            ax2.bar(monthly_trades.index, monthly_trades.values, 
                   width=20, color=self.colors['strategy'], alpha=0.7)
            ax2.set_title('每月交易次数', fontsize=12, fontweight='bold')
            ax2.set_xlabel('月份', fontsize=11)
            ax2.set_ylabel('交易次数', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 买卖方向分布
        if 'direction' in df_trades.columns:
            direction_counts = df_trades['direction'].value_counts()
            colors_pie = [self.colors['positive'], self.colors['negative']]
            wedges, texts, autotexts = ax3.pie(
                direction_counts.values, 
                labels=['买入', '卖出'],
                autopct='%1.1f%%', 
                colors=colors_pie[:len(direction_counts)],
                startangle=90
            )
            ax3.set_title('买卖方向分布', fontsize=12, fontweight='bold')
        
        # 4. 退出原因分布
        sell_trades = df_trades[df_trades['direction'] == 'sell']
        if not sell_trades.empty and 'exit_reason' in sell_trades.columns:
            # 处理枚举类型
            exit_reasons = sell_trades['exit_reason'].apply(
                lambda x: x.value if hasattr(x, 'value') else str(x)
            )
            exit_counts = exit_reasons.value_counts()
            
            colors_exit = plt.cm.Set3(np.linspace(0, 1, len(exit_counts)))
            ax4.pie(exit_counts.values, labels=exit_counts.index,
                   autopct='%1.1f%%', colors=colors_exit, startangle=90)
            ax4.set_title('退出原因分布', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'trade_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_metrics(self, df_daily: pd.DataFrame):
        """绘制风险指标"""
        if df_daily.empty or '总资产' not in df_daily.columns:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        dates = df_daily['date']
        daily_returns = df_daily['总资产'].pct_change().fillna(0)
        
        # 1. 滚动夏普比率 (21日)
        rolling_sharpe = daily_returns.rolling(window=21).apply(
            lambda x: (x.mean() * 252 - 0.02) / (x.std() * np.sqrt(252)) 
            if x.std() > 0 else 0
        )
        ax1.plot(dates, rolling_sharpe, linewidth=1.5, color=self.colors['strategy'])
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='夏普=1')
        ax1.set_title('21日滚动夏普比率', fontsize=12, fontweight='bold')
        ax1.set_ylabel('夏普比率', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 滚动波动率 (21日)
        rolling_vol = daily_returns.rolling(window=21).std() * np.sqrt(252) * 100
        ax2.plot(dates, rolling_vol, linewidth=1.5, color='orange')
        ax2.axhline(y=rolling_vol.mean(), color='red', linestyle='--', 
                   alpha=0.7, label=f'均值: {rolling_vol.mean():.1f}%')
        ax2.set_title('21日滚动波动率', fontsize=12, fontweight='bold')
        ax2.set_ylabel('年化波动率 (%)', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 市场环境分布（如有）
        if '市场环境' in df_daily.columns:
            regime_counts = df_daily['市场环境'].value_counts()
            colors_regime = {'bull': 'green', 'range': 'orange', 'bear': 'red'}
            regime_colors = [colors_regime.get(r, 'gray') for r in regime_counts.index]
            ax3.pie(regime_counts.values, labels=regime_counts.index,
                   autopct='%1.1f%%', colors=regime_colors, startangle=90)
            ax3.set_title('市场环境分布', fontsize=12, fontweight='bold')
        else:
            # 仓位变化
            if '仓位比例' in df_daily.columns:
                ax3.fill_between(dates, df_daily['仓位比例'] * 100, 0,
                               alpha=0.5, color=self.colors['strategy'])
                ax3.set_title('仓位变化', fontsize=12, fontweight='bold')
                ax3.set_ylabel('仓位 (%)', fontsize=11)
                ax3.grid(True, alpha=0.3)
        
        # 4. 累计收益 vs 回撤对比
        cumulative_return = (df_daily['总资产'] / df_daily['总资产'].iloc[0] - 1) * 100
        cumulative_series = (1 + daily_returns).cumprod()
        rolling_max = cumulative_series.expanding().max()
        drawdown_pct = ((cumulative_series - rolling_max) / rolling_max) * 100
        
        ax4.fill_between(dates, cumulative_return, 0,
                        alpha=0.3, color=self.colors['positive'], label='累计收益')
        ax4.fill_between(dates, drawdown_pct, 0,
                        alpha=0.3, color=self.colors['negative'], label='回撤')
        ax4.plot(dates, cumulative_return, color=self.colors['positive'], linewidth=1.5)
        ax4.plot(dates, drawdown_pct, color=self.colors['negative'], linewidth=1.5)
        ax4.set_title('累计收益 vs 回撤', fontsize=12, fontweight='bold')
        ax4.set_ylabel('百分比 (%)', fontsize=11)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        for ax in [ax1, ax2, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'risk_metrics.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, df_daily: pd.DataFrame, df_trades: pd.DataFrame,
                             metrics: Dict, benchmark_values: List[float],
                             initial_capital: float) -> str:
        """生成HTML报告"""
        
        # 计算基准收益
        benchmark_return = 0
        if benchmark_values and len(benchmark_values) > 0:
            benchmark_return = (benchmark_values[-1] - 1) * 100
        
        excess_return = metrics.get('total_return', 0) * 100 - benchmark_return
        
        # 交易统计
        total_trades = len(df_trades[df_trades['direction'] == 'sell']) if not df_trades.empty else 0
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A股策略回测报告 V2.0</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 16px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{ 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white; 
            padding: 40px; 
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ opacity: 0.8; font-size: 1.1em; }}
        
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            padding: 30px;
            background: #f8f9fa;
        }}
        .metric-card {{ 
            background: white;
            border-radius: 12px; 
            padding: 25px; 
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .metric-card:hover {{ transform: translateY(-5px); }}
        .metric-value {{ 
            font-size: 2.2em; 
            font-weight: bold; 
            margin: 10px 0;
        }}
        .metric-label {{ 
            font-size: 0.9em; 
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
        
        .content {{ padding: 30px; }}
        .section {{ margin-bottom: 40px; }}
        .section-title {{ 
            font-size: 1.5em; 
            color: #1a1a2e;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .chart-container {{ 
            text-align: center; 
            margin: 20px 0;
        }}
        .chart-container img {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }}
        
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .summary-table th, .summary-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .summary-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        .summary-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .footer {{ 
            text-align: center; 
            padding: 30px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-success {{ background: #d1fae5; color: #065f46; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; }}
        .badge-info {{ background: #dbeafe; color: #1e40af; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 A股策略回测报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 初始资金: {format_number(initial_capital)}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">总收益率</div>
                <div class="metric-value {'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">
                    {format_percent(metrics.get('total_return', 0))}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">年化收益</div>
                <div class="metric-value {'positive' if metrics.get('annual_return', 0) > 0 else 'negative'}">
                    {format_percent(metrics.get('annual_return', 0))}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">基准收益</div>
                <div class="metric-value {'positive' if benchmark_return > 0 else 'negative'}">
                    {benchmark_return:+.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">超额收益</div>
                <div class="metric-value {'positive' if excess_return > 0 else 'negative'}">
                    {excess_return:+.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">夏普比率</div>
                <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最大回撤</div>
                <div class="metric-value negative">{format_percent(metrics.get('max_drawdown', 0))}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最长水下期</div>
                <div class="metric-value">{metrics.get('max_underwater_days', 0)}天</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">胜率</div>
                <div class="metric-value">{format_percent(metrics.get('win_rate', 0))}</div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">📈 净值曲线 & 基准对比</h2>
                <div class="chart-container">
                    <img src="equity_curve.png" alt="净值曲线">
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">🌊 水下分析（回撤）</h2>
                <div class="chart-container">
                    <img src="underwater_chart.png" alt="水下图">
                </div>
            </div>
            
            {"" if self.simple_mode else '''
            <div class="section">
                <h2 class="section-title">📅 月度收益热力图</h2>
                <div class="chart-container">
                    <img src="monthly_returns.png" alt="月度收益">
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">💰 盈亏分布分析</h2>
                <div class="chart-container">
                    <img src="pnl_distribution.png" alt="盈亏分布">
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">⏱️ 持仓时间分析</h2>
                <div class="chart-container">
                    <img src="holding_analysis.png" alt="持仓分析">
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">💼 交易分析</h2>
                <div class="chart-container">
                    <img src="trade_analysis.png" alt="交易分析">
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">⚠️ 风险指标</h2>
                <div class="chart-container">
                    <img src="risk_metrics.png" alt="风险指标">
                </div>
            </div>
            '''}
            
            <div class="section">
                <h2 class="section-title">📋 绩效摘要</h2>
                <table class="summary-table">
                    <tr><th>指标</th><th>数值</th><th>说明</th></tr>
                    <tr>
                        <td>最终资产</td>
                        <td><strong>{format_number(metrics.get('final_value', 0))}</strong></td>
                        <td>回测结束时的总资产</td>
                    </tr>
                    <tr>
                        <td>交易天数</td>
                        <td>{metrics.get('trading_days', 0)}</td>
                        <td>有效交易日数量</td>
                    </tr>
                    <tr>
                        <td>总交易次数</td>
                        <td>{total_trades}</td>
                        <td>完成的卖出交易数</td>
                    </tr>
                    <tr>
                        <td>年化波动率</td>
                        <td>{format_percent(metrics.get('volatility', 0))}</td>
                        <td>收益的年化标准差</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p>📊 本报告由 A股策略系统 V2.0 自动生成</p>
            <p>⚠️ 历史业绩不代表未来表现，投资有风险，入市需谨慎</p>
        </div>
    </div>
</body>
</html>
        """
        
        html_path = os.path.join(self.output_dir, 'strategy_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path


# 兼容旧接口
class SimpleVisualizer(StrategyVisualizer):
    """简化可视化器（兼容旧代码）"""
    
    def __init__(self, output_dir: str = "./output"):
        super().__init__(output_dir, simple_mode=True)
    
    def generate_simple_report(self, daily_results: List[Dict],
                              trades: List,
                              initial_capital: float = 1_000_000) -> str:
        """生成简单报告（兼容旧接口）"""
        return self.generate_report(daily_results, trades, initial_capital)


if __name__ == "__main__":
    print("可视化模块 V2.0 测试")
    print("请通过 main.py 运行完整回测以生成报告")
