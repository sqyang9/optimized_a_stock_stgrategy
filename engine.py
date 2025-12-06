"""
回测引擎 V3.0
新增: 大盘择时集成、板块共振信号、基准对比数据
"""

import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import StrategyConfig, MarketRegime
from indicators import IndicatorCalculator
from stock_pool import StockPoolManager
from signal_generator import SignalGenerator, TradingSignal
from position_manager import PositionManager, ExitReason
from market_timer import MarketTimer
from utils import setup_logger

logger = setup_logger("Engine")


class BacktestEngineV3:
    """回测引擎 V3.0"""
    
    def __init__(self, config: StrategyConfig, data_dir: str = './data/daily'):
        self.config = config
        self.data_dir = data_dir
        
        # 数据加载器
        try:
            from core.data_loader import RealDataLoader
            self.loader = RealDataLoader(data_dir)
        except ImportError:
            try:
                from data_loader import RealDataLoader
                self.loader = RealDataLoader(data_dir)
            except ImportError:
                logger.error("无法导入 RealDataLoader")
                raise
        
        # 指标计算器
        self.indicator_calc = IndicatorCalculator(config.indicator)
        
        # 股票池管理器
        self.pool_manager = StockPoolManager(config.stock_pool)
        self.industry_map = self.pool_manager.get_code_industry_map()
        
        # 信号生成器
        self.signal_gen = SignalGenerator(
            config.signal, 
            config.entry,
            config.sector_resonance,
            self.industry_map
        )
        
        # 仓位管理器
        self.pos_mgr = PositionManager(
            config.position, 
            config.exit, 
            config.risk, 
            config.backtest.initial_capital
        )
        
        # 大盘择时器
        self.market_timer = MarketTimer(config.market_timing)
        
        # 数据存储
        self.universe_data: Dict[str, pd.DataFrame] = {}
        self.index_data: Optional[pd.DataFrame] = None
        self.daily_results: List[Dict] = []
        self.trade_dates: List = []
        
        # 基准数据（用于可视化对比）
        self.benchmark_values: List[float] = []
    
    def prepare_data(self, start_date: str, end_date: str):
        """预加载数据并计算所有指标"""
        logger.info("正在预加载数据并计算技术指标...")
        
        self.universe_data = {}
        codes = self.loader.get_all_codes()
        all_dates = set()
        
        # 1. 加载个股数据
        for code in tqdm(codes, desc="Loading Stock Data"):
            df = self.loader.load_stock_data(code, start_date=None)
            
            if df is not None and not df.empty:
                # 计算所有指标
                df = self.indicator_calc.calculate_all(df)
                
                # 截取回测时间段
                mask = (df['date'] >= pd.to_datetime(start_date)) & \
                       (df['date'] <= pd.to_datetime(end_date))
                df_slice = df.loc[mask].copy()
                
                if not df_slice.empty:
                    df_slice.set_index('date', inplace=True)
                    self.universe_data[code] = df_slice
                    all_dates.update(df_slice.index)
        
        self.trade_dates = sorted(list(all_dates))
        
        # 2. 加载指数数据（大盘择时）
        self._load_index_data(start_date, end_date)
        
        logger.info(f"数据准备完成: {len(self.universe_data)} 只股票, "
                   f"{len(self.trade_dates)} 个交易日")
        gc.collect()
    
    def _load_index_data(self, start_date: str, end_date: str):
        """加载指数数据"""
        index_code = self.config.market_timing.index_code
        
        # 尝试加载指数数据
        index_df = self.loader.load_stock_data(index_code)
        
        if index_df is None or index_df.empty:
            logger.warning(f"未找到指数 {index_code} 数据，构造市场指数")
            index_df = self._construct_market_index(start_date, end_date)
        
        if index_df is not None and not index_df.empty:
            self.index_data = index_df
            self.market_timer.load_index_data(index_df)
            logger.info(f"指数数据加载完成: {len(index_df)} 条记录")
        else:
            logger.warning("无法加载指数数据，将禁用大盘择时")
            self.config.market_timing.enabled = False
    
    def _construct_market_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """从股票数据构造市场指数（等权平均）"""
        if not self.universe_data:
            return None
        
        all_closes = {}
        for code, df in self.universe_data.items():
            for date in df.index:
                if date not in all_closes:
                    all_closes[date] = []
                all_closes[date].append(df.loc[date, 'close'])
        
        dates = sorted(all_closes.keys())
        avg_prices = [np.mean(all_closes[d]) for d in dates]
        
        base_price = avg_prices[0] if avg_prices else 1000
        index_values = [p / base_price * 1000 for p in avg_prices]
        
        index_df = pd.DataFrame({
            'date': dates,
            'close': index_values,
            'open': index_values,
            'high': index_values,
            'low': index_values,
            'volume': [100000000] * len(dates)
        })
        
        return index_df
    
    def run(self) -> Tuple[PositionManager, List[Dict]]:
        """运行回测"""
        if not self.trade_dates:
            logger.error("无数据，请先调用 prepare_data()")
            return None, []
        
        logger.info("开始回测...")
        
        initial_benchmark = None
        
        for current_date in tqdm(self.trade_dates, desc="Backtesting"):
            date_str = current_date.strftime('%Y-%m-%d')
            
            # 1. 获取当日市场状态
            market_state = self.market_timer.get_market_state(date_str)
            self.pos_mgr.market_regime = market_state.regime
            
            # 2. 构建当日快照
            daily_snapshot = self._build_daily_snapshot(current_date)
            if not daily_snapshot:
                continue
            
            # 3. 更新持仓状态
            self.pos_mgr.update_positions(date_str, daily_snapshot)
            
            # 4. 检查退出信号
            self._process_exits(date_str, daily_snapshot)
            
            # 5. 生成新信号（如果允许开仓）
            if market_state.allow_entry:
                min_score = self.market_timer.get_regime_score_threshold(market_state.regime)
                
                potential_signals = self.signal_gen.scan_daily_snapshot(
                    daily_snapshot, 
                    date_str,
                    min_score=min_score
                )
                
                # 6. 执行入场
                self._process_entries(date_str, potential_signals, market_state)
            
            # 7. 每日结算
            summary = self._daily_settlement(date_str, market_state)
            self.daily_results.append(summary)
            
            # 记录基准值
            if initial_benchmark is None and market_state.index_close > 0:
                initial_benchmark = market_state.index_close
            
            if initial_benchmark and market_state.index_close > 0:
                benchmark_return = market_state.index_close / initial_benchmark
                self.benchmark_values.append(benchmark_return)
            else:
                self.benchmark_values.append(1.0)
        
        logger.info("回测完成")
        return self.pos_mgr, self.daily_results
    
    def _build_daily_snapshot(self, current_date) -> Dict[str, pd.Series]:
        """构建当日数据快照"""
        daily_snapshot = {}
        for code, df in self.universe_data.items():
            if current_date in df.index:
                daily_snapshot[code] = df.loc[current_date]
        return daily_snapshot
    
    def _process_exits(self, date_str: str, daily_snapshot: Dict[str, pd.Series]):
        """处理退出信号"""
        for code in list(self.pos_mgr.positions.keys()):
            if code not in daily_snapshot:
                continue
            
            price_data = daily_snapshot[code]
            should_exit, reason, ratio = self.pos_mgr.check_exit_signals(
                code, price_data, date_str
            )
            
            if should_exit:
                self.pos_mgr.close_position(
                    code=code,
                    date=date_str,
                    price=price_data['close'],
                    reason=reason,
                    sell_ratio=ratio,
                    commission_rate=self.config.backtest.commission_rate,
                    stamp_tax_rate=self.config.backtest.stamp_tax_rate
                )
    
    def _process_entries(self, date_str: str, signals: List[TradingSignal],
                        market_state):
        """处理入场信号"""
        trades_today = 0
        max_trades = self.config.position.max_trades_per_day
        
        for signal in signals:
            if trades_today >= max_trades:
                break
            
            is_safe, msg = self.pos_mgr.check_risk_limits(date_str)
            if not is_safe:
                logger.debug(f"风控限制: {msg}")
                break
            
            industry = self.industry_map.get(signal.code, "Unknown")
            
            pos_pct, shares = self.pos_mgr.calc_position_size(
                entry_price=signal.close,
                stop_price=signal.stop_price,
                entry_type=signal.entry_type,
                signal_score=signal.score,
                position_ratio=signal.position_ratio * market_state.max_position_ratio
            )
            
            if shares <= 0:
                continue
            
            can_buy, msg = self.pos_mgr.check_position_constraints(
                signal.code, industry, pos_pct
            )
            
            if not can_buy:
                logger.debug(f"仓位约束: {signal.code} - {msg}")
                continue
            
            success = self.pos_mgr.open_position(
                code=signal.code,
                name=signal.name,
                date=date_str,
                price=signal.close,
                shares=shares,
                stop_price=signal.stop_price,
                atr=signal.atr,
                entry_type=signal.entry_type,
                signal_score=signal.adjusted_score,
                industry=industry,
                commission_rate=self.config.backtest.commission_rate
            )
            
            if success:
                trades_today += 1
    
    def _daily_settlement(self, date_str: str, market_state) -> Dict:
        """每日结算"""
        summary = self.pos_mgr.get_account_summary()
        summary['date'] = date_str
        summary['市场环境'] = market_state.regime.value
        summary['允许开仓'] = market_state.allow_entry
        summary['指数涨跌'] = market_state.index_change
        
        self.pos_mgr.reset_daily_stats()
        
        return summary
    
    def get_benchmark_data(self) -> List[float]:
        """获取基准数据"""
        return self.benchmark_values
    
    def get_performance_summary(self) -> Dict:
        """获取绩效摘要"""
        if not self.daily_results:
            return {}
        
        initial_capital = self.config.backtest.initial_capital
        final_value = self.daily_results[-1]['总资产']
        total_return = (final_value - initial_capital) / initial_capital
        
        values = [r['总资产'] for r in self.daily_results]
        daily_returns = pd.Series(values).pct_change().dropna()
        
        trading_days = len(self.daily_results)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        risk_free = self.config.backtest.risk_free_rate
        sharpe = (annual_return - risk_free) / volatility if volatility > 0 else 0
        
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # 最长回撤期
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
        
        # 卡尔马比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 基准收益
        benchmark_return = 0
        if self.benchmark_values and len(self.benchmark_values) > 0:
            benchmark_return = self.benchmark_values[-1] - 1
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_underwater_days': max_underwater_days,
            'calmar_ratio': calmar,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'final_value': final_value,
            'trading_days': trading_days,
            'total_trades': len(self.pos_mgr.trades),
            'win_rate': self.pos_mgr.get_account_summary().get('胜率', 0)
        }


if __name__ == "__main__":
    print("回测引擎 V3.0 模块测试")
    print("请通过 main.py 运行完整回测")
