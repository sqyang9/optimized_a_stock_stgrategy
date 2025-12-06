"""
A股科技成长股趋势策略 V1.0 - 技术指标计算模块
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from config import IndicatorConfig, DEFAULT_CONFIG
from utils import setup_logger, rolling_percentile

logger = setup_logger(__name__)


@dataclass
class IndicatorResult:
    """指标计算结果"""
    df: pd.DataFrame          # 包含所有指标的DataFrame
    squeeze_active: bool      # 当前是否处于挤压状态
    squeeze_release: bool     # 当前是否释放信号
    trend_up: bool           # 趋势是否向上
    bbw_percentile: float    # BBW百分位


class IndicatorCalculator:
    """技术指标计算器"""
    
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or DEFAULT_CONFIG.indicator
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标
        
        Args:
            df: 包含 date, open, high, low, close, volume, amount 的DataFrame
        
        Returns:
            添加了所有指标的DataFrame
        """
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # 确保按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        # 计算各类指标
        df = self._calc_moving_averages(df)
        df = self._calc_bollinger_bands(df)
        df = self._calc_bbw_percentile(df)
        df = self._calc_atr(df)
        df = self._calc_rsi(df)
        df = self._calc_volume_indicators(df)
        df = self._calc_price_patterns(df)
        df = self._calc_signals(df)
        
        return df
    
    def _calc_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均线"""
        close = df['close']
        
        # EMA
        df['ema5'] = close.ewm(span=5, adjust=False).mean()
        df['ema20'] = close.ewm(span=self.config.ema_short, adjust=False).mean()
        df['ema60'] = close.ewm(span=self.config.ema_mid, adjust=False).mean()
        df['ema120'] = close.ewm(span=self.config.ema_long, adjust=False).mean()
        
        # SMA
        df['ma5'] = close.rolling(self.config.ma_5).mean()
        df['ma20'] = close.rolling(self.config.ema_short).mean()
        
        return df
    
    def _calc_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带"""
        close = df['close']
        period = self.config.bb_period
        std_mult = self.config.bb_std
        
        # 中轨（20日均线）
        df['bb_mid'] = close.rolling(period).mean()
        
        # 标准差
        std = close.rolling(period).std()
        
        # 上下轨
        df['bb_upper'] = df['bb_mid'] + std_mult * std
        df['bb_lower'] = df['bb_mid'] - std_mult * std
        
        # 布林带宽度 (BBW)
        df['bbw'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # %B（价格在布林带中的位置）
        df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _calc_bbw_percentile(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算BBW百分位（用于判断挤压）"""
        lookback = self.config.bbw_lookback
        
        # 计算BBW在近N日的百分位排名
        def calc_percentile(x):
            if len(x) < lookback:
                return np.nan
            current = x.iloc[-1]
            return (x < current).sum() / len(x)
        
        df['bbw_percentile'] = df['bbw'].rolling(lookback).apply(calc_percentile, raw=False)
        
        # 挤压状态判断
        df['squeeze'] = df['bbw_percentile'] < self.config.squeeze_threshold
        
        # BBW拐头判断
        df['bbw_turning_up'] = (df['bbw'] > df['bbw'].shift(1)) & \
                               (df['bbw'].shift(1) <= df['bbw'].shift(2))
        
        # 挤压持续天数
        df['squeeze_days'] = 0
        squeeze_count = 0
        for i in range(len(df)):
            if df.loc[df.index[i], 'squeeze']:
                squeeze_count += 1
            else:
                squeeze_count = 0
            df.loc[df.index[i], 'squeeze_days'] = squeeze_count
        
        return df
    
    def _calc_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算ATR (Average True Range)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        df['atr'] = df['tr'].rolling(self.config.atr_period).mean()
        
        # ATR百分比
        df['atr_pct'] = df['atr'] / close
        
        return df
    
    def _calc_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算RSI"""
        close = df['close']
        period = self.config.rsi_period
        
        # 价格变化
        delta = close.diff()
        
        # 分离涨跌
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # 平均涨跌
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        # RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI高点（用于背离判断）
        df['rsi_high'] = df['rsi'].rolling(20).max()
        
        return df
    
    def _calc_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量相关指标"""
        volume = df['volume']
        
        # 成交量均线
        df['vol_ma5'] = volume.rolling(5).mean()
        df['vol_ma20'] = volume.rolling(self.config.volume_ma_short).mean()
        df['vol_ma60'] = volume.rolling(self.config.volume_ma_long).mean()
        
        # 量比
        df['volume_ratio'] = volume / df['vol_ma20']
        
        # 相对成交量（相对于突破日）
        df['relative_volume'] = volume / volume.shift(1)
        
        # 成交量是否创新高
        df['vol_new_high_10'] = volume >= volume.rolling(10).max()
        
        # 量能健康度（20日均量/60日均量）
        df['volume_health'] = df['vol_ma20'] / df['vol_ma60']
        
        return df
    
    def _calc_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格形态"""
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        # 涨跌幅
        df['pct_change'] = close.pct_change()
        
        # 实体和影线
        df['body'] = abs(close - open_price)
        df['upper_shadow'] = high - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - low
        
        # 长上影线判断
        df['long_upper_shadow'] = (df['upper_shadow'] > df['body'] * 2) & \
                                   (close < (high + low) / 2)
        
        # 收盘价在当日区间的位置
        df['close_position'] = (close - low) / (high - low + 1e-10)
        
        # 趋势强度
        df['trend_strength'] = (close > df['ema20']) & (df['ema20'] > df['ema60'])
        
        # 价格创新高
        df['price_new_high'] = close >= close.rolling(20).max()
        
        # MA5拐头
        df['ma5_turn_down'] = (df['ma5'] < df['ma5'].shift(1)) & \
                              (df['ma5'].shift(1) >= df['ma5'].shift(2))
        
        return df
    
    def _calc_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算交易信号"""
        close = df['close']
        
        # 趋势条件：收盘价在EMA60之上
        df['trend_up'] = close > df['ema60']
        
        # 突破布林上轨
        df['break_bb_upper'] = (close > df['bb_upper']) & \
                                (df['close'].shift(1) <= df['bb_upper'].shift(1))
        
        # 挤压释放信号
        df['squeeze_release'] = (
            df['squeeze'].shift(1) &                    # 昨日处于挤压
            df['bbw_turning_up'] &                      # BBW向上拐头
            (close > df['bb_upper']) &                  # 收盘突破上轨
            (df['volume_ratio'] > 1.3)                  # 放量
        )
        
        # 回踩支撑判断
        df['at_support'] = (
            (close <= df['ma5'] * 1.01) |               # 接近5日线
            (close <= df['bb_mid'] * 1.01) |            # 接近布林中轨
            (close <= df['ema20'] * 1.01)               # 接近EMA20
        )
        
        # 缩量回踩
        df['pullback_shrink'] = df['volume_ratio'] < 0.7
        
        # 顶部信号 - 放量滞涨
        df['top_stall'] = df['vol_new_high_10'] & (df['pct_change'].abs() < 0.01)
        
        # 顶部信号 - RSI背离
        df['rsi_divergence'] = df['price_new_high'] & (df['rsi'] < df['rsi_high'])
        
        # 破位信号
        df['breakdown'] = (close < df['ma5']) & df['ma5_turn_down']
        
        return df
    
    def get_indicator_summary(self, df: pd.DataFrame) -> IndicatorResult:
        """获取最新指标摘要
        
        Args:
            df: 已计算指标的DataFrame
        
        Returns:
            IndicatorResult对象
        """
        if df is None or df.empty:
            return IndicatorResult(
                df=df,
                squeeze_active=False,
                squeeze_release=False,
                trend_up=False,
                bbw_percentile=0.5
            )
        
        last = df.iloc[-1]
        
        return IndicatorResult(
            df=df,
            squeeze_active=bool(last.get('squeeze', False)),
            squeeze_release=bool(last.get('squeeze_release', False)),
            trend_up=bool(last.get('trend_up', False)),
            bbw_percentile=float(last.get('bbw_percentile', 0.5))
        )


def calc_market_regime(index_df: pd.DataFrame) -> str:
    """判断市场环境
    
    Args:
        index_df: 指数日线数据（需包含ema60, ema120）
    
    Returns:
        'bull', 'range', 或 'bear'
    """
    if index_df is None or index_df.empty:
        return 'range'
    
    calc = IndicatorCalculator()
    df = calc.calculate_all(index_df)
    
    if df.empty:
        return 'range'
    
    last = df.iloc[-1]
    close = last['close']
    ema60 = last['ema60']
    ema120 = last['ema120']
    
    # 牛市：价格 > EMA60 > EMA120
    if close > ema60 and ema60 > ema120:
        return 'bull'
    
    # 熊市：价格 < EMA60 < EMA120
    if close < ema60 and ema60 < ema120:
        return 'bear'
    
    # 其他情况为震荡
    return 'range'


def quick_calc_squeeze(df: pd.DataFrame, bb_period: int = 20, 
                       bbw_lookback: int = 60, threshold: float = 0.35) -> pd.Series:
    """快速计算挤压状态
    
    Args:
        df: 包含close的DataFrame
        bb_period: 布林带周期
        bbw_lookback: BBW百分位回看周期
        threshold: 挤压阈值
    
    Returns:
        布尔Series，True表示处于挤压状态
    """
    close = df['close']
    
    # 布林带
    bb_mid = close.rolling(bb_period).mean()
    bb_std = close.rolling(bb_period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # BBW
    bbw = (bb_upper - bb_lower) / bb_mid
    
    # BBW百分位
    def calc_pct(x):
        if len(x) < bbw_lookback:
            return np.nan
        return (x < x.iloc[-1]).sum() / len(x)
    
    bbw_pct = bbw.rolling(bbw_lookback).apply(calc_pct, raw=False)
    
    return bbw_pct < threshold


if __name__ == "__main__":
    # 测试指标计算
    print("测试技术指标计算模块...")
    
    # 创建模拟数据
    np.random.seed(42)
    n = 200
    
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    close = 50 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    
    df = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.normal(0, 0.005, n)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.015, n))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.015, n))),
        'close': close,
        'volume': np.random.randint(5000000, 20000000, n),
        'amount': close * np.random.randint(5000000, 20000000, n)
    })
    
    # 确保OHLC逻辑正确
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    # 计算指标
    calc = IndicatorCalculator()
    result_df = calc.calculate_all(df)
    
    print(f"\n计算完成，共 {len(result_df)} 行")
    print(f"\n指标列表:")
    print(result_df.columns.tolist())
    
    # 检查关键指标
    last = result_df.iloc[-1]
    print(f"\n最新数据:")
    print(f"  收盘价: {last['close']:.2f}")
    print(f"  EMA20: {last['ema20']:.2f}")
    print(f"  EMA60: {last['ema60']:.2f}")
    print(f"  布林上轨: {last['bb_upper']:.2f}")
    print(f"  布林中轨: {last['bb_mid']:.2f}")
    print(f"  布林下轨: {last['bb_lower']:.2f}")
    print(f"  BBW: {last['bbw']:.4f}")
    print(f"  BBW百分位: {last['bbw_percentile']:.2%}")
    print(f"  ATR: {last['atr']:.2f}")
    print(f"  RSI: {last['rsi']:.1f}")
    print(f"  挤压状态: {last['squeeze']}")
    print(f"  趋势向上: {last['trend_up']}")
    
    # 获取摘要
    summary = calc.get_indicator_summary(result_df)
    print(f"\n指标摘要:")
    print(f"  挤压状态: {summary.squeeze_active}")
    print(f"  挤压释放: {summary.squeeze_release}")
    print(f"  趋势向上: {summary.trend_up}")
    print(f"  BBW百分位: {summary.bbw_percentile:.2%}")
    
    # 测试市场环境判断
    regime = calc_market_regime(df)
    print(f"\n市场环境: {regime}")
    
    # 统计信号出现次数
    print(f"\n信号统计:")
    print(f"  挤压释放次数: {result_df['squeeze_release'].sum()}")
    print(f"  顶部滞涨次数: {result_df['top_stall'].sum()}")
    print(f"  RSI背离次数: {result_df['rsi_divergence'].sum()}")
    
    print("\n技术指标模块测试完成!")
