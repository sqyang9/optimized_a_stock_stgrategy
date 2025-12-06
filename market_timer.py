"""
市场择时模块 - 大盘环境判断
基于指数均线判断牛熊震荡，控制策略仓位和开仓权限
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config import MarketTimingConfig, MarketRegime
from utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class MarketState:
    """市场状态"""
    regime: MarketRegime           # 当前市场环境
    index_close: float             # 指数收盘价
    index_ma_short: float          # 短期均线
    index_ma_long: float           # 长期均线
    index_change: float            # 当日涨跌幅
    volatility: float              # 近期波动率
    allow_entry: bool              # 是否允许开仓
    max_position_ratio: float      # 最大仓位比例
    reason: str                    # 状态说明


class MarketTimer:
    """大盘择时器"""
    
    def __init__(self, config: MarketTimingConfig):
        self.config = config
        self.index_data: Optional[pd.DataFrame] = None
        self.last_state: Optional[MarketState] = None
        
        # 暂停开仓的标记（因指数大跌触发）
        self._pause_until_date: Optional[str] = None
    
    def load_index_data(self, index_df: pd.DataFrame):
        """加载指数数据并计算均线
        
        Args:
            index_df: 包含 date, close 的指数数据
        """
        if index_df is None or index_df.empty:
            logger.warning("指数数据为空")
            return
        
        df = index_df.copy()
        
        # 确保按日期排序
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        # 计算均线
        df['ma_short'] = df['close'].rolling(self.config.ma_short).mean()
        df['ma_long'] = df['close'].rolling(self.config.ma_long).mean()
        
        # 计算涨跌幅
        df['pct_change'] = df['close'].pct_change()
        
        # 计算波动率（20日滚动标准差）
        df['volatility'] = df['pct_change'].rolling(20).std()
        
        # 设置日期索引
        df.set_index('date', inplace=True)
        
        self.index_data = df
        logger.info(f"指数数据加载完成: {len(df)} 条记录")
    
    def get_market_state(self, date: str) -> MarketState:
        """获取指定日期的市场状态
        
        Args:
            date: 日期字符串 'YYYY-MM-DD'
        
        Returns:
            MarketState 对象
        """
        # 默认状态（无数据时）
        default_state = MarketState(
            regime=MarketRegime.RANGE,
            index_close=0,
            index_ma_short=0,
            index_ma_long=0,
            index_change=0,
            volatility=0,
            allow_entry=True,
            max_position_ratio=1.0,
            reason="无指数数据，使用默认状态"
        )
        
        if not self.config.enabled:
            default_state.reason = "大盘择时已禁用"
            return default_state
        
        if self.index_data is None or self.index_data.empty:
            return default_state
        
        # 查找当日数据
        try:
            current_date = pd.to_datetime(date)
            if current_date not in self.index_data.index:
                # 找最近的交易日
                available_dates = self.index_data.index[self.index_data.index <= current_date]
                if len(available_dates) == 0:
                    return default_state
                current_date = available_dates[-1]
            
            row = self.index_data.loc[current_date]
        except Exception as e:
            logger.debug(f"获取指数数据失败: {date}, {e}")
            return default_state
        
        # 提取数据
        close = row['close']
        ma_short = row.get('ma_short', close)
        ma_long = row.get('ma_long', close)
        pct_change = row.get('pct_change', 0)
        volatility = row.get('volatility', 0)
        
        # 处理NaN
        if pd.isna(ma_short):
            ma_short = close
        if pd.isna(ma_long):
            ma_long = close
        if pd.isna(pct_change):
            pct_change = 0
        if pd.isna(volatility):
            volatility = 0
        
        # 判断市场环境
        regime, regime_reason = self._determine_regime(close, ma_short, ma_long)
        
        # 判断是否允许开仓
        allow_entry, max_pos, entry_reason = self._check_entry_permission(
            date, regime, pct_change, volatility
        )
        
        state = MarketState(
            regime=regime,
            index_close=close,
            index_ma_short=ma_short,
            index_ma_long=ma_long,
            index_change=pct_change,
            volatility=volatility,
            allow_entry=allow_entry,
            max_position_ratio=max_pos,
            reason=f"{regime_reason} | {entry_reason}"
        )
        
        self.last_state = state
        return state
    
    def _determine_regime(self, close: float, ma_short: float, 
                         ma_long: float) -> Tuple[MarketRegime, str]:
        """判断市场环境
        
        牛市: 价格 > MA_short > MA_long
        熊市: 价格 < MA_short < MA_long  
        震荡: 其他情况
        """
        if close > ma_short > ma_long:
            return MarketRegime.BULL, "牛市(价格>MA20>MA60)"
        
        if close < ma_short < ma_long:
            return MarketRegime.BEAR, "熊市(价格<MA20<MA60)"
        
        # 其他情况判断趋势强度
        if close > ma_short and ma_short > ma_long * 0.98:
            return MarketRegime.BULL, "弱牛市(价格>MA20,MA20接近MA60)"
        
        if close < ma_short and ma_short < ma_long * 1.02:
            return MarketRegime.BEAR, "弱熊市(价格<MA20,MA20接近MA60)"
        
        return MarketRegime.RANGE, "震荡市"
    
    def _check_entry_permission(self, date: str, regime: MarketRegime,
                                pct_change: float, volatility: float
                               ) -> Tuple[bool, float, str]:
        """检查是否允许开仓及最大仓位
        
        Returns:
            (是否允许开仓, 最大仓位比例, 原因说明)
        """
        # 检查是否在暂停期内
        if self._pause_until_date:
            if date <= self._pause_until_date:
                return False, 0.0, f"暂停开仓至{self._pause_until_date}"
            else:
                self._pause_until_date = None
        
        # 检查指数大跌
        if pct_change <= self.config.index_drop_threshold:
            # 设置第二天暂停开仓
            try:
                next_date = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                self._pause_until_date = next_date
            except:
                pass
            return False, 0.0, f"指数大跌{pct_change:.1%}，暂停开仓"
        
        # 检查高波动
        if volatility > self.config.high_volatility_threshold:
            max_pos = 0.5  # 高波动时最多50%仓位
            return True, max_pos, f"高波动({volatility:.1%})，限制仓位50%"
        
        # 根据市场环境决定
        if regime == MarketRegime.BEAR:
            if not self.config.allow_entry_in_bear:
                return False, 0.0, "熊市禁止开仓"
            return True, self.config.bear_max_position, f"熊市限制仓位{self.config.bear_max_position:.0%}"
        
        if regime == MarketRegime.RANGE:
            return True, self.config.range_max_position, f"震荡市限制仓位{self.config.range_max_position:.0%}"
        
        # 牛市
        return True, 1.0, "牛市正常开仓"
    
    def get_regime_score_threshold(self, regime: MarketRegime) -> int:
        """获取不同市场环境下的信号门槛
        
        牛市: 3分即可入场
        震荡: 需要4分
        熊市: 需要5分
        """
        thresholds = {
            MarketRegime.BULL: 3,
            MarketRegime.RANGE: 4,
            MarketRegime.BEAR: 5
        }
        return thresholds.get(regime, 4)
    
    def should_reduce_position(self, date: str) -> Tuple[bool, float, str]:
        """判断是否应该减仓
        
        Returns:
            (是否减仓, 减仓比例, 原因)
        """
        state = self.get_market_state(date)
        
        # 指数大跌时建议减仓
        if state.index_change <= -0.02:
            return True, 0.3, f"指数跌{state.index_change:.1%}，建议减仓30%"
        
        # 熊市且已持仓时建议逐步减仓
        if state.regime == MarketRegime.BEAR:
            return True, 0.2, "熊市环境，建议减仓20%"
        
        return False, 0.0, ""


def create_mock_index_data(start_date: str, end_date: str, 
                          initial_price: float = 4000) -> pd.DataFrame:
    """创建模拟指数数据（测试用）"""
    dates = pd.date_range(start_date, end_date, freq='B')
    n = len(dates)
    
    # 生成随机走势
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.012, n)
    prices = initial_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.003, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.008, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.008, n))),
        'volume': np.random.randint(100000000, 300000000, n)
    })
    
    return df


if __name__ == "__main__":
    from config import MarketTimingConfig
    
    print("测试市场择时模块...")
    
    # 创建模拟数据
    index_df = create_mock_index_data('2024-01-01', '2024-12-31')
    
    # 初始化择时器
    config = MarketTimingConfig()
    timer = MarketTimer(config)
    timer.load_index_data(index_df)
    
    # 测试几个日期
    test_dates = ['2024-03-15', '2024-06-15', '2024-09-15', '2024-12-15']
    
    for date in test_dates:
        state = timer.get_market_state(date)
        print(f"\n{date}:")
        print(f"  市场环境: {state.regime.value}")
        print(f"  指数收盘: {state.index_close:.2f}")
        print(f"  MA20: {state.index_ma_short:.2f}")
        print(f"  MA60: {state.index_ma_long:.2f}")
        print(f"  允许开仓: {state.allow_entry}")
        print(f"  最大仓位: {state.max_position_ratio:.0%}")
        print(f"  状态: {state.reason}")
    
    print("\n市场择时模块测试完成!")
