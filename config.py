"""
A股科技成长股趋势策略 V2.0 - 配置文件
新增: 大盘择时配置、板块共振配置、时间止损配置
"""

from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class MarketRegime(Enum):
    """市场环境"""
    BULL = "bull"      # 牛市
    RANGE = "range"    # 震荡
    BEAR = "bear"      # 熊市


class SignalGrade(Enum):
    """信号等级"""
    A = "A"  # 优先关注 (>=5分)
    B = "B"  # 正常关注 (3-4分)
    C = "C"  # 暂不考虑 (<3分)


class EntryType(Enum):
    """入场类型"""
    PULLBACK = "pullback"        # 回踩确认
    AUCTION = "auction"          # 竞价狙击
    INTRADAY = "intraday"        # 盘中确认
    BREAKOUT = "breakout"        # 突破首日（不追）


@dataclass
class StockPoolConfig:
    """股票池配置"""
    # 行业范围
    sectors: List[str] = field(default_factory=lambda: [
        "半导体", "新能源", "人工智能", "军工", "创新药", "高端制造",
        "消费电子", "云计算", "自动驾驶", "机器人"
    ])
    
    # 市值要求（亿元）
    min_market_cap: float = 200.0
    
    # 成交额要求（亿元，20日均值）
    min_avg_amount: float = 5.0
    
    # 上市天数要求
    min_list_days: int = 60
    
    # 排除近期涨跌停的天数
    exclude_limit_days: int = 5
    
    # 涨跌停幅度阈值
    limit_threshold: float = 0.095  # 9.5%视为涨跌停


@dataclass
class IndicatorConfig:
    """技术指标配置"""
    # 均线周期
    ema_short: int = 20
    ema_mid: int = 60
    ema_long: int = 120
    ma_5: int = 5
    
    # 布林带参数
    bb_period: int = 20
    bb_std: float = 2.0
    
    # BBW百分位计算周期
    bbw_lookback: int = 60
    
    # 挤压阈值（百分位）
    squeeze_threshold: float = 0.35  # 35%
    
    # ATR周期
    atr_period: int = 14
    
    # RSI周期
    rsi_period: int = 14
    
    # 成交量均线
    volume_ma_short: int = 20
    volume_ma_long: int = 60


@dataclass
class MarketTimingConfig:
    """【新增】大盘择时配置"""
    # 是否启用大盘过滤
    enabled: bool = True
    
    # 基准指数代码
    index_code: str = "000300"  # 沪深300
    
    # 趋势判断参数
    ma_short: int = 20   # 短期均线
    ma_long: int = 60    # 长期均线
    
    # 牛市条件：指数 > MA20 > MA60
    # 熊市条件：指数 < MA20 < MA60
    # 震荡：其他情况
    
    # 熊市下是否允许开仓
    allow_entry_in_bear: bool = False
    
    # 震荡市最大仓位比例
    range_max_position: float = 0.5
    
    # 熊市最大仓位比例
    bear_max_position: float = 0.3
    
    # 指数单日跌幅预警阈值（触发后暂停开仓1天）
    index_drop_threshold: float = -0.02  # -2%
    
    # 指数波动率阈值（高波动时减仓）
    high_volatility_threshold: float = 0.025  # 日波动率 > 2.5%


@dataclass
class SectorResonanceConfig:
    """【新增】板块共振配置"""
    # 是否启用板块共振
    enabled: bool = True
    
    # 同板块最少同时信号数（包含自身）
    min_sector_signals: int = 2
    
    # 板块共振时的加分
    resonance_bonus_score: int = 1
    
    # 孤立信号（无板块共振）的仓位折扣
    isolated_signal_position_ratio: float = 0.5


@dataclass
class SignalConfig:
    """信号配置"""
    # 挤压释放条件
    squeeze_days_min: int = 5           # 挤压最少持续天数
    breakout_volume_ratio: float = 1.3  # 突破量能倍数
    
    # 评分条件
    strong_volume_ratio: float = 1.8    # 强势量能倍数
    min_breakout_pct: float = 0.02      # 最小突破涨幅2%
    
    # 信号等级阈值
    grade_a_threshold: int = 5
    grade_b_threshold: int = 3
    
    # 观察池条件
    watch_max_change_pct: float = 0.07  # 日涨跌幅上限7%
    watch_volume_ratio: float = 0.8     # 量能健康阈值


@dataclass
class EntryConfig:
    """入场配置"""
    # 回踩入场
    pullback_volume_ratio: float = 0.7   # 回踩缩量阈值
    pullback_max_days: int = 3           # 回踩最大等待天数
    
    # 竞价入场
    auction_open_low: float = 0.01       # 高开下限1%
    auction_open_high: float = 0.03      # 高开上限3%
    auction_bid_ask_ratio: float = 1.5   # 委买委卖比
    auction_volume_ratio: float = 2.0    # 量比阈值
    auction_price_buffer: float = 1.003  # 挂单价格系数
    auction_no_chase_gap: float = 0.05   # 高开超5%不追
    
    # 盘中确认
    intraday_retracement_max: float = 0.5  # 最大回撤50%涨幅
    intraday_confirm_time: str = "10:30"   # 确认时间
    
    # 不追涨停
    no_chase_to_limit: float = 0.03      # 距涨停<3%不买


@dataclass
class ExitConfig:
    """出场配置"""
    # 初始止损
    initial_stop_atr_mult: float = 2.0   # ATR倍数 (从2.5调整为2.0)
    max_stop_pct: float = 0.06           # 最大止损6%
    
    # 移动止损
    breakeven_atr: float = 1.5           # 保本触发ATR
    breakeven_buffer: float = 1.005      # 保本价=入场价×1.005
    
    trailing_start_atr: float = 2.5      # 追踪启动ATR (从3.0调整为2.5)
    trailing_atr_mult: float = 1.5       # 追踪ATR倍数 (从2.0调整为1.5)
    
    tight_start_atr: float = 4.0         # 收紧启动ATR (从5.0调整为4.0)
    tight_atr_mult: float = 1.0          # 收紧ATR倍数 (从1.5调整为1.0)
    
    # 分批止盈
    tp1_atr: float = 2.5                 # TP1触发ATR (从3.0调整为2.5)
    tp1_sell_pct: float = 0.5            # TP1卖出50% (从30%调整为50%)
    tp1_new_stop_atr: float = 0.5        # TP1后止损位 (从1.0调整为0.5)
    
    tp2_atr: float = 4.0                 # TP2触发ATR (从5.0调整为4.0)
    tp2_sell_pct: float = 0.3            # TP2卖出30%
    tp2_new_stop_atr: float = 1.0        # TP2后止损位 (从1.5调整为1.0)
    
    # 【新增】时间止损
    time_stop_enabled: bool = True       # 是否启用时间止损
    time_stop_days: int = 5              # 持仓天数上限（未脱离成本区）
    time_stop_profit_threshold: float = 0.01  # 盈利阈值（低于此值视为未脱离成本区）
    
    # 顶部信号
    top_volume_high_days: int = 10       # 放量滞涨判断周期
    top_stall_pct: float = 0.01          # 滞涨涨幅阈值1%
    top_shadow_ratio: float = 2.0        # 长上影线比例


@dataclass
class PositionConfig:
    """仓位配置"""
    # 单笔风险
    single_risk_pct: float = 0.015       # 1.5%
    
    # 仓位限制
    max_single_position: float = 0.15    # 单票最大15% (从20%调整)
    max_sector_position: float = 0.35    # 单板块最大35% (从40%调整)
    max_total_position: float = 0.80     # 总仓位上限80%
    min_cash_reserve: float = 0.20       # 最小现金20%
    max_holdings: int = 8                # 最大持仓数
    
    # 入场仓位比例（相对于计算仓位）
    entry_position_ratio: Dict[str, float] = field(default_factory=lambda: {
        "pullback": 0.6,    # 回踩60%
        "auction": 0.4,     # 竞价40%
        "intraday": 0.5,    # 盘中50%
        "breakout": 0.5,    # 突破50%
    })
    
    # 市场环境仓位调整
    regime_position_mult: Dict[str, float] = field(default_factory=lambda: {
        "bull": 1.0,
        "range": 0.625,     # 50%/80%
        "bear": 0.375,      # 30%/80%
    })
    
    # 市场环境入场门槛
    regime_min_score: Dict[str, int] = field(default_factory=lambda: {
        "bull": 3,
        "range": 4,
        "bear": 5,
    })

    # 每日最大交易次数
    max_trades_per_day: int = 3


@dataclass
class RiskConfig:
    """风控配置"""
    # 亏损限制
    max_daily_loss: float = 0.03         # 单日最大亏损3%
    max_weekly_loss: float = 0.06        # 单周最大亏损6%
    
    # 连续止损暂停
    consecutive_stops: int = 3           # 连续3次止损
    pause_days: int = 1                  # 暂停1天
    
    # 黑天鹅阈值
    index_crash_threshold: float = -0.03  # 指数跌幅3%
    
    # 涨跌停处理
    limit_up_hold: bool = True           # 涨停继续持有
    limit_down_queue: bool = True        # 跌停排队卖出


@dataclass
class BacktestConfig:
    """回测配置"""
    # 回测期间
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    
    # 初始资金
    initial_capital: float = 1_000_000.0
    
    # 交易成本
    commission_rate: float = 0.0003      # 万三佣金
    stamp_tax_rate: float = 0.001        # 千一印花税（卖出）
    slippage_pct: float = 0.001          # 滑点0.1%
    
    # 基准指数
    benchmark: str = "000300"            # 沪深300
    
    # 无风险利率（年化）
    risk_free_rate: float = 0.02


@dataclass
class StrategyConfig:
    """策略总配置"""
    stock_pool: StockPoolConfig = field(default_factory=StockPoolConfig)
    indicator: IndicatorConfig = field(default_factory=IndicatorConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    # 【新增】
    market_timing: MarketTimingConfig = field(default_factory=MarketTimingConfig)
    sector_resonance: SectorResonanceConfig = field(default_factory=SectorResonanceConfig)


# 创建默认配置实例
DEFAULT_CONFIG = StrategyConfig()


if __name__ == "__main__":
    # 测试配置
    config = StrategyConfig()
    print("策略配置加载成功 (V2.0)")
    print(f"股票池行业: {config.stock_pool.sectors}")
    print(f"最小市值: {config.stock_pool.min_market_cap}亿")
    print(f"单票最大仓位: {config.position.max_single_position*100}%")
    print(f"初始止损ATR倍数: {config.exit.initial_stop_atr_mult}")
    print(f"大盘择时启用: {config.market_timing.enabled}")
    print(f"板块共振启用: {config.sector_resonance.enabled}")
    print(f"时间止损启用: {config.exit.time_stop_enabled}")
