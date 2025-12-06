"""
仓位管理模块 V2.0
新增: 时间止损、动态追踪止损、分批止盈优化
"""

import pandas as pd
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from config import PositionConfig, ExitConfig, RiskConfig, EntryType, MarketRegime
from utils import setup_logger, format_percent, calc_position_shares, round_to_lot

logger = setup_logger(__name__)


class ExitReason(Enum):
    """退出原因"""
    STOP_LOSS = "止损"
    TAKE_PROFIT_1 = "止盈1"
    TAKE_PROFIT_2 = "止盈2"
    TRAILING_STOP = "追踪止损"
    TIME_STOP = "时间止损"
    TOP_SIGNAL = "顶部信号"
    BREAKEVEN_STOP = "保本止损"
    MANUAL = "手动"
    MARKET_RISK = "市场风险"


@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    code: str
    name: str
    direction: str  # 'buy' or 'sell'
    price: float
    shares: int
    amount: float
    commission: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: Optional[ExitReason] = None
    hold_days: int = 0


@dataclass
class Position:
    """持仓信息"""
    code: str
    name: str
    entry_date: str
    entry_price: float
    shares: int
    entry_type: EntryType
    initial_stop: float
    current_stop: float
    atr_at_entry: float
    industry: str = "Unknown"
    signal_score: int = 0
    
    # 动态更新字段
    highest_close: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    hold_days: int = 0
    
    # 止盈状态
    tp1_triggered: bool = False
    tp2_triggered: bool = False
    original_shares: int = 0  # 原始持仓数
    
    def __post_init__(self):
        if self.original_shares == 0:
            self.original_shares = self.shares
        if self.highest_close == 0:
            self.highest_close = self.entry_price

    @property
    def market_value(self) -> float:
        return self.current_price * self.shares
    
    @property
    def profit_pct(self) -> float:
        if self.entry_price == 0:
            return 0
        return (self.current_price - self.entry_price) / self.entry_price
    
    @property
    def profit_atr(self) -> float:
        """盈利ATR倍数"""
        if self.atr_at_entry == 0:
            return 0
        return (self.current_price - self.entry_price) / self.atr_at_entry


class PositionManager:
    """仓位管理器 V2.0"""
    
    def __init__(self, pos_config: PositionConfig, exit_config: ExitConfig, 
                 risk_config: RiskConfig, initial_capital: float):
        self.pos_config = pos_config
        self.exit_config = exit_config
        self.risk_config = risk_config
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        
        # 风控状态
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.consecutive_stops = 0
        self.pause_until_date: Optional[str] = None
        self.market_regime = MarketRegime.RANGE
        
        # 统计
        self.total_wins = 0
        self.total_losses = 0
    
    def open_position(self, code: str, name: str, date: str, price: float,
                     shares: int, stop_price: float, atr: float,
                     entry_type: EntryType, signal_score: int = 0,
                     industry: str = "Unknown",
                     commission_rate: float = 0.0003) -> bool:
        """开仓
        
        Returns:
            是否成功开仓
        """
        if code in self.positions:
            logger.warning(f"已持有 {code}，无法重复开仓")
            return False
        
        cost = price * shares * (1 + commission_rate)
        
        if cost > self.cash:
            logger.warning(f"资金不足: 需要 {cost:.2f}, 可用 {self.cash:.2f}")
            return False
        
        # 扣除资金
        self.cash -= cost
        
        # 创建持仓
        position = Position(
            code=code,
            name=name,
            entry_date=date,
            entry_price=price,
            shares=shares,
            entry_type=entry_type,
            initial_stop=stop_price,
            current_stop=stop_price,
            atr_at_entry=atr,
            industry=industry,
            signal_score=signal_score,
            current_price=price,
            highest_close=price
        )
        
        self.positions[code] = position
        
        # 记录交易
        trade = TradeRecord(
            date=date,
            code=code,
            name=name,
            direction='buy',
            price=price,
            shares=shares,
            amount=cost,
            commission=price * shares * commission_rate
        )
        self.trades.append(trade)
        
        logger.info(f"开仓: {code} {name} @ {price:.2f} x {shares}股, "
                   f"止损: {stop_price:.2f}")
        
        return True
    
    def close_position(self, code: str, date: str, price: float,
                      reason: ExitReason, sell_ratio: float = 1.0,
                      commission_rate: float = 0.0003,
                      stamp_tax_rate: float = 0.001) -> Optional[TradeRecord]:
        """平仓（支持部分平仓）
        
        Args:
            sell_ratio: 卖出比例 (0-1)
        
        Returns:
            交易记录
        """
        if code not in self.positions:
            logger.warning(f"未持有 {code}")
            return None
        
        pos = self.positions[code]
        
        # 计算卖出股数
        sell_shares = round_to_lot(int(pos.shares * sell_ratio), 100)
        if sell_shares <= 0:
            sell_shares = pos.shares  # 至少卖1手或全部
        
        # 计算卖出所得
        gross_value = price * sell_shares
        commission = gross_value * commission_rate
        stamp_tax = gross_value * stamp_tax_rate
        net_value = gross_value - commission - stamp_tax
        
        # 计算盈亏
        entry_cost = pos.entry_price * sell_shares
        pnl = net_value - entry_cost
        pnl_pct = (price - pos.entry_price) / pos.entry_price
        
        # 更新资金
        self.cash += net_value
        
        # 计算持仓天数
        try:
            entry_dt = datetime.strptime(pos.entry_date, '%Y-%m-%d')
            exit_dt = datetime.strptime(date, '%Y-%m-%d')
            hold_days = (exit_dt - entry_dt).days
        except:
            hold_days = pos.hold_days
        
        # 记录交易
        trade = TradeRecord(
            date=date,
            code=code,
            name=pos.name,
            direction='sell',
            price=price,
            shares=sell_shares,
            amount=net_value,
            commission=commission + stamp_tax,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            hold_days=hold_days
        )
        self.trades.append(trade)
        
        # 更新持仓或删除
        remaining_shares = pos.shares - sell_shares
        if remaining_shares <= 0:
            del self.positions[code]
            logger.info(f"平仓: {code} @ {price:.2f} x {sell_shares}股, "
                       f"盈亏: {pnl:.2f} ({pnl_pct:+.2%}), 原因: {reason.value}")
        else:
            pos.shares = remaining_shares
            logger.info(f"减仓: {code} @ {price:.2f} x {sell_shares}股, "
                       f"剩余: {remaining_shares}股, 原因: {reason.value}")
        
        # 更新统计
        if pnl > 0:
            self.total_wins += 1
            self.consecutive_stops = 0
        else:
            self.total_losses += 1
            if reason == ExitReason.STOP_LOSS:
                self.consecutive_stops += 1
        
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        
        return trade
    
    def update_positions(self, date_str: str, daily_snapshot: Dict[str, pd.Series]):
        """更新所有持仓状态"""
        for code, pos in self.positions.items():
            if code in daily_snapshot:
                data = daily_snapshot[code]
                current_price = data['close']
                
                # 更新价格
                pos.current_price = current_price
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.shares
                
                # 更新最高价
                if current_price > pos.highest_close:
                    pos.highest_close = current_price
                
                # 更新持仓天数
                pos.hold_days += 1
                
                # 更新止损位（追踪止损）
                self._update_trailing_stop(pos)
    
    def _update_trailing_stop(self, pos: Position):
        """更新追踪止损"""
        if pos.atr_at_entry <= 0:
            return
        
        profit_atr = pos.profit_atr
        
        # 保本止损（盈利 > 1.5 ATR时启动）
        if profit_atr >= self.exit_config.breakeven_atr:
            breakeven_stop = pos.entry_price * self.exit_config.breakeven_buffer
            if breakeven_stop > pos.current_stop:
                pos.current_stop = breakeven_stop
        
        # 追踪止损（盈利 > 2.5 ATR时启动）
        if profit_atr >= self.exit_config.trailing_start_atr:
            trailing_stop = pos.highest_close - self.exit_config.trailing_atr_mult * pos.atr_at_entry
            if trailing_stop > pos.current_stop:
                pos.current_stop = trailing_stop
        
        # 收紧止损（盈利 > 4 ATR时启动）
        if profit_atr >= self.exit_config.tight_start_atr:
            tight_stop = pos.highest_close - self.exit_config.tight_atr_mult * pos.atr_at_entry
            if tight_stop > pos.current_stop:
                pos.current_stop = tight_stop
    
    def check_exit_signals(self, code: str, price_data: Union[Dict, pd.Series],
                          current_date: str = None) -> Tuple[bool, Optional[ExitReason], float]:
        """检查退出信号
        
        Returns:
            (是否退出, 退出原因, 卖出比例)
        """
        if code not in self.positions:
            return False, None, 0.0
        
        pos = self.positions[code]
        
        try:
            current_close = price_data['close']
            current_atr = price_data.get('atr', pos.atr_at_entry)
        except (KeyError, AttributeError, TypeError):
            logger.error(f"Price data format error for {code}")
            return False, None, 0.0
        
        # 更新当前价格
        pos.current_price = current_close
        
        # 1. 硬止损
        if current_close <= pos.current_stop:
            return True, ExitReason.STOP_LOSS, 1.0
        
        # 2. 时间止损
        if self.exit_config.time_stop_enabled:
            if pos.hold_days >= self.exit_config.time_stop_days:
                profit_pct = pos.profit_pct
                if profit_pct < self.exit_config.time_stop_profit_threshold:
                    return True, ExitReason.TIME_STOP, 1.0
        
        # 计算盈利ATR倍数
        if pos.atr_at_entry <= 0:
            return False, None, 0
        
        profit_atr = (current_close - pos.entry_price) / pos.atr_at_entry
        
        # 3. 分批止盈 TP1
        if not pos.tp1_triggered and profit_atr >= self.exit_config.tp1_atr:
            pos.tp1_triggered = True
            # 上移止损
            new_stop = pos.entry_price + self.exit_config.tp1_new_stop_atr * pos.atr_at_entry
            pos.current_stop = max(pos.current_stop, new_stop)
            return True, ExitReason.TAKE_PROFIT_1, self.exit_config.tp1_sell_pct
        
        # 4. 分批止盈 TP2
        if pos.tp1_triggered and not pos.tp2_triggered and profit_atr >= self.exit_config.tp2_atr:
            pos.tp2_triggered = True
            # 收紧止损
            new_stop = pos.highest_close - self.exit_config.tp2_new_stop_atr * pos.atr_at_entry
            pos.current_stop = max(pos.current_stop, new_stop)
            return True, ExitReason.TAKE_PROFIT_2, self.exit_config.tp2_sell_pct
        
        # 5. 追踪止损触发
        if pos.tp1_triggered and current_close <= pos.current_stop:
            return True, ExitReason.TRAILING_STOP, 1.0
        
        # 6. 顶部信号检测
        if self._check_top_signal(price_data):
            return True, ExitReason.TOP_SIGNAL, 0.5  # 卖出50%
        
        return False, None, 0.0
    
    def _check_top_signal(self, price_data: pd.Series) -> bool:
        """检测顶部信号"""
        # 放量滞涨
        if price_data.get('top_stall', False):
            return True
        
        # RSI背离
        if price_data.get('rsi_divergence', False):
            return True
        
        # 长上影线
        if price_data.get('long_upper_shadow', False):
            return True
        
        return False
    
    def calc_position_size(self, entry_price: float, stop_price: float,
                          entry_type: EntryType, signal_score: int = 0,
                          position_ratio: float = 1.0) -> Tuple[float, int]:
        """计算仓位大小
        
        Args:
            entry_price: 入场价
            stop_price: 止损价
            entry_type: 入场类型
            signal_score: 信号评分
            position_ratio: 仓位系数（板块共振等调整）
        
        Returns:
            (仓位比例, 股数)
        """
        # 基于风险的仓位计算
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.05  # 默认5%风险
        
        total_value = self.get_total_value()
        risk_amount = total_value * self.pos_config.single_risk_pct
        
        # 理论股数
        theoretical_shares = risk_amount / risk_per_share
        
        # 入场类型调整
        entry_ratio = self.pos_config.entry_position_ratio.get(
            entry_type.value, 0.5
        )
        
        # 市场环境调整
        regime_mult = self.pos_config.regime_position_mult.get(
            self.market_regime.value, 0.625
        )
        
        # 最终股数
        adjusted_shares = theoretical_shares * entry_ratio * regime_mult * position_ratio
        final_shares = round_to_lot(int(adjusted_shares), 100)
        
        # 检查单票上限
        position_value = entry_price * final_shares
        max_position_value = total_value * self.pos_config.max_single_position
        
        if position_value > max_position_value:
            final_shares = round_to_lot(int(max_position_value / entry_price), 100)
        
        # 检查现金
        if entry_price * final_shares > self.cash:
            final_shares = round_to_lot(int(self.cash * 0.95 / entry_price), 100)
        
        position_pct = (entry_price * final_shares) / total_value if total_value > 0 else 0
        
        return position_pct, final_shares
    
    def check_position_constraints(self, code: str, industry: str, 
                                   position_pct: float) -> Tuple[bool, str]:
        """检查开仓限制"""
        # 1. 是否已持有
        if code in self.positions:
            return False, "已持有该股票"
        
        # 2. 持仓数限制
        if len(self.positions) >= self.pos_config.max_holdings:
            return False, "持仓数已满"
        
        # 3. 单票仓位限制
        if position_pct > self.pos_config.max_single_position:
            return False, f"单票超限 {position_pct:.1%}"
        
        # 4. 板块仓位限制
        if industry and industry != "Unknown":
            current_sector_pos = self._calc_sector_position(industry)
            if current_sector_pos + position_pct > self.pos_config.max_sector_position:
                return False, f"板块({industry})仓位超限"
        
        # 5. 总仓位限制
        current_total_pos = self._calc_total_position()
        if current_total_pos + position_pct > self.pos_config.max_total_position:
            return False, f"总仓位超限"
        
        return True, "OK"
    
    def check_risk_limits(self, date_str: str) -> Tuple[bool, str]:
        """检查风控限制"""
        # 暂停交易检查
        if self.pause_until_date:
            if date_str <= self.pause_until_date:
                return False, f"暂停交易至 {self.pause_until_date}"
            else:
                self.pause_until_date = None
        
        # 连续止损检查
        if self.consecutive_stops >= self.risk_config.consecutive_stops:
            # 设置暂停
            try:
                pause_date = datetime.strptime(date_str, '%Y-%m-%d')
                from datetime import timedelta
                pause_until = pause_date + timedelta(days=self.risk_config.pause_days)
                self.pause_until_date = pause_until.strftime('%Y-%m-%d')
            except:
                pass
            self.consecutive_stops = 0
            return False, f"连续止损{self.risk_config.consecutive_stops}次，暂停交易"
        
        # 单日亏损检查
        daily_loss_pct = self.daily_pnl / self.initial_capital
        if daily_loss_pct <= -self.risk_config.max_daily_loss:
            return False, f"单日亏损达限 {daily_loss_pct:.1%}"
        
        return True, "OK"
    
    def _calc_sector_position(self, industry: str) -> float:
        """计算板块仓位"""
        total_val = self.get_total_value()
        if total_val == 0:
            return 0
        
        sector_val = sum(
            p.market_value for p in self.positions.values() 
            if p.industry == industry
        )
        return sector_val / total_val
    
    def _calc_total_position(self) -> float:
        """计算总仓位"""
        total_val = self.get_total_value()
        if total_val == 0:
            return 0
        
        position_val = sum(p.market_value for p in self.positions.values())
        return position_val / total_val
    
    def get_total_value(self) -> float:
        """获取总资产"""
        return self.cash + sum(p.market_value for p in self.positions.values())
    
    def get_account_summary(self) -> Dict:
        """获取账户摘要"""
        total_val = self.get_total_value()
        position_val = sum(p.market_value for p in self.positions.values())
        
        return {
            '总资产': total_val,
            '现金': self.cash,
            '持仓市值': position_val,
            '仓位比例': position_val / total_val if total_val > 0 else 0,
            '总收益': (total_val - self.initial_capital) / self.initial_capital,
            '持仓数': len(self.positions),
            '交易次数': len(self.trades),
            '胜率': self.total_wins / (self.total_wins + self.total_losses) if (self.total_wins + self.total_losses) > 0 else 0
        }
    
    def reset_daily_stats(self):
        """重置每日统计"""
        self.daily_pnl = 0.0
    
    def reset_weekly_stats(self):
        """重置每周统计"""
        self.weekly_pnl = 0.0
    
    def update_stops(self, date_str: str, daily_snapshot: Dict[str, pd.Series]):
        """更新所有持仓的止损位（兼容旧接口）"""
        self.update_positions(date_str, daily_snapshot)


if __name__ == "__main__":
    from config import PositionConfig, ExitConfig, RiskConfig, EntryType
    
    print("测试仓位管理模块 V2.0...")
    
    # 初始化
    pos_config = PositionConfig()
    exit_config = ExitConfig()
    risk_config = RiskConfig()
    
    pm = PositionManager(pos_config, exit_config, risk_config, 1000000)
    
    # 测试开仓
    success = pm.open_position(
        code='000001',
        name='测试股票',
        date='2024-01-10',
        price=50.0,
        shares=1000,
        stop_price=47.0,
        atr=1.5,
        entry_type=EntryType.BREAKOUT,
        signal_score=5,
        industry='半导体'
    )
    print(f"\n开仓结果: {success}")
    print(f"账户摘要: {pm.get_account_summary()}")
    
    # 模拟价格更新
    for day, price in enumerate([51, 52, 53, 52.5, 54, 55, 53], start=1):
        mock_data = pd.Series({
            'close': price,
            'atr': 1.5,
            'top_stall': False,
            'rsi_divergence': False,
            'long_upper_shadow': False
        })
        
        pm.update_positions(f'2024-01-{10+day}', {'000001': mock_data})
        
        should_exit, reason, ratio = pm.check_exit_signals(
            '000001', mock_data, f'2024-01-{10+day}'
        )
        
        pos = pm.positions.get('000001')
        if pos:
            print(f"Day {day}: 价格={price}, 止损={pos.current_stop:.2f}, "
                  f"盈利ATR={pos.profit_atr:.1f}, TP1={pos.tp1_triggered}")
        
        if should_exit:
            pm.close_position('000001', f'2024-01-{10+day}', price, reason, ratio)
            break
    
    print(f"\n最终账户摘要: {pm.get_account_summary()}")
    print("\n仓位管理模块测试完成!")
