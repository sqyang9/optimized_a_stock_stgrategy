"""
信号生成模块 V2.0
新增: 板块共振检测、信号质量评估
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from config import SignalConfig, EntryConfig, SignalGrade, EntryType, SectorResonanceConfig
from utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class TradingSignal:
    """交易信号"""
    code: str
    name: str
    date: str
    signal_type: str
    grade: SignalGrade
    score: int
    entry_type: EntryType
    entry_price_ref: float
    stop_price: float
    close: float
    atr: float
    industry: str = "Unknown"
    details: Dict = field(default_factory=dict)
    
    # 板块共振相关
    sector_signal_count: int = 0        # 同板块信号数
    has_sector_resonance: bool = False  # 是否有板块共振
    adjusted_score: int = 0             # 调整后评分
    position_ratio: float = 1.0         # 仓位系数


class SignalGenerator:
    """信号生成器 V2.0"""
    
    def __init__(self, signal_config: SignalConfig, entry_config: EntryConfig,
                 sector_config: SectorResonanceConfig = None,
                 industry_map: Dict[str, str] = None):
        self.signal_config = signal_config
        self.entry_config = entry_config
        self.sector_config = sector_config or SectorResonanceConfig()
        self.industry_map = industry_map or {}
    
    def set_industry_map(self, industry_map: Dict[str, str]):
        """设置股票-行业映射"""
        self.industry_map = industry_map
    
    def scan_daily_snapshot(self, 
                           daily_snapshot: Dict[str, pd.Series], 
                           date_str: str,
                           min_score: int = 3) -> List[TradingSignal]:
        """
        扫描当日快照，生成交易信号
        
        Args:
            daily_snapshot: {code: pd.Series} 包含当日的所有指标数据
            date_str: 当前日期
            min_score: 最低信号评分门槛（根据市场环境调整）
        
        Returns:
            排序后的交易信号列表
        """
        # 第一轮：生成原始信号
        raw_signals = self._generate_raw_signals(daily_snapshot, date_str)
        
        if not raw_signals:
            return []
        
        # 第二轮：板块共振分析
        if self.sector_config.enabled:
            raw_signals = self._apply_sector_resonance(raw_signals)
        
        # 第三轮：过滤和排序
        filtered_signals = [
            s for s in raw_signals 
            if s.adjusted_score >= min_score
        ]
        
        # 按调整后评分排序（高到低）
        filtered_signals.sort(key=lambda x: (x.adjusted_score, x.score), reverse=True)
        
        logger.debug(f"{date_str}: 生成 {len(filtered_signals)} 个有效信号")
        
        return filtered_signals
    
    def _generate_raw_signals(self, daily_snapshot: Dict[str, pd.Series],
                              date_str: str) -> List[TradingSignal]:
        """生成原始信号（不含板块共振调整）"""
        signals = []
        
        for code, data in daily_snapshot.items():
            # 1. 基础过滤
            if data.get('volume', 0) == 0:
                continue
            
            # 2. 核心逻辑：挤压释放
            if not data.get('squeeze_release', False):
                continue
            
            # 3. 计算基础评分 (0-7分)
            score, details = self._calculate_score(data)
            
            # 4. 最低门槛过滤（避免生成过多无效信号）
            if score < 2:
                continue
            
            # 5. 获取行业
            industry = self.industry_map.get(code, "Unknown")
            
            # 6. 计算止损位
            atr = data.get('atr', data['close'] * 0.03)
            stop_price = data['close'] - 2.0 * atr
            
            # 7. 确定信号等级
            grade = self._determine_grade(score)
            
            signal = TradingSignal(
                code=code,
                name=f"Stock_{code}",
                date=date_str,
                signal_type="squeeze_release",
                grade=grade,
                score=score,
                entry_type=EntryType.BREAKOUT,
                entry_price_ref=data['close'],
                stop_price=stop_price,
                close=data['close'],
                atr=atr,
                industry=industry,
                details=details,
                adjusted_score=score,  # 初始化为原始分
                position_ratio=1.0
            )
            signals.append(signal)
        
        return signals
    
    def _calculate_score(self, data: pd.Series) -> Tuple[int, Dict]:
        """计算信号评分
        
        评分维度:
        - 量能配合: 0-2分
        - 趋势强度: 0-2分  
        - 突破力度: 0-1分
        - 挤压质量: 0-1分
        - 价格位置: 0-1分
        
        总分: 0-7分
        """
        score = 0
        details = {}
        
        # A. 量能配合 (0-2分)
        vol_ratio = data.get('volume_ratio', 0)
        if vol_ratio > self.signal_config.strong_volume_ratio:
            score += 2
            details['volume'] = 'strong'
        elif vol_ratio > self.signal_config.breakout_volume_ratio:
            score += 1
            details['volume'] = 'normal'
        
        # B. 趋势强度 (0-2分)
        trend_up = data.get('trend_up', False)
        trend_strength = data.get('trend_strength', False)
        
        if trend_strength:
            score += 2
            details['trend'] = 'strong'
        elif trend_up:
            score += 1
            details['trend'] = 'up'
        
        # C. 突破力度 (0-1分)
        pct_change = data.get('pct_change', 0)
        if pct_change > self.signal_config.min_breakout_pct:
            score += 1
            details['breakout'] = True
        
        # D. 挤压质量 (0-1分)
        squeeze_days = data.get('squeeze_days', 0)
        if squeeze_days >= self.signal_config.squeeze_days_min:
            score += 1
            details['squeeze_quality'] = True
        
        # E. 价格位置 (0-1分) - 新增
        # 收盘价在当日区间的位置，越高越好
        close_position = data.get('close_position', 0.5)
        if close_position > 0.7:  # 收在高位
            score += 1
            details['close_position'] = 'high'
        
        return score, details
    
    def _determine_grade(self, score: int) -> SignalGrade:
        """确定信号等级"""
        if score >= self.signal_config.grade_a_threshold:
            return SignalGrade.A
        elif score >= self.signal_config.grade_b_threshold:
            return SignalGrade.B
        else:
            return SignalGrade.C
    
    def _apply_sector_resonance(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """应用板块共振分析
        
        检查同板块有多少只股票同时触发信号，
        有共振的信号加分，孤立信号降低仓位。
        """
        if not signals:
            return signals
        
        # 统计各板块信号数量
        sector_counts = defaultdict(int)
        for sig in signals:
            if sig.industry and sig.industry != "Unknown":
                sector_counts[sig.industry] += 1
        
        # 更新每个信号的板块共振信息
        for sig in signals:
            industry = sig.industry
            
            if industry and industry != "Unknown":
                count = sector_counts.get(industry, 0)
                sig.sector_signal_count = count
                
                # 判断是否有板块共振
                if count >= self.sector_config.min_sector_signals:
                    sig.has_sector_resonance = True
                    # 加分
                    sig.adjusted_score = sig.score + self.sector_config.resonance_bonus_score
                    sig.position_ratio = 1.0
                    sig.details['sector_resonance'] = True
                else:
                    sig.has_sector_resonance = False
                    sig.adjusted_score = sig.score
                    # 孤立信号降低仓位
                    sig.position_ratio = self.sector_config.isolated_signal_position_ratio
                    sig.details['isolated'] = True
            else:
                # 无行业信息的股票
                sig.adjusted_score = sig.score
                sig.position_ratio = self.sector_config.isolated_signal_position_ratio
        
        # 更新等级
        for sig in signals:
            sig.grade = self._determine_grade(sig.adjusted_score)
        
        return signals
    
    def get_sector_summary(self, signals: List[TradingSignal]) -> Dict[str, Dict]:
        """获取板块信号汇总
        
        Returns:
            {板块名: {count: 数量, codes: [代码列表], avg_score: 平均分}}
        """
        sector_summary = defaultdict(lambda: {'count': 0, 'codes': [], 'total_score': 0})
        
        for sig in signals:
            industry = sig.industry or "Unknown"
            sector_summary[industry]['count'] += 1
            sector_summary[industry]['codes'].append(sig.code)
            sector_summary[industry]['total_score'] += sig.score
        
        # 计算平均分
        for sector, info in sector_summary.items():
            if info['count'] > 0:
                info['avg_score'] = info['total_score'] / info['count']
            else:
                info['avg_score'] = 0
            del info['total_score']
        
        return dict(sector_summary)


class WatchlistManager:
    """观察池管理器
    
    管理突破后待回踩确认的股票
    """
    
    def __init__(self, max_watch_days: int = 5):
        self.max_watch_days = max_watch_days
        self.watchlist: Dict[str, Dict] = {}  # {code: {signal, added_date, days}}
    
    def add_to_watch(self, signal: TradingSignal):
        """添加到观察池"""
        self.watchlist[signal.code] = {
            'signal': signal,
            'added_date': signal.date,
            'days': 0,
            'highest_since': signal.close,
            'pullback_target': signal.entry_price_ref * 0.97  # 回踩目标
        }
        logger.debug(f"添加到观察池: {signal.code}")
    
    def update_daily(self, date_str: str, daily_snapshot: Dict[str, pd.Series]):
        """每日更新观察池"""
        expired = []
        
        for code, info in self.watchlist.items():
            info['days'] += 1
            
            # 超时移除
            if info['days'] > self.max_watch_days:
                expired.append(code)
                continue
            
            # 更新最高价
            if code in daily_snapshot:
                current_close = daily_snapshot[code]['close']
                if current_close > info['highest_since']:
                    info['highest_since'] = current_close
        
        # 移除过期
        for code in expired:
            del self.watchlist[code]
            logger.debug(f"移除过期观察: {code}")
    
    def check_pullback_entry(self, code: str, 
                             current_data: pd.Series) -> Optional[TradingSignal]:
        """检查是否触发回踩入场"""
        if code not in self.watchlist:
            return None
        
        info = self.watchlist[code]
        current_close = current_data['close']
        volume_ratio = current_data.get('volume_ratio', 1.0)
        
        # 回踩条件：
        # 1. 价格回到目标区域
        # 2. 缩量
        # 3. 没有破位
        original_signal = info['signal']
        
        if (current_close <= info['pullback_target'] and
            volume_ratio < 0.8 and
            current_close > original_signal.stop_price):
            
            # 创建回踩入场信号
            pullback_signal = TradingSignal(
                code=code,
                name=original_signal.name,
                date=current_data.name if hasattr(current_data, 'name') else '',
                signal_type="pullback_entry",
                grade=original_signal.grade,
                score=original_signal.score,
                entry_type=EntryType.PULLBACK,
                entry_price_ref=current_close,
                stop_price=original_signal.stop_price,
                close=current_close,
                atr=original_signal.atr,
                industry=original_signal.industry,
                details={'original_signal': original_signal.date},
                adjusted_score=original_signal.adjusted_score,
                position_ratio=0.6  # 回踩入场用60%仓位
            )
            
            # 从观察池移除
            del self.watchlist[code]
            
            return pullback_signal
        
        return None
    
    def get_watchlist_codes(self) -> List[str]:
        """获取观察池股票代码"""
        return list(self.watchlist.keys())


if __name__ == "__main__":
    from config import SignalConfig, EntryConfig, SectorResonanceConfig
    import numpy as np
    
    print("测试信号生成模块 V2.0...")
    
    # 模拟数据
    np.random.seed(42)
    
    # 创建模拟股票数据
    industry_map = {
        '000001': '半导体',
        '000002': '半导体',
        '000003': '半导体',
        '000004': '新能源',
        '000005': '人工智能',
    }
    
    # 模拟当日快照
    daily_snapshot = {}
    for code in industry_map.keys():
        close = 50 + np.random.randn() * 5
        daily_snapshot[code] = pd.Series({
            'close': close,
            'volume': 10000000,
            'volume_ratio': 1.5 + np.random.rand(),
            'squeeze_release': True if np.random.rand() > 0.3 else False,
            'trend_up': True if np.random.rand() > 0.4 else False,
            'trend_strength': True if np.random.rand() > 0.6 else False,
            'pct_change': 0.02 + np.random.rand() * 0.03,
            'squeeze_days': int(5 + np.random.rand() * 5),
            'close_position': 0.5 + np.random.rand() * 0.5,
            'atr': close * 0.03,
        })
    
    # 初始化信号生成器
    signal_config = SignalConfig()
    entry_config = EntryConfig()
    sector_config = SectorResonanceConfig()
    
    generator = SignalGenerator(signal_config, entry_config, sector_config, industry_map)
    
    # 生成信号
    signals = generator.scan_daily_snapshot(daily_snapshot, '2024-01-15', min_score=3)
    
    print(f"\n生成 {len(signals)} 个信号:")
    for sig in signals:
        print(f"  {sig.code} ({sig.industry}): 原始分={sig.score}, "
              f"调整分={sig.adjusted_score}, 等级={sig.grade.value}, "
              f"共振={sig.has_sector_resonance}, 仓位系数={sig.position_ratio:.0%}")
    
    # 板块汇总
    summary = generator.get_sector_summary(signals)
    print(f"\n板块信号汇总:")
    for sector, info in summary.items():
        print(f"  {sector}: {info['count']}个信号, 平均分={info['avg_score']:.1f}")
    
    print("\n信号生成模块测试完成!")
