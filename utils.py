"""
A股科技成长股趋势策略 V1.0 - 工具函数
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Union
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """设置模块日志"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def format_number(value: float, precision: int = 2) -> str:
    """格式化数字显示"""
    if abs(value) >= 1e8:
        return f"{value/1e8:.{precision}f}亿"
    elif abs(value) >= 1e4:
        return f"{value/1e4:.{precision}f}万"
    else:
        return f"{value:.{precision}f}"


def format_percent(value: float, precision: int = 2) -> str:
    """格式化百分比显示"""
    return f"{value*100:.{precision}f}%"


def format_date(date: Union[str, datetime, pd.Timestamp]) -> str:
    """统一日期格式为YYYY-MM-DD"""
    if isinstance(date, str):
        return pd.to_datetime(date).strftime('%Y-%m-%d')
    return date.strftime('%Y-%m-%d')


def parse_date(date_str: str) -> datetime:
    """解析日期字符串"""
    return pd.to_datetime(date_str).to_pydatetime()


def get_trade_dates(start: str, end: str, trade_cal: pd.Series = None) -> List[str]:
    """获取交易日列表
    
    Args:
        start: 开始日期
        end: 结束日期
        trade_cal: 交易日历Series，index为日期，值为是否交易日
    
    Returns:
        交易日列表
    """
    if trade_cal is None:
        # 简化处理：排除周末
        dates = pd.date_range(start, end, freq='B')
        return [d.strftime('%Y-%m-%d') for d in dates]
    else:
        return trade_cal[start:end][trade_cal == 1].index.tolist()


def shift_trade_date(date: str, days: int, trade_dates: List[str]) -> Optional[str]:
    """获取指定日期前后N个交易日的日期
    
    Args:
        date: 基准日期
        days: 偏移天数（正数向后，负数向前）
        trade_dates: 交易日列表
    
    Returns:
        偏移后的日期，若超出范围返回None
    """
    try:
        idx = trade_dates.index(date)
        new_idx = idx + days
        if 0 <= new_idx < len(trade_dates):
            return trade_dates[new_idx]
    except ValueError:
        pass
    return None


def is_limit_up(close: float, prev_close: float, threshold: float = 0.095) -> bool:
    """判断是否涨停"""
    if prev_close == 0:
        return False
    return (close - prev_close) / prev_close >= threshold


def is_limit_down(close: float, prev_close: float, threshold: float = 0.095) -> bool:
    """判断是否跌停"""
    if prev_close == 0:
        return False
    return (close - prev_close) / prev_close <= -threshold


def calc_return(entry_price: float, exit_price: float) -> float:
    """计算收益率"""
    if entry_price == 0:
        return 0.0
    return (exit_price - entry_price) / entry_price


def calc_pnl(entry_price: float, exit_price: float, shares: int, 
             commission: float = 0.0003, stamp_tax: float = 0.001) -> Tuple[float, float]:
    """计算盈亏（含手续费）
    
    Returns:
        (净盈亏, 总费用)
    """
    entry_cost = entry_price * shares * (1 + commission)
    exit_value = exit_price * shares * (1 - commission - stamp_tax)
    net_pnl = exit_value - entry_cost
    total_fee = entry_price * shares * commission + exit_price * shares * (commission + stamp_tax)
    return net_pnl, total_fee


def percentile_rank(value: float, series: pd.Series) -> float:
    """计算值在序列中的百分位排名
    
    Args:
        value: 待计算的值
        series: 参考序列
    
    Returns:
        百分位排名 (0-1)
    """
    if len(series) == 0:
        return 0.5
    return (series < value).sum() / len(series)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """计算滚动百分位排名"""
    def calc_pct(x):
        if len(x) < window:
            return np.nan
        return percentile_rank(x.iloc[-1], x)
    
    return series.rolling(window).apply(calc_pct, raw=False)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法"""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def round_to_lot(shares: int, lot_size: int = 100) -> int:
    """将股数取整到手（A股100股=1手）"""
    return (shares // lot_size) * lot_size


def calc_position_shares(capital: float, price: float, 
                        position_pct: float, lot_size: int = 100) -> int:
    """计算仓位对应的股数
    
    Args:
        capital: 总资金
        price: 股价
        position_pct: 仓位比例 (0-1)
        lot_size: 最小交易单位
    
    Returns:
        股数（已取整到手）
    """
    if price <= 0:
        return 0
    target_value = capital * position_pct
    shares = int(target_value / price)
    return round_to_lot(shares, lot_size)


def validate_dataframe(df: pd.DataFrame, required_cols: List[str], 
                       name: str = "DataFrame") -> bool:
    """验证DataFrame是否包含必需列"""
    if df is None or df.empty:
        logger.warning(f"{name} 为空")
        return False
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.warning(f"{name} 缺少列: {missing}")
        return False
    return True


def clean_stock_code(code: str) -> str:
    """清理股票代码，统一格式
    
    Examples:
        600519.SH -> 600519
        000001.SZ -> 000001
        sh600519 -> 600519
    """
    code = str(code).upper()
    # 移除市场后缀
    for suffix in ['.SH', '.SZ', '.BJ', '.SS']:
        code = code.replace(suffix, '')
    # 移除市场前缀
    for prefix in ['SH', 'SZ', 'BJ']:
        if code.startswith(prefix):
            code = code[2:]
    return code.zfill(6)


def add_market_suffix(code: str) -> str:
    """添加市场后缀
    
    Args:
        code: 6位股票代码
    
    Returns:
        带后缀的代码（如 600519.SH）
    """
    code = clean_stock_code(code)
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    elif code.startswith(('0', '2', '3')):
        return f"{code}.SZ"
    elif code.startswith(('4', '8')):
        return f"{code}.BJ"
    return code


def get_market(code: str) -> str:
    """获取股票所属市场"""
    code = clean_stock_code(code)
    if code.startswith(('6', '9')):
        return "SH"
    elif code.startswith(('0', '2', '3')):
        return "SZ"
    elif code.startswith(('4', '8')):
        return "BJ"
    return "UNKNOWN"


class Timer:
    """计时器"""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.name:
            logger.info(f"{self.name} 耗时: {self.elapsed:.2f}秒")


class ProgressBar:
    """简易进度条"""
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        self.current += n
        pct = self.current / self.total * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = elapsed / self.current * (self.total - self.current) if self.current > 0 else 0
        print(f"\r{self.desc}: {pct:.1f}% [{self.current}/{self.total}] "
              f"已用:{elapsed:.0f}s 预计:{eta:.0f}s", end="")
        if self.current >= self.total:
            print()
    
    def close(self):
        print()


def create_summary_table(data: dict, title: str = "") -> str:
    """创建简单的汇总表格"""
    lines = []
    if title:
        lines.append(f"\n{'='*50}")
        lines.append(f" {title}")
        lines.append(f"{'='*50}")
    
    max_key_len = max(len(str(k)) for k in data.keys())
    for key, value in data.items():
        lines.append(f" {str(key):<{max_key_len}} : {value}")
    
    lines.append(f"{'='*50}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试数字格式化
    assert format_number(123456789) == "1.23亿"
    assert format_number(12345) == "1.23万"
    assert format_percent(0.1234) == "12.34%"
    
    # 测试股票代码处理
    assert clean_stock_code("600519.SH") == "600519"
    assert clean_stock_code("sh600519") == "600519"
    assert add_market_suffix("600519") == "600519.SH"
    assert add_market_suffix("000001") == "000001.SZ"
    
    # 测试涨跌停判断
    assert is_limit_up(11.0, 10.0)
    assert not is_limit_up(10.5, 10.0)
    assert is_limit_down(9.0, 10.0)
    
    # 测试仓位计算
    shares = calc_position_shares(1000000, 50, 0.2)
    assert shares == 4000  # 20万/50元=4000股
    
    # 测试百分位
    s = pd.Series([1, 2, 3, 4, 5])
    assert percentile_rank(3, s) == 0.4  # 有2个小于3，2/5=0.4
    
    print("所有测试通过!")
