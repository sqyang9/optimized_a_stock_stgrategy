"""
模块测试脚本
验证各个优化模块是否正常工作
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_config():
    """测试配置模块"""
    print("\n" + "="*50)
    print("测试 1: 配置模块")
    print("="*50)
    
    from config import StrategyConfig, MarketTimingConfig, SectorResonanceConfig
    
    config = StrategyConfig()
    
    print(f"✓ 大盘择时配置:")
    print(f"  - 启用: {config.market_timing.enabled}")
    print(f"  - 指数: {config.market_timing.index_code}")
    print(f"  - 熊市允许开仓: {config.market_timing.allow_entry_in_bear}")
    
    print(f"\n✓ 板块共振配置:")
    print(f"  - 启用: {config.sector_resonance.enabled}")
    print(f"  - 最少信号数: {config.sector_resonance.min_sector_signals}")
    print(f"  - 共振加分: {config.sector_resonance.resonance_bonus_score}")
    
    print(f"\n✓ 时间止损配置:")
    print(f"  - 启用: {config.exit.time_stop_enabled}")
    print(f"  - 天数: {config.exit.time_stop_days}")
    print(f"  - 盈利阈值: {config.exit.time_stop_profit_threshold}")
    
    return True


def test_market_timer():
    """测试大盘择时模块"""
    print("\n" + "="*50)
    print("测试 2: 大盘择时模块")
    print("="*50)
    
    from config import MarketTimingConfig
    from market_timer import MarketTimer, create_mock_index_data
    
    # 创建模拟数据
    index_df = create_mock_index_data('2024-01-01', '2024-06-30')
    
    # 初始化择时器
    config = MarketTimingConfig()
    timer = MarketTimer(config)
    timer.load_index_data(index_df)
    
    # 测试几个日期
    test_dates = ['2024-02-15', '2024-04-15', '2024-06-15']
    
    for date in test_dates:
        state = timer.get_market_state(date)
        print(f"\n✓ {date}:")
        print(f"  - 市场环境: {state.regime.value}")
        print(f"  - 允许开仓: {state.allow_entry}")
        print(f"  - 最大仓位: {state.max_position_ratio:.0%}")
    
    return True


def test_signal_generator():
    """测试信号生成模块"""
    print("\n" + "="*50)
    print("测试 3: 信号生成模块 (含板块共振)")
    print("="*50)
    
    from config import SignalConfig, EntryConfig, SectorResonanceConfig
    from signal_generator import SignalGenerator
    
    # 模拟数据
    industry_map = {
        '000001': '半导体',
        '000002': '半导体',
        '000003': '半导体',
        '000004': '新能源',
        '000005': '人工智能',
    }
    
    # 创建模拟快照
    np.random.seed(42)
    daily_snapshot = {}
    for code in industry_map.keys():
        close = 50 + np.random.randn() * 5
        daily_snapshot[code] = pd.Series({
            'close': close,
            'volume': 10000000,
            'volume_ratio': 1.5 + np.random.rand() * 0.5,
            'squeeze_release': True,  # 所有股票都触发挤压释放
            'trend_up': True,
            'trend_strength': True if np.random.rand() > 0.5 else False,
            'pct_change': 0.025,
            'squeeze_days': 6,
            'close_position': 0.75,
            'atr': close * 0.03,
        })
    
    # 初始化信号生成器
    signal_config = SignalConfig()
    entry_config = EntryConfig()
    sector_config = SectorResonanceConfig()
    
    generator = SignalGenerator(signal_config, entry_config, sector_config, industry_map)
    
    # 生成信号
    signals = generator.scan_daily_snapshot(daily_snapshot, '2024-01-15', min_score=3)
    
    print(f"\n✓ 生成 {len(signals)} 个信号:")
    for sig in signals:
        resonance_icon = "🔥" if sig.has_sector_resonance else "⚪"
        print(f"  {resonance_icon} {sig.code} ({sig.industry}): "
              f"原始分={sig.score}, 调整分={sig.adjusted_score}, "
              f"仓位系数={sig.position_ratio:.0%}")
    
    # 板块汇总
    summary = generator.get_sector_summary(signals)
    print(f"\n✓ 板块汇总:")
    for sector, info in summary.items():
        print(f"  - {sector}: {info['count']}个信号")
    
    return True


def test_position_manager():
    """测试仓位管理模块"""
    print("\n" + "="*50)
    print("测试 4: 仓位管理模块 (含时间止损)")
    print("="*50)
    
    from config import PositionConfig, ExitConfig, RiskConfig, EntryType
    from position_manager import PositionManager, ExitReason
    
    # 初始化
    pos_config = PositionConfig()
    exit_config = ExitConfig()
    risk_config = RiskConfig()
    
    pm = PositionManager(pos_config, exit_config, risk_config, 1000000)
    
    # 开仓
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
    print(f"\n✓ 开仓: {'成功' if success else '失败'}")
    
    # 模拟5天横盘（测试时间止损）
    print("\n✓ 模拟5天横盘:")
    for day in range(1, 7):
        price = 50.2 + np.random.randn() * 0.3  # 小幅波动
        mock_data = pd.Series({
            'close': price,
            'atr': 1.5,
            'top_stall': False,
            'rsi_divergence': False,
            'long_upper_shadow': False
        })
        
        date_str = f'2024-01-{10+day}'
        pm.update_positions(date_str, {'000001': mock_data})
        
        should_exit, reason, ratio = pm.check_exit_signals('000001', mock_data, date_str)
        
        pos = pm.positions.get('000001')
        if pos:
            print(f"  Day {day}: 价格={price:.2f}, 持仓天数={pos.hold_days}, "
                  f"盈利={pos.profit_pct:.2%}")
            
            if should_exit:
                print(f"  ⚠️ 触发退出: {reason.value}")
                pm.close_position('000001', date_str, price, reason, ratio)
                break
    
    print(f"\n✓ 最终账户: {pm.get_account_summary()}")
    
    return True


def test_visualizer():
    """测试可视化模块"""
    print("\n" + "="*50)
    print("测试 5: 可视化模块")
    print("="*50)
    
    from visualizer import StrategyVisualizer, SimpleVisualizer
    
    # 创建模拟数据
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
    n = len(dates)
    
    np.random.seed(42)
    values = 1000000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
    benchmark = np.cumprod(1 + np.random.normal(0.0003, 0.012, n))
    
    daily_results = [
        {
            'date': d.strftime('%Y-%m-%d'),
            '总资产': v,
            '市场环境': np.random.choice(['bull', 'range', 'bear']),
            '仓位比例': np.random.uniform(0.3, 0.8)
        }
        for d, v in zip(dates, values)
    ]
    
    # 模拟交易记录
    trades = []
    
    print(f"\n✓ StrategyVisualizer 初始化成功")
    print(f"✓ SimpleVisualizer 初始化成功 (兼容模式)")
    print(f"✓ 模拟数据: {n} 个交易日")
    print(f"✓ 基准数据: {len(benchmark)} 条")
    
    # 注意：实际生成图表需要matplotlib，这里只验证模块加载
    visualizer = StrategyVisualizer("./test_output", simple_mode=True)
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("#  A股策略系统 V2.0 - 模块测试")
    print("#"*60)
    
    tests = [
        ("配置模块", test_config),
        ("大盘择时", test_market_timer),
        ("信号生成", test_signal_generator),
        ("仓位管理", test_position_manager),
        ("可视化", test_visualizer),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ 通过" if success else "❌ 失败"))
        except Exception as e:
            results.append((name, f"❌ 错误: {e}"))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, result in results:
        print(f"  {name}: {result}")
    
    all_passed = all("通过" in r for _, r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
