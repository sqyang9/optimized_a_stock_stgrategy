"""
A股科技成长股趋势策略 V2.0 - 主程序
新增: 大盘择时、板块共振、优化止盈止损
"""

import argparse
import os
import sys
from datetime import datetime

from config import StrategyConfig
from engine import BacktestEngineV3
from visualizer import StrategyVisualizer
from utils import setup_logger

logger = setup_logger("Main")


def main():
    parser = argparse.ArgumentParser(description="A股策略系统 V2.0")
    parser.add_argument('--start', default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--data', default='./data/daily', help='CSV数据目录')
    parser.add_argument('--capital', type=float, default=1_000_000, help='初始资金')
    parser.add_argument('--simple', action='store_true', help='简单报告模式')
    parser.add_argument('--no-timing', action='store_true', help='禁用大盘择时')
    parser.add_argument('--no-resonance', action='store_true', help='禁用板块共振')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  A股科技成长股趋势策略 V2.0")
    print("  优化特性: 大盘择时 | 板块共振 | 时间止损 | 追踪止盈")
    print("="*60 + "\n")
    
    # 1. 加载配置
    config = StrategyConfig()
    config.backtest.initial_capital = args.capital
    config.backtest.start_date = args.start
    config.backtest.end_date = args.end
    
    # 根据参数调整配置
    if args.no_timing:
        config.market_timing.enabled = False
        logger.info("大盘择时已禁用")
    
    if args.no_resonance:
        config.sector_resonance.enabled = False
        logger.info("板块共振已禁用")
    
    print(f"回测参数:")
    print(f"  - 起止日期: {args.start} ~ {args.end}")
    print(f"  - 初始资金: {args.capital:,.0f} 元")
    print(f"  - 数据目录: {args.data}")
    print(f"  - 大盘择时: {'启用' if config.market_timing.enabled else '禁用'}")
    print(f"  - 板块共振: {'启用' if config.sector_resonance.enabled else '禁用'}")
    print(f"  - 时间止损: {'启用' if config.exit.time_stop_enabled else '禁用'}")
    print()
    
    # 2. 检查数据目录
    if not os.path.exists(args.data):
        logger.error(f"数据目录不存在: {args.data}")
        print(f"\n错误: 数据目录 '{args.data}' 不存在")
        print("请确保数据文件已放置在正确位置")
        sys.exit(1)
    
    # 3. 初始化引擎
    logger.info("初始化回测引擎...")
    try:
        engine = BacktestEngineV3(config, data_dir=args.data)
    except Exception as e:
        logger.error(f"引擎初始化失败: {e}")
        print(f"\n错误: {e}")
        sys.exit(1)
    
    # 4. 准备数据
    logger.info("准备数据中...")
    try:
        engine.prepare_data(args.start, args.end)
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        print(f"\n错误: {e}")
        sys.exit(1)
    
    if not engine.universe_data:
        print("\n错误: 没有找到有效的股票数据")
        sys.exit(1)
    
    # 5. 运行回测
    logger.info("开始回测...")
    print("\n正在运行回测，请稍候...\n")
    
    try:
        pos_mgr, daily_results = engine.run()
    except Exception as e:
        logger.error(f"回测运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if not daily_results:
        print("\n错误: 回测未产生结果")
        sys.exit(1)
    
    # 6. 输出结果
    print("\n" + "="*60)
    print("  回测结果")
    print("="*60)
    
    # 获取绩效摘要
    perf = engine.get_performance_summary()
    
    print(f"\n【收益指标】")
    print(f"  总收益率:     {perf.get('total_return', 0)*100:+.2f}%")
    print(f"  年化收益:     {perf.get('annual_return', 0)*100:+.2f}%")
    print(f"  基准收益:     {perf.get('benchmark_return', 0)*100:+.2f}%")
    print(f"  超额收益:     {perf.get('excess_return', 0)*100:+.2f}%")
    
    print(f"\n【风险指标】")
    print(f"  最大回撤:     {perf.get('max_drawdown', 0)*100:.2f}%")
    print(f"  最长水下期:   {perf.get('max_underwater_days', 0)} 天")
    print(f"  年化波动率:   {perf.get('volatility', 0)*100:.2f}%")
    print(f"  夏普比率:     {perf.get('sharpe_ratio', 0):.2f}")
    print(f"  卡尔马比率:   {perf.get('calmar_ratio', 0):.2f}")
    
    print(f"\n【交易统计】")
    print(f"  交易天数:     {perf.get('trading_days', 0)}")
    print(f"  总交易次数:   {perf.get('total_trades', 0)}")
    print(f"  胜率:         {perf.get('win_rate', 0)*100:.1f}%")
    print(f"  最终资产:     {perf.get('final_value', 0):,.2f} 元")
    
    print("\n" + "="*60 + "\n")
    
    # 7. 创建输出目录
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 8. 导出交易记录
    if pos_mgr.trades:
        import pandas as pd
        trades_df = pd.DataFrame([t.__dict__ for t in pos_mgr.trades])
        
        # 处理枚举类型
        if 'exit_reason' in trades_df.columns:
            trades_df['exit_reason'] = trades_df['exit_reason'].apply(
                lambda x: x.value if hasattr(x, 'value') else str(x) if x else ''
            )
        if 'entry_type' in trades_df.columns:
            trades_df['entry_type'] = trades_df['entry_type'].apply(
                lambda x: x.value if hasattr(x, 'value') else str(x) if x else ''
            )
        
        trades_path = os.path.join(output_dir, "trades.csv")
        trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
        print(f"交易记录已保存: {trades_path}")
    
    # 9. 导出每日结果
    import pandas as pd
    daily_df = pd.DataFrame(daily_results)
    daily_path = os.path.join(output_dir, "daily_results.csv")
    daily_df.to_csv(daily_path, index=False, encoding='utf-8-sig')
    print(f"每日结果已保存: {daily_path}")
    
    # 10. 生成可视化报告
    logger.info("生成可视化报告...")
    
    visualizer = StrategyVisualizer(output_dir, simple_mode=args.simple)
    
    try:
        report_path = visualizer.generate_report(
            daily_results=daily_results,
            trades=pos_mgr.trades,
            initial_capital=args.capital,
            benchmark_values=engine.get_benchmark_data(),
            performance_summary=perf
        )
        print(f"可视化报告已生成: {report_path}")
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("  回测完成!")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
