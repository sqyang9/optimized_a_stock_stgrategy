import pandas as pd
import os
from typing import Dict, List, Optional
from utils import setup_logger

logger = setup_logger(__name__)

class RealDataLoader:
    def __init__(self, data_dir='./data/daily'):
        self.data_dir = data_dir
        self.cache = {}
        # 预加载所有文件名，建立股票代码清单
        if os.path.exists(data_dir):
            self.all_codes = [f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv')]
        else:
            self.all_codes = []
            logger.error(f"数据目录不存在: {data_dir}")

    def load_stock_data(self, code: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """读取单只股票CSV并标准化"""
        if code in self.cache:
            df = self.cache[code]
        else:
            file_path = os.path.join(self.data_dir, f"{code}.csv")
            if not os.path.exists(file_path):
                return None
            try:
                # 假设你的CSV包含 date, open, high, low, close, volume
                # 如果包含 'money' 或 'amount'，请根据实际列名调整
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                
                # 统一列名（根据你的 enhanced_backtest_analyzer.py 里的习惯）
                df.rename(columns={
                    '日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', 
                    '收盘': 'close', '成交量': 'volume', '成交额': 'amount',
                    '涨跌幅': 'pct_change'
                }, inplace=True)
                
                df['date'] = pd.to_datetime(df['date'])
                df.sort_values('date', inplace=True)
                df.reset_index(drop=True, inplace=True)
                self.cache[code] = df
            except Exception as e:
                logger.error(f"读取 {code} 失败: {e}")
                return None

        # 时间切片
        mask = pd.Series(True, index=df.index)
        if start_date:
            mask &= (df['date'] >= pd.to_datetime(start_date))
        if end_date:
            mask &= (df['date'] <= pd.to_datetime(end_date))
        
        sliced_df = df.loc[mask].copy()
        return sliced_df if not sliced_df.empty else None

    def get_all_codes(self):
        return self.all_codes