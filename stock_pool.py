import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from config import StockPoolConfig
from utils import setup_logger

logger = setup_logger(__name__)

# 将硬编码数据移出逻辑区，便于管理（实际项目中建议放入 JSON/CSV）
FALLBACK_STOCKS_DATA = {
    '半导体': [
        ('688981', '中芯国际'), ('002371', '北方华创'), ('688012', '中微公司'), 
        ('603501', '韦尔股份'), ('688008', '澜起科技'), ('002049', '紫光国微')
        # ... (此处省略其他，实际代码中保留完整列表)
    ],
    '新能源': [
        ('300750', '宁德时代'), ('002594', '比亚迪'), ('601012', '隆基绿能'),
        ('300274', '阳光电源'), ('002459', '晶澳科技')
        # ...
    ],
    '人工智能': [
        ('002230', '科大讯飞'), ('688111', '金山办公'), ('300496', '中科创达'),
        ('300033', '同花顺'), ('002415', '海康威视')
        # ...
    ],
    # ... 其他板块
}

@dataclass
class StockInfo:
    code: str
    name: str
    industry: str
    market: str = 'A'

class StockPoolManager:
    def __init__(self, config: StockPoolConfig = None):
        self.config = config
        self._base_pool: Dict[str, StockInfo] = {}
        # 初始化时直接构建基础池
        self._init_fallback_pool()

    def _init_fallback_pool(self):
        """初始化默认股票池"""
        for industry, stocks in FALLBACK_STOCKS_DATA.items():
            for code, name in stocks:
                self._base_pool[code] = StockInfo(
                    code=code,
                    name=name,
                    industry=industry,
                    market='SH' if code.startswith('6') else 'SZ'
                )

    def get_code_industry_map(self) -> Dict[str, str]:
        """
        获取 代码->行业 映射表
        供 Engine 和 PositionManager 进行板块风控使用
        """
        return {code: info.industry for code, info in self._base_pool.items()}

    def filter_daily_candidates(self, date: str) -> List[str]:
        """
        (V3简化版) 获取当日待观察的股票代码列表
        在实际回测中，Engine会遍历所有数据，这里可以作为预筛选
        """
        return list(self._base_pool.keys())
    
    def get_stock_info(self, code: str) -> Optional[StockInfo]:
        return self._base_pool.get(code)