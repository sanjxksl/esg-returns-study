"""
load_panel is the single entry point for all analysis scripts.

It loads the CSV, assigns FF48 industries, builds derived financial variables,
and winsorizes everything so every analysis starts from a clean, identical panel.
"""

import os
import pandas as pd
import numpy as np
from .ff48 import assign_ff48

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_HERE, '..', 'data', 'esg_financial_panel_2013_2023.csv')


def _winsorize(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    return series.clip(series.quantile(lower), series.quantile(upper))


def load_panel(csv_path: str = _DEFAULT_CSV) -> pd.DataFrame:
    """
    Load and fully prepare the ESG financial performance panel.

    Returns a DataFrame with roughly 20,758 firm-year observations from 2013 to 2023.
    In addition to the raw CSV columns, the following are added:

        ff48_id, industry       Fama-French 48 industry classification
        leverage                (dltt + dlc) / at, missing debt fields treated as zero
        rd_intensity            xrd / at, missing R&D treated as zero
        capital_intensity       ppent / at
        equity_ratio            ceq / at
        profit_margin           ni / sale, NaN where sale <= 0
        sales_growth            year-over-year change in sale, NaN where lagged sale <= 0
        earnings_news           future_roa minus roa
        fyear_str               fiscal year as string for fixed effects
        ff48_id_str             FF48 id as string for fixed effects

    All continuous variables are winsorized at the 1st and 99th percentiles.
    """
    panel = pd.read_csv(csv_path)
    panel['datadate'] = pd.to_datetime(panel['datadate'])
    panel['sic'] = pd.to_numeric(panel['sic'], errors='coerce')

    required = ['dltt', 'dlc', 'xrd', 'ppent']
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not found. The CSV must be the extended version "
            "produced by data_prepare.ipynb."
        )

    ff48_result = panel['sic'].apply(assign_ff48)
    panel['ff48_id'] = ff48_result.apply(lambda x: x[0]).astype(int)
    panel['industry'] = ff48_result.apply(lambda x: x[1])

    panel = panel.sort_values(['gvkey', 'fyear']).copy()

    panel['leverage'] = (panel['dltt'].fillna(0) + panel['dlc'].fillna(0)) / panel['at']
    panel['rd_intensity'] = panel['xrd'].fillna(0) / panel['at']
    panel['capital_intensity'] = panel['ppent'].fillna(0) / panel['at']
    panel['equity_ratio'] = panel['ceq'] / panel['at']
    panel['profit_margin'] = np.where(panel['sale'] > 0, panel['ni'] / panel['sale'], np.nan)

    lag_sale = panel.groupby('gvkey')['sale'].shift(1)
    panel['sales_growth'] = np.where(lag_sale > 0, (panel['sale'] - lag_sale) / lag_sale, np.nan)

    winsorize_cols = [
        'roa', 'size', 'leverage', 'rd_intensity', 'capital_intensity',
        'equity_ratio', 'profit_margin', 'sales_growth',
        'future_roa', 'future_annual_ret',
    ]
    for col in winsorize_cols:
        if col in panel.columns:
            panel[col] = _winsorize(panel[col])

    panel['earnings_news'] = panel['future_roa'] - panel['roa']
    panel['fyear_str'] = panel['fyear'].astype(str)
    panel['ff48_id_str'] = panel['ff48_id'].astype(str)

    return panel
