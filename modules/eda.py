import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from typing import Dict, List, Union, Tuple, Optional

def run_time_series_eda(df: pd.DataFrame, target_col: str, date_col: str) -> Dict:
    """
    時系列データの探索的データ分析を実行
    
    Parameters:
    ----------
    df : pandas.DataFrame
        分析するデータフレーム
    target_col : str
        対象変数の列名
    date_col : str
        日付列の名前
        
    Returns:
    -------
    dict
        EDA結果を含む辞書
    """
    # 結果を格納する辞書
    results = {}
    
    # 結果をプロット
    fig = plt.figure(figsize=(10, 8))
    
    # ACFプロット
    plt.subplot(211)
    plt.stem(lags, acf_values[1:], linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.axhspan(-conf_int, conf_int, alpha=0.2, color='grey')
    plt.title('自己相関関数 (ACF)')
    plt.xlabel('ラグ')
    plt.ylabel('相関')
    
    # PACFプロット
    plt.subplot(212)
    plt.stem(lags, pacf_values[1:], linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.axhspan(-conf_int, conf_int, alpha=0.2, color='grey')
    plt.title('偏自己相関関数 (PACF)')
    plt.xlabel('ラグ')
    plt.ylabel('偏相関')
    
    plt.tight_layout()
    
    return fig, acf_values, pacf_values, p, q

def detect_seasonality(df: pd.DataFrame, target_col: str, date_col: str) -> Tuple[bool, int]:
    """
    季節性を検出
    
    Parameters:
    ----------
    df : pandas.DataFrame
        分析するデータフレーム
    target_col : str
        対象変数の列名
    date_col : str
        日付列の名前
        
    Returns:
    -------
    tuple
        (季節性の有無, 周期)
    """
    # データを準備
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    
    # 対象変数のシリーズ
    series = df[target_col]
    
    # 自己相関関数を計算
    max_lag = min(365, len(series) // 2)  # 最大1年または系列長の半分
    acf_values = acf(series, nlags=max_lag)
    
    # 信頼区間
    conf_int = 1.96 / np.sqrt(len(series))
    
    # 有意な自己相関を持つラグを探す
    significant_lags = [i for i, v in enumerate(acf_values) if i > 0 and abs(v) > conf_int]
    
    if not significant_lags:
        return False, 0
    
    # 季節性の周期を推定
    # 最初の有意なピークを見つける
    candidates = []
    prev_val = 0
    
    for lag in significant_lags:
        curr_val = acf_values[lag]
        # ピークを検出（前の値より大きく、かつ信頼区間よりも大きい）
        if curr_val > prev_val and curr_val > conf_int:
            candidates.append(lag)
        prev_val = curr_val
    
    if not candidates:
        return False, 0
    
    # 最初の強いピークを季節周期として採用
    season_period = candidates[0]
    
    # 一般的な季節周期に近いものに調整
    common_periods = [7, 12, 24, 52, 365]  # 週、月、日内時間、週（年）、年
    
    # 最も近い一般的な周期を探す
    if season_period > 3:  # 少なくとも3以上の周期であること
        closest_period = min(common_periods, key=lambda x: abs(x - season_period))
        # 元の推定値の50%以内なら調整
        if abs(closest_period - season_period) / season_period < 0.5:
            season_period = closest_period
    
    # 季節性があると判断する閾値
    has_seasonality = acf_values[season_period] > 2 * conf_int
    
    return has_seasonality, season_period 