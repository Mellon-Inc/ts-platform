import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import streamlit as st
from statsmodels.tsa.stattools import acf, pacf
import sys
import os

# utils.pyからの機能をインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import detect_frequency, create_time_features

def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    時間ベースの特徴量を追加
    
    Parameters:
    ----------
    df : pandas.DataFrame
        処理するデータフレーム
    date_col : str
        日付列の名前
        
    Returns:
    -------
    pandas.DataFrame
        時間特徴量が追加されたデータフレーム
    """
    df_new = df.copy()
    
    # 日付を確実に日付型に変換
    df_new[date_col] = pd.to_datetime(df_new[date_col])
    
    # 時間特徴量を作成
    time_features = create_time_features(df_new[date_col])
    
    # 特徴量をデータフレームに追加
    for feature_name, feature_values in time_features.items():
        df_new[feature_name] = feature_values
    
    return df_new

def add_lag_features(df: pd.DataFrame, 
                     target_col: str, 
                     lag_periods: Optional[List[int]] = None) -> pd.DataFrame:
    """
    ラグ特徴量を追加
    
    Parameters:
    ----------
    df : pandas.DataFrame
        処理するデータフレーム
    target_col : str
        ターゲット変数の列名
    lag_periods : list of int, optional
        追加するラグ期間のリスト
        
    Returns:
    -------
    pandas.DataFrame
        ラグ特徴量が追加されたデータフレーム
    """
    df_new = df.copy()
    
    # データの頻度に基づいて適切なラグを決定
    if lag_periods is None:
        freq = detect_frequency(df_new.index if isinstance(df_new.index, pd.DatetimeIndex) else df_new[df_new.columns[0]])
        
        if freq == 'hourly':
            lag_periods = [1, 2, 3, 6, 12, 24]
        elif freq == 'daily':
            lag_periods = [1, 2, 3, 7, 14, 28]
        elif freq == 'weekly':
            lag_periods = [1, 2, 4, 8, 12]
        elif freq == 'monthly':
            lag_periods = [1, 2, 3, 6, 12]
        elif freq == 'quarterly':
            lag_periods = [1, 2, 4, 8]
        else:  # yearly or unknown
            lag_periods = [1, 2, 3]
    
    # ラグ特徴量を追加
    for lag in lag_periods:
        df_new[f'lag_{lag}'] = df_new[target_col].shift(lag)
    
    return df_new

def add_rolling_features(df: pd.DataFrame, 
                         target_col: str, 
                         windows: Optional[List[int]] = None) -> pd.DataFrame:
    """
    移動平均などの特徴量を追加
    
    Parameters:
    ----------
    df : pandas.DataFrame
        処理するデータフレーム
    target_col : str
        ターゲット変数の列名
    windows : list of int, optional
        移動平均の窓サイズのリスト
        
    Returns:
    -------
    pandas.DataFrame
        移動特徴量が追加されたデータフレーム
    """
    df_new = df.copy()
    
    # データの頻度に基づいて適切な窓サイズを決定
    if windows is None:
        freq = detect_frequency(df_new.index if isinstance(df_new.index, pd.DatetimeIndex) else df_new[df_new.columns[0]])
        
        if freq == 'hourly':
            windows = [6, 12, 24, 48]
        elif freq == 'daily':
            windows = [7, 14, 30, 90]
        elif freq == 'weekly':
            windows = [4, 8, 13, 26]
        elif freq == 'monthly':
            windows = [3, 6, 12, 24]
        elif freq == 'quarterly':
            windows = [2, 4, 8]
        else:  # yearly or unknown
            windows = [2, 3, 5]
    
    # 移動平均特徴量を追加
    for window in windows:
        df_new[f'rolling_mean_{window}'] = df_new[target_col].rolling(window=window).mean()
        df_new[f'rolling_std_{window}'] = df_new[target_col].rolling(window=window).std()
        df_new[f'rolling_min_{window}'] = df_new[target_col].rolling(window=window).min()
        df_new[f'rolling_max_{window}'] = df_new[target_col].rolling(window=window).max()
    
    return df_new

def extract_seasonality_features(df: pd.DataFrame, 
                                target_col: str,
                                date_col: str) -> pd.DataFrame:
    """
    季節性特徴量を抽出して追加
    
    Parameters:
    ----------
    df : pandas.DataFrame
        処理するデータフレーム
    target_col : str
        ターゲット変数の列名
    date_col : str
        日付列の名前
        
    Returns:
    -------
    pandas.DataFrame
        季節性特徴量が追加されたデータフレーム
    """
    df_new = df.copy()
    
    # 日付を確実に日付型に変換
    df_new[date_col] = pd.to_datetime(df_new[date_col])
    
    # 頻度を検出
    freq = detect_frequency(df_new[date_col])
    
    # 頻度に基づいて季節性フラグを作成
    if freq in ['hourly', 'daily']:
        # 曜日ダミー変数
        for i in range(7):
            df_new[f'day_{i}'] = (df_new[date_col].dt.dayofweek == i).astype(int)
        
        # 月ダミー変数
        for i in range(1, 13):
            df_new[f'month_{i}'] = (df_new[date_col].dt.month == i).astype(int)
            
        # 季節ダミー変数
        # 1:春(3-5月), 2:夏(6-8月), 3:秋(9-11月), 4:冬(12-2月)
        season_map = {1: 4, 2: 4, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 4}
        df_new['season'] = df_new[date_col].dt.month.map(season_map)
        
        for i in range(1, 5):
            df_new[f'season_{i}'] = (df_new['season'] == i).astype(int)
        
    elif freq in ['weekly', 'monthly']:
        # 月ダミー変数
        for i in range(1, 13):
            df_new[f'month_{i}'] = (df_new[date_col].dt.month == i).astype(int)
        
        # 四半期ダミー変数
        for i in range(1, 5):
            df_new[f'quarter_{i}'] = (df_new[date_col].dt.quarter == i).astype(int)
            
    elif freq in ['quarterly', 'yearly']:
        # 四半期ダミー変数
        for i in range(1, 5):
            df_new[f'quarter_{i}'] = (df_new[date_col].dt.quarter == i).astype(int)
    
    return df_new

def detect_optimal_lags(df: pd.DataFrame, target_col: str, max_lag: int = 30) -> Tuple[int, int]:
    """
    ACFとPACFに基づいて最適なラグを検出
    
    Parameters:
    ----------
    df : pandas.DataFrame
        処理するデータフレーム
    target_col : str
        ターゲット変数の列名
    max_lag : int
        考慮する最大ラグ数
        
    Returns:
    -------
    tuple
        (AR次数p, MA次数q)の推定値
    """
    # 欠損値を削除
    series = df[target_col].dropna()
    
    # ACF, PACFを計算
    acf_values = acf(series, nlags=max_lag)
    pacf_values = pacf(series, nlags=max_lag)
    
    # 95%信頼区間
    confidence_interval = 1.96 / np.sqrt(len(series))
    
    # PACFから有意なラグを検出（AR次数の推定）
    significant_p_lags = [i for i, v in enumerate(pacf_values) if abs(v) > confidence_interval and i > 0]
    if significant_p_lags:
        p = max(significant_p_lags)
        p = min(p, 5)  # 実用的な上限を設定
    else:
        p = 1
    
    # ACFから有意なラグを検出（MA次数の推定）
    significant_q_lags = [i for i, v in enumerate(acf_values) if abs(v) > confidence_interval and i > 0]
    if significant_q_lags:
        q = max(significant_q_lags)
        q = min(q, 5)  # 実用的な上限を設定
    else:
        q = 1
    
    return p, q

def get_feature_importance(df: pd.DataFrame, 
                          target_col: str, 
                          features: List[str],
                          importance_method: str = 'random_forest') -> pd.DataFrame:
    """
    特徴量の重要度を計算
    
    Parameters:
    ----------
    df : pandas.DataFrame
        処理するデータフレーム
    target_col : str
        ターゲット変数の列名
    features : list of str
        分析する特徴量のリスト
    importance_method : str
        重要度計算方法
        
    Returns:
    -------
    pandas.DataFrame
        特徴量重要度のデータフレーム
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # 特徴量と目的変数を準備
    X = df[features].dropna()
    y = df.loc[X.index, target_col]
    
    # ランダムフォレストでモデル構築
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 特徴量重要度のデータフレームを作成
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    })
    
    return importance_df.sort_values('importance', ascending=False) 