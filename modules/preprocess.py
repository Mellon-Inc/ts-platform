import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from typing import Optional, List, Dict, Union, Tuple

def clean_data(df: pd.DataFrame, 
               date_col: str, 
               target_col: str, 
               handle_missing: bool = True, 
               handle_outliers: bool = False) -> pd.DataFrame:
    """
    データクリーニングを実行
    
    Parameters:
    ----------
    df : pandas.DataFrame
        クリーニングするデータフレーム
    date_col : str
        日付列の名前
    target_col : str
        ターゲット変数の列名
    handle_missing : bool
        欠損値を処理するかどうか
    handle_outliers : bool
        外れ値を処理するかどうか
        
    Returns:
    -------
    pandas.DataFrame
        クリーニングされたデータフレーム
    """
    # コピーを作成
    df_clean = df.copy()
    
    # 日付列を変換
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    
    # 日付でソート
    df_clean = df_clean.sort_values(by=date_col)
    
    # 欠損値の処理
    if handle_missing:
        # 数値列の欠損値を線形補間
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].interpolate(method='linear')
        
        # カテゴリ列の欠損値を最頻値で埋める
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df_clean[col].isna().any() and col != date_col:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # 外れ値の処理
    if handle_outliers:
        for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
            if col != date_col:
                # IQR法で外れ値を検出
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 外れ値を境界値にクリップ
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    return df_clean

def normalize_data(df: pd.DataFrame, 
                   target_col: str, 
                   exclude_cols: Optional[List[str]] = None, 
                   method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
    """
    数値データを正規化
    
    Parameters:
    ----------
    df : pandas.DataFrame
        正規化するデータフレーム
    target_col : str
        ターゲット変数の列名
    exclude_cols : list
        正規化から除外する列のリスト
    method : str
        正規化方法 ('minmax' または 'robust')
        
    Returns:
    -------
    tuple
        (正規化されたデータフレーム, スケーラー辞書)
    """
    # コピーを作成
    df_norm = df.copy()
    
    if exclude_cols is None:
        exclude_cols = []
    
    # 日付列などを除外
    exclude_cols = list(set(exclude_cols + [col for col in df.columns if df[col].dtype == 'datetime64[ns]']))
    
    # 数値列を抽出
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # スケーラーを保存
    scalers = {}
    
    # 各数値列を正規化
    for col in numeric_cols:
        if method == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            scaler = RobustScaler()
        
        # 列を2D配列に変換してスケーラーをフィット
        values = df[col].values.reshape(-1, 1)
        scaler.fit(values)
        
        # 変換
        df_norm[col] = scaler.transform(values)
        
        # スケーラーを保存
        scalers[col] = scaler
    
    return df_norm, scalers

def make_stationary(df: pd.DataFrame, 
                    target_col: str, 
                    method: str = 'diff', 
                    diff_order: int = 1) -> Tuple[pd.DataFrame, Dict]:
    """
    時系列データを定常化
    
    Parameters:
    ----------
    df : pandas.DataFrame
        定常化するデータフレーム
    target_col : str
        ターゲット変数の列名
    method : str
        定常化方法 ('diff', 'log', 'pct_change')
    diff_order : int
        差分の次数 (methodが'diff'の場合)
        
    Returns:
    -------
    tuple
        (定常化されたデータフレーム, 変換情報)
    """
    # コピーを作成
    df_stationary = df.copy()
    transform_info = {'method': method, 'original_data': df[target_col].copy()}
    
    if method == 'diff':
        transform_info['diff_order'] = diff_order
        for i in range(diff_order):
            df_stationary[target_col] = df_stationary[target_col].diff()
        # 欠損値を削除または0で埋める
        df_stationary = df_stationary.iloc[diff_order:]
        
    elif method == 'log':
        # 負の値がある場合は最小値を調整
        min_val = df[target_col].min()
        if min_val <= 0:
            adjustment = abs(min_val) + 1
            df_stationary[target_col] = np.log(df[target_col] + adjustment)
            transform_info['adjustment'] = adjustment
        else:
            df_stationary[target_col] = np.log(df[target_col])
            
    elif method == 'pct_change':
        df_stationary[target_col] = df_stationary[target_col].pct_change() * 100
        df_stationary = df_stationary.iloc[1:]  # 最初の行は削除（NaN)
    
    return df_stationary, transform_info

def inverse_transform(transformed_data: Union[pd.Series, np.ndarray], 
                      transform_info: Dict) -> Union[pd.Series, np.ndarray]:
    """
    変換されたデータを元に戻す
    
    Parameters:
    ----------
    transformed_data : pandas.Series or numpy.ndarray
        変換されたデータ
    transform_info : dict
        変換情報を含む辞書
        
    Returns:
    -------
    pandas.Series or numpy.ndarray
        元に戻されたデータ
    """
    method = transform_info['method']
    
    if method == 'diff':
        # 差分の場合、元データを使って累積和を計算
        # 注: これは単純化されたアプローチです。実際の逆変換はより複雑かもしれません
        original = transform_info['original_data']
        last_original = original.iloc[-1]
        
        if isinstance(transformed_data, pd.Series):
            return last_original + transformed_data.cumsum()
        else:
            return last_original + np.cumsum(transformed_data)
        
    elif method == 'log':
        if 'adjustment' in transform_info:
            adj = transform_info['adjustment']
            return np.exp(transformed_data) - adj
        else:
            return np.exp(transformed_data)
        
    elif method == 'pct_change':
        # パーセント変化の逆変換
        original = transform_info['original_data']
        start_value = original.iloc[-1]
        
        result = [start_value]
        for change in transformed_data:
            next_value = result[-1] * (1 + change/100)
            result.append(next_value)
        
        if isinstance(transformed_data, pd.Series):
            return pd.Series(result[1:])
        else:
            return np.array(result[1:]) 