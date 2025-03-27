import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any, Union
import os

@st.cache_data(ttl=3600)
def load_data(file_obj: Any) -> pd.DataFrame:
    """
    ファイルからデータを読み込む
    
    Parameters:
    ----------
    file_obj : file object
        読み込むファイルオブジェクト
        
    Returns:
    -------
    pandas.DataFrame
        読み込まれたデータフレーム
    """
    try:
        # ファイル名を取得
        file_name = file_obj.name
        file_extension = os.path.splitext(file_name)[1].lower()
        
        # ファイル拡張子に基づいて適切な読み込み方法を選択
        if file_extension == '.csv':
            df = pd.read_csv(file_obj)
        elif file_extension in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(file_obj)
            except ImportError:
                st.error("Excelファイルを読み込むには'openpyxl'が必要です。'pip install openpyxl'でインストールしてください。")
                return None
        else:
            st.error(f"サポートされていないファイル形式です: {file_extension}")
            return None
            
        return df
    except Exception as e:
        st.error(f"データの読み込みエラー: {str(e)}")
        return None

def infer_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    データ型を推測して変換
    
    Parameters:
    ----------
    df : pandas.DataFrame
        変換するデータフレーム
        
    Returns:
    -------
    pandas.DataFrame
        データ型が変換されたデータフレーム
    """
    df_copy = df.copy()
    
    # 日付列の検出と変換
    for col in df_copy.columns:
        # 列名に'date'や'time'が含まれる場合
        if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'year']):
            try:
                df_copy[col] = pd.to_datetime(df_copy[col])
            except:
                pass
        
        # 数値への変換を試みる
        elif df_copy[col].dtype == 'object':
            try:
                df_copy[col] = pd.to_numeric(df_copy[col])
            except:
                pass
    
    return df_copy

def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    データの概要情報を取得
    
    Parameters:
    ----------
    df : pandas.DataFrame
        分析するデータフレーム
        
    Returns:
    -------
    dict
        データの概要情報を含む辞書
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isna().sum().sum(),
        'missing_percentage': (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB単位
        'column_types': df.dtypes.value_counts().to_dict()
    }
    
    return summary

def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    日付列を推測
    
    Parameters:
    ----------
    df : pandas.DataFrame
        分析するデータフレーム
        
    Returns:
    -------
    str or None
        推測された日付列の名前。見つからない場合はNone
    """
    # datetime型の列を探す
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if datetime_cols:
        # 最初のdatetime列を返す
        return datetime_cols[0]
    
    # 'date'や'time'を含む列名を探す
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'year']):
            try:
                # 日付に変換できるか試す
                pd.to_datetime(df[col])
                return col
            except:
                continue
    
    return None

def suggest_target_column(df: pd.DataFrame, date_col: Optional[str] = None) -> Optional[str]:
    """
    対象変数を推測
    
    Parameters:
    ----------
    df : pandas.DataFrame
        分析するデータフレーム
    date_col : str, optional
        日付列の名前
        
    Returns:
    -------
    str or None
        推測された対象変数の列名。見つからない場合はNone
    """
    # 数値列を抽出
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_cols:
        return None
    
    # 日付列を除外
    if date_col and date_col in numeric_cols:
        numeric_cols.remove(date_col)
    
    if not numeric_cols:
        return None
    
    # 名前に'target'や'value'を含む列を優先
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in ['target', 'value', 'sales', 'price', 'revenue']):
            return col
    
    # デフォルトでは最初の数値列を返す
    return numeric_cols[0] 