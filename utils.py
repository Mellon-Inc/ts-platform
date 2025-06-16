import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Union, Tuple, Any, Optional
from modules.data_loader import load_data

# セッション状態の初期化
def initialize_session_state():
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'seasonal_strength' not in st.session_state:
        st.session_state.seasonal_strength = 0

# データ型に基づいて列を推薦する関数
def recommend_columns(df):
    date_cols = []
    numeric_cols = []
    
    for col in df.columns:
        # 日付列の推定
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
        except:
            pass
        
        # 数値列の確認
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    return date_cols, numeric_cols

# キャッシュ機能の強化
@st.cache_data(ttl=3600)
def load_cached_data(file):
    """データ読み込みをキャッシュする"""
    return load_data(file)

def show_success(message: str):
    """成功メッセージを表示"""
    st.success(message)
    
def show_error(message: str):
    """エラーメッセージを表示"""
    st.error(message)
    
def show_info(message: str):
    """情報メッセージを表示"""
    st.info(message)
    
def show_warning(message: str):
    """警告メッセージを表示"""
    st.warning(message)

def create_downloadable_csv(df: pd.DataFrame, filename: str = "data.csv") -> None:
    """データフレームをダウンロード可能なCSVに変換"""
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"{filename}をダウンロード",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def create_time_features(date_series: pd.Series) -> Dict[str, pd.Series]:
    """日付から様々な時間特徴量を作成"""
    features = {}
    features['year'] = date_series.dt.year
    features['quarter'] = date_series.dt.quarter
    features['month'] = date_series.dt.month
    features['day'] = date_series.dt.day
    features['dayofweek'] = date_series.dt.dayofweek
    features['is_weekend'] = date_series.dt.dayofweek.isin([5, 6]).astype(int)
    features['week'] = date_series.dt.isocalendar().week
    features['dayofyear'] = date_series.dt.dayofyear
    
    return features

def detect_frequency(date_series: pd.Series) -> str:
    """時系列データの頻度を自動検出"""
    # 日付をソート
    date_series = pd.Series(date_series).sort_values()
    
    # 差分の計算
    diff = date_series.diff().dropna()
    
    if len(diff) == 0:
        return 'unknown'
    
    # 最頻値を取得
    most_common_diff = diff.value_counts().index[0]
    
    # 頻度を判定
    if most_common_diff <= pd.Timedelta(hours=1):
        return 'hourly'
    elif most_common_diff <= pd.Timedelta(days=1):
        return 'daily'
    elif most_common_diff <= pd.Timedelta(days=7):
        return 'weekly'
    elif most_common_diff <= pd.Timedelta(days=31):
        return 'monthly'
    elif most_common_diff <= pd.Timedelta(days=92):
        return 'quarterly'
    else:
        return 'yearly'

def plot_time_series(df: pd.DataFrame, x: str, y: str, title: str = "時系列プロット") -> None:
    """時系列データのプロット"""
    fig = px.line(df, x=x, y=y, title=title)
    st.plotly_chart(fig, use_container_width=True)

def plot_multiple_series(df: pd.DataFrame, x: str, y_list: List[str], title: str = "複数時系列プロット") -> None:
    """複数の時系列データのプロット"""
    fig = px.line(df, x=x, y=y_list, title=title)
    st.plotly_chart(fig, use_container_width=True)

def format_large_number(num: float) -> str:
    """大きな数値を読みやすいフォーマットに変換"""
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}" 