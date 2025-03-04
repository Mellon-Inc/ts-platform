import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Union, Tuple, Any, Optional

# 設定ファイルと共通ユーティリティをインポート
from config import APP_TITLE, APP_DESCRIPTION, AVAILABLE_MODELS, DEFAULT_PARAMS, DEFAULT_FORECAST_PERIODS
from utils import (load_cached_data, show_success, show_error, show_info, show_warning,
                  create_downloadable_csv, detect_frequency, plot_time_series, plot_multiple_series)

# 各モジュールをインポート
from modules.preprocess import clean_data, normalize_data, make_stationary, inverse_transform
from modules.features import (add_time_features, add_lag_features, add_rolling_features, 
                            detect_optimal_lags, get_feature_importance)

# ページ設定とタイトル
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide"
)

# セッション状態の初期化
def init_session_state():
    """アプリケーションのセッション状態を初期化"""
    if 'page' not in st.session_state:
        st.session_state.page = 'upload'
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'clean_data' not in st.session_state:
        st.session_state.clean_data = None
    if 'featured_data' not in st.session_state:
        st.session_state.featured_data = None
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    if 'date_col' not in st.session_state:
        st.session_state.date_col = None
    if 'data_frequency' not in st.session_state:
        st.session_state.data_frequency = None
    if 'eda_results' not in st.session_state:
        st.session_state.eda_results = {}
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {}
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'forecast' not in st.session_state:
        st.session_state.forecast = None
    if 'transform_info' not in st.session_state:
        st.session_state.transform_info = None
    if 'scalers' not in st.session_state:
        st.session_state.scalers = None

# アプリの初期化
init_session_state()

# アプリのタイトル
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# サイドバーナビゲーション
def navigation():
    """アプリケーションのナビゲーションメニュー"""
    st.sidebar.title("ナビゲーション")
    
    pages = {
        "データのアップロード": "upload",
        "データの探索的分析": "eda",
        "モデルのトレーニング": "training",
        "予測の生成": "forecast"
    }
    
    # 現在のページを特定
    current_page_name = [name for name, code in pages.items() if code == st.session_state.page][0]
    
    # ページ選択ラジオボタン
    selected_page = st.sidebar.radio("ページを選択", list(pages.keys()), index=list(pages.keys()).index(current_page_name))
    
    # 選択されたページに応じて状態を更新
    if pages[selected_page] != st.session_state.page:
        if pages[selected_page] == 'upload':
            st.session_state.page = 'upload'
            st.rerun()
        elif pages[selected_page] == 'eda':
            if st.session_state.data is not None:
                st.session_state.page = 'eda'
                st.rerun()
            else:
                st.sidebar.error("先にデータをアップロードしてください")
        elif pages[selected_page] == 'training':
            if st.session_state.data is not None:
                st.session_state.page = 'training'
                st.rerun()
            else:
                st.sidebar.error("先にデータをアップロードしてください")
        elif pages[selected_page] == 'forecast':
            if st.session_state.trained_model is not None:
                st.session_state.page = 'forecast'
                st.rerun()
            else:
                st.sidebar.error("先にモデルをトレーニングしてください")

# データアップロードページ
def show_upload_page():
    """データのアップロードと前処理を行うページ"""
    st.header("データのアップロード")
    
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # データの読み込み
            df = load_cached_data(uploaded_file)
            
            if df is not None:
                st.success("ファイルが正常に読み込まれました")
                
                # データプレビュー
                st.subheader("データプレビュー")
                st.dataframe(df.head())
                
                # 列の概要
                st.subheader("列の情報")
                col_info = pd.DataFrame({
                    '列名': df.columns,
                    'データ型': df.dtypes,
                    '非欠損値数': df.count().values,
                    '欠損値数': df.isna().sum().values,
                    '欠損率 (%)': df.isna().sum().values / len(df) * 100
                })
                st.dataframe(col_info)
                
                # 変数の選択
                st.subheader("変数の選択")
                
                # 日付カラムの候補を絞り込む（'date'や'time'を含む列名を優先）
                date_col_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'year', 'month'])]
                if not date_col_candidates:
                    date_col_candidates = df.columns.tolist()
                
                # 最も可能性の高い日付カラムを初期選択
                default_date_index = 0
                for i, col in enumerate(date_col_candidates):
                    if 'date' in col.lower():
                        default_date_index = i
                        break
                
                date_col = st.selectbox("日付変数を選択", date_col_candidates, index=default_date_index)
                
                # 数値カラムのみを抽出
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                
                # 目標変数（ターゲット）の選択
                target_col = st.selectbox("予測対象の変数を選択", numeric_cols)
                
                # 前処理オプション
                st.subheader("前処理オプション")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    handle_missing = st.checkbox("欠損値を処理する", value=True)
                    handle_outliers = st.checkbox("外れ値を処理する", value=False)
                
                with col2:
                    normalize = st.checkbox("データを正規化する", value=False)
                    make_data_stationary = st.checkbox("データを定常化する", value=False)
                    
                if make_data_stationary:
                    stationary_method = st.selectbox(
                        "定常化の方法", 
                        ["差分 (Difference)", "対数変換 (Log)", "変化率 (Percent Change)"],
                        index=0
                    )
                    method_map = {
                        "差分 (Difference)": "diff",
                        "対数変換 (Log)": "log",
                        "変化率 (Percent Change)": "pct_change"
                    }
                
                # 前処理を適用するボタン
                if st.button("前処理を適用"):
                    with st.spinner("データを前処理中..."):
                        # 日付型に変換
                        try:
                            df[date_col] = pd.to_datetime(df[date_col])
                        except Exception as e:
                            st.error(f"日付列の変換に失敗しました: {str(e)}")
                            st.stop()
                        
                        # データクリーニング
                        clean_df = clean_data(df, date_col, target_col, handle_missing, handle_outliers)
                        
                        # データの頻度を検出
                        data_frequency = detect_frequency(clean_df[date_col])
                        
                        # 正規化
                        transform_info = None
                        scalers = None
                        
                        if normalize:
                            normalized_df, scalers = normalize_data(
                                clean_df, 
                                target_col, 
                                exclude_cols=[date_col], 
                                method='minmax'
                            )
                            clean_df = normalized_df
                        
                        # 定常化
                        if make_data_stationary:
                            method = method_map[stationary_method]
                            stationary_df, transform_info = make_stationary(
                                clean_df, 
                                target_col, 
                                method=method, 
                                diff_order=1 if method == 'diff' else None
                            )
                            clean_df = stationary_df
                        
                        # セッション状態を更新
                        st.session_state.data = df.copy()
                        st.session_state.clean_data = clean_df.copy()
                        st.session_state.target_col = target_col
                        st.session_state.date_col = date_col
                        st.session_state.data_frequency = data_frequency
                        st.session_state.transform_info = transform_info
                        st.session_state.scalers = scalers
                    
                    # 成功メッセージとデータプレビュー
                    show_success("前処理が完了しました！")
                    st.subheader("前処理後のデータプレビュー")
                    st.dataframe(clean_df.head())
                    
                    # 処理されたデータをダウンロード可能に
                    create_downloadable_csv(clean_df, "preprocessed_data.csv")
                    
                    # 基本的な時系列プロット
                    st.subheader("時系列プロット")
                    fig = px.line(
                        clean_df, 
                        x=date_col, 
                        y=target_col, 
                        title=f"{target_col} vs 時間"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # EDAページに進むボタン
                    if st.button("データ分析に進む"):
                        # データが存在することを確認してから遷移
                        if st.session_state.data is not None and st.session_state.clean_data is not None:
                            st.session_state.page = 'eda'
                            # 明示的にセッション状態を保存して再読み込み
                            st.experimental_rerun()
                        else:
                            st.error("データが正しく処理されていないため、EDAページに進めません。")
        
        except Exception as e:
            show_error(f"データの読み込み中にエラーが発生しました: {str(e)}")

# EDA (探索的データ分析) ページ
def show_eda_page():
    """探索的データ分析を行うページ"""
    st.header("データの探索的分析")
    
    # データと列の確認
    if st.session_state.clean_data is None:
        show_error("処理済みデータが見つかりません。先にデータを前処理してください。")
        return
    
    df = st.session_state.clean_data
    target_col = st.session_state.target_col
    date_col = st.session_state.date_col
    
    # EDAのタブ
    tabs = st.tabs(["時系列特性", "分布と統計量", "相関分析", "特徴量生成"])
    
    # タブ1: 時系列特性
    with tabs[0]:
        st.subheader("時系列特性の分析")
        
        # トレンド、季節性、定常性の分析
        col1, col2 = st.columns(2)
        
        with col1:
            # 基本的な時系列プロット
            st.subheader("時間に対する目標変数")
            plot_time_series(df, date_col, target_col)
            
            # トレンド検出
            from scipy import stats
            
            # 線形回帰でトレンド検出
            x = np.array(range(len(df)))
            y = df[target_col].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # トレンドラインを追加
            trend_line = intercept + slope * x
            
            fig = px.scatter(
                df, 
                x=date_col, 
                y=target_col, 
                opacity=0.7,
                title=f"{target_col}のトレンド分析"
            )
            
            fig.add_scatter(
                x=df[date_col], 
                y=trend_line, 
                mode='lines', 
                name='トレンドライン',
                line=dict(color='red', width=2)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # トレンドの有意性
            has_trend = p_value < 0.05 and abs(r_value) > 0.3
            
            if has_trend:
                trend_dir = "上昇" if slope > 0 else "下降"
                st.info(f"有意な{trend_dir}トレンドが検出されました (p-value: {p_value:.4f}, r²: {r_value**2:.4f})")
            else:
                st.info(f"有意なトレンドは検出されませんでした (p-value: {p_value:.4f}, r²: {r_value**2:.4f})")
        
        with col2:
            # 季節性分析
            st.subheader("季節性分析")
            
            # データの頻度に基づいて適切な分解方法を選択
            import statsmodels.api as sm
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            try:
                # 欠損値が含まれるとエラーになるため、補間
                temp_df = df.copy()
                temp_df[target_col] = temp_df[target_col].interpolate()
                
                # インデックスを日付に設定
                temp_df = temp_df.set_index(date_col)
                
                # 季節分解
                freq_map = {
                    'hourly': 24,
                    'daily': 7,
                    'weekly': 52,
                    'monthly': 12,
                    'quarterly': 4,
                    'yearly': 1
                }
                
                freq = st.session_state.data_frequency
                period = freq_map.get(freq, 12)  # デフォルトは12
                
                decomposition = seasonal_decompose(
                    temp_df[target_col], 
                    model='additive', 
                    period=period,
                    extrapolate_trend='freq'
                )
                
                # 分解結果のプロット
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid
                
                # 季節性成分の強さを評価
                seasonal_strength = abs(seasonal).mean() / (abs(trend).mean() + abs(seasonal).mean() + abs(residual).mean())
                has_seasonality = seasonal_strength > 0.1
                
                # 各成分をデータフレームに格納
                decomp_df = pd.DataFrame({
                    'trend': trend,
                    'seasonal': seasonal,
                    'residual': residual
                })
                
                # 各成分をプロット
                for component in ['trend', 'seasonal', 'residual']:
                    fig = px.line(
                        decomp_df, 
                        y=component,
                        title=f"{component.capitalize()} 成分"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if has_seasonality:
                    st.info(f"季節性パターンが検出されました (季節性強度: {seasonal_strength:.4f})")
                else:
                    st.info(f"顕著な季節性パターンは検出されませんでした (季節性強度: {seasonal_strength:.4f})")
                
                # 定常性検定
                from statsmodels.tsa.stattools import adfuller
                
                # ADF検定を実行
                result = adfuller(df[target_col].dropna())
                
                adf_stat = result[0]
                p_value = result[1]
                critical_values = result[4]
                
                is_stationary = p_value < 0.05
                
                st.subheader("定常性検定 (ADF)")
                st.write(f"ADF統計量: {adf_stat:.4f}")
                st.write(f"p値: {p_value:.4f}")
                st.write("臨界値:")
                for key, value in critical_values.items():
                    st.write(f"   {key}: {value:.4f}")
                
                if is_stationary:
                    st.success("時系列は定常的です (p < 0.05)")
                else:
                    st.warning("時系列は非定常です (p >= 0.05)")
                
                # EDA結果をセッションに保存
                st.session_state.eda_results = {
                    'has_trend': has_trend,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'none',
                    'trend_strength': abs(r_value),
                    'has_seasonality': has_seasonality,
                    'seasonality_strength': seasonal_strength,
                    'is_stationary': is_stationary,
                    'optimal_period': period
                }
                
            except Exception as e:
                st.error(f"季節性分析中にエラーが発生しました: {str(e)}")
    
    # タブ2: 分布と統計量
    with tabs[1]:
        st.subheader("データの分布と統計量")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ヒストグラム
            fig = px.histogram(
                df, 
                x=target_col, 
                nbins=30,
                title=f"{target_col}のヒストグラム"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 箱ひげ図
            fig = px.box(
                df, 
                y=target_col,
                title=f"{target_col}の箱ひげ図"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # QQプロット
            from scipy import stats
            
            # 正規分布に対するQQプロット
            quantiles = stats.probplot(df[target_col].dropna(), dist='norm')
            
            fig = go.Figure()
            fig.add_scatter(
                x=quantiles[0][0], 
                y=quantiles[0][1],
                mode='markers',
                name='データ点'
            )
            
            # 理論線を追加
            fig.add_scatter(
                x=quantiles[0][0],
                y=quantiles[0][0] * quantiles[1][0] + quantiles[1][1],
                mode='lines',
                name='理論線',
                line=dict(color='red')
            )
            
            fig.update_layout(
                title="正規Q-Qプロット",
                xaxis_title="理論上の分位数",
                yaxis_title="実際の分位数"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 基本統計量
            st.subheader("基本統計量")
            
            stats_df = pd.DataFrame({
                '統計量': [
                    '件数', '欠損値', '平均', '標準偏差', '最小値', '25%分位', 
                    '中央値', '75%分位', '最大値', '歪度', '尖度'
                ],
                '値': [
                    df[target_col].count(),
                    df[target_col].isna().sum(),
                    df[target_col].mean(),
                    df[target_col].std(),
                    df[target_col].min(),
                    df[target_col].quantile(0.25),
                    df[target_col].median(),
                    df[target_col].quantile(0.75),
                    df[target_col].max(),
                    df[target_col].skew(),
                    df[target_col].kurtosis()
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True)
    
    # タブ3: 相関分析
    with tabs[2]:
        st.subheader("自己相関分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ACF (自己相関関数)
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            try:
                plot_acf(df[target_col].dropna(), ax=ax, lags=30)
                ax.set_title('自己相関関数 (ACF)')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"ACFの計算中にエラーが発生しました: {str(e)}")
        
        with col2:
            # PACF (偏自己相関関数)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            try:
                plot_pacf(df[target_col].dropna(), ax=ax, lags=30)
                ax.set_title('偏自己相関関数 (PACF)')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"PACFの計算中にエラーが発生しました: {str(e)}")
        
        # 最適ラグの検出
        st.subheader("ARIMAモデルの最適なパラメータ推定")
        
        try:
            # ACFとPACFに基づいて最適なパラメータを推定
            p, q = detect_optimal_lags(df, target_col)
            
            st.info(f"推定されたARIMAパラメータ: p={p}, q={q}")
            st.session_state.eda_results['optimal_p'] = p
            st.session_state.eda_results['optimal_q'] = q
            
            # これをARIMAモデルのデフォルトパラメータとして設定
            DEFAULT_PARAMS['ARIMA'] = {'p': p, 'd': 1 if not st.session_state.eda_results.get('is_stationary', False) else 0, 'q': q}
            DEFAULT_PARAMS['SARIMA']['p'] = p
            DEFAULT_PARAMS['SARIMA']['q'] = q
            
        except Exception as e:
            st.error(f"最適パラメータの推定中にエラーが発生しました: {str(e)}")
    
    # タブ4: 特徴量生成
    with tabs[3]:
        st.subheader("特徴量エンジニアリング")
        
        # 特徴量生成オプション
        add_time_features_option = st.checkbox("時間ベースの特徴量を追加", value=True)
        add_lag_features_option = st.checkbox("ラグ特徴量を追加", value=True)
        add_rolling_features_option = st.checkbox("移動平均特徴量を追加", value=True)
        
        # カスタムラグ期間
        custom_lags = None
        if add_lag_features_option:
            use_custom_lags = st.checkbox("カスタムラグを指定", value=False)
            if use_custom_lags:
                custom_lags_input = st.text_input("カンマ区切りのラグ期間", "1,2,3,7")
                try:
                    custom_lags = [int(x.strip()) for x in custom_lags_input.split(",")]
                except ValueError:
                    st.error("有効な整数のリストを入力してください")
        
        # カスタム移動平均窓サイズ
        custom_windows = None
        if add_rolling_features_option:
            use_custom_windows = st.checkbox("カスタム移動平均窓サイズを指定", value=False)
            if use_custom_windows:
                custom_windows_input = st.text_input("カンマ区切りの窓サイズ", "3,7,14,30")
                try:
                    custom_windows = [int(x.strip()) for x in custom_windows_input.split(",")]
                except ValueError:
                    st.error("有効な整数のリストを入力してください")
        
        # 特徴量生成ボタン
        if st.button("特徴量を生成"):
            with st.spinner("特徴量を生成中..."):
                featured_df = df.copy()
                
                # 時間ベースの特徴量を追加
                if add_time_features_option:
                    featured_df = add_time_features(featured_df, date_col)
                
                # ラグ特徴量を追加
                if add_lag_features_option:
                    featured_df = add_lag_features(featured_df, target_col, lag_periods=custom_lags)
                
                # 移動平均特徴量を追加
                if add_rolling_features_option:
                    featured_df = add_rolling_features(featured_df, target_col, windows=custom_windows)
                
                # 特徴量を更新
                st.session_state.featured_data = featured_df
            
            # 成功メッセージとデータプレビュー
            show_success("特徴量が生成されました！")
            st.subheader("生成された特徴量のプレビュー")
            st.dataframe(featured_df.head())
            
            # 特徴量の重要度を計算（ランダムフォレストを使用）
            try:
                # 数値列のみ抽出し、target_colとdate_colを除外
                feature_cols = [col for col in featured_df.select_dtypes(include=['float64', 'int64']).columns 
                                if col != target_col and col != date_col]
                
                if len(feature_cols) > 0:
                    importance_df = get_feature_importance(
                        featured_df.dropna(), 
                        target_col, 
                        feature_cols
                    )
                    
                    st.subheader("特徴量の重要度")
                    
                    # トップ15の特徴量だけ表示
                    top_features = importance_df.head(15)
                    
                    fig = px.bar(
                        top_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="特徴量の重要度 (トップ15)",
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ダウンロード可能にする
                    create_downloadable_csv(featured_df, "featured_data.csv")
            except Exception as e:
                st.error(f"特徴量重要度の計算中にエラーが発生しました: {str(e)}")

# モデルトレーニングページ
def show_training_page():
    """モデルトレーニングを行うページ"""
    st.header("モデルのトレーニング")
    
    # データの確認
    if st.session_state.clean_data is None:
        show_error("処理済みデータが見つかりません。先にデータを前処理してください。")
        return
    
    # 通常の前処理済みデータか、特徴量エンジニアリング済みデータを使用
    if st.session_state.featured_data is not None and st.button("特徴量エンジニアリング済みデータを使用"):
        df = st.session_state.featured_data
        st.success("特徴量エンジニアリング済みデータを使用します")
    else:
        df = st.session_state.clean_data
    
    target_col = st.session_state.target_col
    date_col = st.session_state.date_col
    
    # モデル選択と設定のタブ
    tabs = st.tabs(["モデル選択", "トレーニング", "評価"])
    
    # タブ1: モデル選択
    with tabs[0]:
        st.subheader("モデルの選択")
        
        # EDA結果に基づいて最適なモデルを推奨
        eda_results = st.session_state.eda_results
        
        recommended_model = None
        if eda_results:
            has_trend = eda_results.get('has_trend', False)
            has_seasonality = eda_results.get('has_seasonality', False)
            is_stationary = eda_results.get('is_stationary', False)
            
            if has_seasonality:
                recommended_model = 'SARIMA'
            elif has_trend and not is_stationary:
                recommended_model = 'Prophet'
            elif is_stationary:
                recommended_model = 'ARIMA'
            else:
                recommended_model = 'Random Forest'
        
        # モデル選択
        model_options = list(AVAILABLE_MODELS.keys())
        default_index = model_options.index(recommended_model) if recommended_model in model_options else 0
        
        selected_model = st.selectbox(
            "予測モデルを選択", 
            model_options,
            index=default_index,
            format_func=lambda x: AVAILABLE_MODELS[x]
        )
        
        if recommended_model:
            st.info(f"EDA結果に基づく推奨モデル: {AVAILABLE_MODELS[recommended_model]}")
        
        # モデルパラメータの設定
        st.subheader("モデルパラメータ")
        
        model_params = DEFAULT_PARAMS.get(selected_model, {}).copy()
        
        if selected_model == 'ARIMA':
            col1, col2, col3 = st.columns(3)
            with col1:
                model_params['p'] = st.number_input("AR次数 (p)", min_value=0, max_value=10, value=model_params.get('p', 1))
            with col2:
                model_params['d'] = st.number_input("差分次数 (d)", min_value=0, max_value=2, value=model_params.get('d', 1))
            with col3:
                model_params['q'] = st.number_input("MA次数 (q)", min_value=0, max_value=10, value=model_params.get('q', 1))
        
        elif selected_model == 'SARIMA':
            col1, col2, col3 = st.columns(3)
            with col1:
                model_params['p'] = st.number_input("AR次数 (p)", min_value=0, max_value=5, value=model_params.get('p', 1))
                model_params['P'] = st.number_input("季節AR次数 (P)", min_value=0, max_value=5, value=model_params.get('P', 1))
            with col2:
                model_params['d'] = st.number_input("差分次数 (d)", min_value=0, max_value=2, value=model_params.get('d', 1))
                model_params['D'] = st.number_input("季節差分次数 (D)", min_value=0, max_value=1, value=model_params.get('D', 0))
            with col3:
                model_params['q'] = st.number_input("MA次数 (q)", min_value=0, max_value=5, value=model_params.get('q', 1))
                model_params['Q'] = st.number_input("季節MA次数 (Q)", min_value=0, max_value=5, value=model_params.get('Q', 1))
            
            model_params['m'] = st.number_input(
                "季節周期 (m)", 
                min_value=2, 
                max_value=365, 
                value=model_params.get('m', eda_results.get('optimal_period', 12))
            )
        
        elif selected_model == 'Prophet':
            col1, col2 = st.columns(2)
            with col1:
                yearly_options = ['auto', True, False]
                yearly_index = yearly_options.index(model_params.get('yearly_seasonality', 'auto'))
                model_params['yearly_seasonality'] = st.selectbox(
                    "年次季節性", 
                    yearly_options, 
                    index=yearly_index
                )
                
                daily_options = ['auto', True, False]
                daily_index = daily_options.index(model_params.get('daily_seasonality', 'auto'))
                model_params['daily_seasonality'] = st.selectbox(
                    "日次季節性", 
                    daily_options, 
                    index=daily_index
                )
            
            with col2:
                weekly_options = ['auto', True, False]
                weekly_index = weekly_options.index(model_params.get('weekly_seasonality', 'auto'))
                model_params['weekly_seasonality'] = st.selectbox(
                    "週次季節性", 
                    weekly_options, 
                    index=weekly_index
                )
                
                model_params['seasonality_mode'] = st.selectbox(
                    "季節性モード", 
                    ['additive', 'multiplicative'], 
                    index=0
                )
        
        elif selected_model == 'Random Forest':
            col1, col2 = st.columns(2)
            with col1:
                model_params['n_estimators'] = st.slider(
                    "木の数", 
                    min_value=10, 
                    max_value=500, 
                    value=model_params.get('n_estimators', 100),
                    step=10
                )
            
            with col2:
                model_params['max_depth'] = st.slider(
                    "最大深さ", 
                    min_value=1, 
                    max_value=30, 
                    value=model_params.get('max_depth', 10)
                )
        
        # トレーニング設定
        st.subheader("トレーニング設定")
        
        # テストデータ分割
        test_size = st.slider("テストデータの割合", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        
        # モデル選択とパラメータをセッションに保存
        if st.button("トレーニング設定を保存"):
            st.session_state.selected_model = selected_model
            st.session_state.model_params = model_params
            st.session_state.test_size = test_size
            show_success("モデル設定が保存されました！トレーニングタブに進んでください。")

        # トレーニングの実行
        if st.button("モデルをトレーニング"):
            if st.session_state.selected_model is None:
                show_error("モデル設定を先に保存してください")
            else:
                with st.spinner("モデルをトレーニング中..."):
                    # トレーニングデータの準備
                    df = st.session_state.featured_data if st.session_state.featured_data is not None else st.session_state.clean_data
                    target_col = st.session_state.target_col
                    date_col = st.session_state.date_col
                    model_type = st.session_state.selected_model
                    params = st.session_state.model_params
                    
                    # トレーニング実行
                    trained_model = train_model(df, target_col, date_col, model_type, params, test_size)
                    
                    if trained_model is not None:
                        st.session_state.trained_model = trained_model
                        
                        # 評価実行
                        evaluation_results = evaluate_model(
                            trained_model, 
                            df, 
                            target_col, 
                            date_col, 
                            model_type
                        )
                        
                        st.session_state.evaluation_results = evaluation_results
                        
                        show_success("モデルのトレーニングと評価が完了しました！")
                        
                        # 評価メトリクスの表示
                        st.subheader("モデル評価")
                        metrics_df = pd.DataFrame({
                            'メトリクス': ['RMSE', 'MAE', 'MAPE (%)', 'R²'],
                            '値': [
                                evaluation_results['rmse'],
                                evaluation_results['mae'],
                                evaluation_results['mape'],
                                evaluation_results['r2']
                            ]
                        })
                        st.dataframe(metrics_df)
                        
                        # プロットの表示
                        plot_evaluation(evaluation_results)
                        
                        if st.button("予測ページに進む"):
                            st.session_state.page = 'forecast'
                            st.rerun()

def show_forecast_page():
    """将来予測を生成するページ"""
    st.header("予測の生成")
    
    if st.session_state.trained_model is None:
        show_error("先にモデルをトレーニングしてください")
        return
    
    # 予測期間の設定
    st.subheader("予測期間の設定")
    
    freq = st.session_state.data_frequency
    default_periods = DEFAULT_FORECAST_PERIODS.get(freq, 12)
    
    forecast_periods = st.number_input(
        "予測する期間数", 
        min_value=1, 
        max_value=1000, 
        value=default_periods
    )
    
    # 信頼区間の設定
    confidence_level = st.slider(
        "信頼区間 (%)", 
        min_value=50, 
        max_value=99, 
        value=95, 
        step=5
    ) / 100
    
    # 予測実行
    if st.button("予測を生成"):
        with st.spinner("予測を生成中..."):
            model = st.session_state.trained_model
            df = st.session_state.featured_data if st.session_state.featured_data is not None else st.session_state.clean_data
            target_col = st.session_state.target_col
            date_col = st.session_state.date_col
            model_type = st.session_state.selected_model
            
            forecast_results = generate_forecast(
                model, 
                df, 
                target_col, 
                date_col, 
                model_type, 
                forecast_periods, 
                confidence_level
            )
            
            if forecast_results is not None:
                st.session_state.forecast = forecast_results
                
                # 予測結果の表示
                st.subheader("予測結果")
                
                # 予測データを表示
                forecast_df = forecast_results['forecast_df']
                st.dataframe(forecast_df)
                
                # 予測のプロット
                st.subheader("予測グラフ")
                plot_forecast_results(forecast_results, df, target_col, date_col)
                
                # 予測データのダウンロード
                st.markdown("### 予測データのダウンロード")
                create_downloadable_csv(forecast_df, "forecast_data.csv")

def plot_forecast_results(forecast_results, historical_df, target_col, date_col):
    """予測結果のプロット"""
    # 予測データ
    forecast_df = forecast_results['forecast_df']
    
    # 履歴データの最後の部分（グラフ表示用）
    hist_periods = min(len(historical_df), 365)  # 最大で1年分
    hist_df = historical_df.sort_values(by=date_col).tail(hist_periods)
    
    # 履歴データと予測データを結合したプロット
    fig = go.Figure()
    
    # 履歴データの追加
    fig.add_trace(go.Scatter(
        x=hist_df[date_col],
        y=hist_df[target_col],
        mode='lines',
        name='実績値',
        line=dict(color='blue')
    ))
    
    # 予測データの追加
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['prediction'],
        mode='lines',
        name='予測値',
        line=dict(color='red', dash='dash')
    ))
    
    # 信頼区間の追加
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['upper_bound'],
        mode='lines',
        name='上限',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['lower_bound'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        name='95%信頼区間',
        line=dict(width=0)
    ))
    
    # プロットのレイアウト設定
    fig.update_layout(
        title='予測結果',
        xaxis_title='日付',
        yaxis_title=target_col,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 季節性の分解を表示 (Prophetモデルの場合)
    if 'components_df' in forecast_results:
        st.subheader("時系列の分解")
        components_df = forecast_results['components_df']
        
        # トレンドのプロット
        fig = px.line(
            components_df, 
            x='ds', 
            y='trend',
            title='トレンド成分'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 季節性のプロット
        seasonality_components = [col for col in components_df.columns if any(s in col for s in ['yearly', 'weekly', 'daily'])]
        
        for component in seasonality_components:
            fig = px.line(
                components_df, 
                x='ds', 
                y=component,
                title=f'{component} 季節性'
            )
            st.plotly_chart(fig, use_container_width=True)

def plot_evaluation(evaluation_results):
    """モデル評価結果のプロット"""
    y_true = evaluation_results['y_true']
    y_pred = evaluation_results['y_pred']
    dates = evaluation_results['dates']
    
    # 実測値と予測値のプロット
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode='lines',
        name='実測値'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode='lines',
        name='予測値'
    ))
    
    fig.update_layout(
        title='実測値 vs 予測値',
        xaxis_title='日付',
        yaxis_title='値',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 残差プロット
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=residuals,
        mode='markers',
        name='残差',
        marker=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=[dates.min(), dates.max()],
        y=[0, 0],
        mode='lines',
        name='ゼロライン',
        line=dict(color='black', dash='dash')
    ))
    
    fig.update_layout(
        title='残差プロット',
        xaxis_title='日付',
        yaxis_title='残差',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 実測値と予測値の散布図
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='実測値 vs 予測値',
        marker=dict(color='blue')
    ))
    
    # 45度線（理想的な予測ライン）
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='理想ライン',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='実測値 vs 予測値の散布図',
        xaxis_title='実測値',
        yaxis_title='予測値',
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# モデルトレーニング関数
def train_model(df, target_col, date_col, model_type, params, test_size=0.2):
    """モデルをトレーニングする関数"""
    try:
        # データを時系列順にソート
        df = df.sort_values(by=date_col)
        
        # テストデータとトレーニングデータに分割
        train_size = int(len(df) * (1 - test_size))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # 返却するモデル情報
        model_info = {
            'model': None,
            'model_type': model_type,
            'train_df': train_df,
            'test_df': test_df,
            'params': params,
            'scaler': None
        }
        
        # モデル別のトレーニング処理
        if model_type == 'ARIMA':
            from statsmodels.tsa.arima.model import ARIMA
            
            # ARIMAモデルをトレーニング
            model = ARIMA(
                train_df[target_col], 
                order=(params['p'], params['d'], params['q'])
            )
            fitted_model = model.fit()
            model_info['model'] = fitted_model
        
        elif model_type == 'SARIMA':
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # SARIMAXモデルをトレーニング
            model = SARIMAX(
                train_df[target_col],
                order=(params['p'], params['d'], params['q']),
                seasonal_order=(params['P'], params['D'], params['Q'], params['m'])
            )
            fitted_model = model.fit(disp=False)
            model_info['model'] = fitted_model
        
        elif model_type == 'Prophet':
            from prophet import Prophet
            
            # Prophetのデータフレーム形式に変換
            prophet_df = pd.DataFrame({
                'ds': train_df[date_col],
                'y': train_df[target_col]
            })
            
            # Prophetモデルの初期化とトレーニング
            model = Prophet(
                yearly_seasonality=params.get('yearly_seasonality', 'auto'),
                weekly_seasonality=params.get('weekly_seasonality', 'auto'),
                daily_seasonality=params.get('daily_seasonality', 'auto'),
                seasonality_mode=params.get('seasonality_mode', 'additive')
            )
            
            # 休日データがあれば追加
            if 'holidays' in params:
                model.add_country_holidays(country_name=params['holidays'])
            
            # 追加の季節性があれば追加
            if 'add_seasonality' in params:
                for s in params['add_seasonality']:
                    model.add_seasonality(**s)
            
            # モデルをトレーニング
            model.fit(prophet_df)
            model_info['model'] = model
        
        elif model_type == 'Random Forest':
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # 特徴量とターゲットの準備
            X_cols = [col for col in train_df.columns if col != target_col and col != date_col]
            
            if not X_cols:
                # 時間ベースの特徴量を自動追加
                for df_part in [train_df, test_df]:
                    df_part['year'] = df_part[date_col].dt.year
                    df_part['month'] = df_part[date_col].dt.month
                    df_part['day'] = df_part[date_col].dt.day
                    df_part['dayofweek'] = df_part[date_col].dt.dayofweek
                    df_part['quarter'] = df_part[date_col].dt.quarter
                
                X_cols = ['year', 'month', 'day', 'dayofweek', 'quarter']
            
            X_train = train_df[X_cols]
            y_train = train_df[target_col]
            
            # 特徴量のスケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # ランダムフォレストモデルの初期化とトレーニング
            model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            model_info['model'] = model
            model_info['scaler'] = scaler
            model_info['feature_cols'] = X_cols
        
        return model_info
    
    except Exception as e:
        st.error(f"モデルトレーニング中にエラーが発生しました: {str(e)}")
        return None

# モデル評価関数
def evaluate_model(model_info, df, target_col, date_col, model_type):
    """トレーニング済みモデルを評価する関数"""
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        test_df = model_info['test_df']
        
        # モデル別の予測処理
        if model_type in ['ARIMA', 'SARIMA']:
            # 予測の開始と終了インデックス
            start_idx = len(model_info['train_df'])
            end_idx = start_idx + len(test_df) - 1
            
            # 予測の実行
            predictions = model_info['model'].predict(start=start_idx, end=end_idx)
            
        elif model_type == 'Prophet':
            # 予測用データフレームの作成
            future = pd.DataFrame({'ds': test_df[date_col]})
            
            # 予測の実行
            forecast = model_info['model'].predict(future)
            predictions = forecast['yhat'].values
            
        elif model_type == 'Random Forest':
            # 特徴量の抽出とスケーリング
            X_test = test_df[model_info['feature_cols']]
            X_test_scaled = model_info['scaler'].transform(X_test)
            
            # 予測の実行
            predictions = model_info['model'].predict(X_test_scaled)
        
        # 実測値
        y_true = test_df[target_col].values
        
        # 評価指標の計算
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        
        # MAPE (Mean Absolute Percentage Error)の計算
        mask = y_true != 0  # ゼロ除算を避ける
        mape = np.mean(np.abs((y_true[mask] - predictions[mask]) / y_true[mask])) * 100
        
        # 評価結果の返却
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'y_true': y_true,
            'y_pred': predictions,
            'dates': test_df[date_col].values
        }
    
    except Exception as e:
        st.error(f"モデル評価中にエラーが発生しました: {str(e)}")
        return None

# 予測生成関数
def generate_forecast(model_info, df, target_col, date_col, model_type, forecast_periods, confidence_level=0.95):
    """将来予測を生成する関数"""
    try:
        # 最後の日付を取得
        last_date = df[date_col].max()
        
        # データ頻度の取得
        freq = st.session_state.data_frequency
        
        # 頻度に基づく日付のオフセットマップ
        freq_map = {
            'hourly': 'H',
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q',
            'yearly': 'Y'
        }
        
        offset = freq_map.get(freq, 'D')
        
        # 将来日付の生成
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=forecast_periods,
            freq=offset
        )
        
        # モデル別の予測処理
        if model_type in ['ARIMA', 'SARIMA']:
            # 予測と信頼区間
            forecast_result = model_info['model'].get_forecast(steps=forecast_periods)
            predictions = forecast_result.predicted_mean
            
            # 信頼区間
            conf_int = forecast_result.conf_int(alpha=1-confidence_level)
            lower_bounds = conf_int.iloc[:, 0]
            upper_bounds = conf_int.iloc[:, 1]
            
            # 予測データフレームの作成
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'prediction': predictions.values,
                'lower_bound': lower_bounds.values,
                'upper_bound': upper_bounds.values
            })
            
            result = {'forecast_df': forecast_df}
            
        elif model_type == 'Prophet':
            # 予測用データフレームの作成
            future = pd.DataFrame({'ds': pd.concat([df[date_col], pd.Series(future_dates)])})
            
            # 予測の実行
            forecast = model_info['model'].predict(future)
            
            # 予測期間のみを抽出
            forecast_result = forecast.tail(forecast_periods)
            
            # 予測データフレームの作成
            forecast_df = pd.DataFrame({
                'date': forecast_result['ds'],
                'prediction': forecast_result['yhat'],
                'lower_bound': forecast_result['yhat_lower'],
                'upper_bound': forecast_result['yhat_upper']
            })
            
            # 成分分解を含む結果
            result = {
                'forecast_df': forecast_df,
                'components_df': forecast_result[['ds', 'trend'] + [col for col in forecast_result.columns if 'seasonality' in col]]
            }
            
        elif model_type == 'Random Forest':
            # 将来の特徴量を生成
            future_df = pd.DataFrame({'date': future_dates})
            
            # 時間ベースの特徴量を追加
            future_df['year'] = future_df['date'].dt.year
            future_df['month'] = future_df['date'].dt.month
            future_df['day'] = future_df['date'].dt.day
            future_df['dayofweek'] = future_df['date'].dt.dayofweek
            future_df['quarter'] = future_df['date'].dt.quarter
            
            # モデルの特徴量だけを抽出
            X_future = future_df[model_info['feature_cols']]
            
            # スケーリング
            X_future_scaled = model_info['scaler'].transform(X_future)
            
            # 予測の実行
            predictions = model_info['model'].predict(X_future_scaled)
            
            # ランダムフォレストの場合、予測区間を推定
            # (単純な近似として標準偏差を使用)
            from scipy import stats
            
            # テストデータの残差から標準偏差を計算
            test_predictions = model_info['model'].predict(
                model_info['scaler'].transform(model_info['test_df'][model_info['feature_cols']])
            )
            residuals = model_info['test_df'][target_col].values - test_predictions
            std_dev = residuals.std()
            
            # 信頼区間の計算（正規分布を仮定）
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_score * std_dev
            
            # 予測データフレームの作成
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'prediction': predictions,
                'lower_bound': predictions - margin,
                'upper_bound': predictions + margin
            })
            
            result = {'forecast_df': forecast_df}
        
        # 変換情報がある場合は元のスケールに戻す
        if st.session_state.transform_info is not None:
            result['forecast_df']['prediction'] = inverse_transform(
                result['forecast_df']['prediction'],
                st.session_state.transform_info
            )
            result['forecast_df']['lower_bound'] = inverse_transform(
                result['forecast_df']['lower_bound'],
                st.session_state.transform_info
            )
            result['forecast_df']['upper_bound'] = inverse_transform(
                result['forecast_df']['upper_bound'],
                st.session_state.transform_info
            )
        
        return result
    
    except Exception as e:
        st.error(f"予測生成中にエラーが発生しました: {str(e)}")
        return None

# メイン関数
def main():
    """アプリケーションのメインエントリポイント"""
    # ナビゲーション表示
    navigation()
    
    # 現在のページに応じて表示
    if st.session_state.page == 'upload':
        show_upload_page()
    elif st.session_state.page == 'eda':
        show_eda_page()
    elif st.session_state.page == 'training':
        show_training_page()
    elif st.session_state.page == 'forecast':
        show_forecast_page()

# データロード関数
def load_data(file):
    """データファイルを読み込む"""
    try:
        df = load_cached_data(file)
        return df
    except Exception as e:
        show_error(f"データの読み込みに失敗しました: {e}")
        return None

# アプリケーション実行
if __name__ == "__main__":
    main()