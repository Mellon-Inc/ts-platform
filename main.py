import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# アップロードページモジュールをインポート
from modules.upload import show_upload_page

# preprocessing モジュールをインポート
from modules.preprocessing import show_preprocessing_page

# eda モジュールをインポート
from modules.eda import show_eda_page

from modules.training import show_training_page

from modules.forecast import show_forecast_page

# 設定ファイルと共通ユーティリティをインポート
from config import APP_TITLE, APP_DESCRIPTION

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
    
    # サイドバーにステップを表示
    st.sidebar.subheader("分析ステップ")
    
    # 各ステップへのボタン
    steps = {
        "1. データのアップロード": "upload",
        "2. データの前処理": "preprocessing",
        "3. 探索的データ分析 (EDA)": "eda",
        "4. モデルのトレーニング": "training",
        "5. 予測の生成": "forecast"
    }
    
    # 各ステップの依存関係を定義
    # 現在の状態に基づいて情報メッセージを表示
    if st.session_state.data is None:
        st.sidebar.info("データをアップロードしてから前処理を行えます")
    elif st.session_state.clean_data is None:
        st.sidebar.info("データの前処理を完了してからEDAを行えます")
    elif not st.session_state.eda_results:
        st.sidebar.info("EDAを完了してからモデルトレーニングを行えます")
    elif st.session_state.trained_model is None:
        st.sidebar.info("モデルをトレーニングしてから予測を生成できます")
    
    for step_name, step_page in steps.items():
        # データアップロード前は他のステップを無効化
        if st.session_state.data is None and step_page != "upload":
            st.sidebar.button(step_name, disabled=True, use_container_width=True)
        # 前処理完了前はEDA以降を無効化
        elif st.session_state.clean_data is None and step_page not in ["upload", "preprocessing"]:
            st.sidebar.button(step_name, disabled=True, use_container_width=True)
        # EDA完了前はモデルトレーニング以降を無効化
        elif not st.session_state.eda_results and step_page not in ["upload", "preprocessing", "eda"]:
            st.sidebar.button(step_name, disabled=True, use_container_width=True)
        # モデルトレーニング完了前は予測を無効化
        elif st.session_state.trained_model is None and step_page == "forecast":
            st.sidebar.button(step_name, disabled=True, use_container_width=True)
        else:
            if st.sidebar.button(step_name, use_container_width=True):
                st.session_state.page = step_page
                

# メイン関数
def main():
    """アプリケーションのメインエントリポイント"""
    # ナビゲーション表示
    navigation()
    
    # 現在のページに応じて表示
    if st.session_state.page == 'upload':
        show_upload_page()
    elif st.session_state.page == 'preprocessing':
        show_preprocessing_page()
    elif st.session_state.page == 'eda':
        show_eda_page()
    elif st.session_state.page == 'training':
        show_training_page()
    elif st.session_state.page == 'forecast':
        show_forecast_page()

# アプリケーション実行
if __name__ == "__main__":
    main()