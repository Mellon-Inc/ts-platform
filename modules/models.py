import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional

# 外部ライブラリの可用性チェック
try:
    import pmdarima as pm
    pmdarima_available = True
except (ImportError, ValueError):
    pmdarima_available = False

try:
    from prophet import Prophet
    prophet_available = True
except (ImportError, ValueError):
    prophet_available = False

# 線形回帰・重回帰パラメータ設定
def setup_regression_parameters(model_type, df, target_col, date_col):
    feature_cols = st.multiselect(
        '説明変数となる列を選択してください:', 
        [col for col in df.columns if col not in [target_col, date_col]], 
        default=[]
    )
    if model_type == '線形回帰' and len(feature_cols) > 0:
        st.warning('線形回帰では最初の説明変数のみが使用されます')
        feature_cols = feature_cols[:1]
    
    return {'feature_cols': feature_cols}

# ARIMAパラメータ設定
def setup_arima_parameters(df, target_col):
    use_auto_arima = False
    p, d, q = 1, 1, 1
    
    if not pmdarima_available:
        st.error('pmdarimaライブラリが利用できないため、Auto ARIMAは使用できません。手動でパラメータを設定してください。')
    else:
        use_auto_arima = st.checkbox('Auto ARIMAを使用する', value=True)
        
    if not use_auto_arima:
        p = st.slider('ARパラメータ (p):', 0, 5, 1)
        d = st.slider('差分パラメータ (d):', 0, 2, 1)
        q = st.slider('MAパラメータ (q):', 0, 5, 1)
        
    return {'feature_cols': [], 'use_auto_arima': use_auto_arima, 'p': p, 'd': d, 'q': q}

# Prophetパラメータ設定
def setup_prophet_parameters():
    if not prophet_available:
        st.error('Prophetライブラリが利用できません。インストールしてください。')
        return {'feature_cols': []}
    
    seasonality_mode = st.selectbox('季節性モード:', ['additive', 'multiplicative'])
    yearly_seasonality = st.checkbox('年次季節性', value=True)
    weekly_seasonality = st.checkbox('週次季節性', value=True)
    daily_seasonality = st.checkbox('日次季節性', value=False)
    
    # カスタム周期性の追加
    add_custom_seasonality = st.checkbox('カスタム周期性を追加', value=False)
    custom_seasonality = {}
    
    if add_custom_seasonality:
        with st.expander('カスタム周期性の設定'):
            custom_name = st.text_input('周期性の名前（例：月次、四半期など）:', 'custom')
            custom_period = st.number_input('周期の長さ（日数）:', min_value=1, value=30)
            custom_fourier = st.slider('フーリエ項の数:', 1, 20, 5)
            custom_mode = st.selectbox('周期性モード:', ['additive', 'multiplicative'])
            
            custom_seasonality = {
                'name': custom_name,
                'period': custom_period,
                'fourier_order': custom_fourier,
                'mode': custom_mode
            }
    
    return {
        'feature_cols': [],
        'seasonality_mode': seasonality_mode,
        'yearly_seasonality': yearly_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'daily_seasonality': daily_seasonality,
        'add_custom_seasonality': add_custom_seasonality,
        'custom_seasonality': custom_seasonality
    }

# モデルパラメータ設定UI
def setup_model_parameters(model_type, df, target_col, date_col):
    if model_type in ['線形回帰', '重回帰']:
        return setup_regression_parameters(model_type, df, target_col, date_col)
    elif model_type == 'ARIMA':
        return setup_arima_parameters(df, target_col)
    elif model_type == 'Prophet':
        return setup_prophet_parameters()
    
    return {'feature_cols': []}

# 線形回帰モデルの実行
def run_linear_regression(df, target_col, date_col, feature_cols, forecast_period):
    from sklearn.linear_model import LinearRegression
    
    # 特徴量の準備
    X = df[['days']] if not feature_cols else df[feature_cols]
    y = df[target_col]
    
    # モデルの学習
    model = LinearRegression()
    model.fit(X, y)
    
    # 予測用データの準備
    last_day = int(df['days'].max())
    future_days = np.array(range(last_day + 1, last_day + forecast_period + 1)).reshape(-1, 1)
    
    # 予測の実行
    if not feature_cols:
        predictions = model.predict(future_days)
    else:
        # 特徴量がある場合は最後の値を使用
        last_features = df[feature_cols].iloc[-1:].values
        future_features = np.tile(last_features, (forecast_period, 1))
        predictions = model.predict(future_features)
    
    return predictions

# ARIMAモデルの実行
def run_arima(df, target_col, params, forecast_period):
    from statsmodels.tsa.arima.model import ARIMA
    
    # データの準備
    y = df[target_col]
    
    if params['use_auto_arima'] and pmdarima_available:
        # Auto ARIMAの実行
        auto_model = pm.auto_arima(
            y,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=None,
            trace=True
        )
        
        # 最適なパラメータを取得
        p, d, q = auto_model.order
        st.write(f'Auto ARIMAで選択されたパラメータ: p={p}, d={d}, q={q}')
        
        # 予測の実行
        forecast, conf_int = auto_model.predict(n_periods=forecast_period, return_conf_int=True)
        return forecast, auto_model
    else:
        # 手動パラメータでARIMAを実行
        model = ARIMA(y, order=(params['p'], params['d'], params['q']))
        model_fit = model.fit()
        
        # 予測の実行
        forecast = model_fit.forecast(steps=forecast_period)
        return forecast, model_fit

# Prophetモデルの実行
def run_prophet(df, target_col, date_col, params, forecast_period):
    if not prophet_available:
        st.error('Prophetライブラリが利用できません。')
        return []
    
    # Prophetのデータ形式に変換
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    
    # モデルの初期化
    model = Prophet(
        seasonality_mode=params['seasonality_mode'],
        yearly_seasonality=params['yearly_seasonality'],
        weekly_seasonality=params['weekly_seasonality'],
        daily_seasonality=params['daily_seasonality']
    )
    
    # カスタム季節性の追加
    if params['add_custom_seasonality']:
        model.add_seasonality(
            name=params['custom_seasonality']['name'],
            period=params['custom_seasonality']['period'],
            fourier_order=params['custom_seasonality']['fourier_order'],
            mode=params['custom_seasonality']['mode']
        )
    
    # モデルの学習
    model.fit(prophet_df)
    
    # 将来データフレームの作成
    future = model.make_future_dataframe(periods=forecast_period)
    
    # 予測の実行
    forecast = model.predict(future)
    
    # 予測結果の取得
    predictions = forecast['yhat'].iloc[-forecast_period:].values
    
    return predictions

# 予測実行
def run_prediction(df, model_type, target_col, date_col, params, forecast_period):
    # 日付を日数に変換
    df[date_col] = pd.to_datetime(df[date_col])
    df['days'] = (df[date_col] - df[date_col].min()).dt.days

    model_fit = None
    
    # モデル別の予測実行
    if model_type in ['線形回帰', '重回帰']:
        predictions = run_linear_regression(df, target_col, date_col, params['feature_cols'], forecast_period)
    elif model_type == 'ARIMA':
        predictions, model_fit = run_arima(df, target_col, params, forecast_period)
    elif model_type == 'Prophet' and prophet_available:
        predictions = run_prophet(df, target_col, date_col, params, forecast_period)
    else:
        st.error('選択されたモデルは現在利用できません。')
        return

    # 予測結果の可視化
    future_dates = visualize_predictions(df, target_col, date_col, predictions, forecast_period)
    
    # モデルがARIMAの場合、診断プロットを表示
    if model_type == 'ARIMA' and not params['use_auto_arima'] and model_fit:
        st.subheader('モデル診断')
        try:
            fig_diag = model_fit.plot_diagnostics(figsize=(10, 8))
            st.pyplot(fig_diag)
        except:
            st.warning('診断プロットの生成に失敗しました。')

# 予測結果の可視化
def visualize_predictions(df, target_col, date_col, predictions, forecast_period):
    future_dates = pd.date_range(
        start=df[date_col].max(),
        periods=forecast_period + 1,
        freq='D'
    )[1:]

    # 結果のデータフレーム作成
    hist_data = df[[date_col, target_col]].reset_index(drop=True)
    future_data = pd.DataFrame({
        date_col: future_dates,
        target_col: predictions
    }).reset_index(drop=True)
    
    # データを分割して別々のトレースとして表示
    fig = go.Figure()
    
    # 履歴データのプロット
    fig.add_trace(go.Scatter(
        x=hist_data[date_col],
        y=hist_data[target_col],
        mode='lines',
        name='実測値',
        line=dict(color='royalblue', width=2)
    ))
    
    # 予測データのプロット
    fig.add_trace(go.Scatter(
        x=future_data[date_col],
        y=future_data[target_col],
        mode='lines',
        name='予測値',
        line=dict(color='firebrick', width=2, dash='dot')
    ))
    
    # 区切り線 - Timestamp型を数値に変換して使用
    last_date = df[date_col].max()
    
    # 区切り線を追加（エラーを修正）
    fig.add_shape(
        type="line",
        x0=last_date,
        y0=0,
        x1=last_date,
        y1=1,
        yref="paper",
        line=dict(color='gray', width=1, dash='dash')
    )
    
    # 注釈を追加
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="予測開始",
        showarrow=False,
        xanchor="right"
    )
    
    # レイアウト設定
    fig.update_layout(
        title=f'{target_col}の時系列予測',
        xaxis_title='日付',
        yaxis_title=target_col,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        hovermode='x unified'
    )
    
    # グラフ表示
    st.plotly_chart(fig, use_container_width=True)
    
    return future_dates

def get_model_info(model_type: str) -> Dict:
    """
    モデルの情報を取得
    
    Parameters:
    ----------
    model_type : str
        モデルの種類
        
    Returns:
    -------
    dict
        モデル情報を含む辞書
    """
    model_info = {
        'ARIMA': {
            'name': 'ARIMA (自己回帰和分移動平均)',
            'description': 'ARIMAモデルは、時系列データの自己相関構造をモデル化するための統計的手法です。季節性をもたないデータに適しています。',
            'params': ['p (自己回帰次数)', 'd (差分次数)', 'q (移動平均次数)'],
            'suitable_for': ['トレンドがあるデータ', '定常でないデータ', '短期予測'],
            'not_suitable_for': ['季節性が強いデータ', '非線形の関係', '複数の外部要因がある場合'],
            'advantages': ['解釈がしやすい', '少ないデータでも機能する', '信頼区間を自然に提供'],
            'disadvantages': ['季節性を処理できない', '外部要因を考慮しない', '非線形関係を捉えられない']
        },
        'SARIMA': {
            'name': 'SARIMA (季節性ARIMA)',
            'description': 'SARIMAはARIMAを拡張したもので、季節性の要素を含む時系列データに適しています。',
            'params': ['p, d, q (非季節成分)', 'P, D, Q (季節成分)', 'm (季節周期)'],
            'suitable_for': ['季節性のあるデータ', 'トレンドがあるデータ', '中期予測'],
            'not_suitable_for': ['複雑な非線形パターン', '多変量データ', '不規則なパターン'],
            'advantages': ['季節性を直接モデル化できる', '時系列の伝統的な手法', '解釈がしやすい'],
            'disadvantages': ['複雑なパターンを捉えられない', '多くのパラメータ調整が必要', '外部要因の考慮が難しい']
        },
        'Prophet': {
            'name': 'Prophet (Facebookの予測モデル)',
            'description': 'ProphetはFacebookが開発した強力な予測ツールで、季節性、休日の効果、トレンドの変化点を扱えます。',
            'params': ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality', 'seasonality_mode'],
            'suitable_for': ['強い季節性のあるデータ', '休日の効果があるデータ', 'トレンドの変化点があるデータ'],
            'not_suitable_for': ['高頻度データ', '複雑な依存関係', '多変量データ'],
            'advantages': ['使いやすさ', '柔軟性', '変化点の自動検出', '休日効果の組み込み'],
            'disadvantages': ['ブラックボックス的な側面', '細かいパラメータ調整が難しい', '特定のデータ構造が必要']
        },
        'Random Forest': {
            'name': 'ランダムフォレスト',
            'description': 'ランダムフォレストは多数の決定木の予測を組み合わせるアンサンブル学習法です。非線形パターンの捕捉に優れています。',
            'params': ['n_estimators (木の数)', 'max_depth (木の深さ)'],
            'suitable_for': ['非線形関係', '多変量データ', '特徴量の重要度分析'],
            'not_suitable_for': ['長期予測', 'データ量が少ない場合', '外挿'],
            'advantages': ['強力な予測能力', '過学習に強い', '特徴量の重要度評価'],
            'disadvantages': ['解釈が難しい', 'メモリ消費が大きい', '時間的依存関係の直接的なモデル化ができない']
        },
        'Linear Regression': {
            'name': '線形回帰',
            'description': '線形回帰は最も基本的な予測手法で、特徴量と目的変数の間の線形関係をモデル化します。',
            'params': [],
            'suitable_for': ['線形関係', 'シンプルなデータ', '解釈性重視の分析'],
            'not_suitable_for': ['非線形パターン', '複雑な時系列', '季節性が強いデータ'],
            'advantages': ['シンプルで解釈しやすい', '計算が速い', 'メモリ効率が良い'],
            'disadvantages': ['表現力が限られる', '外れ値に敏感', '複雑なパターンを捉えられない']
        },
        'XGBoost': {
            'name': 'XGBoost (勾配ブースティング)',
            'description': 'XGBoostは勾配ブースティングフレームワークで、高い予測精度と効率的な計算が特徴です。',
            'params': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree'],
            'suitable_for': ['構造化データ', '非線形関係', '高い予測精度が必要な場合'],
            'not_suitable_for': ['解釈性が重要な場合', '小さなデータセット', '時間的関係のみの場合'],
            'advantages': ['高い予測精度', '過学習に対する様々な制御', 'スケーラビリティ'],
            'disadvantages': ['パラメータチューニングが複雑', '解釈が難しい', '時系列の特性を直接モデル化しない']
        }
    }
    
    if model_type in model_info:
        return model_info[model_type]
    else:
        return {
            'name': '不明なモデル',
            'description': '指定されたモデルの情報はありません。',
            'params': [],
            'suitable_for': [],
            'not_suitable_for': [],
            'advantages': [],
            'disadvantages': []
        }

def build_model(model_type: str, params: Dict = None) -> Any:
    """
    指定されたタイプのモデルを構築
    
    Parameters:
    ----------
    model_type : str
        モデルの種類
    params : dict, optional
        モデルのパラメータ
        
    Returns:
    -------
    object
        構築されたモデル
    """
    if params is None:
        params = {}
    
    if model_type == 'ARIMA':
        from statsmodels.tsa.arima.model import ARIMA
        p = params.get('p', 1)
        d = params.get('d', 1)
        q = params.get('q', 1)
        
        return ARIMA(order=(p, d, q))
    
    elif model_type == 'SARIMA':
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        p = params.get('p', 1)
        d = params.get('d', 1)
        q = params.get('q', 1)
        P = params.get('P', 1)
        D = params.get('D', 0)
        Q = params.get('Q', 1)
        m = params.get('m', 12)
        
        return SARIMAX(order=(p, d, q), seasonal_order=(P, D, Q, m))
    
    elif model_type == 'Prophet':
        try:
            from prophet import Prophet
            
            yearly_seasonality = params.get('yearly_seasonality', 'auto')
            weekly_seasonality = params.get('weekly_seasonality', 'auto')
            daily_seasonality = params.get('daily_seasonality', 'auto')
            seasonality_mode = params.get('seasonality_mode', 'additive')
            
            return Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                seasonality_mode=seasonality_mode
            )
        except ImportError:
            st.error("Prophetがインストールされていません。以下のコマンドでインストールしてください：`pip install prophet`")
            return None
    
    elif model_type == 'Random Forest':
        from sklearn.ensemble import RandomForestRegressor
        
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', None)
        
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    elif model_type == 'Linear Regression':
        from sklearn.linear_model import LinearRegression
        
        return LinearRegression()
    
    elif model_type == 'XGBoost':
        try:
            import xgboost as xgb
            
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', 6)
            learning_rate = params.get('learning_rate', 0.1)
            subsample = params.get('subsample', 0.8)
            colsample_bytree = params.get('colsample_bytree', 0.8)
            
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42
            )
        except ImportError:
            st.error("XGBoostがインストールされていません。以下のコマンドでインストールしてください：`pip install xgboost`")
            st.info("MacOSユーザーの場合: ターミナルで `brew install libomp` を実行してOpenMPランタイムをインストールしてください。")
            return None
    
    else:
        st.error(f"未対応のモデルタイプです: {model_type}")
        return None

def get_recommended_model(df: pd.DataFrame, target_col: str, date_col: str) -> str:
    """
    データに基づいて推奨モデルを提案
    
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
    str
        推奨モデル名
    """
    # データサイズ
    data_size = len(df)
    
    # 季節性チェック
    from modules.eda import detect_seasonality
    has_seasonality, _ = detect_seasonality(df, target_col, date_col)
    
    # 定常性チェック
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(df[target_col].dropna())
    is_stationary = adf_result[1] < 0.05  # p値が0.05未満なら定常とみなす
    
    # 線形トレンドチェック
    from scipy.stats import linregress
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy['time_idx'] = range(len(df_copy))
    slope, _, r_value, p_value, _ = linregress(df_copy['time_idx'], df_copy[target_col])
    has_linear_trend = p_value < 0.05 and abs(r_value) > 0.3
    
    # 外部特徴量の数
    external_features = len(df.columns) - 2  # 日付列とターゲット列を除く
    
    # 推奨モデル選択ロジック
    if data_size < 50:
        # データが少ない場合
        if has_seasonality:
            return 'SARIMA'
        else:
            return 'ARIMA'
    elif has_seasonality and data_size >= 100:
        # 季節性があり、十分なデータがある場合
        if external_features > 0:
            # 外部特徴量がある場合
            return 'Random Forest'
        else:
            return 'Prophet'
    elif has_linear_trend and not has_seasonality:
        # 線形トレンドがあり、季節性がない場合
        if external_features > 0:
            return 'Linear Regression'
        else:
            return 'ARIMA'
    elif not is_stationary and not has_seasonality:
        # 非定常で季節性がない場合
        return 'ARIMA'
    elif external_features > 3:
        # 多くの外部特徴量がある場合
        return 'XGBoost'
    else:
        # デフォルト
        return 'Random Forest'