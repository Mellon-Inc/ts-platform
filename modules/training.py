import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import lightgbm
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

def show_training_page():
    """モデルのトレーニングと評価を行うページ"""
    st.title("時系列予測モデルのトレーニング")
    
    # データが存在するか確認
    if st.session_state.clean_data is None or not st.session_state.eda_results:
        st.warning("前処理とEDAを完了させてください。")
        return
    
    # データとEDA結果の取得
    data = st.session_state.clean_data
    date_col = st.session_state.date_col
    target_col = st.session_state.target_col
    eda_results = st.session_state.eda_results
    
    # データの確認
    st.subheader("トレーニングデータの確認")
    st.write(f"データサイズ: {data.shape[0]}行 × {data.shape[1]}列")
    st.write("データプレビュー:")
    st.dataframe(data.head())
    
    # モデル選択セクション
    st.subheader("予測モデルの選択")
    
    # EDA結果からの推奨モデル選択
    recommended_model = recommend_model(eda_results)
    
    # モデル選択オプション
    model_options = {
        "Prophet": "Facebookが開発した時系列予測モデル - 複雑な季節性と休日効果を扱えます",
        "LGBM": "Light Gradient Boosting Machine - 非線形パターンの予測に強い機械学習モデル"
    }
    
    # 推奨モデルの表示
    st.info(f"EDA結果に基づく推奨モデル: **{recommended_model}**")
    
    # モデル選択
    selected_model = st.selectbox(
        "使用するモデルを選択してください:", 
        list(model_options.keys()),
        index=list(model_options.keys()).index(recommended_model if recommended_model != "ARIMA" else "Prophet")
    )
    
    st.write(f"選択されたモデル: {selected_model} - {model_options[selected_model]}")
    
    # 時系列分割の設定
    st.subheader("トレーニング・テスト分割設定")
    
    # データサイズに基づいてデフォルト値を設定
    data_size = len(data)
    default_test_size = max(int(data_size * 0.2), 1)  # デフォルトで20%
    
    # テストデータサイズの設定
    test_size = st.slider(
        "テストデータのサイズ:", 
        min_value=1, 
        max_value=int(data_size * 0.5),
        value=default_test_size,
        help="検証に使用するデータポイントの数"
    )
    
    # トレーニングデータとテストデータの分割
    train_data = data.iloc[:-test_size].copy()
    test_data = data.iloc[-test_size:].copy()
    
    # 分割の可視化
    fig = go.Figure()
    
    # トレーニングデータ
    fig.add_trace(go.Scatter(
        x=train_data[date_col], 
        y=train_data[target_col],
        mode='lines',
        name='トレーニングデータ',
        line=dict(color='blue')
    ))
    
    # テストデータ
    fig.add_trace(go.Scatter(
        x=test_data[date_col], 
        y=test_data[target_col],
        mode='lines',
        name='テストデータ',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='トレーニング・テストデータの分割',
        xaxis_title=date_col,
        yaxis_title=target_col,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # モデル固有のパラメータ設定
    st.subheader("モデルパラメータ設定")
    
    if selected_model == "Prophet":
        # EDA結果から推奨パラメータを取得
        seasonality_mode, n_changepoints = recommend_prophet_params(eda_results)
        
        # Prophet パラメータ設定
        col1, col2 = st.columns(2)
        with col1:
            seasonality_mode = st.selectbox(
                "季節性モード:", 
                ["additive", "multiplicative"], 
                index=0 if seasonality_mode == "additive" else 1
            )
        with col2:
            n_changepoints = st.number_input(
                "変化点の数:", 
                min_value=0, 
                max_value=50, 
                value=n_changepoints
            )
        
        # 季節性の有効化設定
        st.write("一般的な季節性の設定:")
        col1, col2, col3 = st.columns(3)
        with col1:
            yearly = st.checkbox("年次季節性", value=has_yearly_seasonality(eda_results))
        with col2:
            weekly = st.checkbox("週次季節性", value=has_weekly_seasonality(eda_results))
        with col3:
            daily = st.checkbox("日次季節性", value=has_daily_seasonality(eda_results))
        
        # EDAで検出された周期を表示
        periods = eda_results.get('periods', [])
        if periods:
            st.write("EDAで検出された周期成分:")
            custom_periods = {}
            
            # 既存の標準季節性と重複しない周期のみを表示
            filtered_periods = []
            for p in periods:
                # 年次、週次、日次の季節性と重複しないものを表示
                if not (350 <= p <= 380 or 6.5 <= p <= 7.5 or 23.5 <= p <= 24.5):
                    filtered_periods.append(p)
            
            # 周期の数に応じて列数を調整
            num_cols = min(3, len(filtered_periods))
            if filtered_periods:
                cols = st.columns(num_cols)
                
                for i, period in enumerate(filtered_periods):
                    period_name = f"周期 {period:.1f}"
                    if 25 <= period <= 35:
                        period_name += " (約1ヶ月)"
                    elif 85 <= period <= 95:
                        period_name += " (約3ヶ月)"
                    elif 170 <= period <= 190:
                        period_name += " (約6ヶ月)"
                    
                    with cols[i % num_cols]:
                        custom_periods[period] = st.checkbox(period_name, value=True)
            else:
                st.info("標準季節性以外の特別な周期は検出されませんでした。")
        
        model_params = {
            "seasonality_mode": seasonality_mode,
            "n_changepoints": n_changepoints,
            "yearly_seasonality": yearly,
            "weekly_seasonality": weekly,
            "daily_seasonality": daily,
            "custom_periods": custom_periods if 'custom_periods' in locals() else {}
        }
        
    elif selected_model == "LGBM":
        # EDA結果から推奨パラメータを取得
        num_leaves, max_depth, learning_rate = recommend_lgbm_params(eda_results)
        
        # LGBM パラメータ設定
        col1, col2, col3 = st.columns(3)
        with col1:
            num_leaves = st.number_input("葉の数:", min_value=2, max_value=256, value=num_leaves)
        with col2:
            max_depth = st.number_input("最大深さ:", min_value=-1, max_value=20, value=max_depth)
        with col3:
            learning_rate = st.number_input("学習率:", min_value=0.001, max_value=0.5, value=learning_rate, format="%.3f")
        
        # 特徴量エンジニアリングの設定
        st.write("時系列特徴量の生成:")
        col1, col2 = st.columns(2)
        with col1:
            lag_features = st.checkbox("ラグ特徴量", value=True)
        with col2:
            window_features = st.checkbox("ウィンドウ統計量", value=True)
        
        model_params = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "lag_features": lag_features,
            "window_features": window_features
        }
    
    # トレーニングボタン
    if st.button("モデルをトレーニング", use_container_width=True):
        with st.spinner(f"{selected_model}モデルをトレーニング中..."):
            # モデルのトレーニングと評価
            if selected_model == "Prophet":
                model, predictions, metrics = train_prophet_model(
                    train_data, test_data, date_col, target_col, model_params
                )
            elif selected_model == "LGBM":
                model, predictions, metrics = train_lgbm_model(
                    train_data, test_data, date_col, target_col, model_params
                )
            
            # 結果の表示
            st.success("モデルのトレーニングが完了しました！")
            
            # 評価指標の表示
            st.subheader("モデル評価結果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE (平均二乗誤差の平方根)", f"{metrics['rmse']:.4f}")
            with col2:
                st.metric("MAE (平均絶対誤差)", f"{metrics['mae']:.4f}")
            with col3:
                st.metric("R² (決定係数)", f"{metrics['r2']:.4f}")
            
            # 予測と実測値の比較プロット
            fig_comparison = plot_comparison(test_data[date_col], test_data[target_col], predictions)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # 残差プロット
            fig_residuals = plot_residuals(test_data[date_col], test_data[target_col], predictions)
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            # セッション状態に結果を保存
            st.session_state.trained_model = model
            st.session_state.selected_model = selected_model
            st.session_state.model_params = model_params
            st.session_state.evaluation_results = {
                "test_data": test_data,
                "predictions": predictions,
                "metrics": metrics
            }
            st.session_state.model_trained = True  # トレーニング完了フラグを追加
    
    # 予測ページへ進むボタン - 関数内に配置し、トレーニング完了フラグに基づいて表示
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        if st.button("予測ページへ進む", key="goto_forecast", use_container_width=True):
            st.session_state.page = 'forecast'
            st.rerun()

# モデル推奨関数
def recommend_model(eda_results):
    """EDA結果に基づいて最適なモデルを推奨する"""
    # 新しい実装や修正の提案
    # ...

# Prophetパラメータ推奨関数
def recommend_prophet_params(eda_results):
    """EDA結果に基づいてProphetのパラメータを推奨"""
    # デフォルト値
    seasonality_mode = "additive"
    n_changepoints = 25
    
    # 季節性の強さに基づいて季節性モードを設定
    seasonal = eda_results.get('seasonal', np.array([]))
    data = eda_results.get('data', np.array([]))
    
    if len(seasonal) > 0 and len(data) > 0:
        seasonal_sum = np.sum(seasonal, axis=1) if seasonal.ndim > 1 else seasonal
        seasonal_contrib = np.var(seasonal_sum) / np.var(data)
        
        # 季節性が強く、データが非負の場合は乗法的季節性を推奨
        if seasonal_contrib > 0.3 and np.all(data >= 0):
            seasonality_mode = "multiplicative"
        else:
            seasonality_mode = "additive"
    
    # トレンドの変化点の数を設定
    trend = eda_results.get('trend', np.array([]))
    if len(trend) > 0:
        # トレンドの変化の多さに基づいて変化点の数を調整
        trend_diff = np.diff(trend)
        sign_changes = np.sum(np.diff(np.signbit(trend_diff)))
        n_changepoints = max(5, min(25, int(sign_changes * 2)))
    
    return seasonality_mode, n_changepoints

# LGBMパラメータ推奨関数
def recommend_lgbm_params(eda_results):
    """EDA結果に基づいてLGBMのパラメータを推奨"""
    # デフォルト値
    num_leaves = 31
    max_depth = -1
    learning_rate = 0.1
    
    # 残差の複雑さに基づいてパラメータを調整
    resid = eda_results.get('resid', np.array([]))
    
    if len(resid) > 0:
        # 残差の複雑さに基づいてモデルの複雑さを調整
        resid_complexity = np.abs(np.diff(resid)).mean()
        
        if resid_complexity > 0.5:
            # 複雑な残差パターンの場合、より複雑なモデルを推奨
            num_leaves = 63
            max_depth = 8
            learning_rate = 0.05
        elif resid_complexity > 0.2:
            # 中程度の複雑さ
            num_leaves = 31
            max_depth = 6
            learning_rate = 0.1
        else:
            # シンプルなパターン
            num_leaves = 15
            max_depth = 4
            learning_rate = 0.15
    
    return num_leaves, max_depth, learning_rate

# 季節性の有無を判定する関数
def has_yearly_seasonality(eda_results):
    """年次季節性の有無を判定"""
    periods = eda_results.get('periods', [])
    return any(350 <= p <= 380 for p in periods)

def has_weekly_seasonality(eda_results):
    """週次季節性の有無を判定"""
    periods = eda_results.get('periods', [])
    return any(6.5 <= p <= 7.5 for p in periods)

def has_daily_seasonality(eda_results):
    """日次季節性の有無を判定"""
    periods = eda_results.get('periods', [])
    return any(23.5 <= p <= 24.5 for p in periods)

def train_prophet_model(train_data, test_data, date_col, target_col, params):
    """Prophetモデルのトレーニングと評価"""
    # Prophet用にデータ形式を変換
    train_df = train_data[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    test_df = test_data[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    
    # モデルの初期化
    model = Prophet(
        seasonality_mode=params["seasonality_mode"],
        n_changepoints=params["n_changepoints"],
        yearly_seasonality=params["yearly_seasonality"],
        weekly_seasonality=params["weekly_seasonality"],
        daily_seasonality=params["daily_seasonality"]
    )
    
    # EDAで検出された周期を追加
    if 'periods' in st.session_state.eda_results:
        periods = st.session_state.eda_results['periods']
        data_freq = st.session_state.data_frequency if 'data_frequency' in st.session_state else 'D'
        
        for period in periods:
            # 既存の季節性（年次、週次、日次）と重複しないものだけ追加
            if (not params["yearly_seasonality"] or not (350 <= period <= 380)) and \
               (not params["weekly_seasonality"] or not (6.5 <= period <= 7.5)) and \
               (not params["daily_seasonality"] or not (23.5 <= period <= 24.5)):
                
                # 周期の名前と詳細を設定
                period_name = f"custom_period_{int(period)}"
                fourier_order = min(5, max(3, int(period/10)))  # 周期の長さに応じてフーリエ次数を調整
                
                # 期間に意味のある名前をつける
                if 25 <= period <= 35:
                    period_name = "monthly"
                elif 85 <= period <= 95:
                    period_name = "quarterly"
                elif 170 <= period <= 190:
                    period_name = "half_yearly"
                
                # Prophetにカスタム周期を追加
                model.add_seasonality(
                    name=period_name,
                    period=period,
                    fourier_order=fourier_order,
                    mode=params["seasonality_mode"]
                )
    
    # モデルのトレーニング
    model.fit(train_df)
    
    # テストデータでの予測
    forecast = model.predict(test_df[['ds']])
    
    # 評価指標の計算
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    return model, y_pred, metrics

def train_lgbm_model(train_data, test_data, date_col, target_col, params):
    """LGBMモデルのトレーニングと評価"""
    # データリークを防ぐために全データを連結してから特徴量作成
    all_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    all_features = create_features(all_data, date_col, target_col, params)
    
    # トレーニングデータとテストデータを再分割
    train_features = all_features[:len(train_data)]
    test_features = all_features[len(train_data):]
    
    # トレーニングデータとテストデータの準備
    X_train = train_features.drop([target_col], axis=1)
    if date_col in X_train.columns:
        X_train = X_train.drop([date_col], axis=1)
    y_train = train_features[target_col]
    
    X_test = test_features.drop([target_col], axis=1)
    if date_col in X_test.columns:
        X_test = X_test.drop([date_col], axis=1)
    y_test = test_features[target_col]
    
    # 特徴量の型を確認し、数値型に変換
    numeric_columns = []
    for col in X_train.columns:
        try:
            # 数値型への変換を試みる
            X_train[col] = pd.to_numeric(X_train[col])
            X_test[col] = pd.to_numeric(X_test[col])
            numeric_columns.append(col)
        except:
            st.warning(f"列 '{col}' は数値型に変換できないため、除外されます。")
            continue
    
    # 数値型の列のみを選択
    X_train = X_train[numeric_columns]
    X_test = X_test[numeric_columns]
    
    # 特徴量が少なくとも1つあることを確認
    if X_train.shape[1] == 0:
        st.error("モデルトレーニングに有効な特徴量がありません。データを確認してください。")
        # 単純な特徴量を追加（時間的な順序を表す）
        X_train['index'] = np.arange(len(X_train))
        X_test['index'] = np.arange(len(X_test))
    
    # 検証データを作成（早期停止用）- データリーク防止
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # LGBMモデルの初期化
    model = LGBMRegressor(
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=1000  # 最大イテレーション数を増やし、早期停止に任せる
    )
    
    # モデルのトレーニング - テストデータではなく検証データで早期停止
    model.fit(
        X_tr, 
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lightgbm.early_stopping(stopping_rounds=10)],  # 正しい早期停止の構文
        eval_metric='rmse'  # 評価指標を明示的に指定
    )
    
    # テストデータでの予測
    y_pred = model.predict(X_test)
    
    # 評価指標の計算
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    return model, y_pred, metrics

def create_features(data, date_col, target_col, params):
    """時系列特徴量を生成する関数"""
    df = data.copy()
    
    # 日付列がdatetime型かチェックし、必要に応じて変換
    try:
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col])
        
        # 基本的な日付特徴量
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        
        # 追加の日付特徴量
        df['weekofyear'] = df[date_col].dt.isocalendar().week
        df['dayofyear'] = df[date_col].dt.dayofyear
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        # 時間関連の特徴量（データに時間情報がある場合）
        if df[date_col].dt.hour.nunique() > 1:
            df['hour'] = df[date_col].dt.hour
            df['minute'] = df[date_col].dt.minute
            
        # 季節性を捉えるための周期的特徴量
        # 月の周期性
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        # 曜日の周期性
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        # 日の周期性
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
    except (TypeError, ValueError) as e:
        st.warning(f"日付列からの特徴量生成で一部エラーが発生しました: {e}")
        # 基本的な特徴量生成を試みる
        try:
            # 年月日が別々のカラムとして存在する場合の対応
            if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
                pass  # 既にある場合は何もしない
            else:
                # インデックスを特徴量として使用
                df['index_feature'] = np.arange(len(df))
        except Exception as inner_e:
            st.error(f"特徴量生成に失敗しました: {inner_e}")
            df['index_feature'] = np.arange(len(df))
    
    # ラグ特徴量の生成
    if params["lag_features"]:
        # EDAから検出された周期に基づいてラグを選択
        lags = [1]  # 基本的な1日前のラグは常に含める
        
        # EDAから周期情報を取得
        if 'eda_results' in st.session_state and 'periods' in st.session_state.eda_results:
            periods = st.session_state.eda_results['periods']
            
            # 周期からラグを追加（小数点を四捨五入して整数化）
            for period in periods:
                lag = int(round(period))
                if lag > 1 and lag not in lags:  # 重複を避ける
                    lags.append(lag)
            
            # 重要な周期のハーモニクス（半分や倍）も考慮
            for period in periods:
                # 半周期
                half_period = int(round(period / 2))
                if half_period > 1 and half_period not in lags:
                    lags.append(half_period)
                
                # 2倍周期（あまり大きくなりすぎないように制限）
                double_period = int(round(period * 2))
                if double_period > 1 and double_period < 100 and double_period not in lags:
                    lags.append(double_period)
        
        # 周期がない場合や特定できなかった場合のデフォルト値
        if len(lags) <= 1:
            lags.extend([7, 14, 30])
            
        # ラグをソートして重複を除去
        lags = sorted(list(set(lags)))
        
        # ラグ特徴量を生成
        for lag in lags:
            if len(df) > lag:
                df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # 移動平均、標準偏差などの特徴量
    if params["window_features"]:
        windows = [7, 14, 30]
        
        # EDAから検出された周期に基づいてウィンドウサイズを追加
        if 'eda_results' in st.session_state and 'periods' in st.session_state.eda_results:
            periods = st.session_state.eda_results['periods']
            for period in periods:
                window = int(round(period))
                if window > 3 and window not in windows:  # 小さすぎるウィンドウは避ける
                    windows.append(window)
                    
        # 重複を除去してソート
        windows = sorted(list(set(windows)))
        
        for window in windows:
            if len(df) > window:
                df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean().shift(1)
                df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std().shift(1)
                df[f'rolling_min_{window}'] = df[target_col].rolling(window=window).min().shift(1)
                df[f'rolling_max_{window}'] = df[target_col].rolling(window=window).max().shift(1)
                df[f'rolling_median_{window}'] = df[target_col].rolling(window=window).median().shift(1)
    
    # 欠損値を0で埋める
    df = df.fillna(0)
    
    return df

def plot_comparison(x, y, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='実測値',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y_pred,
        mode='lines',
        name='予測値',
        line=dict(color='red')
    ))
    fig.update_layout(
        title='予測と実測値の比較',
        xaxis_title='日付',
        yaxis_title='値',
        template="plotly_white"
    )
    return fig

def plot_residuals(x, y, y_pred):
    residuals = y - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=residuals,
        mode='lines+markers',
        name='残差',
        line=dict(color='green')
    ))
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="black",
        annotation_text="理想的な残差", 
        annotation_position="bottom right"
    )
    fig.update_layout(
        title='残差プロット',
        xaxis_title='日付',
        yaxis_title='残差',
        template="plotly_white"
    )
    return fig
