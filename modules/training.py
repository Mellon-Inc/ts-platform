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
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
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
    # 推奨モデルがNoneの場合のデフォルト値を設定
    if recommended_model is None:
        recommended_model = "Prophet"  # デフォルトモデルをProphetに設定
    
    # モデル選択オプション
    model_options = {
        "Prophet": "Facebookが開発した時系列予測モデル - 複雑な季節性と休日効果を扱えます",
        "LGBM": "Light Gradient Boosting Machine - 非線形パターンの予測に強い機械学習モデル",
        "LGBM (実験的)": "改良版Light Gradient Boosting Machine - より高度な特徴量エンジニアリングと最適化"
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
        
    elif selected_model in ["LGBM", "LGBM (実験的)"]:
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
            
        # 実験的LGBMの場合の追加特徴量
        if selected_model == "LGBM (実験的)":
            st.write("追加の特徴量:")
            col1, col2 = st.columns(2)
            with col1:
                use_trend_feature = st.checkbox("トレンド特徴量", value=False, help="EDAで分解されたトレンド成分を特徴量として使用")
            with col2:
                minmax_features = st.checkbox("最大値・最小値差分特徴量", value=False, help="検出された周期ごとの最大値・最小値との差分を特徴量として使用")

            # 周期と振幅の閾値設定
            st.write("周期・振幅の閾値設定:")
            col1, col2 = st.columns(2)
            with col1:
                period_threshold = st.number_input("周期の閾値:", min_value=1, max_value=100, value=50, help="この値を超える周期のみを使用")
            with col2:
                amplitude_threshold = st.number_input("振幅の閾値:", min_value=1, max_value=100, value=50, help="この値を超える振幅のみを使用")

            # 相関係数の閾値設定
            st.write("相関係数の閾値設定:")
            correlation_threshold = st.slider(
                "相関係数の閾値:",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
                help="この値以上の相関係数を持つ説明変数を除外します"
            )

            # Optunaの設定
            st.write("ハイパーパラメータ最適化:")
            use_optuna = st.checkbox('Optunaによる自動チューニングを使用', value=False)
            if use_optuna:
                n_trials = st.slider('Optunaの試行回数', min_value=10, max_value=200, value=50)
        else:
            # 実験的LGBMでない場合はデフォルト値を設定
            use_trend_feature = False
            minmax_features = False
            use_optuna = False
            n_trials = None
            period_threshold = 50
            amplitude_threshold = 50
            correlation_threshold = 1.0  # 相関フィルタリングを無効化
        
        model_params = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "lag_features": lag_features,
            "window_features": window_features,
            "use_trend_feature": use_trend_feature if selected_model == "LGBM (実験的)" else False,
            "minmax_features": minmax_features if selected_model == "LGBM (実験的)" else False,
            "period_threshold": period_threshold,
            "amplitude_threshold": amplitude_threshold,
            "correlation_threshold": correlation_threshold
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
            elif selected_model == "LGBM (実験的)":
                from .LGBMtraining_experimental import train_lgbm_model_experimental
                
                # データを結合して is_train フラグを追加
                all_data = pd.concat([
                    train_data.assign(is_train=True),
                    test_data.assign(is_train=False)
                ]).reset_index(drop=True)
                
                # 日付列名を 'ds' に変更
                all_data = all_data.rename(columns={date_col: 'ds'})
                
                # モデルのトレーニングと予測
                model, predictions, metrics = train_lgbm_model_experimental(
                    df=all_data,
                    target_col=target_col,
                    model_params=model_params,
                    use_optuna=use_optuna,
                    n_trials=n_trials
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
    # デフォルトはProphet
    recommended_model = "Prophet"
    
    # EDA結果から適切なモデルを判断
    # 残差の複雑さ、季節性、周期性などに基づいて判断
    trend = eda_results.get('trend', np.array([]))
    seasonal = eda_results.get('seasonal', np.array([]))
    resid = eda_results.get('resid', np.array([]))
    
    if len(resid) > 0:
        # 残差の複雑さを評価
        resid_complexity = np.abs(np.diff(resid)).mean()
        
        # 複雑な残差パターンがある場合はLGBMを推奨
        if resid_complexity > 0.2:
            recommended_model = "LGBM"
    
    # 季節性の強さに基づいて判断
    if len(seasonal) > 0 and len(eda_results.get('data', [])) > 0:
        seasonal_sum = np.sum(seasonal, axis=1) if seasonal.ndim > 1 else seasonal
        seasonal_contrib = np.var(seasonal_sum) / np.var(eda_results['data'])
        
        # 強い季節性があるならProphetを推奨
        if seasonal_contrib > 0.3:
            recommended_model = "Prophet"
    
    return recommended_model

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
    
    # タイムゾーン情報を削除
    train_df['ds'] = pd.to_datetime(train_df['ds']).dt.tz_localize(None)
    test_df['ds'] = pd.to_datetime(test_df['ds']).dt.tz_localize(None)
    
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
    
    # 特徴量生成
    features = create_features(all_data, date_col, target_col, params)
    
    # トレーニングデータとテストデータを再分割
    train_features = features[:len(train_data)]
    test_features = features[len(train_data):]
    
    # 直接の値を予測対象にする
    X_train = train_features.copy()
    y_train = train_data[target_col]
    
    X_test = test_features.copy()
    y_test = test_data[target_col]
    
    # モデルのパラメータを調整
    model = LGBMRegressor(
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=2000,     # より多くの学習機会
        min_child_samples=20,  # より細かいパターンを学習
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,       # 再現性のため
        importance_type='gain', # 特徴量重要度の計算方法
        metric='rmse'          # 評価指標
    )
    
    # コールバックを設定
    callbacks = [
        early_stopping(stopping_rounds=100, verbose=True),  # 早期停止
        log_evaluation(period=100)                          # ログ出力
    ]
    
    # モデルのトレーニング
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=callbacks
    )
    
    # 予測と評価
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

def create_features(data, date_col, target_col, params, debug_output=False):
    df = data.copy()
    
    # 差分を計算（新規追加）
    df['target_diff'] = df[target_col].diff()
    
    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col])

    # 基本的な日付特徴量（既存のまま）
    df['year'] = df[date_col].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)
    df['quarter_sin'] = np.sin(2 * np.pi * df[date_col].dt.quarter / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df[date_col].dt.quarter / 4)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / 365)

    # 重要: 日付列を削除
    df = df.drop(columns=[date_col])

    # ターゲット列が存在するときだけラグ・移動平均を生成
    if target_col in df.columns:
        target_data = df[target_col].copy()
        target_diff_data = df['target_diff'].copy()  # 差分データも保存
        df = df.drop(columns=[target_col])  # ターゲット列を特徴量から除外
        
        if params.get("lag_features", False):
            for lag in [1, 3, 5, 7, 14, 30]:
                if len(target_data) > lag:
                    df[f'lag_{lag}'] = target_data.shift(lag)
                    # 差分のラグも追加
                    df[f'diff_lag_{lag}'] = target_diff_data.shift(lag)

        if params.get("window_features", False):
            windows = [7, 14, 30, 60, 180, 240, 360]
            for window in windows:
                if len(target_data) > window:
                    # 通常の特徴量
                    df[f'rolling_mean_{window}'] = target_data.rolling(window=window).mean()
                    df[f'rolling_std_{window}'] = target_data.rolling(window=window).std()
                    # 差分の特徴量を追加
                    df[f'diff_rolling_mean_{window}'] = target_diff_data.rolling(window=window).mean()
                    df[f'diff_rolling_std_{window}'] = target_diff_data.rolling(window=window).std()

    # デバッグ出力をフラグで制御
    if debug_output:
        if 'eda_results' in st.session_state and 'periods' in st.session_state.eda_results:
            st.write("検出された周期:", st.session_state.eda_results['periods'])
        st.write("生成された特徴量:", df.columns.tolist())
        st.write("特徴量サンプル:", df.head())

    # 数値特徴量のスケーリング
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_features:
        if col != target_col:  # ターゲット変数はスケーリングしない
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
    
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
