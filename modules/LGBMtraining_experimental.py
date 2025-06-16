import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .LGBMtuning_experimental import tune_lgbm_parameters
import lightgbm
from tqdm import tqdm

# 警告カウント用のグローバル変数
warning_counter = 0
MAX_WARNINGS = 5

def create_features_experimental(data, date_col, target_col, params, debug_output=False):
    """実験的な特徴量エンジニアリング関数
    
    特徴量の種類：
    1. 基本的な日付特徴量
    2. 検出された周期に基づく特徴量
        - 主要な周期のラグ特徴量
        - 主要な周期のウィンドウ統計量
        - 主要な周期の三角関数特徴量
        - 主要な周期の最大値・最小値差分特徴量
    3. トレンド特徴量（オプション）
    4. EDAで検出された重要な外部特徴量
    """
    df = data.copy()
    
    # is_train列を保持
    is_train = None
    if 'is_train' in df.columns:
        is_train = df['is_train'].copy()
    
    # 相関係数に基づくフィルタリング
    if 'eda_results' in st.session_state and 'target_correlations' in st.session_state.eda_results:
        target_correlations = st.session_state.eda_results['target_correlations']
        correlation_threshold = params.get('correlation_threshold', 1.0)
        
        # 相関係数が閾値以上の列を特定
        high_corr_cols = []
        for col, corr in target_correlations.items():
            if col in df.columns and abs(corr) >= correlation_threshold:
                high_corr_cols.append(col)
        
        if debug_output:
            with st.expander("相関係数フィルタリング情報", expanded=False):
                st.write(f"相関係数閾値: {correlation_threshold}")
                if high_corr_cols:
                    st.write("除外される説明変数:")
                    for col in high_corr_cols:
                        st.write(f"- {col} (相関係数: {abs(target_correlations[col]):.3f})")
                else:
                    st.write("除外される説明変数はありません")
        
        # 高相関の列を除外
        if high_corr_cols:
            df = df.drop(columns=high_corr_cols)
            if debug_output:
                st.write(f"高相関の{len(high_corr_cols)}個の変数を除外しました")
    
    # 日付型への変換とタイムゾーン情報の削除
    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    
    # 基本的な日付特徴量を生成
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    #df['dayofweek_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
    #df['dayofweek_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)
    df['quarter'] = df[date_col].dt.quarter
    
    # トレンド特徴量の追加（UIで選択可能）
    if params.get("use_trend_feature", False):
        if 'eda_results' in st.session_state and 'trend' in st.session_state.eda_results:
            trend = st.session_state.eda_results['trend']
            if len(trend) != len(df):
                st.warning(f"トレンドデータの長さ ({len(trend)}) が入力データの長さ ({len(df)}) と一致しません。トレンド特徴量は使用されません。")
            else:
                df['trend'] = trend
                if debug_output:
                    st.write("トレンド特徴量を追加しました")
    
    if debug_output:
        # 進捗状況をシンプルに表示
        progress_text = st.empty()  # プレースホルダーを作成
        progress_text.write("特徴量を生成中...")
    
    # 特徴量生成のロジック
    feature_counts = {
        'ラグ特徴量': 0,
        'ウィンドウ統計量': 0,
        '最大最小差分特徴量': 0,
        '三角関数特徴量': 0
    }
    
    if 'eda_results' in st.session_state and 'periods' in st.session_state.eda_results:
        # 周期と振幅の閾値を取得
        period_threshold = params.get('period_threshold', float('inf'))
        amplitude_threshold = params.get('amplitude_threshold', float('inf'))
        
        # 周期と振幅のペアを取得
        periods_and_amplitudes = st.session_state.eda_results.get('periods_and_amplitudes', [])
        
        # 閾値に基づいて周期をフィルタリング
        filtered_periods = []
        for period, amplitude in periods_and_amplitudes:
            # 周期と振幅の両方が閾値を超える場合のみ採用
            if period > period_threshold and amplitude > amplitude_threshold:
                filtered_periods.append(period)
        
        periods = sorted([int(round(p)) for p in filtered_periods])
        
        if debug_output:
            with st.expander("周期フィルタリング情報", expanded=False):
                st.write(f"周期閾値: {period_threshold}")
                st.write(f"振幅閾値: {amplitude_threshold}")
                st.write(f"フィルタリング前の周期数: {len(periods_and_amplitudes)}")
                st.write(f"フィルタリング後の周期数: {len(periods)}")
                st.write("※ 周期と振幅の両方が閾値を超える成分のみを使用")
        
        min_date = df[date_col].min()
        df['days_from_start'] = (df[date_col] - min_date).dt.total_seconds() / (24 * 60 * 60)
        
        for period in periods:
            if target_col in df.columns:
                target_data = df[target_col].copy()
                
                # 各特徴量の生成（カウントのみ更新）
                if params.get("lag_features", False) and len(df) > period:
                    df[f'lag_{period}'] = target_data.shift(period)
                    feature_counts['ラグ特徴量'] += 1
                
                if params.get("window_features", False) and len(df) > period:
                    df[f'rolling_mean_{period}'] = target_data.rolling(window=period).mean().shift(1)
                    df[f'rolling_std_{period}'] = target_data.rolling(window=period).std().shift(1)
                    feature_counts['ウィンドウ統計量'] += 2
                
                if params.get("minmax_features", False) and len(df) > period:
                    rolling_max = target_data.rolling(window=period).max()
                    rolling_min = target_data.rolling(window=period).min()
                    df[f'diff_from_max_{period}'] = target_data - rolling_max
                    df[f'diff_from_min_{period}'] = target_data - rolling_min
                    feature_counts['最大最小差分特徴量'] += 2
                
                cycle = 2 * np.pi * df['days_from_start'] / period
                df[f'period_{period}_sin'] = np.sin(cycle)
                df[f'period_{period}_cos'] = np.cos(cycle)
                feature_counts['三角関数特徴量'] += 2
    
    # 不要な列を削除
    columns_to_drop = [date_col, target_col, 'days_from_start']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    if debug_output:
        # 進捗表示を消去
        progress_text.empty()
        
        # 最終的な特徴量の要約のみを表示
        total_features = len(df.columns) - (1 if 'is_train' in df.columns else 0)
        
        # 特徴量数が0でない項目のみを表示
        feature_summary = [f"{name}: {count}個" for name, count in feature_counts.items() if count > 0]
        if feature_summary:
            st.write("生成された特徴量:")
            st.write(f"- {' / '.join(feature_summary)}")
            st.write(f"総特徴量数: {total_features}")
    
    # is_train列を復元
    if is_train is not None:
        df['is_train'] = is_train
    
    return df

def custom_warning_callback():
    """警告回数をカウントするコールバック関数"""
    global warning_counter
    
    def _callback(env):
        global warning_counter
        message = env.evaluation_result_list[0][1]
        if "No further splits with positive gain" in str(message):
            warning_counter += 1
            if warning_counter >= MAX_WARNINGS:
                st.write(f"\n'No further splits'警告が{MAX_WARNINGS}回検出されました。")
                st.write(f"学習を早期終了します。")
                raise lightgbm.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    return _callback

def train_lgbm_model_experimental(df, target_col, model_params, use_optuna=False, n_trials=None):
    """実験的なLGBMモデルのトレーニング関数"""
    global warning_counter
    warning_counter = 0  # カウンターをリセット
    
    # データの準備
    train_mask = df['is_train']
    test_mask = ~df['is_train']
    
    # 特徴量生成パラメータの設定
    feature_params = {
        "lag_features": model_params.get("lag_features", False),
        "window_features": model_params.get("window_features", False),
        "use_trend_feature": model_params.get("use_trend_feature", False),
        "minmax_features": model_params.get("minmax_features", False),
        "period_threshold": model_params.get("period_threshold", float('inf')),
        "amplitude_threshold": model_params.get("amplitude_threshold", float('inf')),
        "correlation_threshold": model_params.get("correlation_threshold", 1.0)  # 相関係数閾値を追加
    }
    
    # 特徴量生成（目的変数を除外）
    feature_df = create_features_experimental(
        data=df,
        date_col='ds',
        target_col=target_col,
        params=feature_params,
        debug_output=True
    )
    
    # 目的変数を除外した特徴量を取得
    feature_columns = [col for col in feature_df.columns 
                      if col not in [target_col, 'is_train', 'ds']]
    
    # 特徴量の確認と表示（エクスパンダー内に配置）
    with st.expander("使用する特徴量を表示", expanded=False):
        for col in feature_columns:
            st.write(f"- {col}")
    
    if len(feature_columns) == 0:
        st.error("使用可能な特徴量が見つかりません。")
        return None, None, None
    
    # 訓練データとテストデータの準備
    X_train = feature_df[train_mask][feature_columns]
    y_train = df[train_mask][target_col]
    X_test = feature_df[test_mask][feature_columns]
    y_test = df[test_mask][target_col]
    
    # データ型の確認と変換
    datetime_cols = X_train.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        st.warning(f"日付型の列が含まれています: {datetime_cols}")
        for col in datetime_cols:
            X_train[col] = X_train[col].astype(float)
            X_test[col] = X_test[col].astype(float)
    
    # デバッグ情報の表示（エクスパンダー内に配置）
    with st.expander("データ分割情報を表示", expanded=False):
        st.write(f"- 訓練データ数: {len(X_train)}")
        st.write(f"- テストデータ数: {len(X_test)}")
        st.write(f"- 特徴量数: {len(feature_columns)}")
    
    # モデルパラメータの設定
    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42,
        'num_leaves': model_params.get('num_leaves', 31),
        'max_depth': model_params.get('max_depth', -1),
        'learning_rate': model_params.get('learning_rate', 0.1),
        'n_estimators': 2000,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'importance_type': 'gain',
        'verbose': -1
    }
    
    # パラメータ情報の表示（エクスパンダー内に配置）
    with st.expander("モデルパラメータを表示", expanded=False):
        st.write(f"- 葉の数 (num_leaves): {base_params['num_leaves']}")
        st.write(f"- 木の深さ (max_depth): {base_params['max_depth']}")
        st.write(f"- 学習率 (learning_rate): {base_params['learning_rate']}")
    
    if use_optuna and n_trials:
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_text.write('チューニング中...')
        
        # プログレスバー更新用のコールバック
        def progress_callback(study, trial):
            progress = (len(study.trials) / n_trials)
            progress_bar.progress(progress)
            progress_text.write(f'チューニング中... ({len(study.trials)}/{n_trials} 試行完了)')
        
        try:
            base_params = tune_lgbm_parameters(
                X_train, y_train,
                X_test, y_test,
                base_params, n_trials,
                progress_callback  # コールバックを追加
            )
            progress_bar.empty()
            progress_text.empty()
        except lightgbm.callback.EarlyStopException:
            progress_bar.empty()
            progress_text.empty()
            st.write("Optunaチューニング中に警告による早期終了")
            if warning_counter >= MAX_WARNINGS:
                st.write("警告回数が上限に達したため、チューニングを終了します。")
                return None, None, None
    
    # モデルのトレーニング
    model = LGBMRegressor(**base_params)
    callbacks = [
        custom_warning_callback(),
        early_stopping(stopping_rounds=100),
        log_evaluation(period=100, show_stdv=True)
    ]
    
    # categorical_featureの自動検出
    categorical_features = [col for col in feature_columns if ('曜日' in col or 'weekday' in col.lower() or 'dayofweek' in col.lower())]
    
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=callbacks,
            categorical_feature=categorical_features if categorical_features else None
        )
    except lightgbm.callback.EarlyStopException:
        st.write("No further splits警告による早期終了")
    
    st.write(f"\n検出された'No further splits'警告の回数: {warning_counter}回")
    
    # 予測と評価
    y_pred = model.predict(X_test)

    # 実測値と予測値をターミナルに縦並びで出力
    print("実測値：")
    for t in y_test:
        print(t)
    print("\n予測値：")
    for p in y_pred:
        print(p)
    
    # 評価指標の計算
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 結果の表示
    st.write('\nモデル評価結果:')
    st.write(f'RMSE: {rmse:.4f}')
    st.write(f'MAE: {mae:.4f}')
    st.write(f'R²: {r2:.4f}')
    
    # 特徴量の重要度を計算
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    })
    
    # 重要度を正規化（パーセンテージに変換）
    total_importance = feature_importance['importance'].sum()
    feature_importance['importance'] = (feature_importance['importance'] / total_importance) * 100
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    st.write('\n特徴量の重要度（上位10件）:')
    st.write('※ 値は全特徴量の重要度に対する割合（%）')
    st.write(feature_importance.head(10))
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    return model, y_pred, metrics 