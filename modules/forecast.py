import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import io
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def show_forecast_page():
    """将来予測を行うページ"""
    st.title("時系列データの将来予測")
    
    # モデルがトレーニングされているか確認
    if not ('model_trained' in st.session_state and st.session_state.model_trained):
        st.warning("予測モデルがまだトレーニングされていません。トレーニングページに戻ってモデルをトレーニングしてください。")
        if st.button("トレーニングページへ戻る", use_container_width=True):
            st.session_state.page = 'training'
            st.rerun()
        return
    
    # セッション状態からモデル情報を取得
    model = st.session_state.trained_model
    selected_model = st.session_state.selected_model
    model_params = st.session_state.model_params
    evaluation_results = st.session_state.evaluation_results
    
    # データ情報の取得
    data = st.session_state.clean_data
    date_col = st.session_state.date_col
    target_col = st.session_state.target_col
    
    # モデル評価結果の表示
    st.subheader("モデル評価結果")
    metrics = evaluation_results["metrics"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", f"{metrics['rmse']:.4f}")
    with col2:
        st.metric("MAE", f"{metrics['mae']:.4f}")
    with col3:
        st.metric("R²", f"{metrics['r2']:.4f}")
    
    # 予測設定
    st.subheader("予測期間の設定")
    
    # データの日付範囲を取得
    last_date = data[date_col].max()
    
    # 日付が文字列型の場合は変換する
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    # データ頻度の推定
    freq = estimate_frequency(data[date_col])
    
    # 予測期間の選択
    forecast_periods = st.slider(
        "予測期間（単位期間）:", 
        min_value=1, 
        max_value=365, 
        value=30,
        help="予測する未来の期間数"
    )
    
    # 予測期間の単位
    period_unit = st.selectbox(
        "期間の単位:", 
        ["日", "週", "月", "四半期", "年"],
        index=freq_to_index(freq)
    )
    
    # 予測期間の境界を表示
    future_date = calculate_future_date(last_date, forecast_periods, period_unit)
    st.info(f"予測期間: {last_date.strftime('%Y-%m-%d')} から {future_date.strftime('%Y-%m-%d')} まで")
    
    # 予測実行ボタン
    if st.button("予測を実行", use_container_width=True):
        with st.spinner("将来予測を計算中..."):
            if selected_model == "Prophet":
                future_df, forecast = forecast_with_prophet(
                    model, last_date, forecast_periods, period_unit, date_col, target_col
                )
            elif selected_model == "LGBM":
                future_df, _ = forecast_with_lgbm_static(
                    model, data, last_date, forecast_periods, period_unit, date_col, target_col, model_params
                )
            elif selected_model == "LGBM (実験的)":
                from .LGBMforecast_experimental import forecast_with_lgbm_experimental
                future_df, _ = forecast_with_lgbm_experimental(
                    model, data, last_date, forecast_periods, period_unit, date_col, target_col, model_params
                )
            
            # 予測結果の表示
            st.success("予測が完了しました！")
            
            # 予測チャートの表示
            st.subheader("予測結果")
            
            # 実データと予測を結合して表示
            fig = plot_forecast(data, future_df, date_col, target_col)
            st.plotly_chart(fig, use_container_width=True)
            
            # 予測データフレームの表示
            st.subheader("予測データテーブル")
            st.dataframe(future_df.head(20))
            
            # 予測結果のダウンロード
            download_forecast(future_df)

def estimate_frequency(date_series):
    """日付列から頻度を推定する"""
    if len(date_series) < 2:
        return 'D'  # デフォルトは日次
        
    # 日付列が日付型であることを確認し、必要に応じて変換
    try:
        date_series_dt = pd.to_datetime(date_series)
        # 日付の差分を計算
        diffs = date_series_dt.sort_values().diff().dt.days.dropna()
    except Exception:
        return 'D'  # 変換エラーの場合はデフォルト値を返す
    
    if diffs.empty:
        return 'D'
    
    # 最頻値を求める
    median_diff = diffs.median()
    
    if median_diff <= 1:
        return 'D'  # 日次
    elif 2 <= median_diff <= 3:
        return 'D'  # ほぼ日次とみなす
    elif 6 <= median_diff <= 8:
        return 'W'  # 週次
    elif 28 <= median_diff <= 31:
        return 'M'  # 月次
    elif 89 <= median_diff <= 92:
        return 'Q'  # 四半期
    elif median_diff >= 350:
        return 'Y'  # 年次
    else:
        return 'D'  # それ以外は日次とする

def freq_to_index(freq):
    """頻度から選択肢のインデックスを返す"""
    if freq == 'D':
        return 0  # 日
    elif freq == 'W':
        return 1  # 週
    elif freq == 'M':
        return 2  # 月
    elif freq == 'Q':
        return 3  # 四半期
    elif freq == 'Y':
        return 4  # 年
    else:
        return 0  # デフォルトは日

def calculate_future_date(last_date, periods, unit):
    """最終日から指定された期間だけ未来の日付を計算"""
    # last_dateが文字列の場合はdatetimeに変換
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    if unit == "日":
        return last_date + timedelta(days=periods)
    elif unit == "週":
        return last_date + timedelta(weeks=periods)
    elif unit == "月":
        # 月は約30日として計算
        return last_date + timedelta(days=periods * 30)
    elif unit == "四半期":
        # 四半期は約90日として計算
        return last_date + timedelta(days=periods * 90)
    elif unit == "年":
        # 年は約365日として計算
        return last_date + timedelta(days=periods * 365)
    else:
        return last_date + timedelta(days=periods)

def forecast_with_prophet(model, last_date, periods, unit, date_col, target_col):
    """Prophetモデルで将来予測を行う"""
    # 頻度を設定
    freq = 'D'
    if unit == "週":
        freq = 'W'
    elif unit == "月":
        freq = 'M'
    elif unit == "四半期":
        freq = 'Q'
    elif unit == "年":
        freq = 'Y'
    
    # 未来の日付リストを作成 - 学習データの終了日以降のデータを確実に生成するため、
    # last_dateの翌日から指定された期間分のデータを生成
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    # タイムゾーン情報を削除
    last_date = pd.to_datetime(last_date).tz_localize(None)
    
    # 強制的に予測用の未来日付を生成（make_future_dataframeではなく直接日付レンジを生成）
    if unit == "日":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    elif unit == "週":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='W')
    elif unit == "月":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='MS')
    elif unit == "四半期":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='QS')
    elif unit == "年":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='YS')
    
    # Prophetの予測用データフレームを作成
    future = pd.DataFrame({'ds': future_dates})
    
    # 予測実行
    forecast = model.predict(future)
    
    # 元のdatetime形式に戻す
    future_df = pd.DataFrame({
        date_col: forecast['ds'],
        target_col: forecast['yhat'],
        'yhat_lower': forecast['yhat_lower'],
        'yhat_upper': forecast['yhat_upper']
    })
    
    # 予測結果をデバッグ出力
    st.write(f"予測期間: {periods} {unit}")
    st.write(f"生成された予測データ数: {len(future_df)}")
    
    return future_df, forecast

def forecast_with_lgbm(model, data, last_date, periods, unit, date_col, target_col, model_params):
    """LGBMモデルで将来予測を行う"""
    # 将来の日付範囲を生成
    if unit == "日":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    elif unit == "週":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='W')
    elif unit == "月":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='MS')
    elif unit == "四半期":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='QS')
    elif unit == "年":
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='YS')
    
    # 将来の日付データフレームを作成
    future_df = pd.DataFrame({date_col: future_dates})
    
    # 特徴量を事前に生成（トレーニングデータのパターンを学習させる）
    history_data = data.copy()
    
    # モデルが使用した特徴量の名前を取得（カラム順序を合わせるため）
    feature_names = model.feature_name_
    
    # 予測結果を格納する配列
    predictions = np.array([])
    
    # 逐次的に予測を行う
    for i in range(len(future_df)):
        # 現在の日付のデータフレームを作成
        current_date = future_df[date_col].iloc[i]
        new_row = pd.DataFrame({date_col: [current_date]})
        
        # ターゲット変数に前回の予測値または最後の実データを設定
        if i == 0:
            new_row[target_col] = history_data[target_col].iloc[-1]  # 最初は最後の実測値
        else:
            new_row[target_col] = predictions[-1]  # それ以降は前回の予測値
        
        # 履歴データに新しい予測行を追加
        history_data = pd.concat([history_data, new_row]).reset_index(drop=True)
        
        # 特徴量を生成
        features_df = create_features(history_data, date_col, target_col, model_params)
        
        # 予測対象行の特徴量を取得
        pred_features = features_df.iloc[-1:].copy()
        
        # 予測に不要な列を削除
        X_pred = pred_features.drop([target_col], axis=1)
        if date_col in X_pred.columns:
            X_pred = X_pred.drop([date_col], axis=1)
        
        # 特徴量をモデルの学習時と同じ順序に並べる
        X_pred = ensure_feature_compatibility(X_pred, feature_names)
        
        # 予測実行
        this_prediction = model.predict(X_pred)[0]
        
        # 予測結果を配列に追加
        if i == 0:
            predictions = np.array([this_prediction])
        else:
            predictions = np.append(predictions, this_prediction)
        
        # 予測結果を履歴データに反映（次のステップの予測のため）
        history_data.iloc[-1, history_data.columns.get_loc(target_col)] = this_prediction
    
    # 最終的な予測結果を将来の日付データフレームに追加
    future_df[target_col] = predictions
    
    # 信頼区間を簡易的に計算
    mae = st.session_state.evaluation_results["metrics"]["mae"]
    future_df['yhat_lower'] = future_df[target_col] - mae * 1.96
    future_df['yhat_upper'] = future_df[target_col] + mae * 1.96
    
    return future_df, None

def forecast_with_lgbm_static(model, data, last_date, forecast_periods, period_unit, date_col, target_col, model_params):
    """LGBMモデルで将来予測を行う（スケール調整版）"""
    # period_unitを適切な頻度文字列に変換
    freq_map = {
        "日": "D",
        "週": "W",
        "月": "M",
        "四半期": "Q",
        "年": "Y"
    }
    freq = freq_map.get(period_unit, "D")

    # 最後の実績値の日付を確認
    last_actual_date = pd.to_datetime(last_date)
    
    # 将来の日付を生成（最後の実績値の翌日から）
    future_dates = pd.date_range(
        start=last_actual_date + pd.Timedelta(days=1),
        periods=forecast_periods,
        freq=freq
    )
    
    # モデルの特徴量名を取得
    feature_names = model.feature_name_
    
    # 予測値の安定化
    predictions = []
    last_actual_value = data[target_col].iloc[-1]
    history_data = data.copy()
    
    for i in range(len(future_dates)):
        current_date = future_dates[i]
        
        # 特徴量生成
        features = create_features_experimental(history_data, date_col, target_col, model_params)
        pred_features = features.iloc[-1:]
        
        # 特徴量の互換性を確保
        pred_features = ensure_feature_compatibility(pred_features, feature_names)
        
        # 差分を予測
        pred_diff = model.predict(pred_features)[0]
        
        # 差分から実際の値を計算
        if i == 0:
            pred_value = last_actual_value + pred_diff
        else:
            pred_value = predictions[-1] + pred_diff
            
        predictions.append(pred_value)
        
        # 履歴データに新しい予測を追加
        new_row = pd.DataFrame({
            date_col: [current_date],
            target_col: [pred_value]
        })
        history_data = pd.concat([history_data, new_row]).reset_index(drop=True)

    # 予測結果のデータフレーム作成
    future_df = pd.DataFrame({
        date_col: future_dates,
        target_col: predictions
    })
    
    # 信頼区間の計算も元に戻す
    mae = st.session_state.evaluation_results["metrics"]["mae"]
    future_df['yhat_lower'] = future_df[target_col] - mae * 1.96
    future_df['yhat_upper'] = future_df[target_col] + mae * 1.96

    return future_df, None

def ensure_feature_compatibility(X_pred, feature_names):
    """予測データの特徴量をモデルの学習時の特徴量と互換性を持たせる"""
    # 数値型に変換
    for col in X_pred.columns:
        if X_pred[col].dtype == 'object':
            try:
                X_pred[col] = pd.to_numeric(X_pred[col])
            except:
                X_pred[col] = 0  # 変換できない場合は0埋め
    
    # モデルが期待する特徴量をすべて含むように調整
    missing_cols = set(feature_names) - set(X_pred.columns)
    for col in missing_cols:
        X_pred[col] = 0  # 欠損している特徴量は0で埋める
    
    # 余分な特徴量を削除
    extra_cols = set(X_pred.columns) - set(feature_names)
    if extra_cols:
        X_pred = X_pred.drop(extra_cols, axis=1)
    
    # モデルが期待する順序に並べ替え
    X_pred = X_pred[feature_names]
    
    return X_pred

def create_features(data, date_col, target_col, params):
    """時系列特徴量を生成する関数（training.pyからコピー）"""
    df = data.copy()
    
    # 日付列がdatetime型かチェックし、必要に応じて変換
    try:
        if df[date_col].dtype != 'datetime64[ns]':
            df[date_col] = pd.to_datetime(df[date_col])
        
        # 日付から特徴量を生成
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
    except (TypeError, ValueError) as e:
        # 日付変換ができない場合は警告
        pass
    
    # ラグ特徴量の生成
    if params["lag_features"]:
        lags = [1, 7, 14, 30]
        for lag in lags:
            if len(df) > lag:
                df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # 移動平均、標準偏差などの特徴量
    if params["window_features"]:
        windows = [7, 14, 30]
        for window in windows:
            if len(df) > window:
                df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean().shift(1)
                df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std().shift(1)
    
    # 欠損値を0で埋める
    df = df.fillna(0)
    
    return df

def create_features_experimental(data, date_col, target_col, params, debug_output=False):
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

def plot_forecast(data, future_df, date_col, target_col):
    """実データと予測データのプロット（ホバー修正版）"""
    fig = go.Figure()
    
    # データの前処理
    data = data.copy()
    future_df = future_df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    future_df[date_col] = pd.to_datetime(future_df[date_col])
    
    # 最後の実績値の日付を取得
    last_actual_date = data[date_col].max()
    
    # スケーリングパラメータが存在する場合、実測値のみを元のスケールに戻す
    if 'scaling_params' in st.session_state and target_col in st.session_state.scaling_params:
        scaling_info = st.session_state.scaling_params[target_col]
        if scaling_info['method'] == 'minmax':
            min_val = scaling_info['min']
            max_val = scaling_info['max']
            # 実測値のみスケーリングを戻す
            data[target_col] = data[target_col] * (max_val - min_val) + min_val
        elif scaling_info['method'] == 'standard':
            mean_val = scaling_info['mean']
            std_val = scaling_info['std']
            # 実測値のみスケーリングを戻す
            data[target_col] = data[target_col] * std_val + mean_val
        elif scaling_info['method'] == 'robust':
            median_val = scaling_info['median']
            iqr = scaling_info['iqr']
            # 実測値のみスケーリングを戻す
            data[target_col] = data[target_col] * iqr + median_val
            
        # デバッグ情報を出力
        with st.expander("スケーリング詳細情報", expanded=False):
            st.write("プロット時のスケーリング情報:")
            st.write(f"実測値の範囲: {data[target_col].min():.4f} - {data[target_col].max():.4f}")
            st.write(f"予測値の範囲: {future_df[target_col].min():.4f} - {future_df[target_col].max():.4f}")
    
    # 実績値のプロット（最後の実績値までのみ）
    actual_mask = data[date_col] <= last_actual_date
    fig.add_trace(go.Scatter(
        x=data[actual_mask][date_col],
        y=data[actual_mask][target_col],
        mode='lines',
        name='実績値',
        line=dict(color='blue'),
        customdata=data[actual_mask][target_col],
        hovertemplate='%{customdata:.4f}<extra>実績値</extra>'
    ))
    
    # 予測値のプロット（最後の実績値の翌日以降のみ）
    future_mask = future_df[date_col] > last_actual_date
    future_plot_df = future_df[future_mask].copy()
    
    if not future_plot_df.empty:
        fig.add_trace(go.Scatter(
            x=future_plot_df[date_col],
            y=future_plot_df[target_col],
            mode='lines',
            name='予測値',
            line=dict(color='red'),
            customdata=future_plot_df[target_col],
            hovertemplate='%{customdata:.4f}<extra>予測値</extra>'
        ))
        
        # 信頼区間の表示（予測期間のみ）
        fig.add_trace(go.Scatter(
            x=future_plot_df[date_col].tolist() + future_plot_df[date_col].tolist()[::-1],
            y=future_plot_df['yhat_upper'].tolist() + future_plot_df['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95%予測区間',
            hoverinfo='none'  # 信頼区間のホバー情報を完全に無効化
        ))
    
    # 予測開始位置の縦線
    fig.add_shape(
        type="line",
        x0=last_actual_date,
        y0=0,
        x1=last_actual_date,
        y1=1,
        yref="paper",
        line=dict(
            color="green",
            width=2,
            dash="dash",
        )
    )
    
    # 予測開始位置のアノテーション
    fig.add_annotation(
        x=last_actual_date,
        y=1,
        yref="paper",
        text="予測開始",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    # レイアウトの更新
    fig.update_layout(
        title='時系列予測結果',
        xaxis_title='日付',
        yaxis_title=target_col,
        template="plotly_white",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def download_forecast(forecast_df):
    """予測結果のダウンロード機能"""
    # CSVとして出力
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    
    # Excelとして出力
    excel_buffer = io.BytesIO()
    forecast_df.to_excel(excel_buffer, index=False)
    excel_data = excel_buffer.getvalue()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="予測結果をCSVダウンロード",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            label="予測結果をExcelダウンロード",
            data=excel_data,
            file_name="forecast_results.xlsx",
            mime="application/vnd.ms-excel",
            use_container_width=True
        )

def train_lgbm_model(train_data, test_data, date_col, target_col, params):
    # データの基本統計量を確認
    st.write("学習データの基本統計量:")
    st.write(train_data[target_col].describe())
    
    # データリークを防ぐために全データを連結してから特徴量作成
    all_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    
    # 差分を計算
    all_data['target_diff'] = all_data[target_col].diff()
    
    # 特徴量生成
    features = create_features_experimental(all_data, date_col, target_col, params)
    
    # トレーニングデータとテストデータを再分割
    train_features = features[:len(train_data)]
    test_features = features[len(train_data):]
    
    # 差分を予測対象にする
    X_train = train_features.drop(['target_diff', target_col], axis=1)
    y_train = train_features['target_diff'].fillna(0)
    
    X_test = test_features.drop(['target_diff', target_col], axis=1)
    y_test = test_features['target_diff'].fillna(0)
    
    # モデルの初期化とトレーニング
    model = LGBMRegressor(
        num_leaves=31,
        max_depth=6,
        learning_rate=0.01,
        n_estimators=2000,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    
    # モデルのフィット
    model.fit(X_train, y_train)
    
    # テストデータでの予測
    y_pred = model.predict(X_test)
    
    # 差分から実際の値に変換
    y_test_actual = test_data[target_col].values
    y_pred_actual = np.zeros_like(y_pred)
    y_pred_actual[0] = train_data[target_col].iloc[-1] + y_pred[0]
    for i in range(1, len(y_pred)):
        y_pred_actual[i] = y_pred_actual[i-1] + y_pred[i]
    
    # 評価指標の計算
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    # 特徴量の重要度を確認（モデルフィット後に実行可能）
    feature_importance = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.write("重要な特徴量:", feature_importance.head(10))
    
    return model, y_pred_actual, metrics