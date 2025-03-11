import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import io

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
                future_df, forecast = forecast_with_lgbm(
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
    
    # 未来の日付リストを作成
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # 予測実行
    forecast = model.predict(future)
    
    # 元のdatetime形式に戻す
    forecast_df = pd.DataFrame({
        date_col: forecast['ds'],
        target_col: forecast['yhat'],
        'yhat_lower': forecast['yhat_lower'],
        'yhat_upper': forecast['yhat_upper']
    })
    
    # 予測期間のみ抽出（日付を確実に比較できるようにする）
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    # 予測結果をデバッグ出力
    st.write(f"予測総レコード数: {len(forecast_df)}")
    st.write(f"最終日付: {last_date}")
    
    # 未来データのみを抽出
    future_df = forecast_df[forecast_df[date_col] > last_date].reset_index(drop=True)
    
    # 抽出結果をデバッグ出力
    st.write(f"将来予測レコード数: {len(future_df)}")
    
    if len(future_df) == 0:
        st.warning("予測期間が正しく設定されていない可能性があります。")
        # 簡易的な対応として、最終日付の翌日から予測期間分のデータを生成
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
        
        # 強制的に予測データを生成
        future_data = model.predict(pd.DataFrame({'ds': future_dates}))
        future_df = pd.DataFrame({
            date_col: future_data['ds'],
            target_col: future_data['yhat'],
            'yhat_lower': future_data['yhat_lower'],
            'yhat_upper': future_data['yhat_upper']
        })
    
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

def plot_forecast(data, future_df, date_col, target_col):
    """実データと予測データのプロット"""
    fig = go.Figure()
    
    # 過去の実データ
    fig.add_trace(go.Scatter(
        x=data[date_col],
        y=data[target_col],
        mode='lines',
        name='実績値',
        line=dict(color='blue')
    ))
    
    # 予測値
    fig.add_trace(go.Scatter(
        x=future_df[date_col],
        y=future_df[target_col],
        mode='lines',
        name='予測値',
        line=dict(color='red')
    ))
    
    # 信頼区間
    fig.add_trace(go.Scatter(
        x=future_df[date_col].tolist() + future_df[date_col].tolist()[::-1],
        y=future_df['yhat_upper'].tolist() + future_df['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95%予測区間'
    ))
    
    # 縦線で実績値と予測値の境界を表示
    last_date = data[date_col].max()
    # 文字列型の場合は変換
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    # add_vline()の代わりにadd_shape()を使用
    fig.add_shape(
        type="line",
        x0=last_date,
        y0=0,
        x1=last_date,
        y1=1,
        yref="paper",
        line=dict(
            color="green",
            width=2,
            dash="dash",
        )
    )
    
    # アノテーションを追加
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="予測開始",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.update_layout(
        title='時系列予測結果',
        xaxis_title='日付',
        yaxis_title=target_col,
        template="plotly_white",
        hovermode='x unified'
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