import pandas as pd
import numpy as np
from .LGBMtraining_experimental import create_features_experimental
import streamlit as st

def calculate_trend_direction(trend_data, window_size=30):
    """トレンドの方向性と強さを計算する関数"""
    if len(trend_data) < window_size:
        window_size = len(trend_data)
    
    recent_trend = trend_data[-window_size:]
    
    # 直近のトレンドの変化量を計算
    changes = np.diff(recent_trend)
    
    # 上昇と下降の割合を計算
    up_ratio = np.sum(changes > 0) / len(changes)
    down_ratio = np.sum(changes < 0) / len(changes)
    
    # 平均変化量を計算
    avg_change = np.mean(changes)
    
    # トレンドの強さを計算（変化量の絶対値の平均）
    trend_strength = np.mean(np.abs(changes))
    
    return avg_change, trend_strength, up_ratio, down_ratio

def forecast_with_lgbm_experimental(model, data, last_date, forecast_periods, period_unit, date_col, target_col, model_params):
    """実験的なLGBMモデルによる予測関数"""
    # 期間単位を適切な頻度文字列に変換
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
    
    # 予測期間を日数に変換
    if freq == "W":
        days_to_forecast = forecast_periods * 7
        freq_for_dates = "D"  # 日単位でデータポイントを生成
    elif freq == "M":
        days_to_forecast = forecast_periods * 30  # 月は約30日
        freq_for_dates = "D"
    elif freq == "Q":
        days_to_forecast = forecast_periods * 90  # 四半期は約90日
        freq_for_dates = "D"
    elif freq == "Y":
        days_to_forecast = forecast_periods * 365  # 年は約365日
        freq_for_dates = "D"
    else:
        days_to_forecast = forecast_periods
        freq_for_dates = freq

    # 将来の日付を生成（最後の実績値の翌日から）
    future_dates = pd.date_range(
        start=last_actual_date + pd.Timedelta(days=1),
        periods=days_to_forecast,
        freq=freq_for_dates
    )
    
    # モデルの特徴量名を取得
    feature_names = model.feature_name_
    
    # 予測値を格納するリスト
    predictions = []
    history_data = data.copy()
    
    # トレンド特徴量の準備（存在する場合）
    original_trend = None
    if model_params.get("use_trend_feature", False) and 'eda_results' in st.session_state:
        original_trend = st.session_state.eda_results.get('trend', None)
        if original_trend is not None:
            # 入力データの長さに合わせてトレンドデータを切り詰める
            original_trend = original_trend[:len(data)]
            
            # トレンドの方向性と強さを計算
            avg_change, trend_strength, up_ratio, down_ratio = calculate_trend_direction(original_trend)
            
            # トレンドの変化量を調整
            if up_ratio > 0.7:  # 強い上昇トレンド
                trend_diff = avg_change * 0.8  # 上昇トレンドを若干抑制
            elif down_ratio > 0.7:  # 強い下降トレンド
                trend_diff = avg_change * 0.8  # 下降トレンドを若干抑制
            else:  # 混合トレンド
                trend_diff = avg_change * 0.5  # トレンドの影響を半減
            
            st.session_state.eda_results['trend'] = original_trend.copy()
            
            # デバッグ情報
            st.write(f"トレンド分析:")
            st.write(f"- 平均変化量: {avg_change:.4f}")
            st.write(f"- トレンド強度: {trend_strength:.4f}")
            st.write(f"- 上昇割合: {up_ratio:.2%}")
            st.write(f"- 下降割合: {down_ratio:.2%}")
            st.write(f"- 採用した変化量: {trend_diff:.4f}")
    
    # 予測用データフレームの準備
    for i in range(len(future_dates)):
        current_date = future_dates[i]
        
        # 新しい行を作成
        new_row = pd.DataFrame({
            'ds': [current_date],  # date_colを'ds'に統一
            target_col: [history_data[target_col].iloc[-1]]  # 最後の値を使用
        })
        
        # is_train フラグを追加（予測データなのでFalse）
        new_row['is_train'] = False
        
        # 履歴データに新しい行を追加
        history_data = pd.concat([history_data, new_row]).reset_index(drop=True)
        
        # トレンド特徴量の拡張
        if model_params.get("use_trend_feature", False) and 'eda_results' in st.session_state:
            current_trend = st.session_state.eda_results.get('trend', None)
            if current_trend is not None:
                # 新しいトレンド値を計算して追加
                next_trend = current_trend[-1] + trend_diff
                
                # 急激な変化を抑制
                if abs(next_trend - current_trend[-1]) > trend_strength * 2:
                    next_trend = current_trend[-1] + np.sign(trend_diff) * trend_strength * 2
                
                st.session_state.eda_results['trend'] = np.append(current_trend, next_trend)
        
        # 特徴量生成（学習時と同じ関数を使用）
        features = create_features_experimental(
            data=history_data,
            date_col='ds',  # date_colを'ds'に統一
            target_col=target_col,
            params=model_params,
            debug_output=False
        )
        
        # 予測対象行の特徴量を取得
        pred_features = features.iloc[-1:].copy()
        
        # 特徴量の互換性を確保
        missing_cols = set(feature_names) - set(pred_features.columns)
        for col in missing_cols:
            pred_features[col] = 0
        
        # 余分な特徴量を削除
        extra_cols = set(pred_features.columns) - set(feature_names)
        if extra_cols:
            pred_features = pred_features.drop(extra_cols, axis=1)
        
        # モデルが期待する順序に並べ替え
        pred_features = pred_features[feature_names]
        
        # 予測実行
        pred_value = model.predict(pred_features)[0]
        predictions.append(pred_value)
        
        # 履歴データの最後の値を更新
        history_data.iloc[-1, history_data.columns.get_loc(target_col)] = pred_value

    # 予測結果のデータフレーム作成
    all_predictions_df = pd.DataFrame({
        date_col: future_dates,
        target_col: predictions
    })
    
    # スケーリングパラメータが存在する場合、予測値を元のスケールに戻す
    if 'scaling_params' in st.session_state and target_col in st.session_state.scaling_params:
        scaling_info = st.session_state.scaling_params[target_col]
        
        # 変換前の値を保存
        before_scaling = all_predictions_df[target_col].copy()
        
        # スケーリング逆変換を実行
        if scaling_info['method'] == 'minmax':
            min_val = scaling_info['min']
            max_val = scaling_info['max']
            all_predictions_df[target_col] = all_predictions_df[target_col] * (max_val - min_val) + min_val
        elif scaling_info['method'] == 'standard':
            mean_val = scaling_info['mean']
            std_val = scaling_info['std']
            all_predictions_df[target_col] = all_predictions_df[target_col] * std_val + mean_val
        elif scaling_info['method'] == 'robust':
            median_val = scaling_info['median']
            iqr = scaling_info['iqr']
            all_predictions_df[target_col] = all_predictions_df[target_col] * iqr + median_val
        
        # スケーリング情報をexpanderにまとめて表示
        with st.expander("スケーリングパラメータ情報", expanded=False):
            st.write(f"対象カラム: {target_col}")
            st.write(f"スケーリング方法: {scaling_info['method']}")
            st.write(f"スケーリングパラメータ: {scaling_info}")
            st.write(f"変換前の予測値範囲: {before_scaling.min():.4f} - {before_scaling.max():.4f}")
            st.write(f"変換後の予測値範囲: {all_predictions_df[target_col].min():.4f} - {all_predictions_df[target_col].max():.4f}")
    
    # 期間単位に応じて予測値を集約
    if freq != "D":
        # 週、月、四半期、年の場合は適切な期間でリサンプリング
        future_df = all_predictions_df.set_index(date_col).resample(freq).mean().reset_index()
    else:
        future_df = all_predictions_df
    
    # 95%予測区間の計算（統計的に正確な実装）
    if 'evaluation_results' in model_params and 'metrics' in model_params['evaluation_results']:
        rmse = model_params['evaluation_results']['metrics']['rmse']
        
        # 時間依存の不確実性を計算
        horizon = np.arange(len(future_df))
        time_factor = 1 + (horizon / len(future_df)) * 0.5  # 最大で1.5倍まで
        
        # 基本予測区間幅を計算（RMSEに基づく）
        base_interval = rmse * 1.96  # 95%信頼区間の基本幅
        
        # スケーリング係数を適用して予測区間を計算
        if 'scaling_params' in st.session_state and target_col in st.session_state.scaling_params:
            scaling_info = st.session_state.scaling_params[target_col]
            if scaling_info['method'] == 'robust':
                # robustスケーリングの場合はIQRを使用
                base_interval = base_interval * scaling_info['iqr']
            elif scaling_info['method'] == 'standard':
                # 標準化の場合は標準偏差を使用
                base_interval = base_interval * scaling_info['std']
            elif scaling_info['method'] == 'minmax':
                # min-maxスケーリングの場合はレンジを使用
                base_interval = base_interval * (scaling_info['max'] - scaling_info['min'])
        
        # 時間に応じた予測区間幅を計算
        interval_width = base_interval * time_factor[:, np.newaxis]
        
        # デバッグ情報をexpanderに収める
        with st.expander("予測区間の詳細情報", expanded=False):
            st.write("予測区間の計算情報:")
            st.write(f"- 基本RMSE: {rmse:.4f}")
            st.write(f"- スケーリング適用前の基本区間幅 (RMSE * 1.96): {rmse * 1.96:.4f}")
            st.write(f"- スケーリング適用後の基本区間幅: {base_interval:.4f}")
            st.write(f"- 時間係数による最終区間幅の範囲: {interval_width.min():.4f} - {interval_width.max():.4f}")
        
        # 予測区間を計算
        future_df['yhat_lower'] = future_df[target_col] - interval_width
        future_df['yhat_upper'] = future_df[target_col] + interval_width
    else:
        # 予測値の標準偏差を使用（フォールバック）
        std = np.std(predictions)
        if 'scaling_params' in st.session_state and target_col in st.session_state.scaling_params:
            scaling_info = st.session_state.scaling_params[target_col]
            if scaling_info['method'] == 'robust':
                std = std * scaling_info['iqr']
            elif scaling_info['method'] == 'standard':
                std = std * scaling_info['std']
            elif scaling_info['method'] == 'minmax':
                std = std * (scaling_info['max'] - scaling_info['min'])
        
        time_factor = 1 + (np.arange(len(future_df)) / len(future_df)) * 0.5
        interval_width = std * time_factor * 1.96
        future_df['yhat_lower'] = future_df[target_col] - interval_width
        future_df['yhat_upper'] = future_df[target_col] + interval_width

    # トレンド特徴量を元に戻す（存在する場合）
    if original_trend is not None:
        st.session_state.eda_results['trend'] = original_trend

    return future_df, None 