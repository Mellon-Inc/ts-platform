import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
from typing import Dict, Any, List, Union, Optional

def generate_forecast(model: Any, 
                     df: pd.DataFrame, 
                     target_col: str, 
                     date_col: str, 
                     model_type: str, 
                     forecast_periods: int, 
                     confidence_level: float = 0.95) -> Dict:
    """
    将来予測を生成
    
    Parameters:
    ----------
    model : object
        トレーニングされたモデル
    df : pandas.DataFrame
        評価用データ
    target_col : str
        対象変数の列名
    date_col : str
        日付列の名前
    model_type : str
        モデルの種類
    forecast_periods : int
        予測する期間数
    confidence_level : float
        信頼区間のレベル (0から1の間)
        
    Returns:
    -------
    dict
        予測結果
    """
    # データの準備
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    
    # 最後の日付
    last_date = df[date_col].max()
    
    # データ頻度を検出
    from utils import detect_frequency
    frequency = detect_frequency(df[date_col])
    
    # 将来の日付を生成
    future_dates = generate_future_dates(last_date, forecast_periods, frequency)
    
    # モデルタイプに応じた予測
    if model_type == 'ARIMA':
        # ARIMAモデルの予測
        forecast = model.forecast(steps=forecast_periods)
        
        # 信頼区間の計算
        pred_ci = model.get_forecast(steps=forecast_periods).conf_int(alpha=1-confidence_level)
        
        # 結果をデータフレームにまとめる
        forecast_df = pd.DataFrame({
            date_col: future_dates,
            'forecast': forecast,
            'lower_bound': pred_ci.iloc[:, 0],
            'upper_bound': pred_ci.iloc[:, 1]
        })
        
    elif model_type == 'SARIMA':
        # SARIMAモデルの予測
        forecast = model.forecast(steps=forecast_periods)
        
        # 信頼区間の計算
        pred_ci = model.get_forecast(steps=forecast_periods).conf_int(alpha=1-confidence_level)
        
        # 結果をデータフレームにまとめる
        forecast_df = pd.DataFrame({
            date_col: future_dates,
            'forecast': forecast,
            'lower_bound': pred_ci.iloc[:, 0],
            'upper_bound': pred_ci.iloc[:, 1]
        })
        
    elif model_type == 'Prophet':
        try:
            # 将来の日付を含むデータフレームを作成
            future = pd.DataFrame({
                'ds': future_dates
            })
            
            # 予測
            forecast = model.predict(future)
            
            # 結果をデータフレームにまとめる
            forecast_df = pd.DataFrame({
                date_col: future_dates,
                'forecast': forecast['yhat'],
                'lower_bound': forecast['yhat_lower'],
                'upper_bound': forecast['yhat_upper']
            })
        except Exception as e:
            st.error(f"Prophet予測中にエラーが発生しました: {str(e)}")
            return None
        
    elif model_type in ['Random Forest', 'XGBoost', 'Linear Regression']:
        try:
            # 機械学習モデルの場合
            ml_model = model['model']
            feature_names = model['features']
            
            # 将来のデータフレームを作成
            future_df = pd.DataFrame({date_col: future_dates})
            
            # 時間特徴量を追加
            from modules.features import add_time_features
            future_df = add_time_features(future_df, date_col)
            
            # ラグ特徴量と移動平均特徴量
            # 注: これらの特徴量は過去データが必要なため、
            # 将来予測のためには手動で計算してデータを追加する必要があります
            
            # トレーニングデータの最後の値をコピー
            for feature in feature_names:
                if feature not in future_df.columns:
                    # 特徴量が見つからない場合
                    if 'lag' in feature.lower():
                        # ラグ特徴量の場合、最後のターゲット値を使用
                        lag_n = int(feature.split('_')[-1])
                        last_values = df[target_col].tail(lag_n).values
                        for i in range(min(lag_n, forecast_periods)):
                            if i < len(last_values):
                                future_df.loc[i, feature] = last_values[-(i+1)]
                            else:
                                future_df.loc[i, feature] = future_df.loc[i-1, feature]
                    
                    elif 'rolling' in feature.lower():
                        # 移動平均特徴量の場合、過去の平均値を使用
                        window = int(feature.split('_')[-1])
                        future_df[feature] = df[target_col].tail(window).mean()
                    
                    else:
                        # その他の特徴量は0で初期化
                        future_df[feature] = 0
            
            # 不足している特徴量を0で埋める
            for feature in feature_names:
                if feature not in future_df.columns:
                    future_df[feature] = 0
            
            # 予測
            future_X = future_df[feature_names]
            forecast = ml_model.predict(future_X)
            
            # 不確実性の推定（ランダムフォレストの場合）
            if model_type == 'Random Forest':
                # 各木の予測を取得
                trees_predictions = np.array([tree.predict(future_X) for tree in ml_model.estimators_])
                
                # 標準偏差を計算
                std = np.std(trees_predictions, axis=0)
                
                # 信頼区間の計算（正規分布を仮定）
                from scipy.stats import norm
                z_score = norm.ppf((1 + confidence_level) / 2)
                
                lower_bound = forecast - z_score * std
                upper_bound = forecast + z_score * std
            else:
                # 他のモデルでは単純なヒューリスティックを使用
                # 例: 過去データの平均絶対誤差から信頼区間を推定
                mae = np.mean(np.abs(model['y_test'] - ml_model.predict(model['X_test'])))
                z_score = norm.ppf((1 + confidence_level) / 2)
                
                lower_bound = forecast - z_score * mae
                upper_bound = forecast + z_score * mae
            
            # 結果をデータフレームにまとめる
            forecast_df = pd.DataFrame({
                date_col: future_dates,
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
        except Exception as e:
            st.error(f"機械学習モデルの予測中にエラーが発生しました: {str(e)}")
            return None
    
    else:
        st.error(f"未対応のモデルタイプです: {model_type}")
        return None
    
    # 過去データと予測を結合
    historical_df = df[[date_col, target_col]].copy()
    historical_df['type'] = '実測値'
    
    forecast_df_with_type = forecast_df.copy()
    forecast_df_with_type[target_col] = forecast_df['forecast']
    forecast_df_with_type['type'] = '予測値'
    
    result_df = pd.concat([
        historical_df,
        forecast_df_with_type[[date_col, target_col, 'type']]
    ])
    
    # 結果を辞書にまとめる
    result = {
        'historical': historical_df,
        'forecast': forecast_df,
        'combined': result_df
    }
    
    return result

def generate_future_dates(last_date: datetime, 
                         periods: int, 
                         frequency: str) -> List[datetime]:
    """
    将来の日付を生成
    
    Parameters:
    ----------
    last_date : datetime
        最後の日付
    periods : int
        生成する期間数
    frequency : str
        時系列データの頻度
        
    Returns:
    -------
    list
        将来の日付のリスト
    """
    future_dates = []
    current_date = last_date
    
    for _ in range(periods):
        if frequency == 'hourly':
            current_date += timedelta(hours=1)
        elif frequency == 'daily':
            current_date += timedelta(days=1)
        elif frequency == 'weekly':
            current_date += timedelta(weeks=1)
        elif frequency == 'monthly':
            # 月を加算
            year = current_date.year
            month = current_date.month + 1
            
            if month > 12:
                year += 1
                month = 1
            
            day = min(current_date.day, get_days_in_month(year, month))
            current_date = datetime(year, month, day)
        elif frequency == 'quarterly':
            # 四半期を加算
            year = current_date.year
            month = current_date.month + 3
            
            if month > 12:
                year += month // 12
                month = month % 12
                if month == 0:
                    month = 12
                    year -= 1
            
            day = min(current_date.day, get_days_in_month(year, month))
            current_date = datetime(year, month, day)
        elif frequency == 'yearly':
            # 年を加算
            current_date = datetime(current_date.year + 1, current_date.month, current_date.day)
        else:
            # デフォルトは日次
            current_date += timedelta(days=1)
        
        future_dates.append(current_date)
    
    return future_dates

def get_days_in_month(year: int, month: int) -> int:
    """
    指定された年月の日数を取得
    
    Parameters:
    ----------
    year : int
        年
    month : int
        月
        
    Returns:
    -------
    int
        その月の日数
    """
    if month == 2:  # 2月
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):  # うるう年
            return 29
        else:
            return 28
    elif month in [4, 6, 9, 11]:  # 4, 6, 9, 11月
        return 30
    else:  # その他の月
        return 31

def plot_forecast(forecast_results: Dict) -> None:
    """
    予測結果をプロット
    
    Parameters:
    ----------
    forecast_results : dict
        予測結果を含む辞書
    """
    if forecast_results is None:
        st.error("予測結果がありません")
        return
    
    # 過去データと予測を取得
    historical_df = forecast_results['historical']
    forecast_df = forecast_results['forecast']
    combined_df = forecast_results['combined']
    
    # 日付列とターゲット列の名前を取得
    date_col = historical_df.columns[0]
    target_col = historical_df.columns[1]
    
    # Plotlyを使用した予測プロット
    fig = go.Figure()
    
    # 過去データをプロット
    fig.add_trace(go.Scatter(
        x=historical_df[date_col],
        y=historical_df[target_col],
        mode='lines',
        name='実測値',
        line=dict(color='blue')
    ))
    
    # 予測をプロット
    fig.add_trace(go.Scatter(
        x=forecast_df[date_col],
        y=forecast_df['forecast'],
        mode='lines',
        name='予測値',
        line=dict(color='red')
    ))
    
    # 信頼区間をプロット
    fig.add_trace(go.Scatter(
        x=forecast_df[date_col],
        y=forecast_df['upper_bound'],
        mode='lines',
        name='上限',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df[date_col],
        y=forecast_df['lower_bound'],
        mode='lines',
        name='下限',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        showlegend=False
    ))
    
    # グラフのレイアウト設定
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
    
    # 予測データテーブル
    st.subheader("予測値")
    
    # 日付フォーマットを整える
    forecast_display = forecast_df.copy()
    forecast_display[date_col] = forecast_display[date_col].dt.strftime('%Y-%m-%d')
    
    st.dataframe(forecast_display)
    
    # CSVダウンロード機能
    from utils import create_downloadable_csv
    create_downloadable_csv(forecast_df, "forecast_results.csv")