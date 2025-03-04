import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Tuple, Union

def evaluate_model(model: Any, 
                  df: pd.DataFrame, 
                  target_col: str, 
                  date_col: str, 
                  model_type: str) -> Dict:
    """
    トレーニングされたモデルを評価
    
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
        
    Returns:
    -------
    dict
        評価指標
    """
    # 結果を格納する辞書
    results = {}
    
    # モデルタイプに応じた評価
    if model_type in ['ARIMA', 'SARIMA']:
        # ARIMA/SARIMAモデルの場合
        
        # 予測
        predictions = model.predict(start=0, end=len(df)-1)
        
        # 実測値
        actual = df[target_col].values
        
        # 日付
        dates = df[date_col].values
        
    elif model_type == 'Prophet':
        # Prophetモデルの場合
        prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
        forecast = model.predict(prophet_df)
        
        # 予測値、実測値、日付
        predictions = forecast['yhat'].values
        actual = prophet_df['y'].values
        dates = prophet_df['ds'].values
        
    elif model_type in ['Random Forest', 'XGBoost', 'Linear Regression']:
        # 機械学習モデルの場合
        ml_model = model['model']
        features = model['features']
        
        # テストデータ
        X_test = model['X_test']
        y_test = model['y_test']
        
        # 予測
        predictions = ml_model.predict(X_test)
        
        # 実測値と日付
        actual = y_test.values
        dates = model['date_test'].values
    
    else:
        st.error(f"未対応のモデルタイプです: {model_type}")
        return {}
    
    # 評価指標の計算
    results['y_true'] = actual
    results['y_pred'] = predictions
    results['dates'] = dates
    
    # RMSE (Root Mean Squared Error)
    results['rmse'] = np.sqrt(mean_squared_error(actual, predictions))
    
    # MAE (Mean Absolute Error)
    results['mae'] = mean_absolute_error(actual, predictions)
    
    # MAPE (Mean Absolute Percentage Error)
    # ゼロ除算を避ける
    mask = actual != 0
    results['mape'] = np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
    
    # R² (決定係数)
    results['r2'] = r2_score(actual, predictions)
    
    return results

def display_evaluation_metrics(evaluation_results: Dict) -> None:
    """
    評価指標を表示
    
    Parameters:
    ----------
    evaluation_results : dict
        評価結果を含む辞書
    """
    if not evaluation_results:
        st.error("評価結果がありません")
        return
    
    # 評価指標の表を作成
    metrics_df = pd.DataFrame({
        'メトリクス': ['RMSE (二乗平均平方根誤差)', 'MAE (平均絶対誤差)', 'MAPE (平均絶対パーセント誤差)', 'R² (決定係数)'],
        '値': [
            evaluation_results['rmse'],
            evaluation_results['mae'],
            f"{evaluation_results['mape']:.2f}%",
            evaluation_results['r2']
        ]
    })
    
    st.dataframe(metrics_df) 