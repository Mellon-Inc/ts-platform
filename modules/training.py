import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple

def train_model(df: pd.DataFrame, 
                target_col: str, 
                date_col: str, 
                model_type: str, 
                params: Dict = None, 
                test_size: float = 0.2) -> Any:
    """
    選択されたモデルをトレーニング
    
    Parameters:
    ----------
    df : pandas.DataFrame
        トレーニングデータ
    target_col : str
        対象変数の列名
    date_col : str
        日付列の名前
    model_type : str
        モデルの種類
    params : dict, optional
        モデルのパラメータ
    test_size : float, optional
        テストデータの割合
        
    Returns:
    -------
    object
        トレーニングされたモデル
    """
    # データを準備
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    
    # デフォルトのパラメータ
    if params is None:
        params = {}
    
    # モデルのトレーニング
    if model_type == 'ARIMA':
        # ARIMAのデフォルトパラメータ
        p = params.get('p', 1)
        d = params.get('d', 1)
        q = params.get('q', 1)
        
        # モデルのトレーニング
        model = ARIMA(df[target_col], order=(p, d, q))
        model_fit = model.fit()
        
        return model_fit
    
    elif model_type == 'SARIMA':
        # SARIMAのデフォルトパラメータ
        p = params.get('p', 1)
        d = params.get('d', 1)
        q = params.get('q', 1)
        P = params.get('P', 1)
        D = params.get('D', 0)
        Q = params.get('Q', 1)
        m = params.get('m', 12)
        
        # モデルのトレーニング
        model = SARIMAX(
            df[target_col], 
            order=(p, d, q), 
            seasonal_order=(P, D, Q, m)
        )
        model_fit = model.fit(disp=False)
        
        return model_fit
    
    elif model_type == 'Prophet':
        try:
            from prophet import Prophet
            
            # Prophetの日付とターゲット列の要件に対応
            prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
            
            # Prophetモデルのパラメータ
            yearly_seasonality = params.get('yearly_seasonality', 'auto')
            weekly_seasonality = params.get('weekly_seasonality', 'auto')
            daily_seasonality = params.get('daily_seasonality', 'auto')
            seasonality_mode = params.get('seasonality_mode', 'additive')
            
            # モデルのトレーニング
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                seasonality_mode=seasonality_mode
            )
            
            model.fit(prophet_df)
            
            return model
        except ImportError:
            st.error("Prophetがインストールされていません。以下のコマンドでインストールしてください：`pip install prophet`")
            return None
        except Exception as e:
            st.error(f"Prophetモデルのトレーニング中にエラーが発生しました: {str(e)}")
            return None
    
    elif model_type == 'XGBoost':
        try:
            import xgboost as xgb
            
            # 機械学習モデルの場合は、特徴量エンジニアリングが必要
            from modules.features import add_time_features, add_lag_features, add_rolling_features
            
            # 特徴量を追加
            featured_df = df.copy()
            featured_df = add_time_features(featured_df, date_col)
            featured_df = add_lag_features(featured_df, target_col)
            featured_df = add_rolling_features(featured_df, target_col)
            
            # 欠損値を含む行を削除
            featured_df = featured_df.dropna()
            
            # 特徴量と目的変数を分離
            X = featured_df.drop([target_col, date_col], axis=1)
            y = featured_df[target_col]
            
            # トレーニングデータとテストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            # XGBoostパラメータ
            xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 6),
                'learning_rate': params.get('learning_rate', 0.1),
                'subsample': params.get('subsample', 0.8),
                'colsample_bytree': params.get('colsample_bytree', 0.8),
                'random_state': 42
            }
            
            # モデルのトレーニング
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            
            return {
                'model': model,
                'features': X.columns.tolist(),
                'X_test': X_test,
                'y_test': y_test,
                'date_test': featured_df.loc[X_test.index, date_col]
            }
        except ImportError:
            st.error("XGBoostがインストールされていません。以下のコマンドでインストールしてください：`pip install xgboost`")
            return None
        except Exception as e:
            st.error(f"XGBoostモデルのトレーニング中にエラーが発生しました: {str(e)}")
            st.info("MacOSユーザーの場合: ターミナルで `brew install libomp` を実行してOpenMPランタイムをインストールしてください。")
            return None
    
    elif model_type in ['Random Forest', 'Linear Regression']:
        # 機械学習モデルの場合は、特徴量エンジニアリングが必要
        from modules.features import add_time_features, add_lag_features, add_rolling_features
        
        # 特徴量を追加
        featured_df = df.copy()
        featured_df = add_time_features(featured_df, date_col)
        featured_df = add_lag_features(featured_df, target_col)
        featured_df = add_rolling_features(featured_df, target_col)
        
        # 欠損値を含む行を削除
        featured_df = featured_df.dropna()
        
        # 特徴量と目的変数を分離
        X = featured_df.drop([target_col, date_col], axis=1)
        y = featured_df[target_col]
        
        # トレーニングデータとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # モデルの選択
        if model_type == 'Random Forest':
            model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=42
            )
        else:  # Linear Regression
            model = LinearRegression()
        
        # モデルのトレーニング
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'features': X.columns.tolist(),
            'X_test': X_test,
            'y_test': y_test,
            'date_test': featured_df.loc[X_test.index, date_col]
        }
    
    else:
        st.error(f"未対応のモデルタイプです: {model_type}")
        return None 