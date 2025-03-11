"""
アプリケーション設定と定数
"""

# アプリ設定
APP_TITLE = "Mellon"
APP_DESCRIPTION = "Integrated Time-Series Analysis Platform"

# モデル設定
AVAILABLE_MODELS = {
    'ARIMA': 'ARIMA (自己回帰和分移動平均)',
    'SARIMA': 'SARIMA (季節性ARIMA)',
    'Prophet': 'Prophet (Facebook製予測モデル)',
    'Random Forest': 'ランダムフォレスト',
    'Linear Regression': '線形回帰'
}

# デフォルトパラメータ
DEFAULT_PARAMS = {
    'ARIMA': {
        'p': 1,
        'd': 1, 
        'q': 1
    },
    'SARIMA': {
        'p': 1, 
        'd': 1, 
        'q': 1,
        'P': 1, 
        'D': 0, 
        'Q': 1,
        'm': 12
    },
    'Prophet': {
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto'
    },
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': 10
    },
    'Linear Regression': {}
}

# 予測期間のデフォルト設定
DEFAULT_FORECAST_PERIODS = {
    'hourly': 24,    # 24時間
    'daily': 30,     # 30日
    'weekly': 12,    # 12週
    'monthly': 12,   # 12か月
    'quarterly': 4,  # 4四半期
    'yearly': 3      # 3年
}

# グラフのカラーパレット
COLORS = {
    'actual': '#1f77b4',    # 青
    'predicted': '#ff7f0e',  # オレンジ
    'forecast': '#2ca02c',   # 緑
    'lower_bound': '#d62728', # 赤
    'upper_bound': '#9467bd'  # 紫
}

# 評価指標
METRICS = [
    'RMSE',  # 二乗平均平方根誤差
    'MAE',   # 平均絶対誤差
    'MAPE',  # 平均絶対パーセント誤差
    'R2'     # 決定係数
] 