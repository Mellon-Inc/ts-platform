import streamlit as st
import optuna
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from lightgbm import early_stopping, log_evaluation

def plot_optimization_history(study):
    """Optunaの最適化履歴をプロットする"""
    fig = go.Figure()
    
    # 目的関数の値の履歴
    fig.add_trace(go.Scatter(
        x=list(range(len(study.trials))),
        y=[t.value for t in study.trials],
        mode='markers',
        name='Trial value'
    ))
    
    # ベストスコアの履歴
    best_values = [min([t.value for t in study.trials[:i+1]]) for i in range(len(study.trials))]
    fig.add_trace(go.Scatter(
        x=list(range(len(study.trials))),
        y=best_values,
        mode='lines',
        name='Best value'
    ))
    
    fig.update_layout(
        title='Optimization History',
        xaxis_title='Trial number',
        yaxis_title='Objective value (RMSE)',
        showlegend=True
    )
    
    return fig

def objective(trial, X_train, y_train, X_test, y_test, base_params):
    """Optunaの目的関数"""
    # ベースパラメータをコピー
    params = base_params.copy()
    
    # チューニング対象のパラメータを設定
    params.update({
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    })
    
    # モデルのトレーニング
    model = LGBMRegressor(**params)
    callbacks = [
        early_stopping(stopping_rounds=100),
        log_evaluation(period=100, show_stdv=True)
    ]
    
    model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             callbacks=callbacks)
    
    # 予測と評価
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rmse

def tune_lgbm_parameters(X_train, y_train, X_test, y_test, base_params, n_trials=100, progress_callback=None):
    """LightGBMのパラメータをOptunaでチューニング"""
    # random_stateを設定してstudy作成
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)  # 再現性のために固定シード値を設定
    )
    
    # 目的関数を部分適用
    objective_with_data = lambda trial: objective(
        trial, X_train, y_train, X_test, y_test, base_params
    )
    
    # 最適化の実行（コールバック付き）
    study.optimize(objective_with_data, n_trials=n_trials, callbacks=[progress_callback] if progress_callback else None)
    
    # 最適化の結果を表示（エクスパンダー内に配置）
    with st.expander("チューニング結果の詳細を表示", expanded=False):
        st.write('Best trial:')
        st.write(f'  RMSE: {study.best_trial.value:.4f}')
        st.write('  Params:')
        for key, value in study.best_trial.params.items():
            st.write(f'    {key}: {value}')
        
        # 最適化の履歴をプロット
        fig = plot_optimization_history(study)
        st.plotly_chart(fig)
    
    # ベストパラメータを返す（ユーザー指定のパラメータを優先）
    best_params = base_params.copy()
    best_params.update(study.best_trial.params)
    
    return best_params 