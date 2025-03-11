import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def show_eda_page():
    """EDAページで時系列分解を可視化"""
    st.title("探索的データ分析 (EDA)")

    # データが存在するか確認
    if st.session_state.clean_data is None:
        st.warning("データが前処理されていません。")
        return

    # データの読み込み
    data = st.session_state.clean_data
    date_col = st.session_state.date_col
    target_col = st.session_state.target_col

    # 時系列データの可視化
    st.subheader("時系列データ")
    if date_col and target_col:
        fig = px.line(data, x=date_col, y=target_col)
        fig.update_layout(
            template="plotly_white",
            xaxis_title=date_col,
            yaxis_title=target_col,
            title_font=dict(size=20),
            legend_title_font=dict(size=16)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("日付列または対象列が指定されていません。")
        return

    # 対象列のデータを取得
    target_data = data[target_col].values

    # FFTを使用して周波数成分を特定
    st.subheader("FFTによる周波数成分の特定")
    fft_result = fft(target_data)
    freqs = np.fft.fftfreq(len(fft_result))
    
    # 正の周波数のみを抽出（負の周波数は対称なので不要）
    positive_mask = freqs > 0
    positive_freqs = freqs[positive_mask]
    positive_amplitude = np.abs(fft_result)[positive_mask]
    
    # 周波数を周期に変換してより直感的に表示
    periods = 1.0 / positive_freqs
    
    # FFT結果の可視化（周期ベース）
    fig_fft = go.Figure()
    
    # 周期ベースのプロット（より直感的）
    fig_fft.add_trace(go.Scatter(
        x=periods, 
        y=positive_amplitude, 
        mode='lines', 
        name='周期ベース',
        line=dict(color='royalblue', width=2)
    ))
    
    # 重要な周期にマーカーを追加
    threshold = 0.1 * np.max(positive_amplitude)
    significant_indices = positive_amplitude > threshold
    
    fig_fft.add_trace(go.Scatter(
        x=periods[significant_indices], 
        y=positive_amplitude[significant_indices], 
        mode='markers', 
        name='重要な周期',
        marker=dict(color='red', size=10)
    ))
    
    # 周期の単位を追加
    fig_fft.update_layout(
        title='周期成分分析',
        xaxis_title='周期（データポイント数）',
        yaxis_title='振幅',
        template="plotly_white",
        title_font=dict(size=20),
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis_type="log",  # 対数スケールで表示して広範囲の周期を見やすく
        hovermode="closest"
    )
    
    # ホバー情報を充実させる
    fig_fft.update_traces(
        hovertemplate='周期: %{x:.1f}<br>振幅: %{y:.1f}'
    )
    
    st.plotly_chart(fig_fft, use_container_width=True)
    
    # 重要な周期の説明
    significant_periods = periods[significant_indices]
    if len(significant_periods) > 0:
        sorted_periods = np.sort(significant_periods)
        period_info = ", ".join([f"{p:.1f}" for p in sorted_periods])
        st.info(f"検出された重要な周期（データポイント数）: {period_info}")
        
        # データ頻度に基づいた解釈を追加
        if st.session_state.data_frequency:
            freq = st.session_state.data_frequency
            st.write("データ頻度に基づく解釈:")
            for p in sorted_periods:
                if freq == 'D':
                    if 350 <= p <= 380:
                        st.write(f"- 周期 {p:.1f}: 約1年の季節性")
                    elif 28 <= p <= 31:
                        st.write(f"- 周期 {p:.1f}: 約1ヶ月の季節性")
                    elif 6.5 <= p <= 7.5:
                        st.write(f"- 周期 {p:.1f}: 週次の季節性")
                elif freq == 'M':
                    if 11.5 <= p <= 12.5:
                        st.write(f"- 周期 {p:.1f}: 年次の季節性")
                    elif 2.9 <= p <= 3.1:
                        st.write(f"- 周期 {p:.1f}: 四半期の季節性")
                elif freq == 'H':
                    if 23.5 <= p <= 24.5:
                        st.write(f"- 周期 {p:.1f}: 日次の季節性")
    else:
        st.info("重要な周期成分は検出されませんでした。")

    # 周波数成分の特定（元のコード）
    threshold = 0.1 * np.max(np.abs(fft_result))
    significant_freqs = freqs[np.abs(fft_result) > threshold]

    # MSTLを使用して時系列分解
    st.subheader("MSTLによる時系列分解")
    
    if len(significant_freqs) > 0:
        try:
            # 正の周波数のみを使用し、周期に変換
            positive_freqs = [f for f in significant_freqs if f > 0]
            if positive_freqs:
                periods = [int(1/f) for f in positive_freqs if 1/f > 2]  # 周期が2以上のものだけ使用
                
                if periods:
                    mstl = MSTL(target_data, periods=periods)
                    result = mstl.fit()
                    
                    # 結果を統合して可視化（より美しく改善）
                    fig_decomp = make_subplots(
                        rows=3, 
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('元データとトレンド成分', '季節成分', '残差成分')
                    )
                    
                    # 元データとトレンド成分
                    fig_decomp.add_trace(
                        go.Scatter(
                            y=target_data, 
                            mode='lines', 
                            name='元データ',
                            line=dict(color='#1f77b4', width=1.5, dash='solid'),
                            fill='tozeroy',
                            fillcolor='rgba(31, 119, 180, 0.1)'
                        ),
                        row=1, col=1
                    )
                    
                    fig_decomp.add_trace(
                        go.Scatter(
                            y=result.trend, 
                            mode='lines', 
                            name='トレンド成分',
                            line=dict(color='#d62728', width=3, dash='solid'),
                            opacity=0.9
                        ),
                        row=1, col=1
                    )
                    
                    # 季節成分の可視化
                    colors = px.colors.qualitative.Bold  # より鮮やかなカラーパレットを使用
                    
                    # 各季節成分を個別に表示
                    for i, period in enumerate(periods):
                        color_idx = i % len(colors)
                        fig_decomp.add_trace(
                            go.Scatter(
                                y=result.seasonal[:, i], 
                                mode='lines', 
                                name=f'季節成分 (周期: {period})',
                                line=dict(color=colors[color_idx], width=2),
                                opacity=0.8
                            ),
                            row=2, col=1
                        )
                    
                    # 残差成分
                    fig_decomp.add_trace(
                        go.Scatter(
                            y=result.resid, 
                            mode='lines', 
                            name='残差成分',
                            line=dict(color='#7f7f7f', width=1.5),
                            opacity=0.8,
                            fill='tozeroy',
                            fillcolor='rgba(127, 127, 127, 0.1)'
                        ),
                        row=3, col=1
                    )
                    
                    # レイアウトの調整
                    fig_decomp.update_layout(
                        height=900,
                        template="plotly_white",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='rgba(0, 0, 0, 0.2)',
                            borderwidth=1
                        ),
                        margin=dict(l=60, r=60, t=120, b=60),
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(family="Arial, sans-serif", size=14, color="#333333")
                    )
                    
                    # 各サブプロットの軸を調整
                    fig_decomp.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(211, 211, 211, 0.5)',
                        zeroline=False,
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0, 0, 0, 0.3)'
                    )
                    
                    fig_decomp.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(211, 211, 211, 0.5)',
                        zeroline=True,
                        zerolinewidth=1.5,
                        zerolinecolor='rgba(0, 0, 0, 0.2)',
                        showline=True,
                        linewidth=1,
                        linecolor='rgba(0, 0, 0, 0.3)'
                    )
                    
                    # ホバー情報を充実させる
                    fig_decomp.update_traces(
                        hovertemplate='<b>%{y:.2f}</b>'
                    )
                    
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    
                    # 分析結果をセッションに保存
                    st.session_state.eda_results = {
                        'trend': result.trend,
                        'seasonal': result.seasonal,
                        'seasonal_components': {f'seasonal_{period}': result.seasonal[:, i] for i, period in enumerate(periods)},
                        'resid': result.resid,
                        'periods': periods,
                        'data': target_data
                    }
                    
                    # 分解結果の統計情報（より視覚的に）
                    st.subheader("成分分析")
                    
                    col1, col2, col3 = st.columns(3)
                    trend_contrib = np.var(result.trend)/np.var(target_data)*100
                    seasonal_sum = np.sum(result.seasonal, axis=1)
                    seasonal_contrib = np.var(seasonal_sum)/np.var(target_data)*100
                    resid_contrib = np.var(result.resid)/np.var(target_data)*100
                    
                    with col1:
                        st.metric("トレンド成分の寄与度", f"{trend_contrib:.1f}%", 
                                 delta=f"{trend_contrib-33.3:.1f}pp" if trend_contrib > 33.3 else f"{trend_contrib-33.3:.1f}pp")
                    with col2:
                        st.metric("季節成分の寄与度", f"{seasonal_contrib:.1f}%", 
                                 delta=f"{seasonal_contrib-33.3:.1f}pp" if seasonal_contrib > 33.3 else f"{seasonal_contrib-33.3:.1f}pp")
                    with col3:
                        st.metric("残差成分の寄与度", f"{resid_contrib:.1f}%", 
                                 delta=f"{resid_contrib-33.3:.1f}pp" if resid_contrib > 33.3 else f"{resid_contrib-33.3:.1f}pp")
                    
                    # 残差の定常性検定
                    st.subheader("残差の定常性分析")
                    
                    # ADF検定の実施
                    adf_result = adfuller(result.resid)
                    adf_pvalue = adf_result[1]
                    
                    # 結果の表示
                    if adf_pvalue < 0.05:
                        st.success(f"残差は定常的です（ADF検定 p値: {adf_pvalue:.4f}）")
                    else:
                        st.warning(f"残差は非定常的である可能性があります（ADF検定 p値: {adf_pvalue:.4f}）")
                    
                    # 残差の自己相関と偏自己相関の可視化
                    st.write("残差の自己相関と偏自己相関")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    plot_acf(result.resid, ax=ax1, lags=40)
                    ax1.set_title("残差の自己相関関数 (ACF)")
                    
                    plot_pacf(result.resid, ax=ax2, lags=40)
                    ax2.set_title("残差の偏自己相関関数 (PACF)")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 残差のQQプロットと分布
                    st.write("残差の分布分析")
                    
                    fig_resid = make_subplots(rows=1, cols=2, 
                                             subplot_titles=("残差のヒストグラム", "残差のQQプロット"),
                                             specs=[[{"type": "xy"}, {"type": "xy"}]])
                    
                    # ヒストグラム
                    fig_resid.add_trace(
                        go.Histogram(
                            x=result.resid,
                            nbinsx=30,
                            name="残差",
                            marker_color='rgba(0, 0, 255, 0.6)'
                        ),
                        row=1, col=1
                    )
                    
                    # QQプロット用のデータ準備
                    from scipy import stats
                    resid_sorted = np.sort(result.resid)
                    norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(resid_sorted)))
                    
                    # QQプロット
                    fig_resid.add_trace(
                        go.Scatter(
                            x=norm_quantiles,
                            y=resid_sorted,
                            mode='markers',
                            name='QQプロット',
                            marker=dict(color='rgba(255, 0, 0, 0.6)')
                        ),
                        row=1, col=2
                    )
                    
                    # 理論線の追加
                    slope, intercept = np.polyfit(norm_quantiles, resid_sorted, 1)
                    line_x = np.array([min(norm_quantiles), max(norm_quantiles)])
                    line_y = slope * line_x + intercept
                    
                    fig_resid.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode='lines',
                            name='理論線',
                            line=dict(color='black', dash='dash')
                        ),
                        row=1, col=2
                    )
                    
                    fig_resid.update_layout(
                        height=500,
                        template="plotly_white",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_resid, use_container_width=True)
                                        
                    st.success("時系列分解が完了しました。次のステップに進むことができます。")
                else:
                    st.info("有効な周期が見つかりませんでした。")
            else:
                st.info("正の周波数成分が見つかりませんでした。")
        except Exception as e:
            st.error(f"時系列分解中にエラーが発生しました: {str(e)}")
    else:
        st.info("特定された周波数成分がありません。MSTLを実行できません。")

    # 次のステップへの遷移ボタン
    st.success("EDAが完了しました。次のステップに進みましょう。")
    if st.button("トレーニングページへ進む", use_container_width=True):
        st.session_state.page = 'training'  # トレーニングページに遷移
        st.rerun()