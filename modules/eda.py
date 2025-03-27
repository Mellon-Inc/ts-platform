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
import japanize_matplotlib

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

    # タブの作成
    tab1, tab2 = st.tabs(["時系列分解", "相関分析"])

    with tab1:
        # 時系列データの可視化
        st.subheader("時系列データ")
        if date_col and target_col:
            # データが存在するか確認
            if data[target_col].isnull().all() or len(data) == 0:
                st.warning("対象列にデータがありません。")
                return
            
            # 対象列の選択機能を追加
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            selected_target = st.selectbox(
                "分析する対象列を選択してください",
                options=list(numeric_cols),
                index=list(numeric_cols).index(target_col) if target_col in numeric_cols else 0,
                help="時系列分解を行う対象列を選択してください。"
            )
                
            # グラフ作成
            try:
                fig = px.line(data, x=date_col, y=selected_target, title=f"{selected_target}の時系列推移")
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title=date_col,
                    yaxis_title=selected_target,
                    title_font=dict(size=20),
                    legend_title_font=dict(size=16)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"グラフ表示中にエラーが発生しました: {str(e)}")
                st.info("データ形式を確認してください。日付列が正しく設定されているか確認してください。")
        else:
            st.warning("日付列または対象列が指定されていません。")
            return

        # 対象列のデータを取得
        target_data = data[selected_target].values

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

            # Prophetを使用して時系列分解
            st.subheader("Prophetによる時系列分解")
            
            try:
                # 周期を整数に丸めて使用
                mstl_periods = [int(round(p)) for p in sorted_periods]
                # 重複を除去して昇順にソート
                mstl_periods = sorted(list(set(mstl_periods)))
                
                # 0より大きい周期のみを使用
                valid_periods = [p for p in mstl_periods if 0 < p <= len(target_data)]
                
                if valid_periods and len(valid_periods) > 0:
                    # Prophetのインポート
                    from prophet import Prophet
                    import pandas as pd
                    
                    # データをProphet用に準備
                    prophet_data = pd.DataFrame({
                        'ds': pd.to_datetime(data[date_col]).dt.tz_localize(None),  # タイムゾーン情報を削除
                        'y': target_data  # ターゲット列
                    })
                    
                    st.info("Prophetモデルをトレーニングしています...")
                    
                    # Prophetモデルのトレーニング
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        seasonality_mode='additive'
                    )
                    
                    # 任意の周期性を追加
                    for period in valid_periods:
                        if period > 2:  # 意味のある周期のみ追加
                            model.add_seasonality(
                                name=f'custom_{period}',
                                period=period,
                                fourier_order=5  # 複雑さの調整
                            )
                    
                    # モデルのフィット
                    model.fit(prophet_data)
                    
                    # 予測と成分の取得
                    forecast = model.predict(prophet_data)
                    
                    # トレンド、季節性、残差の抽出
                    trend = forecast['trend'].values
                    seasonality = forecast['yearly'] + forecast['weekly']
                    
                    # カスタム周期性を追加
                    for period in valid_periods:
                        if period > 2 and f'custom_{period}' in forecast.columns:
                            seasonality += forecast[f'custom_{period}']
                    
                    # 残差の計算
                    residuals = target_data - trend - seasonality.values
                    
                    # 分析結果をセッションに保存
                    st.session_state.eda_results = {
                        'trend': trend,
                        'seasonal': seasonality.values.reshape(-1, 1),
                        'seasonal_components': {
                            'yearly': forecast['yearly'].values,
                            'weekly': forecast['weekly'].values
                        },
                        'resid': residuals,
                        'periods': valid_periods,
                        'data': target_data
                    }
                    
                    # カスタムプロットで可視化
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
                            line=dict(color='#1f77b4', width=1.5),
                            fill='tozeroy',
                            fillcolor='rgba(31, 119, 180, 0.1)'
                        ),
                        row=1, col=1
                    )
                    
                    fig_decomp.add_trace(
                        go.Scatter(
                            y=trend, 
                            mode='lines', 
                            name='トレンド成分',
                            line=dict(color='#d62728', width=3),
                            opacity=0.9
                        ),
                        row=1, col=1
                    )
                    
                    # 季節成分
                    fig_decomp.add_trace(
                        go.Scatter(
                            y=seasonality, 
                            mode='lines', 
                            name='季節成分',
                            line=dict(color='#2ca02c', width=2),
                            opacity=0.8
                        ),
                        row=2, col=1
                    )
                    
                    # 残差成分
                    fig_decomp.add_trace(
                        go.Scatter(
                            y=residuals, 
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
                            x=1
                        ),
                        margin=dict(l=60, r=60, t=120, b=60)
                    )
                    
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    
                    # 分解結果の統計情報
                    st.subheader("成分分析")
                    
                    col1, col2, col3 = st.columns(3)
                    trend_contrib = np.var(trend)/np.var(target_data)*100
                    seasonal_contrib = np.var(seasonality)/np.var(target_data)*100
                    resid_contrib = np.var(residuals)/np.var(target_data)*100
                    
                    with col1:
                        st.metric("トレンド成分の寄与度", f"{trend_contrib:.1f}%", 
                                 delta=f"{trend_contrib-33.3:.1f}pp" if trend_contrib > 33.3 else f"{trend_contrib-33.3:.1f}pp")
                    with col2:
                        st.metric("季節成分の寄与度", f"{seasonal_contrib:.1f}%", 
                                 delta=f"{seasonal_contrib-33.3:.1f}pp" if seasonal_contrib > 33.3 else f"{seasonal_contrib-33.3:.1f}pp")
                    with col3:
                        st.metric("残差成分の寄与度", f"{resid_contrib:.1f}%", 
                                 delta=f"{resid_contrib-33.3:.1f}pp" if resid_contrib > 33.3 else f"{resid_contrib-33.3:.1f}pp")
                    
                    st.success("Prophetによる時系列分解が完了しました。")
                else:
                    st.info("有効な周期が見つかりませんでした。Prophetによる分解を実行できません。")
            except Exception as e:
                st.error(f"Prophet実行中にエラーが発生しました: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                
                # Prophet未インストールの場合
                if "No module named 'prophet'" in str(e):
                    st.warning("Prophetモジュールがインストールされていません。以下のコマンドでインストールしてください:")
                    st.code("pip install prophet")
        else:
            st.info("重要な周期成分は検出されませんでした。")
    with tab2:
        st.subheader("変数間の相関分析")
        
        # 数値列のみを抽出
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # 目的変数の選択機能を追加
            selected_target = st.selectbox(
                "相関分析の目的変数を選択してください",
                options=list(numeric_cols),
                index=list(numeric_cols).index(target_col) if target_col in numeric_cols else 0,
                help="相関分析の中心となる目的変数を選択してください。"
            )
            
            # 列の選択機能
            st.write("分析する変数を選択してください：")
            selected_cols = st.multiselect(
                "変数選択",
                options=list(numeric_cols),
                default=list(numeric_cols),
                help="相関分析を行う変数を選択してください。複数選択可能です。"
            )
            
            if len(selected_cols) > 1:
                # 相関行列の計算
                corr_matrix = data[selected_cols].corr()
                
                # 相関係数の絶対値に基づいて列を並び替え
                if selected_target in selected_cols:
                    # 対象変数との相関に基づいて並び替え
                    ordering = abs(corr_matrix[selected_target]).sort_values(ascending=False).index
                else:
                    # 平均相関係数の絶対値に基づいて並び替え
                    ordering = abs(corr_matrix).mean().sort_values(ascending=False).index
                
                # 相関行列を並び替え
                corr_matrix = corr_matrix.loc[ordering, ordering]
                
                # 相関係数の表示オプション
                st.dataframe(
                    corr_matrix.style
                    .background_gradient(cmap='RdBu', vmin=-1, vmax=1)
                    .format("{:.3f}")
                )
                
                # 対象変数との相関分析
                if selected_target in selected_cols:
                    st.subheader(f"{selected_target}との相関係数")
                    target_corr = corr_matrix[selected_target].sort_values(ascending=False)
                    target_corr = target_corr.drop(selected_target)  # 自身との相関を除外
                    
                    # 相関係数の棒グラフ
                    fig_target_corr = go.Figure()
                    
                    # 棒グラフのトレース追加
                    fig_target_corr.add_trace(
                        go.Bar(
                            x=target_corr.index,
                            y=target_corr.values,
                            marker_color=np.where(target_corr > 0, 'rgb(26, 118, 255)', 'rgb(255, 65, 54)'),
                            hovertemplate="変数: %{x}<br>相関係数: %{y:.3f}<extra></extra>",
                            text=target_corr.values.round(2),  # 相関係数を表示
                            textposition='auto'  # 自動的に配置
                        )
                    )
                    
                    # レイアウトの調整
                    fig_target_corr.update_layout(
                        title=dict(
                            text=f"{selected_target}と他の変数との相関係数",
                            font=dict(size=20)
                        ),
                        template="plotly_white",
                        showlegend=False,
                        xaxis_tickangle=-45,
                        xaxis_title="変数",
                        yaxis_title="相関係数",
                        yaxis=dict(
                            range=[-1, 1],
                            tickformat=".2f"
                        ),
                        margin=dict(l=60, r=60, t=80, b=80)
                    )
                    
                    st.plotly_chart(fig_target_corr, use_container_width=True)
                    
                    # 強い相関を持つ変数の散布図
                    correlation_threshold = st.slider(
                        "散布図を表示する相関係数の閾値",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        help="この値以上の絶対値の相関係数を持つ変数の散布図を表示します"
                    )
                    
                    strong_corr = target_corr[abs(target_corr) > correlation_threshold]
                    if not strong_corr.empty:
                        st.subheader(f"相関係数の絶対値が{correlation_threshold}以上の変数との散布図")
                        for var in strong_corr.index:
                            fig_scatter = px.scatter(
                                data,
                                x=var,
                                y=selected_target,
                                trendline="ols",
                                title=f"{selected_target} vs {var} (相関係数: {target_corr[var]:.3f})"
                            )
                            
                            fig_scatter.update_layout(
                                template="plotly_white",
                                title_font=dict(size=16),
                                showlegend=True,
                                xaxis_title=var,
                                yaxis_title=selected_target
                            )
                            
                            st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info(f"相関係数の絶対値が{correlation_threshold}以上の変数は見つかりませんでした。")
                else:
                    st.warning("対象変数が選択されていません。")
            else:
                st.warning("相関分析を行うには2つ以上の変数を選択してください。")
        else:
            st.warning("数値型の列が不足しているため、相関分析を実行できません。")
    # 次のステップへの遷移ボタン
    st.success("EDAが完了しました。次のステップに進みましょう。")
    if st.button("トレーニングページへ進む", use_container_width=True):
        st.session_state.page = 'training'  # トレーニングページに遷移
        st.rerun()

def calculate_periods_from_fft(target_data, fft_result, freqs):
    """FFTの結果から適切な周期を計算する（改良版）"""
    N = len(target_data)
    
    # FFT結果から振幅スペクトルを計算
    amplitude = np.abs(fft_result)
    
    # ナイキスト周波数までの部分のみを使用
    amplitude = amplitude[1:N//2]
    frequencies = freqs[1:N//2]
    
    # 振幅の閾値を設定（最大値の10%）
    threshold = 0.1 * np.max(amplitude)
    
    # 重要な周波数成分を特定
    significant_indices = amplitude > threshold
    significant_freqs = frequencies[significant_indices]
    significant_amps = amplitude[significant_indices]
    
    # 周期を計算（正しい計算方法）
    periods = []
    for f in significant_freqs:
        if f > 0:  # 念のためのチェック
            period = 1.0 / f  # 周波数の逆数が周期
            if 2 <= period <= N/2:  # 有効な周期範囲をチェック
                periods.append(period)
    
    # 周期を近い値でグループ化
    periods = sorted(periods)
    unique_periods = []
    if periods:
        current_period = periods[0]
        unique_periods.append(current_period)
        
        for period in periods[1:]:
            # 前の周期と20%以上異なる場合のみ追加
            if period > current_period * 1.2:
                current_period = period
                unique_periods.append(current_period)
    
    # 周期を整数に丸める
    final_periods = [int(round(p)) for p in unique_periods]
    final_periods = sorted(list(set(final_periods)))  # 重複を除去
    
    return final_periods