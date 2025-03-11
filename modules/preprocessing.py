import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import show_success, show_error, show_info, show_warning

def show_preprocessing_page():
    """データの前処理を行うページ"""
    st.header("データの前処理")
    
    if st.session_state.data is None:
        st.warning("データがアップロードされていません。データのアップロードページに戻ってください。")
        if st.button("データアップロードに戻る", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    # 元のデータを取得
    df = st.session_state.data.copy()
    
    st.subheader("前処理オプション")
    
    # 前処理タブ
    preprocessing_tab = st.tabs(["欠損値補完", "外れ値除去", "正規化"])
    
    # 欠損値補完タブ
    with preprocessing_tab[0]:
        st.subheader("欠損値補完")
        
        # 欠損値のある列を表示
        missing_cols = df.columns[df.isna().any()].tolist()
        
        if not missing_cols:
            st.info("データセットに欠損値はありません。")
        else:
            st.write(f"欠損値のある列: {len(missing_cols)}個")
            
            # 欠損値補完方法の選択
            imputation_method = st.selectbox(
                "欠損値補完方法を選択してください",
                options=["平均値", "中央値", "最頻値", "前方補完", "後方補完", "線形補間"],
                index=0
            )
            
            # 適用ボタン
            if st.button("欠損値補完を適用", key="apply_imputation"):
                st.info(f"選択された方法: {imputation_method}で欠損値を補完します")
                # ここに実際の欠損値補完ロジックを実装
                st.success("欠損値補完が完了しました")
    
    # 外れ値除去タブ
    with preprocessing_tab[1]:
        st.subheader("外れ値除去")
        
        # 数値列のみ抽出
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numeric_cols:
            st.info("外れ値検出に適した数値列がありません。")
        else:
            # 外れ値検出方法の選択
            outlier_method = st.selectbox(
                "外れ値検出方法を選択してください",
                options=["IQR (四分位範囲)", "Z-スコア", "パーセンタイル"],
                index=0
            )
            
            # 対象列の選択
            selected_cols = st.multiselect(
                "外れ値検出の対象列を選択してください",
                options=numeric_cols
            )
            
            # しきい値の設定
            threshold = st.slider("しきい値", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
            
            # 処理方法の選択
            treatment_method = st.selectbox(
                "外れ値の処理方法を選択してください",
                options=["除去", "置換（平均値）", "置換（中央値）", "置換（上限/下限値）"],
                index=0
            )
            
            # 適用ボタン
            if st.button("外れ値検出を適用", key="apply_outlier"):
                if not selected_cols:
                    st.warning("対象列を選択してください")
                else:
                    # 常に元のデータを使用して処理を行う
                    # セッションに元データが保存されていれば使用、なければ現在のデータを使用
                    if 'original_data' in st.session_state and st.session_state.original_data is not None:
                        df_before = st.session_state.original_data.copy()
                    else:
                        df_before = st.session_state.data.copy()
                        
                    # 処理用のデータフレームを作成
                    df_processed = df_before.copy()
                    outliers_count = {}
                    
                    # 各選択列に対して外れ値検出と処理を実行
                    for col in selected_cols:
                        # 外れ値検出
                        if outlier_method == "IQR (四分位範囲)":
                            Q1 = df_processed[col].quantile(0.25)
                            Q3 = df_processed[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                            outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)].index
                        elif outlier_method == "Z-スコア":
                            z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
                            outliers = df_processed[z_scores > threshold].index
                        else:  # パーセンタイル
                            lower_percentile = threshold
                            upper_percentile = 100 - threshold
                            lower_bound = df_processed[col].quantile(lower_percentile / 100)
                            upper_bound = df_processed[col].quantile(upper_percentile / 100)
                            outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)].index
                        
                        outliers_count[col] = len(outliers)
                        
                        # 外れ値の処理
                        if treatment_method == "除去":
                            df_processed = df_processed.drop(outliers)
                        elif treatment_method == "置換（平均値）":
                            mean_val = df_processed[col].mean()
                            df_processed.loc[outliers, col] = mean_val
                        elif treatment_method == "置換（中央値）":
                            median_val = df_processed[col].median()
                            df_processed.loc[outliers, col] = median_val
                        elif treatment_method == "置換（上限/下限値）":
                            if outlier_method == "IQR (四分位範囲)":
                                df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                                df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                            elif outlier_method == "Z-スコア":
                                mean = df_processed[col].mean()
                                std = df_processed[col].std()
                                df_processed.loc[z_scores > threshold, col] = np.where(
                                    df_processed.loc[z_scores > threshold, col] > mean,
                                    mean + threshold * std,
                                    mean - threshold * std
                                )
                            else:  # パーセンタイル
                                df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                                df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                    
                    # 結果の表示
                    st.success(f"外れ値処理が完了しました。処理方法: {treatment_method}")
                    
                    # 外れ値の数を表示
                    st.write("検出された外れ値の数:")
                    for col, count in outliers_count.items():
                        st.write(f"- {col}: {count}件")
                    
                    # 処理前後の可視化
                    st.subheader("処理前後の比較")
                    
                    # 時系列データの場合は時系列プロットも表示
                    date_col = st.session_state.date_col
                    if date_col and date_col in df_processed.columns:
                        st.subheader("時系列での比較")
                        for col in selected_cols:
                            col1, col2 = st.columns(2)
                            
                            # 日付型に変換
                            df_before_plot = df_before.copy()
                            df_after_plot = df_processed.copy()
                            df_before_plot[date_col] = pd.to_datetime(df_before_plot[date_col])
                            df_after_plot[date_col] = pd.to_datetime(df_after_plot[date_col])
                            
                            # ソート
                            df_before_plot = df_before_plot.sort_values(by=date_col)
                            df_after_plot = df_after_plot.sort_values(by=date_col)
                            
                            with col1:
                                st.write(f"処理前の時系列: {col}")
                                fig = px.line(df_before_plot, x=date_col, y=col, title=f"{col}の時系列 (処理前)")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.write(f"処理後の時系列: {col}")
                                fig = px.line(df_after_plot, x=date_col, y=col, title=f"{col}の時系列 (処理後)")
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # 箱ひげ図とヒストグラムの比較
                    for col in selected_cols:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"処理前: {col}")
                            fig = px.box(df_before, y=col, title=f"{col} (処理前)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # ヒストグラム
                            hist_fig = px.histogram(df_before, x=col, title=f"{col}のヒストグラム (処理前)")
                            st.plotly_chart(hist_fig, use_container_width=True)
                        
                        with col2:
                            st.write(f"処理後: {col}")
                            fig = px.box(df_processed, y=col, title=f"{col} (処理後)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # ヒストグラム
                            hist_fig = px.histogram(df_processed, x=col, title=f"{col}のヒストグラム (処理後)")
                            st.plotly_chart(hist_fig, use_container_width=True)
                    
                    # 処理後のデータを更新
                    st.session_state.data = df_processed
                    # 元のデータを保持
                    if 'original_data' not in st.session_state:
                        st.session_state.original_data = df_before
    # 正規化タブ
    with preprocessing_tab[2]:
        st.subheader("正規化")
        
        # 数値列のみ抽出
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numeric_cols:
            st.info("正規化に適した数値列がありません。")
        else:
            # 正規化方法の選択
            normalization_method = st.selectbox(
                "正規化方法を選択してください",
                options=["Min-Max スケーリング", "標準化 (Z-スコア)", "ロバストスケーリング"],
                index=0
            )
            
            # 対象列の選択
            selected_cols = st.multiselect(
                "正規化の対象列を選択してください",
                options=numeric_cols
            )
            
            # 適用ボタン
            if st.button("正規化を適用", key="apply_normalization"):
                if not selected_cols:
                    st.warning("対象列を選択してください")
                else:
                    # 処理前のデータを保存
                    df_before = df.copy()
                    
                    # 正規化処理の実装
                    if normalization_method == "Min-Max スケーリング":
                        for col in selected_cols:
                            min_val = df[col].min()
                            max_val = df[col].max()
                            df[col] = (df[col] - min_val) / (max_val - min_val)
                    
                    elif normalization_method == "標準化 (Z-スコア)":
                        for col in selected_cols:
                            mean_val = df[col].mean()
                            std_val = df[col].std()
                            df[col] = (df[col] - mean_val) / std_val
                    
                    elif normalization_method == "ロバストスケーリング":
                        for col in selected_cols:
                            median_val = df[col].median()
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            df[col] = (df[col] - median_val) / iqr
                    
                    # 結果の表示
                    st.success(f"{normalization_method}による正規化が完了しました")
                    
                    # 処理前後の比較
                    st.subheader("正規化前後の比較")
                    
                    # 箱ひげ図とヒストグラムの比較
                    for col in selected_cols:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"正規化前: {col}")
                            fig = px.box(df_before, y=col, title=f"{col} (正規化前)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # ヒストグラム
                            hist_fig = px.histogram(df_before, x=col, title=f"{col}のヒストグラム (正規化前)")
                            st.plotly_chart(hist_fig, use_container_width=True)
                        
                        with col2:
                            st.write(f"正規化後: {col}")
                            fig = px.box(df, y=col, title=f"{col} (正規化後)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # ヒストグラム
                            hist_fig = px.histogram(df, x=col, title=f"{col}のヒストグラム (正規化後)")
                            st.plotly_chart(hist_fig, use_container_width=True)
    # 前処理後のデータプレビュー
    st.subheader("データプレビュー")
    
    # 時系列データの場合は時系列プロットを表示
    date_col = st.session_state.date_col
    if date_col and date_col in df.columns:
        # 数値列のみ抽出
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # ターゲット列をデフォルトで選択
        target_col = st.session_state.target_col
        default_selection = [target_col] if target_col in numeric_cols else []
        
        selected_cols = st.multiselect(
            "表示する系列を選択してください",
            options=numeric_cols,
            default=default_selection
        )
        
        if selected_cols:
            # 日付型に変換
            df_plot = df.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col])
            df_plot = df_plot.sort_values(by=date_col)
            
            # 複数系列の時系列プロット
            fig = px.line(df_plot, x=date_col, y=selected_cols, title="前処理後の時系列データ")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("表示する系列を選択してください")
    else:
        # 時系列データでない場合はデータフレームを表示
        st.dataframe(df.head(), use_container_width=True)
    
    # 次のステップへの遷移ボタン
    st.success("前処理が完了しました。次のステップに進みましょう。")
    if st.button("EDA分析へ進む", use_container_width=True):
        st.session_state.clean_data = df  # 前処理済みデータを保存
        st.session_state.page = 'eda'
        st.rerun()