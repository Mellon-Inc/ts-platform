import streamlit as st
import pandas as pd
from utils import load_cached_data, show_error
from modules.data_loader import suggest_target_column, detect_date_column
from utils import detect_frequency, plot_multiple_series

def show_upload_page():
    """データのアップロードと前処理を行うページ"""
    st.header("データのアップロード")
    
    uploaded_file = st.file_uploader("CSVまたはExcelファイルをアップロードしてください", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Excelファイルの場合、シート選択オプションを表示
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                try:
                    xls = pd.ExcelFile(uploaded_file)
                    sheet_name = st.selectbox("シートを選択してください", xls.sheet_names)
                    # データの読み込み（Excelの場合）
                    df = load_cached_data(uploaded_file)
                    st.session_state['data_source'] = {'type': 'excel', 'file': uploaded_file, 'sheet': sheet_name}
                except ImportError:
                    st.error("Excelファイルを読み込むには'openpyxl'が必要です。'pip install openpyxl'でインストールしてください。")
                    return
            else:
                # CSVファイルの場合
                df = load_cached_data(uploaded_file)
                st.session_state['data_source'] = {'type': 'csv', 'file': uploaded_file}
            
            if df is not None:
                st.success("ファイルが正常に読み込まれました")
                
                # データプレビュー
                st.subheader("データプレビュー")
                st.dataframe(df.head(), use_container_width=True)
                
                # 列の概要
                st.subheader("列の情報")
                col_info = pd.DataFrame({
                    '列名': df.columns,
                    'データ型': df.dtypes,
                    '非欠損値数': df.count().values,
                    '欠損値数': df.isna().sum().values,
                    '欠損率 (%)': df.isna().sum().values / len(df) * 100
                })
                st.dataframe(col_info, use_container_width=True)
                
                # 日付列と対象列の推測
                date_col = detect_date_column(df)
                target_col = suggest_target_column(df, date_col)
                
                # 日付列のバリデーション
                if date_col:
                    try:
                        # 日付列を日付型に変換してみる
                        pd.to_datetime(df[date_col])
                        st.success(f"日付列「{date_col}」が正常に検出されました。")
                    except Exception as e:
                        st.warning(f"日付列「{date_col}」の変換中にエラーが発生しました: {str(e)}")
                        date_col = None
                        st.info("日付列を手動で選択してください。")
                        date_col = st.selectbox("日付列を選択", df.columns.tolist())
                        try:
                            pd.to_datetime(df[date_col])
                            st.success(f"日付列「{date_col}」が正常に変換されました。")
                        except:
                            st.error("選択された列は日付として解釈できません。別の列を選択してください。")
                            date_col = None
                else:
                    st.warning("日付列が自動検出されませんでした。手動で選択してください。")
                    date_col = st.selectbox("日付列を選択", df.columns.tolist())
                    try:
                        pd.to_datetime(df[date_col])
                        st.success(f"日付列「{date_col}」が正常に変換されました。")
                    except:
                        st.error("選択された列は日付として解釈できません。別の列を選択してください。")
                        date_col = None
                
                # セッションステートにデータを保存
                st.session_state.data = df
                st.session_state.date_col = date_col
                st.session_state.target_col = target_col
                
                # 時系列データの可視化
                if date_col and target_col:
                    st.subheader("時系列データの可視化")
                    try:
                        # 日付列を日付型に変換
                        df_plot = df.copy()
                        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
                        df_plot = df_plot.sort_values(by=date_col)
                        
                        # 数値列を抽出して選択可能にする
                        numeric_cols = df_plot.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        if date_col in numeric_cols:
                            numeric_cols.remove(date_col)
                        
                        # デフォルトでターゲット列を選択
                        default_selection = [target_col] if target_col in numeric_cols else []
                        selected_cols = st.multiselect(
                            "表示する系列を選択してください",
                            options=numeric_cols,
                            default=default_selection
                        )
                        
                        if selected_cols:
                            # 複数系列の時系列プロット
                            plot_multiple_series(df_plot, date_col, selected_cols)
                        else:
                            st.info("表示する系列を選択してください")
                        
                        # データ頻度の検出
                        st.session_state.data_frequency = detect_frequency(df_plot[date_col])
                        st.info(f"データ頻度: {st.session_state.data_frequency}")
                    except Exception as e:
                        st.warning(f"時系列プロットの生成中にエラーが発生しました: {str(e)}")
                # 前処理ステップへの遷移ボタン
                st.success("データの準備ができました。前処理ステップに進みましょう。")
                if st.button("データ前処理へ進む", use_container_width=True):
                    st.session_state.page = 'preprocessing'
                    st.rerun()
                        
        except Exception as e:
            show_error(f"データの読み込み中にエラーが発生しました: {str(e)}") 