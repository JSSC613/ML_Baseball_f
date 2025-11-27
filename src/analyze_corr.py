# src/analyze_corr.py
import pandas as pd
from plotly import express as px
import plotly.express as px
import os

# 設定路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')

def show_correlation():
    print(f"正在讀取資料: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("找不到資料檔，請先執行 data_processing.py")
        return

    df = pd.read_csv(DATA_PATH)

    # 定義我們要分析的特徵 (與訓練時相同 + 目標欄位)
    features = [
        'home_target', # 加入目標變數，看看誰跟勝負最相關
        'diff_win_rate', 'diff_pyth', 'diff_run_diff', 
        'diff_sp_era', 'diff_sp_whip', 'diff_eqa',
        'home_pre_win_rate', 'vis_pre_win_rate',
        'home_pyth', 'vis_pyth',
        'home_sp_era', 'vis_sp_era',
        'home_sp_whip', 'vis_sp_whip',
        'home_roll_b_eqa', 'vis_roll_b_eqa'
    ]

    # 檢查欄位是否存在，只取存在的
    valid_cols = [c for c in features if c in df.columns]
    
    # 計算相關係數矩陣
    print("正在計算相關係數...")
    corr_matrix = df[valid_cols].corr()

    # 繪製互動式熱力圖
    print("正在產生圖表...")
    fig = px.imshow(
        corr_matrix, 
        text_auto=".2f",  # 顯示兩位小數
        aspect="auto",
        color_continuous_scale='RdBu_r', # 紅藍配色 (紅正相關，藍負相關)
        title="MLB 預測特徵相關性矩陣 (Interactive)",
        labels=dict(x="Feature", y="Feature", color="Correlation")
    )

    # 優化版面
    fig.update_layout(
        width=1000,
        height=800,
        xaxis_tickangle=-45
    )

    fig.show()

if __name__ == "__main__":
    show_correlation()