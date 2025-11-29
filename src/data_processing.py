import pandas as pd
import numpy as np
import os

# 設定路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

def process_pitching_stats(pitching_files):
    """
    讀取 pitching.csv，計算累積數據，並標記每場比賽的先發投手
    """
    print("正在處理投手數據 (ERA, WHIP, Starter Identification)...")
    dfs = []
    for f in pitching_files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
            except Exception as e:
                print(f"警告: 無法讀取 {f}: {e}")
                
    if not dfs: return pd.DataFrame(), pd.DataFrame()
    
    p_df = pd.concat(dfs, ignore_index=True)
    
    # 清理 ID 和 GID
    p_df['gid'] = p_df['gid'].astype(str).str.strip()
    p_df['team'] = p_df['team'].astype(str).str.strip()
    p_df['date'] = pd.to_datetime(p_df['date'])
    
    # --- 1. 計算投手的生涯/賽季累積數據 (不分先發後援都算) ---
    # 這樣即使某個投手從牛棚轉先發，我們也有他的數據
    p_df = p_df.sort_values(by=['id', 'date'])
    grouped = p_df.groupby('id')
    
    # 使用 expanding 計算累積數據 (Shift 1 避免偷看當場數據)
    p_df['cum_er'] = grouped['p_er'].transform(lambda x: x.shift(1).expanding().sum())
    p_df['cum_ipouts'] = grouped['p_ipouts'].transform(lambda x: x.shift(1).expanding().sum())
    p_df['cum_wh'] = grouped['p_h'].transform(lambda x: x.shift(1).expanding().sum()) + \
                     grouped['p_w'].transform(lambda x: x.shift(1).expanding().sum())
    
    # 計算 ERA & WHIP
    # 避免除以 0
    p_df['sp_era'] = (p_df['cum_er'] * 27) / p_df['cum_ipouts']
    p_df['sp_whip'] = (p_df['cum_wh'] * 3) / p_df['cum_ipouts']
    
    # 填補缺失值 (給予聯盟平均水準: ERA 4.50, WHIP 1.35)
    p_df[['sp_era', 'sp_whip']] = p_df[['sp_era', 'sp_whip']].fillna(value={'sp_era': 4.50, 'sp_whip': 1.35})
    p_df = p_df.replace([np.inf, -np.inf], 4.50)

    # --- 2. 提取「先發投手」的資料 ---
    # p_seq = 1 代表該場比賽的先發投手
    starters = p_df[p_df['p_seq'] == 1].copy()
    
    # 只保留我們需要的欄位以便 Merge
    # 這裡的 sp_era 是該投手「賽前」的防禦率
    starter_stats = starters[['gid', 'team', 'id', 'sp_era', 'sp_whip']].rename(columns={'id': 'starter_id'})
    
    return starter_stats

def load_and_process_data():
    print("正在讀取 teamstats...")
    all_teams = []
    pitching_files = []
    
    for year in range(2013, 2025):
        t_file = os.path.join(DATA_DIR, f"{year}teamstats.csv")
        p_file = os.path.join(DATA_DIR, f"{year}pitching.csv")
        
        if os.path.exists(t_file):
            df = pd.read_csv(t_file, low_memory=False)
            df['season'] = year
            all_teams.append(df)
        if os.path.exists(p_file):
            pitching_files.append(p_file)
            
    if not all_teams:
        raise ValueError("錯誤：找不到資料，請確認 data 資料夾")

    df = pd.concat(all_teams, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df['gid'] = df['gid'].astype(str).str.strip()
    df['team'] = df['team'].astype(str).str.strip()
    df['target'] = df['win'].apply(lambda x: 1 if str(x).upper() in ['Y', 'W', '1', 'TRUE'] else 0)
    df['vishome'] = df['vishome'].astype(str).str.lower().str.strip()
    
    # --- 關鍵修正：直接從 Pitching CSV 獲取先發投手數據 ---
    # 不再依賴 teamstats 的打序，解決 DH 找不到投手的問題
    sp_stats_df = process_pitching_stats(pitching_files)
    
    if not sp_stats_df.empty:
        print(f"提取到 {len(sp_stats_df)} 筆先發投手資料，正在合併...")
        # 透過 gid 和 team 合併
        df = pd.merge(df, sp_stats_df, on=['gid', 'team'], how='left')
        
        # 檢查合併後的缺失率
        missing_sp = df['sp_era'].isna().sum()
        print(f"合併後無投手數據的場次: {missing_sp} / {len(df)} ({missing_sp/len(df):.1%})")
        
        # 填補剩餘缺失值
        df[['sp_era', 'sp_whip']] = df[['sp_era', 'sp_whip']].fillna(value={'sp_era': 4.50, 'sp_whip': 1.35})
    else:
        print(" 無法從 pitching.csv 提取數據，所有投手數據將為預設值。")
        df['sp_era'] = 4.50
        df['sp_whip'] = 1.35

    # --- 特徵工程：Sabermetrics (EqA) ---
    print("計算進階攻擊數據 (EqA)...")
    for col in ['b_h', 'b_d', 'b_t', 'b_hr', 'b_ab', 'b_w', 'b_hbp', 'b_sb', 'b_cs']:
        if col not in df.columns: df[col] = 0
        
    df['b_tb'] = df['b_h'] + df['b_d'] + 2*df['b_t'] + 3*df['b_hr']
    numerator = df['b_h'] + df['b_tb'] + 1.5 * (df['b_w'] + df['b_hbp']) + df['b_sb']
    denominator = df['b_ab'] + df['b_w'] + df['b_hbp'] + df['b_cs'] + (df['b_sb'] / 3)
    df['b_eqa'] = np.where(denominator > 0, numerator / denominator, 0.0)

    # --- Rolling Stats ---
    print("計算 Rolling Stats...")
    df = df.sort_values(by=['team', 'season', 'date'])
    
    cols_to_roll = ['b_r', 'b_h', 'p_r', 'p_h', 'd_e', 'b_eqa']
    for col in cols_to_roll:
        df[f'roll_{col}'] = df.groupby(['team', 'season'])[col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
        )
        
    # 累積勝率
    df['cum_wins'] = df.groupby(['team', 'season'])['target'].transform(lambda x: x.shift(1).cumsum())
    df['cum_games'] = df.groupby(['team', 'season']).cumcount()
    df['pre_win_rate'] = np.where(df['cum_games'] > 0, df['cum_wins'] / df['cum_games'], 0.5)
    
    df = df.fillna(0)

    # --- 轉換對戰格式 ---
    print("合併主客隊資料...")
    home_mask = df['vishome'].isin(['h', 'home', '1'])
    vis_mask = df['vishome'].isin(['v', 'vis', 'visitor', '0'])
    
    home_df = df[home_mask].add_prefix('home_')
    vis_df = df[vis_mask].add_prefix('vis_')
    
    matchups = pd.merge(home_df, vis_df, left_on='home_gid', right_on='vis_gid')
    
    if len(matchups) == 0:
        print("合併後資料為空。")
    else:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        matchups.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"資料處理完成！已儲存至 {PROCESSED_DATA_PATH}")
        # 驗證投手數據是否有效 (不應該全是 4.5)
        era_std = matchups['home_sp_era'].std()
        print(f"主隊先發投手 ERA 標準差: {era_std:.4f} (若接近 0 代表數據有問題)")

if __name__ == "__main__":
    load_and_process_data()
