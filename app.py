from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import itertools
import random
import tensorflow as tf
import json
import plotly
import plotly.graph_objects as go
from src.team_info import TEAM_DISPLAY_INFO, MLB_STRUCTURE, get_relation

app = Flask(__name__)

# --- 1. 設定路徑與載入模型 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')

print("正在載入模型...")
try:
    rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    keras_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'keras_model.h5'))
    print(" 模型載入成功")
except Exception as e:
    print(f" 模型載入錯誤: {e}")
    print("請確保已執行 src/train_models.py")

print("正在載入數據...")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    print(" 找不到資料檔")
    df = pd.DataFrame()

# --- 2. 全局特徵工程 (補上缺少的特徵) ---
if not df.empty:
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['home_date'] if 'home_date' in df.columns else pd.to_datetime('2024-01-01'))
    else:
        df['date'] = pd.to_datetime(df['date'])

    if 'home_season' not in df.columns:
        df['home_season'] = df['date'].dt.year

    # 確保必要欄位存在 (避免 KeyError)
    for col in ['home_roll_b_eqa', 'vis_roll_b_eqa', 'home_sp_era', 'vis_sp_era', 'home_sp_whip', 'vis_sp_whip']:
        if col not in df.columns: df[col] = 0  # 若無則補0

    # 計算衍生特徵
    df['home_pyth'] = (df['home_roll_b_r']**1.83) / ((df['home_roll_b_r']**1.83 + df['home_roll_p_r']**1.83) + 1e-9)
    df['vis_pyth'] = (df['vis_roll_b_r']**1.83) / ((df['vis_roll_b_r']**1.83 + df['vis_roll_p_r']**1.83) + 1e-9)

    df['diff_win_rate'] = df['home_pre_win_rate'] - df['vis_pre_win_rate']
    df['diff_run_diff'] = (df['home_roll_b_r'] - df['home_roll_p_r']) - (df['vis_roll_b_r'] - df['vis_roll_p_r'])
    df['diff_pyth'] = df['home_pyth'] - df['vis_pyth']
    df['diff_sp_era'] = df['home_sp_era'] - df['vis_sp_era']
    df['diff_sp_whip'] = df['home_sp_whip'] - df['vis_sp_whip']
    # 新增：EqA 差值
    df['diff_eqa'] = df['home_roll_b_eqa'] - df['vis_roll_b_eqa']

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

# --- 3. 準備球隊列表 ---
teams_list = sorted(df['home_team'].unique()) if not df.empty else []
teams_options = []
for t in teams_list:
    display = f"{t} ({TEAM_DISPLAY_INFO.get(t, 'Unknown')})"
    teams_options.append({"code": t, "display": display})

# --- 4. 準備各隊最新數據 (包含 EqA) ---
last_stats = {}
if not df.empty:
    df_2024 = df[df['home_season'] == 2024].sort_values('date')
    
    for team in teams_list:
        team_games = df_2024[(df_2024['home_team'] == team) | (df_2024['vis_team'] == team)].tail(5)
        
        if not team_games.empty:
            last_game = team_games.iloc[-1]
            
            if last_game['home_team'] == team:
                base_stats = [
                    last_game['home_pre_win_rate'], last_game['home_roll_b_r'],
                    last_game['home_roll_p_r'], last_game['home_roll_b_h'],
                    last_game['home_roll_d_e'], last_game['home_roll_b_eqa'] # 新增 EqA
                ]
            else:
                base_stats = [
                    last_game['vis_pre_win_rate'], last_game['vis_roll_b_r'],
                    last_game['vis_roll_p_r'], last_game['vis_roll_b_h'],
                    last_game['vis_roll_d_e'], last_game['vis_roll_b_eqa'] # 新增 EqA
                ]
            
            sp_eras, sp_whips = [], []
            for _, g in team_games.iterrows():
                if g['home_team'] == team:
                    sp_eras.append(g.get('home_sp_era', 4.50))
                    sp_whips.append(g.get('home_sp_whip', 1.35))
                else:
                    sp_eras.append(g.get('vis_sp_era', 4.50))
                    sp_whips.append(g.get('vis_sp_whip', 1.35))
            
            avg_era = sum(sp_eras) / len(sp_eras) if sp_eras else 4.50
            avg_whip = sum(sp_whips) / len(sp_whips) if sp_whips else 1.35
            
            # last_stats 結構: [Win, R, RA, H, E, EqA, SP_ERA, SP_WHIP]
            # Index:           0    1  2   3  4  5    6       7
            last_stats[team] = base_stats + [avg_era, avg_whip]
        else:
            last_stats[team] = [0.5, 4.0, 4.0, 8.0, 0.5, 0.250, 4.50, 1.35]

# --- 5. 預測單場比賽函數 (關鍵修正：特徵順序) ---
def predict_single_match(home, vis, model_type='xgb'):
    # 預設值: [Win, R, RA, H, E, EqA, ERA, WHIP]
    h_s = last_stats.get(home, [0.5, 4, 4, 8, 0.5, 0.25, 4.5, 1.35])
    v_s = last_stats.get(vis, [0.5, 4, 4, 8, 0.5, 0.25, 4.5, 1.35])
    
    h_pyth = (h_s[1]**1.83) / ((h_s[1]**1.83 + h_s[2]**1.83) + 1e-9)
    v_pyth = (v_s[1]**1.83) / ((v_s[1]**1.83 + v_s[2]**1.83) + 1e-9)
    
    diff_win = h_s[0] - v_s[0]
    diff_pyth = h_pyth - v_pyth
    diff_run = (h_s[1] - h_s[2]) - (v_s[1] - v_s[2])
    diff_era = h_s[6] - v_s[6] # ERA is at index 6
    diff_whip = h_s[7] - v_s[7] # WHIP is at index 7
    diff_eqa = h_s[5] - v_s[5] # EqA is at index 5
    
    # 建立 DataFrame (解決 feature names mismatch)
    # 欄位名稱必須與 train_models.py 中的 features 列表完全一致
    data_dict = {
        'diff_win_rate': [diff_win],
        'diff_pyth': [diff_pyth],
        'diff_run_diff': [diff_run],
        'diff_sp_era': [diff_era],
        'diff_sp_whip': [diff_whip],
        'diff_eqa': [diff_eqa],            # 新增
        'home_pre_win_rate': [h_s[0]],
        'vis_pre_win_rate': [v_s[0]],
        'home_pyth': [h_pyth],
        'vis_pyth': [v_pyth],
        'home_roll_b_r': [h_s[1]],
        'vis_roll_b_r': [v_s[1]],
        'home_roll_p_r': [h_s[2]],
        'vis_roll_p_r': [v_s[2]],
        'home_sp_era': [h_s[6]],
        'vis_sp_era': [v_s[6]],
        'home_sp_whip': [h_s[7]],
        'vis_sp_whip': [v_s[7]],
        'home_roll_b_eqa': [h_s[5]],       # 新增
        'vis_roll_b_eqa': [v_s[5]]         # 新增
    }
    
    features_df = pd.DataFrame(data_dict)
    
    if model_type == 'keras':
        features_scaled = scaler.transform(features_df)
        return float(keras_model.predict(features_scaled, verbose=0)[0][0])
    elif model_type == 'rf':
        return rf_model.predict_proba(features_df)[0][1]
    else:
        return xgb_model.predict_proba(features_df)[0][1]

# --- 6. 賽季模擬邏輯 ---
def simulate_series(team_a, team_b, games_needed):
    wins_a = 0
    wins_b = 0
    prob_a = predict_single_match(team_a, team_b, model_type='xgb')
    
    while wins_a < games_needed and wins_b < games_needed:
        if random.random() < prob_a:
            wins_a += 1
        else:
            wins_b += 1
    winner = team_a if wins_a > wins_b else team_b
    return winner, max(wins_a, wins_b), min(wins_a, wins_b)

def simulate_season():
    if not last_stats: return "Error", []
    teams = list(last_stats.keys())
    standings = {t: 0 for t in teams}
    logs = []

    # 模擬例行賽
    logs.append("=== 2025 例行賽模擬 (162 Games) ===")
    matchups = list(itertools.combinations(teams, 2))
    for t1, t2 in matchups:
        relation = get_relation(t1, t2)
        if relation == 'DIVISION': count = 13
        elif relation == 'LEAGUE': count = 6
        else: count = 3   
        prob_t1 = predict_single_match(t1, t2, model_type='xgb')
        for _ in range(count):
            if random.random() < prob_t1: standings[t1] += 1
            else: standings[t2] += 1
    
    top_teams = sorted(standings.items(), key=lambda x: x[1], reverse=True)[:5]
    for t, w in top_teams:
        logs.append(f"{t}: {w} 勝")
    logs.append("...")
                
    playoff_seeds = {'AL': [], 'NL': []}
    for league, divisions in MLB_STRUCTURE.items():
        div_winners = []
        wild_card_pool = []
        for div_name, div_teams in divisions.items():
            sorted_div = sorted([(t, standings.get(t, 0)) for t in div_teams if t in standings], key=lambda x: x[1], reverse=True)
            if sorted_div:
                div_winners.append(sorted_div[0])
                wild_card_pool.extend(sorted_div[1:])
        div_winners.sort(key=lambda x: x[1], reverse=True)
        wild_card_pool.sort(key=lambda x: x[1], reverse=True)
        if len(wild_card_pool) >= 3:
            playoff_seeds[league] = [t[0] for t in div_winners] + [t[0] for t in wild_card_pool[:3]]

    ws_teams = []
    for league in ['AL', 'NL']:
        seeds = playoff_seeds[league]
        if len(seeds) < 6: continue
        
        logs.append(f"\n[{league} 季後賽]")
        wc1, w1, l1 = simulate_series(seeds[2], seeds[5], 2)
        wc2, w2, l2 = simulate_series(seeds[3], seeds[4], 2)
        logs.append(f"外卡: {seeds[2]} def {seeds[5]} ({w1}-{l1})")
        logs.append(f"外卡: {seeds[3]} def {seeds[4]} ({w2}-{l2})")
        
        ds1, w3, l3 = simulate_series(seeds[0], wc2, 3)
        ds2, w4, l4 = simulate_series(seeds[1], wc1, 3)
        logs.append(f"分區賽: {seeds[0]} def {wc2} ({w3}-{l3})")
        logs.append(f"分區賽: {seeds[1]} def {wc1} ({w4}-{l4})")
        
        cs, w5, l5 = simulate_series(ds1, ds2, 4)
        logs.append(f"冠軍賽: {cs} def {ds1 if cs!=ds1 else ds2} ({w5}-{l5})")
        ws_teams.append(cs)
        
    if len(ws_teams) == 2:
        champion, w6, l6 = simulate_series(ws_teams[0], ws_teams[1], 4)
        logs.append(f"\n 世界大賽: {champion} def {ws_teams[0] if champion!=ws_teams[0] else ws_teams[1]} ({w6}-{l6})")
        return champion, logs
    return "Error", logs

print("正在初始化模擬...")
sim_champion, sim_logs = simulate_season()
print("初始化完成")

# --- 7. 圖表函式 ---
def get_normalized_stats(team_code):
    # last_stats: [Win%, R, RA, H, E, EqA, ERA, WHIP]
    # Index:      0     1  2   3  4  5    6    7
    stats = last_stats.get(team_code, [0.5, 4, 4, 8, 0.5, 0.25, 4.50, 1.35])
    
    win_rate = stats[0]
    scoring = min(stats[1] / 8.0, 1.0)
    pitching = max(0, (9.0 - stats[6]) / 9.0) # 使用 ERA (index 6)
    hitting = min(stats[5] / 0.400, 1.0) # 使用 EqA (index 5), 0.400為高標
    defense = max(0, (2.0 - stats[4]) / 2.0)
    
    return [win_rate, scoring, pitching, hitting, defense]

def create_gauge_chart(prob_home):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_home * 100,
        title = {'text': "綜合預測主隊勝率 (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"},
                {'range': [60, 100], 'color': "lightblue"}
            ],
            'threshold' : {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_radar_chart(home_stats, vis_stats, home_name, vis_name):
    categories = ['勝率', '得分火力', '投手壓制', '攻擊指數(EqA)', '守備穩定']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_stats, theta=categories, fill='toself', name=f'{home_name} (主)'))
    fig.add_trace(go.Scatterpolar(r=vis_stats, theta=categories, fill='toself', name=f'{vis_name} (客)'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=300, margin=dict(l=40, r=40, t=30, b=30)
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# --- 8. 路由 ---
@app.route('/', methods=['GET', 'POST'])
def index():
    global sim_champion, sim_logs
    
    if request.args.get('action') == 'resimulate':
        sim_champion, sim_logs = simulate_season()
        
    prediction_result = None
    gauge_json = None
    radar_json = None
    
    if request.method == 'POST':
        home = request.form.get('home_team')
        vis = request.form.get('vis_team')
        
        rf_prob = predict_single_match(home, vis, 'rf')
        xgb_prob = predict_single_match(home, vis, 'xgb')
        keras_prob = predict_single_match(home, vis, 'keras')
        
        avg_prob = (rf_prob + xgb_prob + keras_prob) / 3
        
        prediction_result = {
            'rf_home': round(rf_prob * 100, 1),
            'rf_vis': round((1-rf_prob) * 100, 1),
            'xgb_home': round(xgb_prob * 100, 1),
            'xgb_vis': round((1-xgb_prob) * 100, 1),
            'keras_home': round(keras_prob * 100, 1),
            'keras_vis': round((1-keras_prob) * 100, 1),
            'home_team': home,
            'vis_team': vis,
            'home_div': TEAM_DISPLAY_INFO.get(home, ''),
            'vis_div': TEAM_DISPLAY_INFO.get(vis, '')
        }
        
        # 繪製圖表
        gauge_json = create_gauge_chart(avg_prob)
        h_norm = get_normalized_stats(home)
        v_norm = get_normalized_stats(vis)
        radar_json = create_radar_chart(h_norm, v_norm, home, vis)

    return render_template('index.html', 
                           teams=teams_options, 
                           result=prediction_result,
                           champion=sim_champion,
                           logs=sim_logs,
                           gauge_json=gauge_json,
                           radar_json=radar_json)

if __name__ == '__main__':
    app.run(debug=True)
