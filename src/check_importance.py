# src/check_importance.py
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

# 設定路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_model.pkl')

# 載入 XGBoost 模型
model = joblib.load(MODEL_PATH)

# 定義特徵名稱 (必須與訓練時順序一致)
features = [
    'diff_win_rate', 'diff_pyth', 'diff_run_diff', 
    'diff_sp_era', 'diff_sp_whip', 'diff_eqa',
    'home_pre_win_rate', 'vis_pre_win_rate',
    'home_pyth', 'vis_pyth',
    'home_roll_b_r', 'vis_roll_b_r',
    'home_roll_p_r', 'vis_roll_p_r',
    'home_sp_era', 'vis_sp_era',
    'home_sp_whip', 'vis_sp_whip',
    'home_roll_b_eqa', 'vis_roll_b_eqa'
]

# 取得重要性並排序
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

# 繪圖
plt.figure(figsize=(10, 8))
importances.plot(kind='barh', color='skyblue')
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# 印出數值
print("=== 特徵重要性排名 (Top 10) ===")
print(importances.sort_values(ascending=False).head(10))