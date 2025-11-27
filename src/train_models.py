# src/train_models.py
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
# 修正這裡：補上 RandomizedSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

def train():
    print("正在載入訓練資料...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("❌ 請先執行 data_processing.py")
        return
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # --- 特徵工程 ---
    # 確保分母不為 0
    df['home_pyth'] = (df['home_roll_b_r']**1.83) / ((df['home_roll_b_r']**1.83 + df['home_roll_p_r']**1.83) + 1e-9)
    df['vis_pyth'] = (df['vis_roll_b_r']**1.83) / ((df['vis_roll_b_r']**1.83 + df['vis_roll_p_r']**1.83) + 1e-9)
    
    df['diff_win_rate'] = df['home_pre_win_rate'] - df['vis_pre_win_rate']
    df['diff_pyth'] = df['home_pyth'] - df['vis_pyth']
    df['diff_run_diff'] = (df['home_roll_b_r'] - df['home_roll_p_r']) - (df['vis_roll_b_r'] - df['vis_roll_p_r'])
    
    # 確保投手欄位存在
    if 'home_sp_era' not in df.columns: df['home_sp_era'] = 4.5
    if 'vis_sp_era' not in df.columns: df['vis_sp_era'] = 4.5
    if 'home_sp_whip' not in df.columns: df['home_sp_whip'] = 1.35
    if 'vis_sp_whip' not in df.columns: df['vis_sp_whip'] = 1.35
    
    df['diff_sp_era'] = df['home_sp_era'] - df['vis_sp_era']
    df['diff_sp_whip'] = df['home_sp_whip'] - df['vis_sp_whip']
    
    # 確保 EqA 欄位存在
    if 'home_roll_b_eqa' not in df.columns: df['home_roll_b_eqa'] = 0.250
    if 'vis_roll_b_eqa' not in df.columns: df['vis_roll_b_eqa'] = 0.250
    
    df['diff_eqa'] = df['home_roll_b_eqa'] - df['vis_roll_b_eqa']
    
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
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
    target = 'home_target'
    
    # 分割 (使用 stratify 確保勝負比例一致)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True, stratify=df[target])
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"特徵數量: {len(features)}")
    print(f"訓練集: {len(X_train)}, 測試集: {len(X_test)}")
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    # --- 1. XGBoost (使用 RandomizedSearchCV 尋找最佳參數) ---
    print("\n🚀 正在最佳化 XGBoost 參數 (這需要幾分鐘)...")
    
    xgb_param_dist = {
        'n_estimators': [100, 300, 500, 800],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }
    
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    
    # 隨機搜尋 20 組參數
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_param_dist,
        n_iter=20,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    
    print(f"✅ XGBoost 最佳參數: {search.best_params_}")
    print(f"✅ XGBoost 最佳驗證分數: {search.best_score_:.4f}")
    
    y_pred_xgb = best_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"🏆 XGBoost 測試集正確率: {acc_xgb:.4f}")
    joblib.dump(best_xgb, os.path.join(MODEL_DIR, 'xgb_model.pkl'))

    # --- 2. Random Forest (簡單參數調整) ---
    print("\n🌲 訓練 Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, random_state=42)
    rf.fit(X_train, y_train)
    print(f"RF Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")
    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_model.pkl'))
    
    # --- 3. Keras (深度學習優化版) ---
    print("\n🧠 訓練 Keras Neural Network (優化版)...")
    # 加入 BatchNormalization 和 EarlyStopping 防止過擬合
    model = Sequential([
        Input(shape=(len(features),)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # 早停機制：如果驗證集準確率 10 次沒提升就停止訓練
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # 自動降低學習率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    model.fit(
        X_train_scaled, y_train, 
        epochs=100, 
        batch_size=64, 
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    _, acc_keras = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"🏆 Keras Accuracy: {acc_keras:.4f}")
    model.save(os.path.join(MODEL_DIR, 'keras_model.h5'))
    
    print("\n=== 訓練完成 ===")

if __name__ == "__main__":
    train()