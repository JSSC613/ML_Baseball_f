### ⚾ MLB 賽事勝率預測與 2025 賽季模擬系統
(MLB Win Probability Predictor & Season Simulator)
這是一個整合 機器學習 (Machine Learning)、深度學習 (Deep Learning) 與 賽伯計量學 (Sabermetrics) 的全端預測系統。
本專案利用 2013 至 2024 年的 MLB 歷史數據（包含比賽結果、球隊數據、投手數據），訓練出能預測單場比賽勝率的模型，並透過 蒙地卡羅方法 (Monte Carlo Simulation) 模擬符合真實賽制的 162 場例行賽與季後賽，最終預測 2025 年的世界大賽冠軍。
## 🚀 專案核心特色 (Project Highlights)
1. 多模型集成預測 (Ensemble Modeling)
為了提高預測的可信度，系統同時運行三種不同原理的模型，並在前端並列顯示結果：
Random Forest (隨機森林)：基於 Bagging 的決策樹集合。
XGBoost (極限梯度提升樹)：目前結構化數據競賽中表現最強的模型。
Deep Learning (Keras)：使用多層感知機 (MLP) 捕捉非線性特徵。
2. 引入 Sabermetrics (賽伯計量學)
本專案不只使用傳統勝率，更深入挖掘反映球隊真實戰力的進階數據：
畢達哥拉斯期望勝率 (Pythagorean Expectation)：使用 
1.83
1.83
 指數公式，根據得失分差校正運氣成分。
WHIP (每局被上壘率)：評估先發投手壓制力的關鍵指標。
得失分差 (Run Differential)：預測長期勝率的最強單一指標。
3. 先發投手權重 (Starting Pitcher Integration)
突破傳統模型只看「球隊」的盲點，本系統解析了 先發投手 (Starting Pitcher) 身分，計算其歷史 ERA (防禦率) 與 WHIP，讓模型能區分「王牌先發」與「後段輪值」的比賽差異。
4. 擬真賽季模擬 (Realistic Season Simulation)
加權賽程：模擬 162 場例行賽，且同分區對戰頻率 (13場) > 同聯盟 (6場) > 跨聯盟 (3場)。
完整季後賽樹狀圖：依據模擬戰績決定種子序，進行外卡戰 (Best of 3) → 分區系列賽 (Best of 5) → 聯盟冠軍賽 (Best of 7) → 世界大賽。
動態隨機性：每場模擬比賽均基於模型預測的勝率進行擲骰，確保每次模擬結果的獨特性。
## 🛠️ 技術架構
語言: Python 3.x
網頁框架: Flask, Jinja2, Bootstrap 5
資料處理: Pandas, NumPy
機器學習: Scikit-Learn, XGBoost
深度學習: TensorFlow / Keras
資料來源: Retrosheet / Baseball Databank
## 📂 專案結構
code
Text
Baseball_ML/
│
├── data/                   # 數據存放區
│   ├── 2013teamstats.csv ~ 2024teamstats.csv
│   ├── 2013gameinfo.csv ~ 2024gameinfo.csv
│   ├── 2013pitching.csv ~ 2024pitching.csv  (新增：投手數據)
│   └── processed_data.csv  # 經過清洗、合併投手數據後的最終訓練集
│
├── models/                 # 訓練好的模型與轉換器
│   ├── rf_model.pkl        # Random Forest 模型
│   ├── xgb_model.pkl       # XGBoost 模型
│   ├── keras_model.h5      # Keras 深度學習模型
│   └── scaler.pkl          # 數據標準化 Scaler
│
├── src/                    # 核心邏輯程式碼
│   ├── data_processing.py  # 資料清洗、投手解析、計算 Rolling Stats
│   ├── train_models.py     # 訓練三種模型、特徵工程、儲存模型
│   └── team_info.py        # MLB 聯盟分區結構定義、賽程權重判斷
│
├── templates/              # 前端頁面
│   └── index.html          # 預測介面與模擬結果顯示
│
├── app.py                  # Flask 啟動檔、即時特徵計算、賽季模擬邏輯
└── requirements.txt        # 套件依賴清單
## 📊 特徵工程 (17 Dimensions)
模型輸入特徵包含以下 17 個維度，分為四大類：
對戰差異 (Differentials)：
diff_win_rate: 勝率差
diff_pyth: 畢達哥拉斯期望勝率差
diff_run_diff: 得失分差的差距
diff_sp_era: 先發投手 ERA 差距
diff_sp_whip: 先發投手 WHIP 差距
基礎實力: 主客隊賽前勝率 (pre_win_rate)、期望勝率 (pyth)。
近期狀況 (Rolling 10 games): 主客隊近 10 場的得分、失分、安打、失誤。
投手數據: 主客隊先發投手的歷史 ERA 與 WHIP。
## 🚀 如何執行
1. 安裝依賴
code
Bash
pip install requirements
2. 資料處理 (Data Processing)
解析原始 CSV，提取先發投手，計算進階數據。
code
Bash
python src/data_processing.py
(成功執行後會產生 data/processed_data.csv)
3. 訓練模型 (Model Training)
讀取處理後的資料，訓練 RF, XGBoost, Keras 並儲存。
code
Bash
python src/train_models.py
(成功執行後 models/ 資料夾會出現 4 個檔案)
4. 啟動系統
code
Bash
python app.py
開啟瀏覽器訪問 http://127.0.0.1:5000。
## 📈 模型表現與限制
準確率: 加入投手數據後，XGBoost 模型在測試集上的準確率約為 55% - 58%。
限制: 棒球比賽具有高度隨機性（單場比賽運氣成分重），且目前模型尚未納入「牛棚疲勞度」與「打者對戰拆分 (Splits)」數據。
模擬意義: 雖然單場預測有極限，但透過 162 場的大量模擬，能夠有效消除隨機雜訊，真實反映球隊的長期實力排名。
Data provided by Retrosheet.
