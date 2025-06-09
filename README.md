# 🏓 桌球選手表現預測模型（TSFEL + XGBoost + Soft Voting）

本專案針對桌球選手之時間序列動作資料進行預測建模，目標為推論選手之性別、持拍手、球齡與等級等資訊。  
透過 TSFEL 特徵提取工具對原始感測資料進行轉換，並結合 XGBoost 分類模型進行訓練與預測。最終採用模型一與模型二的 soft voting 機制進行集成，提升整體預測效果。

---

## 📁 專案結構

```
├── data/
│   ├── 39_Training_Dataset/     # 原始訓練資料
│   └── 39_Test_Dataset/         # 原始測試資料
│
├── TSFEL/
│   ├── train/                   # TSFEL 特徵提取後訓練資料
│   └── test/                    # TSFEL 特徵提取後測試資料
│
├── cutpoint_generator.ipynb                # 資料切段邏輯
├── tsfel_feature_extract.ipynb            # 特徵提取邏輯
├── xgb_model1_cutpoint_20250606.ipynb  # 切分後 TSFEL + XGB 預測
└── xgb_model2_raw_20250606.ipynb  # 未切分 TSFEL + XGB 預測
```

---

## 📦 執行環境

| 項目       | 說明                    |
|------------|-------------------------|
| 作業系統   | Ubuntu 22.04.3 LTS      |
| 顯示卡     | Nvidia RTX A5000（CUDA 12.2） |
| Python 版本 | 3.10.16                 |

### 🔧 主要套件

- `tsfel`：時間序列特徵提取
- `xgboost`：分類模型
- `imblearn`：SMOTE 類別平衡
- `scikit-learn`：K-Fold 分割、集成投票
- `tqdm`：進度條顯示

---

## 🚀 使用說明

### 步驟一: 安裝套件

建立處理環境後，執行:

```bash
pip install -r requirements.txt
```

### 步驟二: 資料預處理與特徵提取

1. 執行 `cutpoint_generator.ipynb`：將原始 txt 資料依據邏輯切分成片段
2. 執行 `tsfel_feature_extract.ipynb`：對切分後與未切分資料進行 TSFEL 特徵提取，並存至 `TSFEL/` 資料夾

### 步驟三: 模型訓練與預測

3. 執行 `xgb_model1_cutpoint_20250606.ipynb`
   - 切點 + TSFEL + SMOTE + K-Fold + XGBoost
4. 執行 `xgb_model2_raw_20250606.ipynb`
   - 未切分 + TSFEL + XGBoost

### 步驟四: 最終 Soft Voting

5. 將上述兩模型預測機率進行 **soft voting 平均**
6. 將最終預測結果匯出為 CSV

---

## 🔬 模型比較與集成

| 模型 | 切分方式 | 特徵 | 特色 | 備註 |
|--------|----------|------|-----------------------|--------|
| 模型一 | 切點分段 | TSFEL | SMOTE + KFold + 機率調整 | 表現優質 |
| 模型二 | 未切分     | TSFEL | 無                 | 補充模型 |
| Soft Voting | - | - | 平均兩者機率 | 最終預測結果 |

