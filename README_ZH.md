# 🏐 排球扣球分析系統

一個基於電腦視覺的排球扣球技術分析系統，使用姿態估計和 3D 視覺化技術。

## ✨ 功能特色

### 核心分析功能
- **📹 視頻上傳** - 支援多種視頻格式（MP4、AVI、MOV、MKV）
- **🤖 姿態提取** - 使用 MediaPipe Pose 進行即時骨架追蹤
- **📊 2D 與 3D 座標** - 提取 2D 和 3D 關節點位置
- **🎬 3D 視覺化** - 使用 Plotly 建立互動式 3D 骨架動畫

### 進階分析功能（最新）
- **🎯 動作階段檢測** - 自動檢測 5 個動作階段（助跑 → 起跳 → 揮臂 → 擊球 → 落地）
- **📐 關節角度分析** - 計算關鍵角度（肩部外展、肘部彎曲、軀幹旋轉）
- **🚀 速度分析** - 計算手腕、手肘和肩部的線性與角速度
- **📏 空間指標** - 跳躍高度、擊球高度、滯空時間、水平位移
- **🏷️ 動作分類** - 將揮臂動作分為 5 種類型（直線式、弓箭式高/低位、快攻式、圓弧式）
- **🔄 多視頻比較** - 並排比較多個視頻，提供相似度指標
- **📄 報告生成** - 生成完整的 HTML 和文字報告

### 匯出與整合
- **💾 數據匯出** - 匯出姿態數據為 CSV/JSON 格式供進一步分析
- **🖥️ 網頁介面** - 使用 Streamlit 建立的友善使用者介面

## 🚀 快速開始

### 系統需求

- Python 3.8 或更高版本
- pip 套件管理器

### 安裝步驟

1. **複製專案儲存庫**
```bash
git clone <repository-url>
cd volleyball-spike-analyzer
```

2. **建立虛擬環境**（建議）
```bash
python -m venv venv

# Windows 系統
venv\Scripts\activate

# macOS/Linux 系統
source venv/bin/activate
```

3. **安裝相依套件**
```bash
pip install -r requirements.txt
```

### 執行應用程式

啟動 Streamlit 網頁介面：

```bash
streamlit run app.py
```

應用程式將在預設瀏覽器中開啟，網址為 `http://localhost:8501`。

## 📖 使用說明

### 1. 上傳視頻

上傳排球扣球視頻，支援 MP4、AVI、MOV 或 MKV 格式。

### 2. 查看 3D 骨架動畫

探索互動式 3D 骨架動畫，包含播放/暫停控制和逐幀導航功能。

### 3. 分析關節角度

檢視肩部、肘部、髖部和膝部的詳細關節角度測量結果。

### 4. 下載結果

匯出分析結果：
- **3D 關節點數據（CSV）** - 完整的 3D 座標數據
- **關節角度（CSV）** - 逐幀角度測量值
- **摘要（JSON）** - 分析的統計摘要

## 📊 分析功能詳解

### 動作階段檢測

系統自動檢測排球扣球動作的 5 個關鍵階段：

1. **助跑（Approach）** - 高水平速度的跑動階段
2. **起跳（Takeoff）** - 跳躍啟動，垂直加速度達到峰值
3. **揮臂（Arm Swing）** - 手臂從開始擺動到速度峰值
4. **擊球（Contact）** - 手部速度峰值附近的球接觸階段
5. **落地（Landing）** - 下降和著地

此外，揮臂階段根據研究文獻進一步細分為 3 個子階段：
- **第一階段（Initiation）** - 手腕和手肘開始上升
- **第二階段（Wind-up）** - 手腕達到最高點
- **第三階段（Final Cocking）** - 手腕下降並加速

### 揮臂動作分類

系統將揮臂動作分為 5 種類型：

1. **直線式（Straight）** - 高弧線動作，帶有停頓
2. **弓箭式高位（Bow-Arrow High）** - 極高弧線，帶有停頓（最大力量）
3. **弓箭式低位（Bow-Arrow Low）** - 中等弧線，帶有停頓（平衡型）
4. **快攻式（Snap）** - 肩膀高度的水平動作（快速攻擊）
5. **圓弧式（Circular）** - 低位連續動作，無停頓（最快速）

### 生物力學指標

**關節角度：**
- 肩部外展角度
- 肩部水平外展角度
- 肘部彎曲角度
- 軀幹旋轉角度
- 軀幹傾斜角度

**速度指標：**
- 手腕線性速度（最大值、平均值、擊球時）
- 手肘線性速度
- 肩部線性速度
- 肩部角速度
- 肘部角速度

**空間指標：**
- 跳躍高度（2 種計算方法）
- 擊球點高度
- 滯空時間
- 水平位移（前後與左右）
- 質心軌跡

### 多視頻比較

並排比較多個扣球視頻：
- 對齊的速度曲線（以擊球幀同步）
- 雷達圖顯示性能指標
- 技術之間的相似度評分
- 包含所有關鍵指標的比較表

### 報告生成

生成完整的分析報告：
- **HTML 報告** - 包含所有指標的互動式樣式報告
- **文字報告** - 快速檢視的純文字摘要
- **CSV 匯出** - 逐幀詳細數據供自訂分析
- **JSON 匯出** - 結構化數據供程式化存取

## 🏗️ 專案結構

```
volleyball-spike-analyzer/
├── app.py                              # Streamlit 主應用程式
├── config/
│   └── config.yaml                     # 配置設定
├── src/
│   ├── core/
│   │   ├── pose_extractor.py          # MediaPipe 姿態提取
│   │   └── skeleton_processor.py      # 骨架數據處理
│   ├── analysis/                       # 分析模組（新）
│   │   ├── phase_detector.py          # 5 階段和 3 階段檢測
│   │   ├── joint_angles.py            # 關節角度計算
│   │   ├── velocity_calculator.py     # 線性與角速度
│   │   ├── spatial_metrics.py         # 跳躍高度、位移
│   │   ├── arm_swing_classifier.py    # 動作類型分類
│   │   └── metrics_summary.py         # 結果匯總
│   ├── comparison/                     # 多視頻比較（新）
│   │   └── multi_video_comparator.py  # 並排比較
│   ├── reporting/                      # 報告生成（新）
│   │   └── report_generator.py        # HTML/文字報告
│   ├── visualization/
│   │   ├── video_overlay.py           # 2D 骨架疊加
│   │   └── skeleton_3d.py             # 3D 視覺化
│   └── utils/
│       ├── video_io.py                # 視頻 I/O 操作
│       └── data_export.py             # 數據匯出工具
├── tests/
│   ├── test_pose_extractor.py         # 單元測試
│   └── test_joint_angles.py           # 角度計算測試
└── data/
    ├── input/                          # 輸入視頻
    ├── output/                         # 處理結果
    └── cache/                          # 臨時快取
```

## ⚙️ 配置設定

透過編輯 `config/config.yaml` 自訂系統設定：

### MediaPipe 設定
- **model_complexity**（0-2）：數值越高越準確但速度越慢
- **min_detection_confidence**（0.0-1.0）：初始檢測的最小信心值
- **min_tracking_confidence**（0.0-1.0）：追蹤的最小信心值

### 視頻處理
- **max_file_size_mb**：最大上傳大小（預設：500MB）
- **output_fps**：輸出視頻幀率（預設：30）
- **resize_width/height**：可選的視頻調整大小（預設：1280x720）

### 視覺化
- **landmark_color**：關節點標記的 RGB 顏色（預設：綠色）
- **connection_color**：骨架線條的 RGB 顏色（預設：紅色）
- **figure_width/height**：3D 圖表大小（預設：1200x800）

### 分析
- **calculate_angles**：啟用關節角度計算
- **calculate_velocities**：啟用速度計算
- **smoothing_window**：軌跡平滑的視窗大小（預設：5）

## 🔬 技術細節

### MediaPipe Pose

本專案使用 [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) 進行人體姿態估計：
- **33 個身體關節點** 的 2D 座標（x, y, 可見度）
- **33 個身體關節點** 的 3D 座標（x, y, z, 可見度）
- CPU 上的即時性能
- 運動動作的高準確度

### 扣球分析的關鍵關節點

系統專注於排球扣球分析的 12 個關鍵關節：
- **上半身**：左/右肩部、手肘、手腕
- **下半身**：左/右髖部、膝部、踝部

### 數據輸出格式

- **CSV**：表格數據帶時間戳，便於在 Excel/Python 中分析
- **JSON**：帶元數據的結構化數據，供程式使用
- **Parquet**：大型數據集的高性能格式（可選）

## 🧪 開發

### 執行測試

```bash
# 執行所有測試
pytest tests/

# 執行並顯示覆蓋率
pytest tests/ --cov=src --cov-report=html

# 執行特定測試檔案
pytest tests/test_pose_extractor.py -v
```

### 程式碼品質

```bash
# 格式化程式碼
black src/ tests/

# 程式碼檢查
flake8 src/ tests/

# 型別檢查
mypy src/
```

### 專案指南

- **型別提示**：所有函式包含型別註解
- **文件字串**：所有公開 API 使用 Google 風格的文件字串
- **錯誤處理**：全面的輸入驗證和例外處理
- **測試**：核心功能的單元測試

## 🗺️ 開發路線圖

### ✅ 第一階段：MVP（已完成）
- [x] 專案結構設定
- [x] 基本姿態提取
- [x] 3D 骨架視覺化
- [x] Streamlit 介面
- [x] 數據匯出功能

### ✅ 第二階段：增強分析（已完成）
- [x] 自動扣球階段檢測（5 個階段：助跑 → 起跳 → 揮臂 → 擊球 → 落地）
- [x] 揮臂子階段檢測（基於研究的第 I/II/III 階段）
- [x] 速度與加速度指標（線性與角速度）
- [x] 跳躍高度估計（2 種方法：髖部位移與滯空時間）
- [x] 擊球高度和水平位移
- [x] 多視頻並排比較
- [x] 動作分類（5 種揮臂類型）
- [x] 完整報告生成（HTML/文字）
- [x] 性能指標儀表板

### 🚀 第三階段：未來增強功能
- [ ] 多人追蹤
- [ ] 技術評分系統與個人化建議
- [ ] 基於分類的動作建議
- [ ] 跨多個訓練階段的歷史趨勢分析
- [ ] 匯出帶有疊加和註解的視頻
- [ ] 多視頻批次處理
- [ ] PDF 報告生成
- [ ] 從網路攝影機/相機即時分析

## 🤝 貢獻

這是一個個人專案，但歡迎貢獻！請隨時：
- 回報錯誤或問題
- 建議新功能
- 提交拉取請求
- 分享您的分析結果

## 📄 授權

MIT 授權 - 可自由用於您的專案並進行修改。

## 🙏 致謝

- [MediaPipe](https://google.github.io/mediapipe/) - 姿態估計框架
- [Streamlit](https://streamlit.io/) - 網頁應用程式框架
- [Plotly](https://plotly.com/) - 互動式 3D 視覺化
- [OpenCV](https://opencv.org/) - 電腦視覺函式庫

## 📧 聯絡方式

如有問題、反饋或合作機會，請在 GitHub 上開啟一個 issue。

---

**用❤️為排球運動員和教練打造**
