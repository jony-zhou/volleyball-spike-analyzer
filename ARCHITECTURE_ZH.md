# 排球扣球分析系統 - 架構與技術說明

## 目錄
1. [系統概述](#系統概述)
2. [整體架構](#整體架構)
3. [核心模組詳解](#核心模組詳解)
4. [數據流程](#數據流程)
5. [演算法與判斷方法](#演算法與判斷方法)
6. [關鍵技術細節](#關鍵技術細節)
7. [擴展開發指南](#擴展開發指南)

---

## 系統概述

### 專案目標
本系統旨在透過電腦視覺技術自動分析排球扣球動作，提供量化的生物力學指標，協助運動員和教練改善技術。

### 技術棧
- **前端介面**：Streamlit（網頁應用框架）
- **姿態估計**：MediaPipe Pose（Google 開發的機器學習模型）
- **數據處理**：NumPy、Pandas、SciPy
- **視覺化**：Plotly（3D 互動圖表）、OpenCV（2D 視頻處理）
- **語言**：Python 3.8+

### 系統特點
- **非侵入式**：僅需視頻，無需穿戴感測器
- **即時處理**：CPU 即可運行，無需昂貴的 GPU
- **全面分析**：從動作階段檢測到生物力學指標的完整分析鏈
- **易於使用**：網頁介面，無需程式設計知識

---

## 整體架構

### 管線式架構（Pipeline Architecture）

系統採用管線式架構，數據依序經過以下階段：

```
視頻輸入
  ↓
[1] 姿態提取（PoseExtractor）
  ↓
[2] 骨架數據處理（SkeletonProcessor）
  ↓
[3] 動作階段檢測（PhaseDetector）
  ↓
[4] 生物力學分析（JointAngles, Velocity, SpatialMetrics）
  ↓
[5] 動作分類（ArmSwingClassifier）
  ↓
[6] 視覺化與匯出（Visualization, DataExporter）
```

### 模組分層

```
┌─────────────────────────────────────┐
│   應用層 (Application Layer)        │
│   - app.py (Streamlit UI)           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   分析層 (Analysis Layer)            │
│   - phase_detector.py                │
│   - joint_angles.py                  │
│   - velocity_calculator.py           │
│   - spatial_metrics.py               │
│   - arm_swing_classifier.py          │
│   - metrics_summary.py               │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   核心層 (Core Layer)                │
│   - pose_extractor.py                │
│   - skeleton_processor.py            │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   工具層 (Utility Layer)             │
│   - video_io.py                      │
│   - data_export.py                   │
│   - visualization (video_overlay,    │
│     skeleton_3d)                     │
└─────────────────────────────────────┘
```

---

## 核心模組詳解

### 1. 姿態提取模組（pose_extractor.py）

#### 功能
- 使用 MediaPipe Pose 從視頻幀中提取人體骨架
- 輸出 33 個關節點的 2D 和 3D 座標

#### 關鍵類別：`PoseExtractor`

**主要方法：**
- `extract_pose(frame)`: 從單幀提取姿態
- `extract_from_video(video_path)`: 從整個視頻提取姿態序列

**MediaPipe 33 個關節點：**
```
0: 鼻子          11-12: 肩膀（左/右）    23-24: 髖部（左/右）
1-8: 眼睛與嘴    13-14: 手肘（左/右）    25-26: 膝蓋（左/右）
9-10: 耳朵       15-16: 手腕（左/右）    27-28: 腳踝（左/右）
                 17-22: 手部關節點        29-32: 腳部關節點
```

**輸出格式：**
```python
{
    'landmarks_2d': np.array(33, 3),  # [x, y, visibility]
    'landmarks_3d': np.array(33, 4)   # [x, y, z, visibility]
}
```

**座標系統：**
- **2D 座標**：標準化座標（0-1），以圖像左上角為原點
- **3D 座標**：以髖部中心為原點的相對座標（單位：米）
  - X 軸：左右（正值向右）
  - Y 軸：上下（正值向上）
  - Z 軸：前後（正值向前）

---

### 2. 骨架處理模組（skeleton_processor.py）

#### 功能
- 平滑軌跡數據（減少抖動）
- 填補缺失幀
- 計算基本的速度和加速度

#### 關鍵類別：`SkeletonProcessor`

**主要方法：**
- `process_sequence(landmarks_sequence, fps)`: 處理完整的關節點序列
- `smooth_trajectory(positions)`: 使用 Savitzky-Golay 濾波器平滑軌跡

**平滑演算法：**
使用 Savitzky-Golay 濾波器（多項式濾波）：
- 窗口大小：5 幀（可配置）
- 多項式階數：2
- 優點：既能平滑噪音，又能保留峰值

---

### 3. 動作階段檢測模組（phase_detector.py）

#### 3.1 完整動作階段檢測（FullMotionPhaseDetector）

**檢測的 5 個階段：**

1. **助跑階段（Approach）**
   - **判斷依據**：水平速度（Z 軸）持續高於閾值（預設 1.0 m/s）
   - **檢測方法**：分析髖部中心的水平速度曲線

2. **起跳階段（Takeoff）**
   - **判斷依據**：垂直加速度達到峰值
   - **檢測方法**：
     - 計算髖部的垂直（Y 軸）加速度
     - 找到加速度峰值點
     - 起跳階段 = 峰值前 0.2 秒到峰值後 0.1 秒

3. **揮臂階段（Arm Swing）**
   - **判斷依據**：手腕開始上升到達到最高速度
   - **檢測方法**：
     - 分析手腕（right_wrist, index 16）的運動軌跡
     - 起點：手腕 Y 座標開始明顯上升
     - 終點：手腕速度達到峰值

4. **擊球階段（Contact）**
   - **判斷依據**：手腕速度峰值附近的時間窗口
   - **檢測方法**：
     - 找到手腕速度的最大值點
     - 擊球階段 = 峰值前後各 0.1 秒（可配置）

5. **落地階段（Landing）**
   - **判斷依據**：擊球後到腳踝 Y 座標停止下降
   - **檢測方法**：
     - 分析腳踝的垂直位置
     - 找到腳踝 Y 座標趨於穩定的點

#### 3.2 揮臂子階段檢測（ArmSwingPhaseDetector）

基於運動生物力學研究，將揮臂階段細分為 3 個子階段：

**第 I 階段（Initiation - 啟動階段）**
- **定義**：手腕和手肘開始上升
- **判斷**：手腕 Y 座標開始增加（相對於起跳點）
- **生物力學意義**：動能開始累積

**第 II 階段（Wind-up - 後仰階段）**
- **定義**：手腕達到最高點
- **判斷**：手腕 Y 座標達到最大值
- **生物力學意義**：位能最大，準備轉換為動能

**第 III 階段（Final Cocking - 最終加速階段）**
- **定義**：手腕下降並加速至擊球
- **判斷**：手腕 Y 座標下降且速度急劇增加
- **生物力學意義**：位能轉換為動能，產生最大擊球速度

**演算法流程：**
```python
1. 在揮臂階段範圍內提取手腕的 Y 座標時間序列
2. 使用 Savitzky-Golay 濾波平滑數據
3. 找到 Y 座標的峰值點 → 第 II 階段的結束點
4. 峰值前的上升段 → 第 I 階段
5. 峰值後到速度峰值 → 第 III 階段
```

---

### 4. 關節角度計算模組（joint_angles.py）

#### 功能
計算關鍵關節角度，用於技術評估和動作分類。

#### 關鍵類別：`JointAngleCalculator`

**計算的角度：**

1. **肩部外展角度（Shoulder Abduction）**
   - **定義**：手臂與軀幹側面的夾角
   - **計算方法**：
     ```
     向量 1：肩部 → 髖部（軀幹向量）
     向量 2：肩部 → 手肘（上臂向量）
     角度 = arccos(向量 1 · 向量 2 / |向量 1| |向量 2|)
     ```

2. **肩部水平外展角度（Shoulder Horizontal Abduction）**
   - **定義**：手臂在水平面上與身體中線的夾角
   - **計算方法**：投影到 XZ 平面後計算角度

3. **肘部彎曲角度（Elbow Flexion）**
   - **定義**：上臂與前臂的夾角
   - **計算方法**：
     ```
     向量 1：肘部 → 肩部
     向量 2：肘部 → 手腕
     角度 = arccos(向量 1 · 向量 2 / |向量 1| |向量 2|)
     ```
   - **典型範圍**：0°（完全伸直）到 150°（完全彎曲）

4. **軀幹旋轉角度（Torso Rotation）**
   - **定義**：肩線與髖線在水平面上的夾角
   - **計算方法**：
     ```
     肩線向量 = 右肩 - 左肩
     髖線向量 = 右髖 - 左髖
     旋轉角 = 兩向量在 XZ 平面的投影夾角
     ```

5. **軀幹傾斜角度（Torso Lean）**
   - **定義**：軀幹與垂直線的夾角
   - **計算方法**：軀幹向量與 Y 軸的夾角

**時間序列分析：**
- `calculate_angles_timeseries()`: 計算每一幀的所有角度
- 輸出 DataFrame，包含時間戳和各角度值
- 用於生成角度曲線圖和階段平均值

---

### 5. 速度計算模組（velocity_calculator.py）

#### 功能
計算關鍵點的線性速度和關節的角速度。

#### 關鍵類別：`VelocityCalculator`

**線性速度計算：**

1. **手腕速度**
   ```python
   位移 = 位置[t+1] - 位置[t]
   速度 = ||位移|| × fps
   ```

2. **手肘速度**

3. **肩部速度**

**角速度計算：**

1. **肩部角速度**
   ```python
   角度變化 = 角度[t+1] - 角度[t]
   角速度 = 角度變化 × fps
   ```

2. **肘部角速度**

**關鍵指標：**
- **最大值**：整個動作中的峰值速度
- **平均值**：整個動作的平均速度
- **擊球時速度**：擊球瞬間的速度（最重要的指標）

**速度分析方法：**
```python
analyze_velocity_profile(skeleton_df, angles_df, phases):
    1. 計算每一幀的速度
    2. 根據階段邊界提取各階段的速度數據
    3. 計算統計量（最大、平均、標準差）
    4. 找到擊球時刻的速度
    返回：速度指標字典
```

---

### 6. 空間指標計算模組（spatial_metrics.py）

#### 功能
計算跳躍高度、擊球高度、位移等空間指標。

#### 關鍵類別：`SpatialMetricsCalculator`

**跳躍高度計算（2 種方法）：**

1. **方法 1：髖部位移法**
   ```
   跳躍高度 = 最高點髖部 Y 座標 - 起跳時髖部 Y 座標
   ```
   - 優點：簡單直接
   - 缺點：受姿態影響（蹲低會低估）

2. **方法 2：滯空時間法**
   ```
   跳躍高度 = (g × t²) / 8
   其中：
   g = 9.81 m/s²（重力加速度）
   t = 滯空時間（起跳到落地的時間）
   ```
   - 優點：更符合物理原理
   - 缺點：需要準確的起跳和落地時間

**系統推薦：**
- 優先使用方法 2（物理法）
- 如果兩者差異過大（>10cm），標記為不可靠

**擊球高度：**
```
擊球高度 = 擊球時手腕的 Y 座標絕對值
```

**水平位移：**
```
前後位移（Forward）= 擊球時 Z 座標 - 起跳時 Z 座標
左右位移（Lateral）= 擊球時 X 座標 - 起跳時 X 座標
總位移 = √(前後² + 左右²)
```

---

### 7. 揮臂動作分類模組（arm_swing_classifier.py）

#### 功能
根據生物力學特徵將揮臂動作分為 5 種類型。

#### 關鍵類別：`ArmSwingClassifier`

**分類方法論：**

基於以下特徵進行分類：
1. **弧線高度**：手腕相對於肩部和前額的高度
2. **停頓動作**：第 III 階段是否有明顯的速度降低
3. **軌跡模式**：手腕的運動軌跡形狀

**5 種動作類型及判斷規則：**

#### 1. 直線式（Straight）
**特徵：**
- 第 II 階段：手腕高於肩部但低於前額
- 第 III 階段：有停頓動作
- 弧線高度：中等

**判斷規則：**
```python
Phase II:
  - wrist > shoulder (Y 軸)
  - wrist < forehead (Y 軸)
  - elbow ≈ shoulder (Y 軸，差距 < 閾值)
Phase III:
  - has_stopping_motion == True
```

**技術特點：**
- 力量型打法
- 適合後排扣球
- 需要較長的準備時間

#### 2. 弓箭式高位（Bow-Arrow High）
**特徵：**
- 第 II 階段：手腕明顯高於前額
- 第 III 階段：有停頓動作
- 弧線高度：極高

**判斷規則：**
```python
Phase II:
  - wrist >> forehead (Y 軸，差距 > 高閾值)
  - elbow > shoulder (Y 軸)
Phase III:
  - has_stopping_motion == True
```

**技術特點：**
- 最大力量輸出
- 最高的擊球點
- 適合主攻手的重扣

#### 3. 弓箭式低位（Bow-Arrow Low）
**特徵：**
- 第 II 階段：手腕略高於肩部
- 第 III 階段：有停頓動作
- 弧線高度：中低

**判斷規則：**
```python
Phase II:
  - shoulder < wrist < forehead (Y 軸)
  - elbow > shoulder (Y 軸，差距中等)
Phase III:
  - has_stopping_motion == True
```

**技術特點：**
- 平衡力量與速度
- 較容易控制
- 適合多數扣球情況

#### 4. 快攻式（Snap）
**特徵：**
- 第 I 階段：手腕起始位置在肩膀高度
- 第 II 階段：手腕保持在肩部附近（水平移動）
- 第 III 階段：有停頓動作

**判斷規則：**
```python
Phase I:
  - wrist ≈ shoulder (Y 軸，水平移動)
Phase II:
  - wrist ≈ shoulder (Y 軸，幾乎不上升)
Phase III:
  - has_stopping_motion == True
```

**技術特點：**
- 最快的出手速度
- 適合快攻戰術
- 犧牲力量換取速度

#### 5. 圓弧式（Circular）
**特徵：**
- 連續的圓弧運動
- 第 III 階段：無停頓動作
- 手腕軌跡較低且流暢

**判斷規則：**
```python
Phase II:
  - wrist < shoulder (Y 軸，低位)
  或
  - wrist ≈ shoulder (Y 軸)
Phase III:
  - has_stopping_motion == False  # 關鍵特徵
```

**技術特點：**
- 最流暢的動作
- 連續性好，適合連續扣球
- 力量最小但速度最快

**停頓動作檢測演算法：**
```python
def detect_stopping_motion(velocity, phase_bounds):
    # 在第 III 階段檢測速度是否有明顯降低
    phase_velocity = velocity[phase_bounds['start']:phase_bounds['end']]
    peak_velocity = max(phase_velocity)

    # 檢查是否有連續幀的速度低於峰值的 20%
    low_velocity_frames = phase_velocity < (peak_velocity * 0.2)
    consecutive_low = count_consecutive_true(low_velocity_frames)

    return consecutive_low >= stopping_duration  # 預設 3 幀
```

---

## 數據流程

### 完整數據流程圖

```
┌─────────────┐
│ 視頻輸入    │
│ (MP4/AVI等) │
└──────┬──────┘
       ↓
┌──────────────────────────────┐
│ Step 1: 姿態提取              │
│ - MediaPipe 處理每一幀        │
│ - 輸出: 33 個關節點 × N 幀    │
└──────┬───────────────────────┘
       ↓
┌──────────────────────────────┐
│ Step 2: 骨架處理              │
│ - 平滑軌跡（Savgol 濾波）     │
│ - 填補缺失幀                  │
│ - 輸出: 平滑的關節點序列       │
└──────┬───────────────────────┘
       ↓
┌──────────────────────────────┐
│ Step 3: 階段檢測              │
│ - 分析速度與加速度            │
│ - 檢測 5 個主要階段           │
│ - 檢測 3 個揮臂子階段         │
│ - 輸出: 階段邊界字典          │
└──────┬───────────────────────┘
       ↓
       ├─────────────┬─────────────┬──────────────┐
       ↓             ↓             ↓              ↓
┌─────────┐   ┌─────────┐   ┌──────────┐  ┌──────────┐
│關節角度 │   │速度計算 │   │空間指標  │  │視覺化    │
│計算     │   │         │   │計算      │  │          │
└────┬────┘   └────┬────┘   └────┬─────┘  └────┬─────┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                   ↓
            ┌─────────────┐
            │動作分類     │
            │（5 種類型） │
            └──────┬──────┘
                   ↓
         ┌───────────────────┐
         │結果匯總與報告生成 │
         │ - HTML 報告       │
         │ - CSV 匯出        │
         │ - JSON 數據       │
         └───────────────────┘
```

### 關鍵數據結構

#### 1. 骨架數據 DataFrame
```python
skeleton_df = pd.DataFrame({
    'frame': [0, 1, 2, ...],          # 幀編號
    'time': [0.0, 0.033, 0.066, ...], # 時間（秒）
    'landmarks_3d': [                  # 33×4 陣列列表
        np.array([[x, y, z, v], ...]),
        np.array([[x, y, z, v], ...]),
        ...
    ]
})
```

#### 2. 階段邊界字典
```python
phases = {
    'approach': {'start': 0, 'end': 45},
    'takeoff': {'start': 45, 'end': 60},
    'arm_swing': {'start': 60, 'end': 90},
    'contact': {'start': 85, 'end': 95},
    'landing': {'start': 95, 'end': 120}
}

arm_swing_phases = {
    'phase_i': {'start': 60, 'end': 72},
    'phase_ii': {'start': 72, 'end': 82},
    'phase_iii': {'start': 82, 'end': 90}
}
```

#### 3. 速度數據字典
```python
velocity_data = {
    'wrist_velocity': {
        'max': 8.5,          # m/s
        'mean': 4.2,
        'at_contact': 8.3
    },
    'shoulder_angular_velocity': {
        'max': 1250,         # deg/s
        'mean': 450
    },
    ...
}
```

#### 4. 分類結果字典
```python
classification = {
    'type': 'BA_HIGH',
    'type_display': 'Bow-Arrow High',
    'confidence': 0.85,
    'has_stopping_motion': True,
    'features': {
        'phase_i': {...},
        'phase_ii': {
            'wrist_vs_shoulder': 'above',
            'wrist_vs_forehead': 'above',
            'elbow_vs_shoulder': 'above'
        },
        'phase_iii': {...}
    },
    'matched_rules': [
        'Wrist significantly above forehead in Phase II',
        'Elbow above shoulder in Phase II',
        'Stopping motion detected in Phase III'
    ]
}
```

---

## 演算法與判斷方法

### 1. 平滑演算法：Savitzky-Golay 濾波

**原理：**
在滑動窗口內擬合多項式，用擬合值替代原始值。

**參數：**
- 窗口大小：5 幀（約 0.17 秒 @ 30fps）
- 多項式階數：2

**優點：**
- 保留峰值特徵
- 平滑噪音
- 適合時間序列數據

**實現：**
```python
from scipy.signal import savgol_filter

smoothed = savgol_filter(
    signal,
    window_length=5,
    polyorder=2
)
```

### 2. 峰值檢測：find_peaks

**用途：**
- 找到速度峰值（擊球時刻）
- 找到加速度峰值（起跳時刻）
- 找到高度峰值（揮臂最高點）

**參數：**
```python
from scipy.signal import find_peaks

peaks, properties = find_peaks(
    signal,
    height=None,      # 最小高度
    distance=10,      # 峰值間最小距離
    prominence=0.5    # 峰值顯著性
)
```

### 3. 向量角度計算

**公式：**
```
cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
θ = arccos(cos(θ))
```

**實現：**
```python
def calculate_angle(v1, v2):
    # 正規化向量
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # 計算點積
    dot_product = np.dot(v1_norm, v2_norm)

    # 限制在 [-1, 1] 範圍內（避免數值誤差）
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 計算角度（弧度）
    angle_rad = np.arccos(dot_product)

    # 轉換為角度
    angle_deg = np.degrees(angle_rad)

    return angle_deg
```

### 4. 速度計算：數值微分

**一階導數（速度）：**
```python
# 使用中心差分
velocity[i] = (position[i+1] - position[i-1]) / (2 * dt)

# NumPy 實現
velocity = np.gradient(position, axis=0) * fps
```

**二階導數（加速度）：**
```python
acceleration = np.gradient(velocity) * fps
```

### 5. 分類決策樹

```
開始分類
  ↓
檢測第 III 階段是否有停頓動作？
  ├─ 無 → 圓弧式（Circular）
  └─ 有 → 繼續
        ↓
     檢測第 II 階段手腕高度
        ├─ 手腕 >> 前額 → 弓箭式高位（BA-High）
        ├─ 手腕 > 肩部 且 < 前額
        │   ├─ 手肘 > 肩部（大差距）→ 弓箭式低位（BA-Low）
        │   └─ 手肘 ≈ 肩部 → 直線式（Straight）
        └─ 手腕 ≈ 肩部 → 快攻式（Snap）
```

---

## 關鍵技術細節

### 1. MediaPipe 座標系統轉換

**原始輸出：**
- 2D 座標：標準化座標（0-1）
- 3D 座標：以髖部中心為原點的相對座標

**問題：**
- Y 軸向下（圖像座標系）需要轉換為向上（物理座標系）
- 3D 座標的單位是相對的，需要校準

**解決方案：**
```python
# Y 軸翻轉
landmarks_3d[:, 1] = -landmarks_3d[:, 1]

# 使用已知身體部位長度校準
# 例如：假設肩寬為 0.45 米
shoulder_width_pixels = right_shoulder - left_shoulder
scale_factor = 0.45 / shoulder_width_pixels
landmarks_3d *= scale_factor
```

### 2. 缺失幀處理

**策略：**
1. 線性插值（短時間缺失，< 5 幀）
2. 三次樣條插值（中等時間缺失，5-15 幀）
3. 標記為無效（長時間缺失，> 15 幀）

```python
from scipy.interpolate import interp1d

# 找到有效幀
valid_frames = [i for i, lm in enumerate(landmarks) if lm is not None]

# 插值
for joint_idx in range(33):
    valid_positions = [landmarks[i][joint_idx] for i in valid_frames]
    interpolator = interp1d(
        valid_frames,
        valid_positions,
        kind='cubic',
        fill_value='extrapolate'
    )
    all_positions = interpolator(range(len(landmarks)))
```

### 3. 性能優化

**策略：**
1. **向量化計算**：使用 NumPy 陣列操作代替循環
2. **批次處理**：一次處理多幀
3. **快取機制**：Streamlit 的 `@st.cache_data` 裝飾器
4. **降採樣**：可選擇處理每 N 幀

```python
# 向量化範例
# 不好：循環
velocities = []
for i in range(len(positions) - 1):
    v = np.linalg.norm(positions[i+1] - positions[i]) * fps
    velocities.append(v)

# 好：向量化
displacements = np.diff(positions, axis=0)
velocities = np.linalg.norm(displacements, axis=1) * fps
```

### 4. 錯誤處理與邊界情況

**常見問題：**
1. 視頻中未檢測到姿態
2. 動作不完整（缺少某些階段）
3. 多人出現在畫面中
4. 遮擋問題

**處理策略：**
```python
# 檢查檢測率
valid_frames = [lm for lm in landmarks if lm is not None]
detection_rate = len(valid_frames) / len(landmarks)

if detection_rate < 0.5:
    logger.warning("Low detection rate")
    # 返回錯誤或警告

# 階段檢測失敗處理
if phases is None:
    # 使用預設值或跳過後續分析
    phases = get_default_phases(len(landmarks))
```

---

## 擴展開發指南

### 1. 新增動作類型

**步驟：**
1. 在 `ArmSwingClassifier` 中定義新類型
2. 添加判斷規則到 `classify_arm_swing()` 方法
3. 添加描述到 `get_motion_description()` 方法
4. 更新測試案例

```python
# 範例：新增「側身扣球」類型
MOTION_TYPES = {
    ...
    'SIDE_ATTACK': 'Side Attack'
}

def classify_arm_swing(self, ...):
    # 添加新的判斷邏輯
    if self._check_side_attack_pattern(features):
        return {
            'type': 'SIDE_ATTACK',
            'type_display': self.MOTION_TYPES['SIDE_ATTACK'],
            ...
        }
```

### 2. 新增生物力學指標

**步驟：**
1. 在適當的計算模組中添加計算方法
2. 更新 `MetricsSummary` 以包含新指標
3. 在 UI 中添加顯示

```python
# 範例：添加「手腕軌跡長度」指標
def calculate_wrist_trajectory_length(skeleton_df):
    wrist_positions = extract_landmark_trajectory(
        skeleton_df,
        LANDMARK_INDICES['right_wrist']
    )

    # 計算相鄰點之間的距離總和
    displacements = np.diff(wrist_positions, axis=0)
    distances = np.linalg.norm(displacements, axis=1)
    total_length = np.sum(distances)

    return total_length
```

### 3. 新增視覺化

**步驟：**
1. 在 `visualization/` 目錄下創建新模組
2. 使用 Plotly 或 Matplotlib 創建圖表
3. 在 `app.py` 中整合

```python
# 範例：添加力矩分析圖
def plot_joint_moments(skeleton_df, angles_df, forces_df):
    fig = go.Figure()

    # 計算關節力矩
    moments = calculate_joint_moments(skeleton_df, angles_df, forces_df)

    # 繪圖
    fig.add_trace(go.Scatter(
        x=moments['time'],
        y=moments['shoulder_moment'],
        name='Shoulder Moment'
    ))

    return fig
```

### 4. 多人追蹤

**挑戰：**
- MediaPipe Pose 一次只能追蹤一個人
- 需要區分多個人並分別分析

**解決方案：**
1. 使用 YOLOv8 或類似模型檢測多個人
2. 為每個人單獨運行 MediaPipe
3. 追蹤 ID 以維持時間一致性

```python
# 偽代碼
detector = PersonDetector()  # YOLOv8
pose_extractor = PoseExtractor()

for frame in video:
    # 檢測所有人
    persons = detector.detect(frame)

    # 為每個人提取姿態
    for person_id, bbox in persons:
        person_frame = crop_frame(frame, bbox)
        landmarks = pose_extractor.extract_pose(person_frame)

        # 儲存到對應的追蹤序列
        tracks[person_id].append(landmarks)
```

### 5. 即時分析

**需求：**
從攝影機即時分析而非上傳視頻。

**實現：**
```python
import cv2

def realtime_analysis():
    cap = cv2.VideoCapture(0)  # 開啟攝影機
    pose_extractor = PoseExtractor()

    buffer = []  # 緩衝最近的幀

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 提取姿態
        landmarks = pose_extractor.extract_pose(frame)
        buffer.append(landmarks)

        # 保持固定長度的緩衝
        if len(buffer) > 300:  # 10 秒 @ 30fps
            buffer.pop(0)

        # 檢測動作（當緩衝足夠時）
        if len(buffer) >= 150:
            phases = detect_phases(buffer)
            if phases and 'contact' in phases:
                # 分析剛完成的動作
                analyze_and_display(buffer, phases)
                buffer.clear()

        # 顯示當前幀
        cv2.imshow('Real-time Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

---

## 總結

本系統採用模組化設計，各模組職責清晰，易於理解和擴展。核心處理流程包括：

1. **姿態提取**：使用 MediaPipe 獲取骨架數據
2. **數據處理**：平滑、插值、標準化
3. **階段檢測**：基於運動學特徵自動分割動作
4. **生物力學分析**：計算角度、速度、空間指標
5. **動作分類**：基於規則的分類系統
6. **視覺化與報告**：多樣化的結果呈現

系統設計充分考慮了排球扣球的運動特性，採用成熟的信號處理和計算幾何演算法，確保分析結果的準確性和可靠性。
