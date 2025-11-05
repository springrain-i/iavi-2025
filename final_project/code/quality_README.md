# 图像质量分析工具（quality.py）

该工具用于在“轴移镜头”采集流程中，先对已有图片做质量体检，再决定是否移动镜头或进入拼接/3D 等后续步骤。脚本会为每张图计算多项质量指标，归一化后得到综合得分，并给出 PASS/WARN/FAIL 判定与问题提示。

## 核心原理

对输入文件夹中的每张图片（转灰度）计算以下指标：

- 清晰度（锐度）
  - 拉普拉斯方差 lap_var：越大越清晰
  - Tenengrad 梯度能量 tenengrad：越大越清晰
- 曝光与对比度
  - 平均灰度 mean
  - 过暗比例 under_pct（灰度 ≤ 5 的像素占比）
  - 过亮比例 over_pct（灰度 ≥ 250 的像素占比）
  - RMS 对比度 contrast（灰度标准差）
- 信息量
  - 香农熵 entropy（256 桶直方图）
- 噪声估计
  - noise_sigma：基于拉普拉斯响应的 MAD 估计
- 纹理/可匹配性
  - 关键点数量 kp_count 与平均响应 kp_response（默认 ORB，可选 SIFT）

### 评分流程

- 对每个指标，按数据集的 p10–p90 百分位做鲁棒归一化到 [0,1]。
  - 越大越好的指标（如 lap_var、tenengrad、entropy、kp_count…）按正向归一化；
  - 越小越好的指标（如 noise_sigma、under_pct、over_pct）按反向归一化；
  - mean 按“越接近中灰（≈127.5）越好”的方式评分。
- 采用加权求和得到综合得分 score（0–1）：
  - lap_var 0.25, tenengrad 0.20, contrast 0.10, entropy 0.10,
    kp_count 0.15, kp_response 0.05, noise_sigma 0.10,
    under_pct 0.025, over_pct 0.025, mean 0.10
- 判定规则：
  - PASS：score ≥ 0.6 且未发现明显问题
  - WARN：0.5 ≤ score < 0.6（或仅轻微问题）
  - FAIL：score < 0.5（常见为虚焦/曝光不当/纹理过少）
- 同时输出问题提示（如：Blur/low sharpness、Under/Overexposed、Low contrast、Few features、Low information content）。

## 输出内容

- CSV 报告（默认 `quality.csv`），包含：
  - image, lap_var, tenengrad, contrast, mean, under_pct, over_pct,
    entropy, noise_sigma, kp_count, kp_response, score, verdict, note
  - 按 score 从高到低排序
- 可选 JSON 报告（`--out-json` 指定）
- 终端概览：打印最佳/最差两张图片的得分与判定

## 使用方法（PowerShell）

在 `final_project/code` 目录下执行：

```powershell
# 基本用法：分析 ..\figures 下的图片，结果写到当前目录的 CSV
python .\quality.py --images-dir ..\figures --out-csv .\quality.csv

# 额外输出 JSON，使用 SIFT 统计特征丰富度，缩放到 50% 加速
python .\quality.py --images-dir ..\figures --out-csv .\out\quality.csv --out-json .\out\quality.json --feature SIFT --scale 0.5
```

说明：
- 支持 PNG/JPG/TIFF 等格式，分析统一在灰度域进行。
- `--scale` 可在大图上显著提速，对排序影响很小。
- SIFT 依赖带 nonfree 的 OpenCV 版本；若不可用会自动使用 ORB。

## 结果解读建议

- 选择 `PASS` 且 `score` 较高的图片作为拼接/参考帧更稳妥。
- 若连续多张 `WARN/FAIL` 且主要问题为清晰度或 `kp_count` 偏低，建议微调焦距/曝光或移动镜头再次采集。
- 曝光问题：
  - `under_pct` 高：适当增加曝光或补光；
  - `over_pct` 高：降低曝光/亮度，避免强反光高亮。
- 低纹理场景可考虑贴点特征或适当改变视角以增加可匹配性。

## 后续可接入的运动策略（思路）

- 若最近 N 张图的中位 `score` < 0.5，或最佳图的 `kp_count` 仍 < 200，则触发一次镜头微移。
- 记录每步的 CSV 结果，基于趋势（滑动窗口均值）选择更优方向（`kp_count` 上升、`under/over_pct` 下降）。

## 默认路径

- 输入：`final_project/figures`
- 输出：`final_project/code/quality.csv`（可通过参数覆盖）

## 常见问题

- “No images found”：检查 `--images-dir` 路径与扩展名过滤。
- 分数整体偏低：可能过暗/过亮或严重虚焦；先检查对焦与曝光。
- 运行缓慢：提高 `--scale`（如 0.5）可明显提速且基本不改变排序。
