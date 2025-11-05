# Lab5 结构光立体重建 - 代码使用说明

以下命令均在 PowerShell（Windows）中运行，默认从本目录启动：`D:\Code\iavi\Lab5_Group13\code`。


1) 采集双目原始图像（可选，已有数据可跳过）

```
python .\grap.py --out capture
```
- 操作：按空格或 y 保存一对 `*_left.png`/`*_right.png`；q 退出。
- 输出：保存到 `capture/` 目录。

1) 生成投影用 Gray 码图案

```
python .\graycode.py --width 1280 --height 720 --out gray_patterns
```
- 输出：`gray_patterns/` 内包含 `ref_white.png/ref_black.png` 和各 bit 图案。

3) 在投影端全屏播放图案（用于采集时）

```
python .\project.py --proj-width 1280 --proj-height 720 --patterns-dir gray_patterns --monitor-index 1 --manual
```
- `--manual`：手动下一张（空格）。无此参数则按 `--delay` 自动轮播。
- `--monitor-index`：Windows 多显示器序号，若不确定可调整试验。

4) 解码获得投影坐标映射（left/right）

```
python .\decode.py --in-dir capture --out-dir img_decode --proj_width 1280 --proj_height 720
```
- 输入假设：`capture/` 内有按以下命名的图像：
	- `1_left.png`（参考白），`2_left.png`（参考黑），其后是 x 轴（竖直条纹）和 y 轴（水平条纹）的 MSB→LSB 序列；右相机同理（`*_right.png`）。
- 输出：`img_decode/` 下生成：
	- `left_x.tiff / left_y.tiff / left_mask.png`
	- `right_x.tiff / right_y.tiff / right_mask.png`

5) 立体重建（点云 + 深度图）

```
python .\reconstruct.py --npz stereo_params.npz --out reconstructed_stereo.ply --depth-out depth.tiff --depth-vis depth.png
```
- 要求：`stereo_params.npz` 为双目标定结果（含 `cameraMatrix1/2, distCoeffs1/2, R, T`）。
- 输出：
	- 点云：`reconstructed_stereo.ply`
	- 深度（float32，米）：`depth.tiff`（无效为 NaN）
	- 深度可视化：`depth.png`（Turbo 伪彩；无效为黑）
- 可选参数：
	- `--depth-out depth_mm.png` 将以毫米的 uint16 写出（无效为 0）。
	- `--min-z / --max-z`（米）过滤深度范围，默认会自动判断单位并转换为米后再筛选。
	- `--vis-min / --vis-max`（米）可手动控制颜色映射范围；不设则用 2%–98% 百分位自适应。
	- `--method exact|tol` 与 `--tol`：从左像素的投影坐标在右图中查找候选的策略（精确/带容差）。

