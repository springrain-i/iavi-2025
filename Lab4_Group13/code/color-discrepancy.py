import cv2
import numpy as np
import os
import argparse
from glob import glob


def load_image(path):
    """加载灰度图像"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def linear_transform_correction(left_img, right_img):
    """基于线性变换的灰度矫正（仅调整亮度和对比度）"""
    # 计算有效像素的掩码（排除过暗区域，避免异常值影响）
    mask_left = left_img > 10
    mask_right = right_img > 10
    valid_mask = mask_left & mask_right

    if not np.any(valid_mask):
        return left_img.copy(), right_img.copy()

    # 提取有效区域的像素值（仅用于计算变换参数）
    left_vals = left_img[valid_mask].astype(np.float32)
    right_vals = right_img[valid_mask].astype(np.float32)

    # 计算均值和标准差（描述亮度和对比度特征）
    mean_l = np.mean(left_vals)
    std_l = np.std(left_vals)
    mean_r = np.mean(right_vals)
    std_r = np.std(right_vals)

    # 避免除零错误
    if std_l < 1e-6 or std_r < 1e-6:
        return left_img.copy(), right_img.copy()

    # 计算变换参数（使右图像匹配左图像的亮度分布）
    scale = std_l / std_r  # 对比度匹配
    offset = mean_l - scale * mean_r  # 亮度匹配

    # 应用变换到右图像
    corrected_right = scale * right_img.astype(np.float32) + offset
    corrected_right = np.clip(corrected_right, 0, 255).astype(np.uint8)

    return left_img.copy(), corrected_right


def histogram_matching_correction(left_img, right_img):
    """基于直方图匹配的灰度矫正"""
    # 计算左图像的累积分布函数
    hist_left, bins_left = np.histogram(left_img.flatten(), 256, [0, 256])
    cdf_left = hist_left.cumsum()
    cdf_left = (255 * cdf_left / cdf_left[-1]).astype(np.uint8)  # 归一化

    # 计算右图像的累积分布函数
    hist_right, bins_right = np.histogram(right_img.flatten(), 256, [0, 256])
    cdf_right = hist_right.cumsum()
    cdf_right = (255 * cdf_right / cdf_right[-1]).astype(np.uint8)  # 归一化

    # 构建映射表
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # 找到右图像CDF中与左图像CDF最接近的值
        idx = np.argmin(np.abs(cdf_left[i] - cdf_right))
        mapping[i] = idx

    # 应用映射到右图像
    corrected_right = mapping[right_img]
    return left_img.copy(), corrected_right


def process_stereo_pair(left_path, right_path, method='linear', output_dir='corrected'):
    """处理一对立体图像"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始图像
    left = load_image(left_path)
    right = load_image(right_path)

    # 确保左右图像尺寸一致
    if left.shape != right.shape:
        raise ValueError(f"左右图像尺寸不一致：{left.shape} vs {right.shape}")

    # 选择矫正方法
    if method == 'linear':
        corrected_left, corrected_right = linear_transform_correction(left, right)
    elif method == 'histogram':
        corrected_left, corrected_right = histogram_matching_correction(left, right)
    else:
        raise ValueError(f"不支持的矫正方法: {method}")

    # 保存结果
    base_name = os.path.splitext(os.path.basename(left_path))[0].replace('_left', '')
    left_out = os.path.join(output_dir, f'{base_name}_left_corrected.png')
    right_out = os.path.join(output_dir, f'{base_name}_right_corrected.png')

    cv2.imwrite(left_out, corrected_left)
    cv2.imwrite(right_out, corrected_right)
    print(f"已保存矫正结果: {left_out}, {right_out}")

    # 生成对比图，为灰度图添加伪彩色便于对比
    left_color = cv2.applyColorMap(left, cv2.COLORMAP_JET)
    right_color = cv2.applyColorMap(right, cv2.COLORMAP_JET)
    corrected_left_color = cv2.applyColorMap(corrected_left, cv2.COLORMAP_JET)
    corrected_right_color = cv2.applyColorMap(corrected_right, cv2.COLORMAP_JET)

    # 添加分隔线
    separator = np.zeros((left.shape[0], 10, 3), dtype=np.uint8)
    comparison = np.hstack((
        left_color, separator,
        right_color, separator,
        corrected_left_color, separator,
        corrected_right_color
    ))
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_comparison.png'), comparison)

    return corrected_left, corrected_right


def process_all_pairs(folder_path, method='linear', output_dir='corrected'):
    """批量处理文件夹中所有立体图像对"""
    left_paths = sorted(glob(os.path.join(folder_path, '*_left.png')))
    if not left_paths:
        print(f"在文件夹 {folder_path} 中未找到左图像（格式：*_left.png）")
        return

    for left_path in left_paths:
        base_name = os.path.basename(left_path).replace('_left.png', '')
        right_path = os.path.join(folder_path, f'{base_name}_right.png')

        if not os.path.exists(right_path):
            print(f"警告：未找到对应右图像 {right_path}，已跳过")
            continue

        print(f"处理图像对：{base_name}")
        try:
            process_stereo_pair(left_path, right_path, method, output_dir)
        except Exception as e:
            print(f"处理失败：{e}")


def main():
    parser = argparse.ArgumentParser(description='立体灰度图像光照矫正工具（保持几何结构）')
    parser.add_argument('--folder', default='figures/figures/use4ply', help='包含左右图像对的文件夹路径')
    parser.add_argument('--method', choices=['linear', 'histogram'], default='linear',
                        help='矫正方法: linear(线性变换) 或 histogram(直方图匹配)')
    parser.add_argument('--output', default='corrected', help='输出目录')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"错误：文件夹 {args.folder} 不存在")
        return

    process_all_pairs(args.folder, args.method, args.output)
    print("所有图像对处理完成")


if __name__ == '__main__':
    main()