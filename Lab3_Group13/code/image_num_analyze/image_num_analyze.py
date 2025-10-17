import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置棋盘格参数
chessboard_size = (11, 8)  # 棋盘格内角点数量 (宽度, 高度)
square_size = 25  # 棋盘格每个方格的实际尺寸(毫米)

# 准备棋盘格的3D点坐标 (世界坐标系)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # 乘以实际尺寸


def calibrate_camera(image_folder):
    """
    对指定文件夹中的图像进行相机标定并计算重投影误差
    """
    # 存储所有图像的3D点和2D点
    objpoints = []  # 3D点 (世界坐标系)
    imgpoints = []  # 2D点 (图像坐标系)
    gray = None  # 初始化灰度图变量

    # 读取指定文件夹中的所有棋盘格图像
    image_pattern = os.path.join(image_folder, '*.png')
    images = glob.glob(image_pattern)

    if len(images) == 0:
        # 如果没有找到png文件，尝试jpg文件
        image_pattern = os.path.join(image_folder, '*.jpg')
        images = glob.glob(image_pattern)

    print(f"在文件夹 {image_folder} 中找到的图像数量：{len(images)}")

    if len(images) == 0:
        print(f"警告：在 {image_folder} 中未找到任何图像！")
        return None, None, None, None, None, len(images), 0

    images = sorted(images)
    valid_images_count = 0

    for idx, fname in enumerate(images):
        # 读取图像
        img = cv2.imread(fname)
        if img is None:
            print(f"警告：无法加载图像 {fname}，已跳过")
            continue

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # 如果找到角点，添加到列表中
        if ret == True:
            objpoints.append(objp)

            # 亚像素级角点精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_images_count += 1
            print(f"  {image_folder}: 已处理第 {idx + 1}/{len(images)} 张图像，成功检测到角点")
        else:
            print(f"  警告：第 {idx + 1}/{len(images)} 张图像 {os.path.basename(fname)} 未检测到角点，已跳过")

    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    final_error = mean_error / len(objpoints)
    print(f"  {image_folder}: 平均重投影误差 = {final_error:.6f} 像素")

    return mtx, dist, rvecs, tvecs, final_error, len(images), valid_images_count


def analyze_image_sets():
    """
    分析不同图片数量对重投影误差的影响
    """
    # 存储结果的列表
    results = []

    # 查找所有image_set_X文件夹
    base_dir = '.'  # 当前目录，可以根据需要修改
    image_set_folders = [f for f in os.listdir(base_dir) if f.startswith('image_set_') and os.path.isdir(f)]

    if not image_set_folders:
        print("未找到任何 image_set_X 文件夹！")
        print("请确保文件夹名称格式为 'image_set_X'，其中X是图片数量")
        return None

    # 按图片数量排序
    image_set_folders.sort(key=lambda x: int(x.split('_')[-1]))

    print("开始分析不同图片数量对重投影误差的影响...")
    print("=" * 60)

    for folder in image_set_folders:
        print(f"\n处理文件夹: {folder}")
        image_count = int(folder.split('_')[-1])

        # 进行相机标定
        mtx, dist, rvecs, tvecs, reprojection_error, total_images, valid_images = calibrate_camera(folder)

        # 存储结果
        result = {
            'folder_name': folder,
            'image_count': image_count,
            'total_images': total_images,
            'valid_images': valid_images,
            'reprojection_error': reprojection_error if reprojection_error is not None else float('nan'),
            'calibration_success': reprojection_error is not None
        }
        results.append(result)

    return results


def save_results(results):
    """
    保存结果为表格和图表
    """
    if not results:
        print("没有有效结果可保存")
        return

    # 创建结果目录
    output_dir = 'reprojection_error_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 只保留标定成功的记录用于绘图
    df_success = df[df['calibration_success']].copy()

    # 生成时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存为CSV表格
    csv_filename = os.path.join(output_dir, f'reprojection_error_analysis_{timestamp}.csv')
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')  # 使用utf-8-sig支持Excel中文
    print(f"\n结果表格已保存为: {csv_filename}")

    # 创建图表
    if len(df_success) > 0:
        # 方法1：使用支持中文的字体
        try:
            # 尝试使用系统中支持中文的字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            fig = plt.figure(figsize=(12, 8))

            # 重投影误差 vs 图片数量
            plt.subplot(2, 1, 1)
            plt.plot(df_success['image_count'], df_success['reprojection_error'], 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Images', fontsize=12)
            plt.ylabel('Reprojection Error (pixels)', fontsize=12)
            plt.title('Effect of Chessboard Image Count on Reprojection Error', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # 为每个数据点添加数值标注
            for i, row in df_success.iterrows():
                plt.annotate(f'{row["reprojection_error"]:.4f}',
                             (row['image_count'], row['reprojection_error']),
                             textcoords="offset points",
                             xytext=(0, 10),
                             ha='center',
                             fontsize=9)

            # 有效图片数量 vs 总图片数量
            plt.subplot(2, 1, 2)
            x_pos = np.arange(len(df))
            bar_width = 0.35

            plt.bar(x_pos - bar_width / 2, df['total_images'], bar_width, label='Total Images', alpha=0.7)
            plt.bar(x_pos + bar_width / 2, df['valid_images'], bar_width, label='Valid Images', alpha=0.7)

            plt.xlabel('Image Set', fontsize=12)
            plt.ylabel('Number of Images', fontsize=12)
            plt.title('Valid Images Statistics for Each Image Set', fontsize=14, fontweight='bold')
            plt.xticks(x_pos, df['folder_name'], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图表
            plot_filename = os.path.join(output_dir, f'reprojection_error_analysis_{timestamp}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"分析图表已保存为: {plot_filename}")

        except Exception as e:
            print(f"使用中文字体时出现错误: {e}")
            print("尝试使用英文图表...")

            # 方法2：如果中文字体失败，使用纯英文图表
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            fig = plt.figure(figsize=(12, 8))

            # 重投影误差 vs 图片数量
            plt.subplot(2, 1, 1)
            plt.plot(df_success['image_count'], df_success['reprojection_error'], 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Images')
            plt.ylabel('Reprojection Error (pixels)')
            plt.title('Effect of Chessboard Image Count on Reprojection Error')
            plt.grid(True, alpha=0.3)

            # 为每个数据点添加数值标注
            for i, row in df_success.iterrows():
                plt.annotate(f'{row["reprojection_error"]:.4f}',
                             (row['image_count'], row['reprojection_error']),
                             textcoords="offset points",
                             xytext=(0, 10),
                             ha='center',
                             fontsize=9)

            # 有效图片数量 vs 总图片数量
            plt.subplot(2, 1, 2)
            x_pos = np.arange(len(df))
            bar_width = 0.35

            plt.bar(x_pos - bar_width / 2, df['total_images'], bar_width, label='Total Images', alpha=0.7)
            plt.bar(x_pos + bar_width / 2, df['valid_images'], bar_width, label='Valid Images', alpha=0.7)

            plt.xlabel('Image Set')
            plt.ylabel('Number of Images')
            plt.title('Valid Images Statistics for Each Image Set')
            plt.xticks(x_pos, df['folder_name'], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图表
            plot_filename = os.path.join(output_dir, f'reprojection_error_analysis_{timestamp}_en.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"英文分析图表已保存为: {plot_filename}")

        # 打印汇总统计
        print("\n" + "=" * 60)
        print("Summary Statistics:")
        print("=" * 60)
        for _, row in df.iterrows():
            status = "Success" if row['calibration_success'] else "Failed"
            error = f"{row['reprojection_error']:.6f}" if not np.isnan(row['reprojection_error']) else "N/A"
            print(f"{row['folder_name']:15} | Images: {row['image_count']:2d} | "
                  f"Valid: {row['valid_images']:2d}/{row['total_images']:2d} | "
                  f"Repro Error: {error:8} | Status: {status}")
    else:
        print("Warning: No successful calibration results, cannot generate charts")


def main():
    """
    主函数
    """
    print("Analysis of Chessboard Image Count on Reprojection Error")
    print("=" * 50)

    # 分析所有image_set_X文件夹
    results = analyze_image_sets()

    if results:
        # 保存结果
        save_results(results)

        # 计算最佳图片数量（重投影误差最小时）
        successful_results = [r for r in results if r['calibration_success']]
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['reprojection_error'])
            print(f"\nBest Result: {best_result['folder_name']}")
            print(f"Minimum Reprojection Error: {best_result['reprojection_error']:.6f} pixels")
            print(f"Recommended Number of Images: {best_result['image_count']}")
    else:
        print("Analysis completed, but no valid results obtained")


if __name__ == "__main__":
    main()