import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -------------------------- 1. 全局参数配置 --------------------------
# 三个实验组的路径
GROUP_PATHS = {
    "High Coverage (H)": r"C:\Users\27198\Desktop\camera_lab2\coverage\high",
    "Mid Coverage (M)": r"C:\Users\27198\Desktop\camera_lab2\coverage\mid",
    "Low Coverage (L)": r"C:\Users\27198\Desktop\camera_lab2\coverage\low"
}
# 棋盘格参数
CHESSBOARD_PATTERN = (11, 8)  # (cols, rows)，内角点列数×行数
SQUARE_SIZE = 2.0 
# 输出结果保存路径
OUTPUT_DIR = r"C:\Users\27198\Desktop\camera_lab2\coverage_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 自动创建输出文件夹


# -------------------------- 2. 核心函数定义 --------------------------
def calculate_chessboard_coverage(img_path, corners_sub):
    """
    计算单张图像中棋盘格的覆盖面积占比（%）
    参数：
        img_path: 图像文件路径
        corners_sub: 亚像素精度的棋盘格角点坐标（格式：(N, 1, 2)，N=rows×cols）
    返回：
        coverage_ratio: 覆盖面积占比（保留2位小数）
    """
    # 读取图像并获取尺寸
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    img_total_area = img_h * img_w  # 图像总像素面积
    
    # 提取角点的x、y坐标，计算棋盘格外接矩形
    corners_x = corners_sub[:, 0, 0]  # 所有角点的x坐标（像素）
    corners_y = corners_sub[:, 0, 1]  # 所有角点的y坐标（像素）
    chess_min_x, chess_max_x = np.min(corners_x), np.max(corners_x)
    chess_min_y, chess_max_y = np.min(corners_y), np.max(corners_y)
    
    # 计算棋盘格外接矩形的像素面积（近似棋盘格实际占用面积）
    chess_pixel_area = (chess_max_x - chess_min_x) * (chess_max_y - chess_min_y)
    # 计算覆盖占比（转为百分比）
    coverage_ratio = (chess_pixel_area / img_total_area) * 100
    return round(coverage_ratio, 2)


def find_chessboard_corners_and_calibrate(image_dir, pattern_size, square_size):
    """
    检测图像中的棋盘格角点，并执行相机标定，返回：
    - 平均重投影误差
    - 每张有效图像的覆盖面积占比
    - 标定结果（内参、外参等，备用）
    """
    cols, rows = pattern_size
    # 1. 初始化棋盘格角点的3D世界坐标（Z=0，位于XY平面）
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size  # 按方格边长缩放
    
    # 2. 初始化存储列表
    obj_points = []  # 存储所有图像的3D世界坐标
    img_points = []  # 存储所有图像的2D亚像素角点坐标
    coverage_ratios = []  # 存储每张有效图像的覆盖面积占比
    valid_img_paths = []  # 存储有效图像的路径（用于后续验证）
    
    # 3. 遍历图像文件夹
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(image_files) == 0:
        raise FileNotFoundError(f"在文件夹 {image_dir} 中未找到图像文件（支持.png/.jpg/.jpeg）")
    
    last_gray = None
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：跳过无法读取的图像 {img_path}")
            continue
        
        # 转为灰度图（角点检测需单通道图像）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_gray = gray
        
        # 4. 粗检测棋盘格角点（整数像素精度）
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE  # 抗光照干扰
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        
        if found:
            # 5. 亚像素角点优化（提升定位精度）
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 6. 计算当前图像的覆盖面积占比
            coverage = calculate_chessboard_coverage(img_path, corners_sub)
            
            # 7. 存储数据
            obj_points.append(objp.copy())
            img_points.append(corners_sub)
            coverage_ratios.append(coverage)
            valid_img_paths.append(img_path)
            print(f"成功处理图像：{img_name} | 覆盖占比：{coverage}%")
        else:
            print(f"警告：在图像 {img_name} 中未检测到完整棋盘格角点，跳过")
    
    # 8. 验证有效数据数量
    if len(obj_points) == 0:
        raise ValueError(f"文件夹 {image_dir} 中无有效棋盘格图像（未检测到完整角点）")
    print(f"=== 文件夹 {os.path.basename(image_dir)} 处理完成 | 有效图像数：{len(valid_img_paths)} ===\n")
    
    # 9. 执行相机标定，计算重投影误差
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, last_gray.shape[::-1], None, None
    )
    
    # 10. 计算平均重投影误差（评估标定精度）
    mean_reprojection_error = 0.0
    for i in range(len(obj_points)):
        # 将3D世界坐标投影到2D图像平面（使用标定后的参数）
        imgpts2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        # 计算真实角点与投影角点的L2范数（误差）
        err = cv2.norm(img_points[i], imgpts2, cv2.NORM_L2) / len(imgpts2)
        mean_reprojection_error += err
    mean_reprojection_error /= len(obj_points)  # 平均误差
    
    # 返回关键结果
    return {
        "mean_reprojection_error": round(mean_reprojection_error, 4),  # 平均重投影误差（保留4位小数）
        "coverage_ratios": coverage_ratios,  # 每张有效图像的覆盖占比
        "valid_image_count": len(valid_img_paths),  # 有效图像数量
        "intrinsic_matrix": mtx,  # 内参矩阵（备用）
        "distortion_coeffs": dist  # 畸变系数（备用）
    }


# -------------------------- 3. 批量处理三个实验组 --------------------------
def batch_analyze_groups(group_paths, pattern_size, square_size):
    """
    批量分析所有实验组，返回汇总结果
    """
    analysis_results = {}
    for group_name, group_path in group_paths.items():
        print(f"==================== 开始分析实验组：{group_name} ====================")
        try:
            # 对当前组执行角点检测和标定
            group_result = find_chessboard_corners_and_calibrate(
                image_dir=group_path,
                pattern_size=pattern_size,
                square_size=square_size
            )
            # 计算当前组的平均覆盖占比（所有有效图像的覆盖占比均值）
            avg_coverage = round(np.mean(group_result["coverage_ratios"]), 2)
            # 补充组信息
            group_result["avg_coverage"] = avg_coverage
            group_result["group_path"] = group_path
            # 存入汇总结果
            analysis_results[group_name] = group_result
            # 打印当前组的核心结果
            print(f"\n【{group_name} 核心结果】")
            print(f"平均覆盖面积占比：{avg_coverage}%")
            print(f"有效图像数量：{group_result['valid_image_count']} 张")
            print(f"平均重投影误差：{group_result['mean_reprojection_error']} 像素\n")
        except Exception as e:
            print(f"【错误】分析实验组 {group_name} 失败：{str(e)}\n")
            analysis_results[group_name] = {"error": str(e)}
    return analysis_results


# -------------------------- 4. 结果可视化（绘制覆盖范围 vs 重投影误差图） --------------------------
def plot_coverage_vs_error(analysis_results, output_dir):
    """
    生成可视化图表：覆盖范围（x轴）vs 重投影误差（y轴）
    """
    # 配置中文字体（避免乱码）
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    # 提取有效数据（过滤分析失败的组）
    valid_groups = []
    avg_coverages = []
    mean_errors = []
    error_stds = []  # 误差的标准差（用于绘制误差线）
    valid_counts = []
    
    for group_name, result in analysis_results.items():
        if "error" not in result:  # 仅保留分析成功的组
            valid_groups.append(group_name)
            avg_coverages.append(result["avg_coverage"])
            mean_errors.append(result["mean_reprojection_error"])
            # 计算重投影误差的标准差（体现组内误差波动）
            error_stds.append(round(np.std(result["coverage_ratios"]), 4))
            valid_counts.append(result["valid_image_count"])
    
    if len(valid_groups) == 0:
        raise ValueError("无有效分析结果，无法生成图表")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制散点+误差线（x轴误差：覆盖占比的标准差；y轴误差：重投影误差的标准差）
    scatter = ax.scatter(
        avg_coverages, mean_errors,
        s=200, c='#2E86AB', alpha=0.8, edgecolors='#1A5276', linewidth=2,
        label='实验组'
    )
    
    # 为每个点添加标签（显示组名、有效图像数）
    for i, (group, count) in enumerate(zip(valid_groups, valid_counts)):
        ax.annotate(
            f"{group}\n(有效图：{count}张)",
            xy=(avg_coverages[i], mean_errors[i]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='#DDDDDD')
        )
    
    # 绘制趋势线（体现覆盖范围与误差的相关性）
    z = np.polyfit(avg_coverages, mean_errors, 1)
    p = np.poly1d(z)
    ax.plot(avg_coverages, p(avg_coverages), "r--", linewidth=2, alpha=0.7, label=f'趋势线: y={z[0]:.6f}x+{z[1]:.4f}')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('棋盘格平均覆盖面积占比（%）', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均重投影误差（像素）', fontsize=12, fontweight='bold')
    ax.set_title('棋盘格覆盖范围对相机标定重投影误差的影响\n（固定视角、光照、棋盘格规格）', fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格（提升可读性）
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 调整布局（避免标签被截断）
    plt.tight_layout()
    
    # 保存图表（高分辨率，支持后续插入报告）
    plot_path = os.path.join(output_dir, "coverage_vs_reprojection_error.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图表已保存至：{plot_path}")
    
    # 返回趋势线参数（用于结论分析）
    return {"slope": z[0], "intercept": z[1]}

# -------------------------- 5. 保存分析结果到文本文件（便于报告引用） --------------------------
def save_analysis_report(analysis_results, trend_params, output_dir):
    """
    保存详细分析报告到文本文件
    """
    report_path = os.path.join(output_dir, "coverage_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("棋盘格覆盖范围对重投影误差影响分析报告\n")
        f.write("="*60 + "\n\n")
        
        # 写入实验参数
        f.write("1. 实验参数\n")
        f.write("-"*30 + "\n")
        f.write(f"棋盘格内角点规格：{CHESSBOARD_PATTERN[0]}列 × {CHESSBOARD_PATTERN[1]}行\n")
        f.write(f"棋盘格方格边长：{SQUARE_SIZE} cm\n")
        f.write(f"图像文件夹路径：\n")
        for group_name, group_path in GROUP_PATHS.items():
            f.write(f"  - {group_name}：{group_path}\n")
        f.write(f"分析时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 写入各组详细结果
        f.write("2. 各组详细分析结果\n")
        f.write("-"*30 + "\n")
        for group_name, result in analysis_results.items():
            f.write(f"\n【{group_name}】\n")
            if "error" in result:
                f.write(f"  状态：分析失败\n")
                f.write(f"  原因：{result['error']}\n")
                continue
            f.write(f"  状态：分析成功\n")
            f.write(f"  平均覆盖面积占比：{result['avg_coverage']}%\n")
            f.write(f"  有效图像数量：{result['valid_image_count']} 张\n")
            f.write(f"  每张图像覆盖占比：{result['coverage_ratios']}\n")
            f.write(f"  平均重投影误差：{result['mean_reprojection_error']} 像素\n")
            f.write(f"  内参矩阵（部分）：\n{result['intrinsic_matrix'][:2, :2]}\n")  # 仅显示焦距部分
        
        # 写入趋势分析
        f.write("\n3. 趋势分析\n")
        f.write("-"*30 + "\n")
        if trend_params:
            slope = trend_params["slope"]
            intercept = trend_params["intercept"]
            f.write(f"覆盖范围与重投影误差的线性趋势：y = {slope:.6f}x + {intercept:.4f}\n")
            if slope < 0:
                f.write("结论：覆盖范围与重投影误差呈负相关——覆盖范围越大，重投影误差越小\n")
            else:
                f.write("结论：覆盖范围与重投影误差呈正相关（异常，需检查实验数据）\n")
        else:
            f.write("无法生成趋势分析（有效实验组数量不足）\n")
    
    print(f"分析报告已保存至：{report_path}")
    return report_path


# -------------------------- 6. 主执行函数 --------------------------
def main():
    """
    主执行流程
    """
    print("=" * 80)
    print("棋盘格覆盖范围对相机标定精度影响分析系统")
    print("=" * 80)
    
    try:
        # 1. 批量分析三个实验组
        print("\n步骤1：开始批量分析三个实验组...")
        analysis_results = batch_analyze_groups(GROUP_PATHS, CHESSBOARD_PATTERN, SQUARE_SIZE)
        
        # 2. 生成可视化图表
        print("\n步骤2：生成覆盖范围 vs 重投影误差图表...")
        trend_params = plot_coverage_vs_error(analysis_results, OUTPUT_DIR)
        
        # 3. 保存详细分析报告
        print("\n步骤3：生成详细分析报告...")
        report_path = save_analysis_report(analysis_results, trend_params, OUTPUT_DIR)
        
        # 4. 输出最终总结
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        
        # 统计成功分析的组数
        successful_groups = [name for name, result in analysis_results.items() if "error" not in result]
        print(f"成功分析组数：{len(successful_groups)}/{len(GROUP_PATHS)}")
        
        # 显示各组核心结果
        for group_name in successful_groups:
            result = analysis_results[group_name]
            print(f"\n{group_name}:")
            print(f"  - 平均覆盖面积：{result['avg_coverage']}%")
            print(f"  - 重投影误差：{result['mean_reprojection_error']} 像素")
            print(f"  - 有效图像数：{result['valid_image_count']} 张")
        
        # 显示输出文件位置
        print(f"\n输出文件：")
        print(f"  - 图表：{os.path.join(OUTPUT_DIR, 'coverage_vs_reprojection_error.png')}")
        print(f"  - 报告：{report_path}")
        
        # 趋势分析结论
        if trend_params:
            slope = trend_params["slope"]
            if slope < 0:
                print(f"\n趋势分析：覆盖范围与重投影误差呈负相关（斜率：{slope:.6f}）")
                print("结论：增大棋盘格在图像中的覆盖范围有助于降低标定误差")
            else:
                print(f"\n趋势分析：覆盖范围与重投影误差呈正相关（斜率：{slope:.6f}）")
                print("注意：此结果与预期不符，请检查实验数据质量")
                
    except Exception as e:
        print(f"\n 分析过程出现错误：{str(e)}")
        print("请检查：")
        print("1. 文件路径是否正确")
        print("2. 棋盘格参数是否与实物一致")
        print("3. 图像中是否包含完整的棋盘格")
        return False
    
    return True


# -------------------------- 7. 程序入口 --------------------------
if __name__ == "__main__":
    # 添加pandas用于时间戳（如果不可用则使用备用方案）
    try:
        import pandas as pd
    except ImportError:
        import datetime
        pd = type('MockPandas', (), {
            'Timestamp': type('Timestamp', (), {
                'now': lambda: type('Now', (), {
                    'strftime': lambda fmt: datetime.datetime.now().strftime(fmt)
                })()
            })()
        })()
    
    # 执行主程序
    success = main()
    
    if success:
        print("\n 所有分析任务已完成！")
    else:
        print("\n 分析过程中断，请检查上述错误信息")