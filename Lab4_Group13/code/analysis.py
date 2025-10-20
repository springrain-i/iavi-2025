import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def load_params(npz_path):
    """加载标定参数"""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}

def stereo_match_with_block_size(left_gray, right_gray, block_size=5, num_disparities=128):
    """使用不同块大小计算视差图"""
    h, w = left_gray.shape[:2]
    
    # 确保块大小为奇数
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)  # 最小块大小为3
    
    # 确保视差数为16的倍数
    num_disp = num_disparities
    if num_disp % 16 != 0:
        num_disp = ((num_disp // 16) + 1) * 16
    
    # 创建SGBM匹配器
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=150,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # 计算视差
    raw_disp = matcher.compute(left_gray, right_gray)
    disp = raw_disp.astype(np.float32) / 16.0  # 转换为浮点数视差
    
    # 计算有效视差比例
    valid_ratio = np.mean(disp > 0)
    
    return disp, valid_ratio

def calculate_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """计算重投影误差"""
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    return mean_error / len(objpoints) if len(objpoints) else 0

def calibrate_with_distance(folder, nx, ny, square_size, distance_label):
    """针对不同距离进行标定并返回误差"""
    from calibration import find_image_pairs, make_object_points
    
    pairs = find_image_pairs(folder)
    if len(pairs) < 5:
        print(f"距离 {distance_label} 样本不足，跳过...")
        return None, None
    
    objp = make_object_points(nx, ny, square_size)
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    
    for left_path, right_path in pairs:
        img_l = cv2.imread(left_path)
        img_r = cv2.imread(right_path)
        if img_l is None or img_r is None:
            continue
            
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, (nx, ny), None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (nx, ny), None)
        
        if ret_l and ret_r:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), term)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), term)
            
            objpoints.append(objp.copy())
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
    
    if len(objpoints) < 5:
        print(f"距离 {distance_label} 有效样本不足，跳过...")
        return None, None
    
    # 单目标定计算误差
    h, w = gray_l.shape[:2]
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, (w, h), None, None)
    left_error = calculate_reprojection_error(objpoints, imgpoints_l, mtx_l, dist_l, rvecs_l, tvecs_l)
    
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, (w, h), None, None)
    right_error = calculate_reprojection_error(objpoints, imgpoints_r, mtx_r, dist_r, rvecs_r, tvecs_r)
    
    mean_error = (left_error + right_error) / 2
    
    # 立体标定
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    ret_stereo, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r, (w, h), criteria=criteria, flags=flags)
    
    return mean_error, ret_stereo

def analyze_block_size_influence(
    left_img_path, 
    right_img_path, 
    params, 
    # 调整块大小区间：3×3到13×13，间隔2
    block_sizes=[3, 5, 7, 9, 11, 13]
):
    """分析不同块大小对立体匹配的影响（细分区间）"""
    # 读取并预处理图像
    left = cv2.imread(left_img_path)
    right = cv2.imread(right_img_path)
    if left is None or right is None:
        print(f"无法读取图像: {left_img_path} 或 {right_img_path}")
        return []
    
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    # 校正图像
    h, w = left_gray.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        params['cameraMatrix1'], params.get('distCoeffs1'),
        params['cameraMatrix2'], params.get('distCoeffs2'),
        (w, h), params['R'], params['T'], alpha=0)
    
    map1x, map1y = cv2.initUndistortRectifyMap(
        params['cameraMatrix1'], params.get('distCoeffs1'), R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        params['cameraMatrix2'], params.get('distCoeffs2'), R2, P2, (w, h), cv2.CV_32FC1)
    
    left_rect = cv2.remap(left_gray, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_gray, map2x, map2y, cv2.INTER_LINEAR)
    
    results = []
    disp_maps = []
    
    # 测试调整后的块大小区间（最大13）
    for block_size in block_sizes:
        print(f"测试块大小: {block_size}x{block_size}")
        disp, valid_ratio = stereo_match_with_block_size(
            left_rect, right_rect, block_size=block_size)
        
        # 计算视差图的方差（反映视差变化剧烈程度）
        disp_valid = disp[disp > 0]
        disp_var = np.var(disp_valid) if len(disp_valid) > 0 else 0
        
        # 新增：计算平均视差
        disp_mean = np.mean(disp_valid) if len(disp_valid) > 0 else 0
        
        results.append({
            'block_size': block_size,
            'valid_ratio': valid_ratio,  # 有效匹配比例
            'disp_variance': disp_var,   # 视差方差（细节/噪声）
            'disp_mean': disp_mean       # 平均视差（匹配稳定性）
        })
        disp_maps.append((block_size, disp))
    
    # 可视化结果：调整为两张子图，优化尺寸
    plt.figure(figsize=(16, 10))  # 总高度减少，避免空白
    
    # 1. 有效视差比例 + 平均视差
    plt.subplot(2, 1, 1)  # 2行1列布局的第一幅图
    # 有效视差比例（主坐标轴）
    ax1 = plt.gca()
    ax1.plot([r['block_size'] for r in results], 
             [r['valid_ratio'] for r in results], 'o-', color='blue', label='有效视差比例')
    ax1.set_ylabel('有效视差比例', color='blue', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('块大小（像素）', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 区间颜色标注
    plt.axvspan(3, 5, color='lightgreen', alpha=0.3, label='小窗口区（3-5）')
    plt.axvspan(7, 9, color='lightyellow', alpha=0.3, label='中窗口区（7-9）')
    plt.axvspan(11, 13, color='lightcoral', alpha=0.3, label='大窗口区（11-13）')  # 上限为13
    # 合并区间标注的图例
    ax1.legend(loc='upper left', handles=ax1.get_legend_handles_labels()[0][:1] + ax1.get_legend_handles_labels()[0][-3:],
               labels=ax1.get_legend_handles_labels()[1][:1] + ax1.get_legend_handles_labels()[1][-3:],
               fontsize=9)
    
    # 平均视差（次坐标轴）
    ax2 = ax1.twinx()
    ax2.plot([r['block_size'] for r in results], 
             [r['disp_mean'] for r in results], 's--', color='green', label='平均视差')
    ax2.set_ylabel('平均视差（像素）', color='green', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right', fontsize=9)
    
    # 2. 视差方差
    plt.subplot(2, 1, 2)  # 2行1列布局的第二幅图
    plt.plot([r['block_size'] for r in results], 
             [r['disp_variance'] for r in results], 'o-', color='orange', label='视差方差')
    plt.ylabel('视差方差', fontsize=10)
    plt.xlabel('块大小（像素）', fontsize=10)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 同步更新区间标注
    plt.axvspan(3, 5, color='lightgreen', alpha=0.3)
    plt.axvspan(7, 9, color='lightyellow', alpha=0.3)
    plt.axvspan(11, 13, color='lightcoral', alpha=0.3)  # 调整上限为13
    
    plt.tight_layout()  # 自动调整间距，避免重叠
    plt.savefig('experiment_results/block_size_analysis.png')
    print("细分块大小分析图已保存为 experiment_results/block_size_analysis.png")
    
    # 视差图对比（调整布局为2行3列，适配6个块大小）
    plt.figure(figsize=(18, 8))  # 调整高度，适配2行布局
    for i, (block_size, disp) in enumerate(disp_maps):
        plt.subplot(2, 3, i+1)  # 2行3列布局（6个图）
        disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.imshow(disp_vis, cmap='jet')
        # 标注区间特征（与第一幅图颜色对应）
        if block_size in [3,5]:
            plt.title(f'小窗口: {block_size}x{block_size}\n（细节丰富）', fontsize=9)
        elif block_size in [7,9]:
            plt.title(f'中等窗口: {block_size}x{block_size}\n（平衡细节与噪声）', fontsize=9)
        else:  # 11,13
            plt.title(f'大窗口: {block_size}x{block_size}\n（平滑但细节少）', fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('experiment_results/block_size_disparity_maps.png')
    print("细分块大小的视差图已保存为 experiment_results/block_size_disparity_maps.png")
    
    return results

def analyze_distance_influence(calibration_folders, nx=11, ny=8, square_size=25.0):
    """分析相机与目标距离对标定质量的影响"""
    results = []
    
    for distance_label, folder in calibration_folders.items():
        if not os.path.exists(folder):
            print(f"距离 {distance_label} 的文件夹不存在: {folder}，跳过...")
            continue
            
        print(f"分析距离 {distance_label} 的标定质量...")
        reproj_error, stereo_error = calibrate_with_distance(
            folder, nx, ny, square_size, distance_label)
        
        if reproj_error is not None and stereo_error is not None:
            results.append({
                'distance': distance_label,
                'reprojection_error': reproj_error,
                'stereo_error': stereo_error
            })
    
    # 可视化结果
    if results:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        distances = [r['distance'] for r in results]
        reproj_errors = [r['reprojection_error'] for r in results]
        plt.bar(distances, reproj_errors, color='blue', alpha=0.7)
        plt.ylabel('平均重投影误差 (像素)')
        plt.xlabel('相机与目标距离')
        plt.title('不同距离下的重投影误差')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        stereo_errors = [r['stereo_error'] for r in results]
        plt.bar(distances, stereo_errors, color='red', alpha=0.7)
        plt.ylabel('立体标定误差')
        plt.xlabel('相机与目标距离')
        plt.title('不同距离下的立体标定误差')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('distance_influence.png')
        print("距离影响分析图已保存为 distance_influence.png")
    else:
        print("没有足够的距离数据进行分析")
    
    return results

def modify_baseline(params, scale_factor):
    """通过缩放基线向量来模拟不同基线长度"""
    modified_params = params.copy()
    # 缩放平移向量T来模拟基线变化
    modified_params['T'] = params['T'] * scale_factor
    return modified_params

def analyze_baseline_influence(original_params, left_img_path, right_img_path, baseline_scales=[1.0, 2.0, 3.0]):
    """分析相机基线距离对点云质量的影响（使用单一标定参数模拟不同基线）"""
    results = []
    baseline_labels = [f"{5*i}cm" for i in range(1, len(baseline_scales)+1)]  # 假设基础基线为5cm
    
    for scale, label in zip(baseline_scales, baseline_labels):
        print(f"分析基线 {label} 的点云质量...")
        # 通过缩放原始基线来模拟不同基线长度
        params = modify_baseline(original_params, scale)
        
        # 读取并预处理图像
        left = cv2.imread(left_img_path)
        right = cv2.imread(right_img_path)
        if left is None or right is None:
            print(f"无法读取图像: {left_img_path} 或 {right_img_path}")
            continue
            
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        
        # 校正图像
        h, w = left_gray.shape[:2]
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            params['cameraMatrix1'], params.get('distCoeffs1'),
            params['cameraMatrix2'], params.get('distCoeffs2'),
            (w, h), params['R'], params['T'], alpha=0)
        
        map1x, map1y = cv2.initUndistortRectifyMap(
            params['cameraMatrix1'], params.get('distCoeffs1'), R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(
            params['cameraMatrix2'], params.get('distCoeffs2'), R2, P2, (w, h), cv2.CV_32FC1)
        
        left_rect = cv2.remap(left_gray, map1x, map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, map2x, map2y, cv2.INTER_LINEAR)
        
        # 计算视差
        disp, valid_ratio = stereo_match_with_block_size(left_rect, right_rect)
        
        # 计算点云
        pts3d = cv2.reprojectImageTo3D(disp, Q)
        valid_mask = (disp > 0) & np.isfinite(pts3d[:, :, 2])
        valid_points = np.sum(valid_mask)
        point_density = valid_points / (h * w)
        
        # 计算点云分布范围
        z_values = pts3d[valid_mask, 2]
        z_range = np.max(z_values) - np.min(z_values) if len(z_values) > 0 else 0
        
        # 计算点云精度指标（视差图的标准差）
        disp_valid = disp[valid_mask]
        disp_std = np.std(disp_valid) if len(disp_valid) > 0 else 0
        
        results.append({
            'baseline': label,
            'scale_factor': scale,
            'valid_ratio': valid_ratio,
            'point_density': point_density,
            'z_range': z_range,
            'disp_std': disp_std
        })
    
    # 可视化结果
    if results:
        plt.figure(figsize=(15, 15))
        
        plt.subplot(4, 1, 1)
        baselines = [r['baseline'] for r in results]
        valid_ratios = [r['valid_ratio'] for r in results]
        plt.plot(baselines, valid_ratios, 'o-', label='有效视差比例')
        plt.ylabel('有效视差比例')
        plt.xlabel('相机基线距离')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(4, 1, 2)
        point_densities = [r['point_density'] for r in results]
        plt.plot(baselines, point_densities, 'o-', color='green', label='点云密度')
        plt.ylabel('点云密度')
        plt.xlabel('相机基线距离')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(4, 1, 3)
        z_ranges = [r['z_range'] for r in results]
        plt.plot(baselines, z_ranges, 'o-', color='red', label='深度范围')
        plt.ylabel('深度范围')
        plt.xlabel('相机基线距离')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        disp_stds = [r['disp_std'] for r in results]
        plt.plot(baselines, disp_stds, 'o-', color='purple', label='视差标准差')
        plt.ylabel('视差标准差')
        plt.xlabel('相机基线距离')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('baseline_influence.png')
        print("基线影响分析图已保存为 baseline_influence.png")
    else:
        print("没有足够的基线数据进行分析")
    
    return results

def main():
    # 创建实验结果目录
    exp_dir = f"experiment_results"
    os.makedirs(exp_dir, exist_ok=True)
    print(f"实验结果将保存到: {exp_dir}")
    
    # 加载标定参数（使用单一参数集）
    try:
        params_path = "stereo_params.npz"  # 你的标定参数文件
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"标定参数文件不存在: {params_path}")
        params = load_params(params_path)
    except Exception as e:
        print(f"加载标定参数失败: {e}")
        return
    
    # 设置图像路径
    left_img = "figures/use4ply/2_left.png"     # 左图路径
    right_img = "figures/use4ply/2_right.png"   # 右图路径
    if not os.path.exists(left_img) or not os.path.exists(right_img):
        print(f"图像文件不存在: {left_img} 或 {right_img}")
        return
    
    # 1. 分析块大小影响（最大块大小调整为13）
    print("\n===== 开始分析块大小影响 =====")
    block_results = analyze_block_size_influence(
        left_img, right_img, params, 
        block_sizes=[3,5,7,9,11,13]  # 移除15，保留到13
    )
    
    # 2. 分析相机与目标距离影响
    print("\n===== 开始分析距离影响 =====")
    # 不同距离的标定图像文件夹
    distance_folders = {
        "近距离(0.5m)": "calibration_data/0.5m",
        "中距离(1.0m)": "calibration_data/1.0m",
        "远距离(2.0m)": "calibration_data/2.0m"
    }
    distance_results = analyze_distance_influence(distance_folders)
    
    # 3. 分析相机基线影响（使用单一参数模拟不同基线）
    print("\n===== 开始分析基线影响 =====")
    # 基线缩放因子，模拟5cm, 10cm, 15cm等不同基线
    baseline_scales = [1.0, 2.0, 3.0]  # 假设基础基线为5cm
    baseline_results = analyze_baseline_influence(params, left_img, right_img, baseline_scales)
    
    # 保存所有结果
    np.savez(os.path.join(exp_dir, "experiment_results.npz"),
             block_results=block_results,
             distance_results=distance_results,
             baseline_results=baseline_results)
    
    # 复制图像结果到实验目录
    for img in ["block_size_analysis.png", "block_size_disparity_maps.png",
                "distance_influence.png", "baseline_influence.png"]:
        if os.path.exists(img):
            shutil.move(img, os.path.join(exp_dir, img))
    
    print("\n所有实验完成！结果已保存到实验目录。")

if __name__ == "__main__":

    main()
