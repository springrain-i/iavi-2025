
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_pattern(pattern):
    cols, rows = map(int, pattern.lower().split('x'))
    return cols, rows

def create_objp(cols, rows, square_size):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp

def analyze_view_angle(image_folder, calib_path, pattern, square_size, threshold=100.0):
    """
    手动筛选图片， 避免距离过近或过远的图片影响结果
    """
    cols, rows = parse_pattern(pattern)
    objp = create_objp(cols, rows, square_size)
    calib = np.load(calib_path)
    mtx = calib['mtx']
    dist = calib['dist']

    images = glob.glob(os.path.join(image_folder, '*.png'))
    print(f"共找到图片数量：{len(images)}")
    if len(images) == 0:
        print(f"警告：未找到任何图像！")
        return []

    results = []
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"警告：无法加载图像 {fname}，已跳过")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 标定
            ok, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                print(f"  solvePnP失败: {fname}")
                continue
            # 重投影误差（过滤偏移过大的点）
            imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
            errors = np.linalg.norm(corners2.reshape(-1,2) - imgpoints2.reshape(-1,2), axis=1)
            mask = errors < threshold
            filtered_errors = errors[mask]
            mean_error = filtered_errors.mean() if len(filtered_errors) > 0 else np.nan
            # 计算view angle
            R, _ = cv2.Rodrigues(rvec)
            normal_cam = R @ np.array([0.0, 0.0, 1.0])
            cosang = float(normal_cam[2] / (np.linalg.norm(normal_cam) + 1e-12))
            tilt_deg = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            results.append({
                'image': os.path.basename(fname),
                'view_angle_deg': tilt_deg,
                'reprojection_error': mean_error,
                'num_points': int(mask.sum()),
                'num_total': int(len(errors))
            })
            print(f"  {fname}: view_angle={tilt_deg:.2f}°, reproj_error={mean_error:.6f} 像素, 有效点数: {int(mask.sum())}/{len(errors)}")
        else:
            print(f"  未检测到角点: {fname}")
    return results

def save_and_plot(results, outdir):
    """
    保存结果为CSV和绘制重投影误差 vs view angle 曲线
    """
    if not results:
        print("没有有效结果可保存")
        return
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_filename = os.path.join(outdir, f'view_angle_analysis.csv')
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"结果表格已保存为: {csv_filename}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['view_angle_deg'], df['reprojection_error'], c='b', s=40, label='Images')
    plt.xlabel('View Angle (deg)', fontsize=12)
    plt.ylabel('Reprojection Error (pixels)', fontsize=12)
    plt.title('Effect of View Angle on Reprojection Error', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # 拟合一条线性趋势线
    if len(df) > 1:
        m, b = np.polyfit(df['view_angle_deg'], df['reprojection_error'], 1)
        xs = np.linspace(df['view_angle_deg'].min(), df['view_angle_deg'].max(), 100)
        plt.plot(xs, m * xs + b, color='red', linestyle='--', label='Linear Fit')
        corr = np.corrcoef(df['view_angle_deg'], df['reprojection_error'])[0, 1]
        plt.legend()
        plt.title(f'Effect of View Angle on Reprojection Error (corr={corr:.3f})', fontsize=14, fontweight='bold')
    plot_filename = os.path.join(outdir, f'view_angle_vs_reproj_error.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"分析图表已保存为: {plot_filename}")

    # 打印统计
    print("\nSummary Statistics:")
    print("=" * 60)
    for _, row in df.iterrows():
        print(f"{row['image']:20} | view_angle: {row['view_angle_deg']:6.2f}° | reproj_error: {row['reprojection_error']:.6f} 像素")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='images', help='图片文件夹')
    parser.add_argument('--calib', default='../camera_params_generated.npz')
    parser.add_argument('--pattern', default='11x8')
    parser.add_argument('--square_size', type=float, default=25.0)
    parser.add_argument('--outdir', default='analysis')
    parser.add_argument('--threshold', type=float, default=2.0, help='重投影误差阈值')
    args = parser.parse_args()

    print("分析view angle对重投影误差的影响（单文件夹，过滤偏移过大点）")
    print("=" * 50)
    results = analyze_view_angle(args.images, args.calib, args.pattern, args.square_size, args.threshold)
    if results:
        save_and_plot(results, args.outdir)
    else:
        print("分析完成，但没有有效结果")

if __name__ == "__main__":
    main()
