import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 实验组路径
GROUP_PATHS = {
    "High Coverage (H)": r"C:\Users\27198\Desktop\camera_lab2\coverage\high",
    "Mid Coverage (M)": r"C:\Users\27198\Desktop\camera_lab2\coverage\mid",
    "Low Coverage (L)": r"C:\Users\27198\Desktop\camera_lab2\coverage\low"
}

# 棋盘格设置
CHESS_SIZE = (11, 8)  
SQUARE_LEN = 2.0 
OUT_DIR = r"C:\Users\27198\Desktop\camera_lab2\coverage_analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

def calc_coverage(img_path, corners):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    total_area = h * w
    
    x_vals = corners[:, 0, 0] 
    y_vals = corners[:, 0, 1]  
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    
    chess_area = (x_max - x_min) * (y_max - y_min)
    coverage = (chess_area / total_area) * 100
    return round(coverage, 2)

def find_corners_and_calibrate(img_dir, pattern, square_len):
    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_len  
    
    obj_list = [] 
    img_list = []  
    cover_list = []  
    valid_paths = [] 
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    last_gray = None
    for img_name in img_files:
        path = os.path.join(img_dir, img_name)
        img = cv2.imread(path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_gray = gray
        
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            cover_ratio = calc_coverage(path, corners_refined)
            
            obj_list.append(objp.copy())
            img_list.append(corners_refined)
            cover_list.append(cover_ratio)
            valid_paths.append(path)
            print(f"处理成功：{img_name} | 覆盖率：{cover_ratio}%")
        else:
            print(f"未检测到角点：{img_name}")
    
    print(f"=== 完成 {os.path.basename(img_dir)} | 有效图像：{len(valid_paths)} ===\n")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, last_gray.shape[::-1], None, None
    )
    
    mean_err = 0.0
    for i in range(len(obj_list)):
        imgpts_proj, _ = cv2.projectPoints(obj_list[i], rvecs[i], tvecs[i], mtx, dist)
        err_val = cv2.norm(img_list[i], imgpts_proj, cv2.NORM_L2) / len(imgpts_proj)
        mean_err += err_val
    mean_err /= len(obj_list)

    return {
        "reproj_err": round(mean_err, 4), 
        "cover_ratios": cover_list,
        "valid_count": len(valid_paths),
        "camera_matrix": mtx,
        "dist_coeffs": dist
    }

def analyze_groups(group_paths, pattern, square_len):
    results = {}
    for name, path in group_paths.items():
        print(f"==================== 分析组：{name} ====================")
        try:
            res = find_corners_and_calibrate(
                img_dir=path,
                pattern=pattern,
                square_len=square_len
            )
            avg_cover = round(np.mean(res["cover_ratios"]), 2)
            res["avg_cover"] = avg_cover
            res["group_path"] = path
            results[name] = res
            print(f"\n【{name} 结果】")
            print(f"平均覆盖率：{avg_cover}%")
            print(f"有效图像数：{res['valid_count']}")
            print(f"重投影误差：{res['reproj_err']}\n")
        except Exception as e:
            print(f"分析失败 {name}：{str(e)}\n")
            results[name] = {"error": str(e)}
    return results

def plot_cover_vs_error(results, out_dir):
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    groups = []
    covers = []
    errors = []
    err_stds = []  
    counts = []
    
    for name, res in results.items():
        if "error" not in res:  
            groups.append(name)
            covers.append(res["avg_cover"])
            errors.append(res["reproj_err"])
            err_stds.append(round(np.std(res["cover_ratios"]), 4))
            counts.append(res["valid_count"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        covers, errors,
        s=200, c='#2E86AB', alpha=0.8, edgecolors='#1A5276', linewidth=2,
        label='实验组'
    )
    
    for i, (group, cnt) in enumerate(zip(groups, counts)):
        ax.annotate(
            f"{group}\n(有效图：{cnt}张)",
            xy=(covers[i], errors[i]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='#DDDDDD')
        )
    
    z = np.polyfit(covers, errors, 1)
    p = np.poly1d(z)
    ax.plot(covers, p(covers), "r--", linewidth=2, alpha=0.7, label=f'趋势: y={z[0]:.6f}x+{z[1]:.4f}')
    
    ax.set_xlabel('棋盘格平均覆盖率（%）', fontsize=12, fontweight='bold')
    ax.set_ylabel('重投影误差（像素）', fontsize=12, fontweight='bold')
    ax.set_title('棋盘格覆盖率对重投影误差的影响', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "coverage_vs_error.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图表保存：{plot_path}")
    return {"slope": z[0], "intercept": z[1]}

def save_report(results, trend, out_dir):
    report_path = os.path.join(out_dir, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("棋盘格覆盖率对重投影误差影响分析\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. 实验设置\n")
        f.write("-"*30 + "\n")
        f.write(f"棋盘格角点：{CHESS_SIZE[0]}列 × {CHESS_SIZE[1]}行\n")
        f.write(f"方格边长：{SQUARE_LEN} cm\n")
        f.write(f"图像路径：\n")
        for name, path in GROUP_PATHS.items():
            f.write(f"  - {name}：{path}\n")
        f.write(f"分析时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("2. 详细结果\n")
        f.write("-"*30 + "\n")
        for name, res in results.items():
            f.write(f"\n【{name}】\n")
            if "error" in res:
                f.write(f"  状态：失败\n")
                f.write(f"  原因：{res['error']}\n")
                continue
            f.write(f"  状态：成功\n")
            f.write(f"  平均覆盖率：{res['avg_cover']}%\n")
            f.write(f"  有效图像数：{res['valid_count']}\n")
            f.write(f"  各图覆盖率：{res['cover_ratios']}\n")
            f.write(f"  重投影误差：{res['reproj_err']}\n")
            f.write(f"  内参矩阵：\n{res['camera_matrix'][:2, :2]}\n")
        
        f.write("\n3. 趋势分析\n")
        f.write("-"*30 + "\n")
        if trend:
            slope = trend["slope"]
            intercept = trend["intercept"]
            f.write(f"线性趋势：y = {slope:.6f}x + {intercept:.4f}\n")
            if slope < 0:
                f.write("结论：覆盖率与误差负相关——覆盖率越大，误差越小\n")
            else:
                f.write("结论：覆盖率与误差正相关（异常，需检查）\n")
        else:
            f.write("无法分析趋势（有效组不足）\n")
    
    print(f"报告保存：{report_path}")
    return report_path

def main():
    print("=" * 80)
    print("棋盘格覆盖率对标定精度影响分析")
    print("=" * 80)
    
    try:
        print("\n步骤1：分析实验组...")
        results = analyze_groups(GROUP_PATHS, CHESS_SIZE, SQUARE_LEN)
        
        print("\n步骤2：生成图表...")
        trend = plot_cover_vs_error(results, OUT_DIR)
        
        print("\n步骤3：生成报告...")
        report_path = save_report(results, trend, OUT_DIR)
        
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        
        success_groups = [name for name, res in results.items() if "error" not in res]
        print(f"成功组数：{len(success_groups)}/{len(GROUP_PATHS)}")
        
        for name in success_groups:
            res = results[name]
            print(f"\n{name}:")
            print(f"  - 平均覆盖率：{res['avg_cover']}%")
            print(f"  - 重投影误差：{res['reproj_err']}")
            print(f"  - 有效图像数：{res['valid_count']}")
        
        print(f"\n输出文件：")
        print(f"  - 图表：{os.path.join(OUT_DIR, 'coverage_vs_error.png')}")
        print(f"  - 报告：{report_path}")
        
        if trend:
            slope = trend["slope"]
            if slope < 0:
                print(f"\n趋势：覆盖率与误差负相关（斜率：{slope:.6f}）")
                print("结论：增大覆盖率可降低标定误差")
            else:
                print(f"\n趋势：覆盖率与误差正相关（斜率：{slope:.6f}）")
                print("注意：结果异常，请检查数据")
                
    except Exception as e:
        print(f"\n错误：{str(e)}")
        print("请检查：")
        print("1. 文件路径")
        print("2. 棋盘格参数")
        print("3. 图像完整性")
        return False
    
    return True

if __name__ == "__main__":
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
    
    success = main()