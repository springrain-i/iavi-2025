import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob  # 用于解析通配符
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import lstsq
# 对于数学文本使用默认字体，中文使用SimHei
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def multi_frame_denoising(image_paths):
    #使用多帧平均法去除图像噪声
    frames = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"无法读取图像: {path}")

        # 转换为RGB并归一化（避免OpenCV默认BGR格式问题）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb.astype(np.float32) / 255.0)

    # 多帧平均去噪
    denoised_image = np.mean(frames, axis=0)
    denoised_image = np.clip(denoised_image, 0, 1)  # 确保像素值在[0,1]

    return denoised_image, frames


def calculate_noise_properties(frames, denoised):
    """
    计算噪声的多个属性，包括均值、标准差和空间分布。
    - 噪声定义为原始帧与去噪帧（信号最佳估计）的差值。
    - 返回原始噪声（非绝对值），以便计算标准差。
    """
    noise_maps = []
    for i, frame in enumerate(frames):
        # 噪声 = 信号 - 最佳估计，这里保留了正负号
        noise = frame - denoised
        noise_maps.append(noise)
        print(f"已计算第{i + 1}张图像的原始噪声（保留符号）")

    # 1. 计算平均噪声矩阵（用于后续统计）
    # 这个矩阵代表了每个像素位置的平均噪声偏差
    avg_noise_matrix = np.mean(noise_maps, axis=0)

    # 2. 计算用于可视化的噪声分布图
    # 使用噪声的绝对值的平均值，更直观地展示噪声强度
    avg_noise_abs = np.mean(np.abs(noise_maps), axis=0)
    # 归一化以便于显示
    avg_noise_visual = (avg_noise_abs - np.min(avg_noise_abs)) / (np.max(avg_noise_abs) - np.min(avg_noise_abs) + 1e-8)

    # 3. 返回原始噪声图列表，用于计算整体标准差
    return avg_noise_matrix, avg_noise_visual, noise_maps


def plot_noise_histogram(noise_matrix, bins=50):
    """绘制噪声数值分布直方图（展示噪声强度的统计特征）"""
    noise_values = noise_matrix.flatten()  # 三维噪声矩阵→一维数组（便于统计）

    plt.figure(figsize=(10, 6))
    # 绘制频率密度直方图（density=True：面积和为1，便于对比不同参数组）
    n, bins, patches = plt.hist(noise_values, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # 添加统计标注（平均值、标准差，帮助分析噪声强度）
    mean_val = np.mean(noise_values)
    std_val = np.std(noise_values)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_val:.6f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f'平均值±标准差')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1)

    # 图表美化
    plt.title('噪声数值分布直方图', fontsize=14, fontweight='bold')
    plt.xlabel('噪声强度（归一化后）', fontsize=12)
    plt.ylabel('频率密度', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='-')
    plt.legend(fontsize=10)

    return plt.gcf()  # 返回图形对象，便于后续保存


def plot_results(original_samples, denoised, avg_noise, param_label):
    """展示原始图、去噪图、噪声分布的对比（标注参数组，便于区分）"""
    plt.figure(figsize=(18, 10))

    # 原始图1（含噪声）
    plt.subplot(2, 3, 1)
    plt.imshow(original_samples[0])
    plt.title(f"原始图像 1（含噪声）\n{param_label}", fontsize=11)
    plt.axis("off")

    # 原始图2（含噪声）
    plt.subplot(2, 3, 2)
    plt.imshow(original_samples[1])
    plt.title(f"原始图像 2（含噪声）\n{param_label}", fontsize=11)
    plt.axis("off")

    # 去噪结果图
    plt.subplot(2, 3, 3)
    plt.imshow(denoised)
    plt.title(f"5帧平均去噪结果\n{param_label}", fontsize=11)
    plt.axis("off")

    # 平均噪声分布图（用viridis色表，噪声强度区分更明显）
    plt.subplot(2, 1, 2)
    im = plt.imshow(avg_noise, cmap='viridis')
    plt.colorbar(im, label='平均噪声强度（归一化）', fraction=0.046, pad=0.04)
    plt.title(f"平均噪声分布（所有图像与去噪图差值的平均）\n{param_label}", fontsize=12, fontweight='bold')
    plt.axis("off")

    plt.tight_layout()
    return plt.gcf()




def fit_and_plot_Exposure(result_root_dir, noise_data,fixed_gain_for_exp_analysis):
    # 分析噪声与 Exposure 的关系，固定 Gain
    def quadratic_model(x, a, b, c):
        return a * x**2 + b * x + c
    
    exp_data = sorted([d for d in noise_data if d['gain'] == fixed_gain_for_exp_analysis], key=lambda x: x['exp'])
    
    if exp_data:
        exposures = np.array([d['exp'] for d in exp_data])
        noise_stds_exp = np.array([d['noise_std'] for d in exp_data])

        # 尝试二次拟合
        try:
            params_quad, _ = curve_fit(quadratic_model, exposures, noise_stds_exp)
            y_fit_quad = quadratic_model(exposures, *params_quad)
            r2_quad = r2_score(noise_stds_exp, y_fit_quad)
            
            # 绘制结果
            plt.figure(figsize=(12, 8))
            plt.scatter(exposures, noise_stds_exp, label='实际噪声数据', color='blue', zorder=5)
            plt.plot(exposures, y_fit_quad, label='二次模型拟合', color='red', linestyle='--')
            
            # 在图表上显示公式和 R^2
            eq_text = f'y = {params_quad[0]:.2e}x^2 + {params_quad[1]:.2e}x + {params_quad[2]:.4f}'
            r2_text = f'R^2 = {r2_quad:.4f}'
            plt.title(f'噪声标准差 vs. 曝光 (固定增益 = {fixed_gain_for_exp_analysis})', fontsize=14)
            plt.xlabel('曝光 (Exposure)', fontsize=12)
            plt.ylabel('噪声标准差 (Noise Std. Dev.)', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.text(0.05, 0.95, f'拟合模型:\n{eq_text}\n{r2_text}', transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
            
            model_fig_path = result_root_dir / f"noise_vs_exposure_model_gain{fixed_gain_for_exp_analysis}.png"
            plt.savefig(model_fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n噪声-曝光模型分析完成。拟合优度 (R²): {r2_quad:.4f}")
            print(f"结果图已保存至: {model_fig_path}")

        except RuntimeError:
            print(f"错误：无法为 增益={fixed_gain_for_exp_analysis} 拟合噪声-曝光模型。数据不足或不合适。")

    else:
        print(f"警告：未找到 增益={fixed_gain_for_exp_analysis} 的数据，无法进行噪声-曝光分析。")

def fit_and_plot_Gain(result_root_dir, noise_data, fixed_exp_for_gain_analysis):
    # 分析噪声与 Gain 的关系，固定 Exposure
    
    def linear_model(x, a, b):
        return a * x + b
    
    def sqrt_model(x, a, b):
        return a * np.sqrt(x) + b
    
    def mixed_model(x, a, b, c):
        """混合模型：sqrt(gain) + linear(gain) + constant"""
        return a * np.sqrt(x) + b * x + c
    
    gain_data = sorted([d for d in noise_data if d['exp'] == fixed_exp_for_gain_analysis], key=lambda x: x['gain'])

    if gain_data:
        gains = np.array([d['gain'] for d in gain_data])
        noise_stds_gain = np.array([d['noise_std'] for d in gain_data])

        # 尝试三种不同的模型
        models = {
            'linear': linear_model,
            'sqrt': sqrt_model,
            'mixed': mixed_model
        }
        
        best_r2 = -np.inf
        best_model_name = None
        best_params = None
        best_y_fit = None
        
        # 尝试拟合所有模型
        for model_name, model_func in models.items():
            try:
                if model_name == 'linear':
                    params, _ = curve_fit(model_func, gains, noise_stds_gain)
                    y_fit = model_func(gains, *params)
                elif model_name == 'sqrt':
                    params, _ = curve_fit(model_func, gains, noise_stds_gain)
                    y_fit = model_func(gains, *params)
                elif model_name == 'mixed':
                    params, _ = curve_fit(model_func, gains, noise_stds_gain)
                    y_fit = model_func(gains, *params)
                
                r2 = r2_score(noise_stds_gain, y_fit)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = model_name
                    best_params = params
                    best_y_fit = y_fit
                    
                print(f"{model_name}模型 R²: {r2:.4f}")
                
            except RuntimeError:
                print(f"无法拟合 {model_name} 模型")
        
        # 绘制结果
        plt.figure(figsize=(12, 8))
        plt.scatter(gains, noise_stds_gain, label='实际噪声数据', color='green', zorder=5, s=50)
        
        # 绘制最佳拟合曲线
        if best_model_name == 'linear':
            plt.plot(gains, best_y_fit, label=f'线性拟合 (R²={best_r2:.4f})', color='red', linewidth=2)
            eq_text = f'y = {best_params[0]:.4f}x + {best_params[1]:.4f}'
        elif best_model_name == 'sqrt':
            plt.plot(gains, best_y_fit, label=f'平方根拟合 (R²={best_r2:.4f})', color='blue', linewidth=2)
            eq_text = f'y = {best_params[0]:.4f}√x + {best_params[1]:.4f}'
        elif best_model_name == 'mixed':
            plt.plot(gains, best_y_fit, label=f'混合拟合 (R²={best_r2:.4f})', color='purple', linewidth=2)
            eq_text = f'y = {best_params[0]:.4f}√x + {best_params[1]:.4f}x + {best_params[2]:.4f}'
        
        # 在图表上显示公式和 R²
        r2_text = f'R² = {best_r2:.4f}'
        plt.title(f'噪声标准差 vs. 增益 (固定曝光 = {fixed_exp_for_gain_analysis})', fontsize=14)
        plt.xlabel('增益 (Gain)', fontsize=12)
        plt.ylabel('噪声标准差 (Noise Std. Dev.)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 添加模型信息文本框
        plt.text(0.05, 0.95, f'最佳模型: {best_model_name}\n{eq_text}\n{r2_text}', 
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))

        model_fig_path = result_root_dir / f"noise_vs_gain_model_exp{fixed_exp_for_gain_analysis}.png"
        plt.savefig(model_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n噪声-增益模型分析完成。最佳模型: {best_model_name}, R²: {best_r2:.4f}")
        print(f"结果图已保存至: {model_fig_path}")

    else:
        print(f"警告：未找到 曝光={fixed_exp_for_gain_analysis} 的数据，无法进行噪声-增益分析。")

def fit_and_plot_3d_model(result_root_dir, noise_data):
    """
    使用二阶多项式拟合三维噪声模型: noise = f(exposure, gain)
    并绘制3D曲面图。
    """
    if not noise_data:
        print("错误：没有噪声数据可用于三维模型拟合。")
        return
    
    exposures = np.array([d['exp'] for d in noise_data])
    gains = np.array([d['gain'] for d in noise_data])
    noise_stds = np.array([d['noise_std'] for d in noise_data])

    # 定义二阶多项式模型
    def poly_2d(X, p00, p10, p01, p20, p11, p02):
        x, y = X
        return p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2
    
    try:
        # 拟合模型
        params, _ = curve_fit(poly_2d, (exposures, gains), noise_stds)
        
        # 计算 R²
        y_fit = poly_2d((exposures, gains), *params)
        r2 = r2_score(noise_stds, y_fit)

        # 打印模型方程
        p00, p10, p01, p20, p11, p02 = params
        print("\n" + "="*20 + " 三维噪声模型结果 " + "="*20)
        print(f"拟合方程: noise_std = f(exp, gain)")
        print(f"  f(x, y) = {p00:.4e} + {p10:.4e}*x + {p01:.4e}*y + {p20:.4e}*x² + {p11:.4e}*x*y + {p02:.4e}*y²")
        print(f"拟合优度 (R²): {r2:.4f}")
        print("="*62)

        # 绘制3D图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制实际数据点
        ax.scatter(exposures, gains, noise_stds, color='r', marker='o', label='实际噪声数据')

        # 创建网格用于绘制曲面
        x_surf = np.linspace(exposures.min(), exposures.max(), 50)
        y_surf = np.linspace(gains.min(), gains.max(), 50)
        x_surf, y_surf = np.meshgrid(x_surf, y_surf)
        z_surf = poly_2d((x_surf, y_surf), *params)
        
        # 绘制拟合曲面
        ax.plot_surface(x_surf, y_surf, z_surf, cmap='viridis', alpha=0.6, label='拟合模型')

        ax.set_xlabel('曝光 (Exposure)')
        ax.set_ylabel('增益 (Gain)')
        ax.set_zlabel('噪声标准差 (Noise Std. Dev.)')
        ax.set_title('三维噪声模型: Noise vs. Exposure & Gain', fontsize=16)
        
        # 保存图像
        model_fig_path = result_root_dir / "3d_noise_model.png"
        plt.savefig(model_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"三维模型图已保存至: {model_fig_path}")

    except RuntimeError as e:
        print(f"错误：无法拟合三维噪声模型。{e}")

def analyze_and_model_noise(result_root_dir, fixed_gain_for_exp_analysis, fixed_exp_for_gain_analysis):
    """
    分析已保存的噪声数据，建立噪声与曝光、增益的量化模型。
    1. 全面二维分析：为每个固定的增益/曝光值，进行独立的噪声建模。
    2. 三维模型分析：建立一个 noise = f(exposure, gain) 的通用模型。
    """


    print("\n" + "="*50)
    print("开始进行噪声建模与分析...")
    print("="*50)

    # 1. 提取所有噪声数据
    noise_data = []
    npy_files = glob.glob(str(result_root_dir / "**" / "average_noise_matrix.npy"), recursive=True)

    if not npy_files:
        print("错误：在 'denoised_results' 文件夹中未找到任何 'average_noise_matrix.npy' 文件。")
        print("请先运行脚本的主处理流程以生成噪声数据。")
        return

    for f in npy_files:
        try:
            parts = Path(f).parent.name.split('_')
            exp = float(parts[0].replace('exp', ''))
            gain = float(parts[1].replace('gain', ''))
            
            noise_matrix = np.load(f)
            noise_std = np.std(noise_matrix) #标准差
            noise_data.append({'exp': exp, 'gain': gain, 'noise_std': noise_std})
        except (ValueError, IndexError) as e:
            print(f"警告：无法解析文件夹名称 '{Path(f).parent.name}'。跳过此文件。错误: {e}")
            continue
    
    if not noise_data:
        print("错误：未能从任何 .npy 文件中成功提取数据。")
        return

    # 2. 分析噪声与 Exposure 的关系 (固定 Gain)
    fit_and_plot_Exposure(result_root_dir, noise_data, fixed_gain_for_exp_analysis)
    # 3. 分析噪声与 Gain 的关系 (固定 Exposure)
    fit_and_plot_Gain(result_root_dir, noise_data, fixed_exp_for_gain_analysis)

    print("二维分析完成，开始三维噪声模型拟合...")
    fit_and_plot_3d_model(result_root_dir,noise_data)

if __name__ == "__main__":
    # 简单 CLI：如果第一个参数为 'model'，跳过图像处理直接建模
    args = sys.argv[1:]
    image_dir = Path("./figure/param_sweep_2025-09-19_16-37-52")  # 图片所在文件夹
    exposure_list = [500.0, 1000.0, 5000.0, 10000.0, 15000.0, 20000.0]  # 曝光参数
    gain_list = [0.0, 5.0, 10.0, 13.0, 15.0, 17.0, 20.0]  # 增益参数
    fixed_gamma = 1.0  # 伽马参数（本次实验固定）

    
    result_root_dir = Path("./denoised_results")  # 结果根文件夹
    result_root_dir.mkdir(exist_ok=True)

    if args and args[0].lower() == "model":
        print("检测到 'model' 参数：跳过图像处理，直接执行建模分析。")
        analyze_and_model_noise(result_root_dir, fixed_gain_for_exp_analysis=5.0, fixed_exp_for_gain_analysis=5000.0)
        sys.exit(0)

    # 正常模式：遍历所有参数组合（曝光×增益）并生成噪声矩阵
    for exp in exposure_list:
        for g in gain_list:  # 变量名改为 g，避免覆盖 gain_list
            wildcard_pattern = str(image_dir / f"img_*_exp{exp}_gain{g}_gamma{fixed_gamma}_#*.png")
            image_paths = glob.glob(wildcard_pattern)
            if len(image_paths) < 5:
                print(f"警告：参数组（exp={exp}, gain={g}, gamma={fixed_gamma}）仅找到{len(image_paths)}张图片，跳过该组！")
                continue
            image_paths.sort()
            print(f"\n找到参数组（exp={exp}, gain={g}, gamma={fixed_gamma}）的{len(image_paths)}张图片：")
            for path in image_paths:
                print(f"- {Path(path).name}")

                # 5. 核心处理流程（去噪→噪声分析→结果保存）
                try:
                    # 5.1 多帧平均去噪
                    denoised, frames = multi_frame_denoising(image_paths)

                    # 5.2 计算噪声特性
                    avg_noise_matrix, avg_noise_visual, individual_noise_maps = calculate_noise_properties(frames, denoised)

                    # 5.3 创建该参数组的专属结果文件夹（避免不同组结果重名）
                    param_label = f"exp{exp}_gain{g}_gamma{fixed_gamma}"
                    param_result_dir = result_root_dir / param_label
                    param_result_dir.mkdir(exist_ok=True)

                    # # 5.4 保存对比结果图（原始图+去噪图+噪声分布）
                    # comparison_fig = plot_results(frames, denoised, avg_noise_visual, param_label)
                    # comparison_path = param_result_dir / "comparison_results.png"
                    # comparison_fig.savefig(str(comparison_path), bbox_inches='tight', dpi=300)
                    # plt.close(comparison_fig)
                    # print(f"对比结果图已保存至: {comparison_path}")

                    # # 5.5 保存噪声直方图
                    # # 注意：直方图现在使用原始噪声矩阵（包含负值），更能反映噪声的真实分布
                    # hist_fig = plot_noise_histogram(avg_noise_matrix)
                    # hist_path = param_result_dir / "noise_value_histogram.png"
                    # hist_fig.savefig(str(hist_path), bbox_inches='tight', dpi=300)
                    # plt.close(hist_fig)
                    # print(f"噪声直方图已保存至: {hist_path}")

                    # # 5.6 保存去噪后的图像
                    # denoised_path = param_result_dir / "denoised_result.png"  # 用png避免jpg压缩损失
                    # denoised_bgr = cv2.cvtColor((denoised * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(str(denoised_path), denoised_bgr)
                    # print(f"去噪图像已保存至: {denoised_path}")

                    # # 5.7 保存平均噪声分布图像（可视化版本）
                    # noise_dist_path = param_result_dir / "average_noise_distribution.png"
                    # noise_dist_bgr = cv2.cvtColor((avg_noise_visual * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(str(noise_dist_path), noise_dist_bgr)
                    # print(f"平均噪声分布图已保存至: {noise_dist_path}")

                    # # 5.8 保存每张原图的单独噪声分布
                    # for i, noise in enumerate(individual_noise_maps):
                    #     # 可视化时仍使用绝对值和归一化
                    #     individual_noise_norm = np.abs(noise)
                    #     individual_noise_norm = (individual_noise_norm - np.min(individual_noise_norm)) / (np.max(individual_noise_norm) - np.min(individual_noise_norm) + 1e-8)
                    #     ind_noise_path = param_result_dir / f"individual_noise_{i + 1}.png"
                    #     ind_noise_bgr = cv2.cvtColor((individual_noise_norm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    #     cv2.imwrite(str(ind_noise_path), ind_noise_bgr)
                    # print(f"所有单张图像的噪声分布已保存至: {param_result_dir}")

                    # 5.9 保存原始平均噪声矩阵（包含正负值，用于精确的统计分析）
                    np.save(str(param_result_dir / "average_noise_matrix.npy"), avg_noise_matrix)
                    print(f"平均噪声矩阵已保存至: {param_result_dir / 'average_noise_matrix.npy'}")

                except Exception as e:
                    print(f"参数组（exp={exp}, gain={g}, gamma={fixed_gamma}）处理失败: {str(e)}")

    print("\n所有参数组处理完成！结果已保存至: ./denoised_results")
    analyze_and_model_noise(result_root_dir, fixed_gain_for_exp_analysis=5.0, fixed_exp_for_gain_analysis=5000.0)