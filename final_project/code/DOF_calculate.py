import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from servo_connect import robust_flash_workflow

# 图片分析
class ScheimpflugAnalyzer:
    def __init__(self):
        self.image = None
        self.gray = None
        self._setup_font()

    # 设置字体
    def _setup_font(self):
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    # 获取指定文件夹中最新的图片文件
    def get_latest_image(self, folder_path):
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg']

        # 读取符合条件的图片
        all_images = []
        for extension in image_extensions:
            all_images.extend(folder.glob(extension))
            all_images.extend(folder.glob(extension.upper()))

        if not all_images:
            raise FileNotFoundError(f"No images found in folder: {folder_path}")

        # 按创建时间排序
        all_images.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        latest_image = all_images[0]

        print(f"Found {len(all_images)} images in folder")
        print(f"Latest image: {latest_image.name}")
        print(f"Last created: {datetime.fromtimestamp(latest_image.stat().st_mtime)}")

        return str(latest_image)

    # 加载图片
    def load_image(self, image_path):
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.image = cv2.imread(str(path))
        if self.image is None:
            raise ValueError("Cannot read image file")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print(f"Successfully loaded image: {path.name}")
        print(f"Image size: {self.image.shape[1]} x {self.image.shape[0]}")
        return self.image

    # 检测物体平面
    def detect_object_plane(self):
        # 假设物体是近似直线的长条形物体
        if self.gray is None:
            raise ValueError("Please load image first")

        height, width = self.gray.shape

        # Canny边缘检测
        edges = cv2.Canny(self.gray, 50, 150)

        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=width * 0.3, maxLineGap=20)

        if lines is None:
            print("No significant lines detected, using default plane")
            # 如果没有检测到明显直线，假设物体是水平的
            return 0, [(0, height // 2), (width, height // 2)], []

        # 找出主要的方向
        angles = []
        valid_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if length > width * 0.2:  # 只考虑较长的直线
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
                valid_lines.append((x1, y1, x2, y2))

        if not angles:
            return 0, [(0, height // 2), (width, height // 2)], []

        # 计算平均角度（去除异常值）
        angles_array = np.array(angles)
        mean_angle = np.mean(angles_array)
        std_angle = np.std(angles_array)

        # 过滤异常值
        filtered_angles = [angle for angle in angles_array
                           if abs(angle - mean_angle) < 2 * std_angle]

        if filtered_angles:
            object_angle = np.mean(filtered_angles)
        else:
            object_angle = mean_angle

        print(f"Detected object plane angle: {object_angle:.2f} degrees")

        # 生成代表物体平面的线段
        center_x, center_y = width // 2, height // 2
        line_length = width * 0.8

        # 根据角度计算线段的端点
        dx = line_length * np.cos(np.radians(object_angle)) / 2
        dy = line_length * np.sin(np.radians(object_angle)) / 2

        line_points = [
            (int(center_x - dx), int(center_y - dy)),
            (int(center_x + dx), int(center_y + dy))
        ]

        return object_angle, line_points, valid_lines

    # 根据Scheimpflug原理计算镜头倾斜角度
    def calculate_tilt_angle(self, object_angle, camera_params):
        focal_length = camera_params['focal_length']  # mm

        # 将物体角度转换为相对于光轴的倾斜角度
        object_tilt_rad = np.radians(object_angle)

        # Scheimpflug 角度计算 (简化模型)
        image_distance = focal_length  # 近似像距

        # 计算镜头倾斜角度
        tilt_angle_rad = np.arctan((image_distance / focal_length) * np.tan(object_tilt_rad))
        tilt_angle_deg = np.degrees(tilt_angle_rad)

        # 限制在合理范围内 (±10度)
        tilt_angle_deg = max(-10, min(10, tilt_angle_deg))

        return tilt_angle_deg

    # 如果物体在垂直方向也有倾斜，计算摆动角度
    def calculate_swing_angle(self, object_lines):
        if not object_lines:
            return 0

        # 分析线条在垂直方向的分布
        vertical_angles = []
        for line in object_lines:
            x1, y1, x2, y2 = line
            if abs(x2 - x1) > abs(y2 - y1):  # 主要是水平线条
                vertical_var = abs(y2 - y1) / abs(x2 - x1)
                vertical_angle = np.arctan(vertical_var) * 180 / np.pi
                vertical_angles.append(vertical_angle)

        if vertical_angles:
            avg_vertical_angle = np.mean(vertical_angles)
            # 摆动角度通常比倾斜角度小
            swing_angle = avg_vertical_angle * 0.3
            swing_angle = max(-5, min(5, swing_angle))
            return swing_angle

        return 0

    # 可视化分析结果
    def visualize_analysis(self, object_angle, tilt_angle, swing_angle, line_points, detected_lines):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 原图与检测到的直线
        axes[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image with Detected Lines')
        axes[0, 0].axis('off')

        # 绘制检测到的所有直线
        for line in detected_lines:
            x1, y1, x2, y2 = line
            axes[0, 0].plot([x1, x2], [y1, y2], 'r-', alpha=0.6, linewidth=1)

        # 绘制主要物体平面
        if line_points:
            x_pts = [line_points[0][0], line_points[1][0]]
            y_pts = [line_points[0][1], line_points[1][1]]
            axes[0, 0].plot(x_pts, y_pts, 'g-', linewidth=3, label='Object Plane')
            axes[0, 0].legend()

        # 边缘检测图
        edges = cv2.Canny(self.gray, 50, 150)
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('Edge Detection')
        axes[0, 1].axis('off')

        # 原理示意图
        axes[1, 0].axis('off')
        schematic_text = f"""Scheimpflug Principle Analysis:

Object Plane Angle: {object_angle:.1f}°

Recommended Adjustments:
Lens Tilt Angle: {tilt_angle:.2f}°
Lens Swing Angle: {swing_angle:.2f}°

Explanation:
- Tilt: Rotate lens around horizontal axis
- Swing: Rotate lens around vertical axis
- This aligns all three planes (lens, image, object)
- Result: Entire object will be in focus"""

        axes[1, 0].text(0.05, 0.95, schematic_text, transform=axes[1, 0].transAxes,
                        verticalalignment='top', fontsize=11,
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

        # 角度可视化
        angles_plot = axes[1, 1]
        angles = ['Object Angle', 'Tilt Angle', 'Swing Angle']
        values = [abs(object_angle), abs(tilt_angle), abs(swing_angle)]
        colors = ['lightcoral', 'lightgreen', 'lightblue']

        bars = angles_plot.bar(angles, values, color=colors, alpha=0.7)
        angles_plot.set_ylabel('Angle (degrees)')
        angles_plot.set_title('Recommended Lens Adjustments')

        # 在柱状图上显示数值
        for bar, value in zip(bars, values):
            angles_plot.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                             f'{value:.2f}°', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    # 完整的分析流程
    def analyze_and_calculate_tilt(self, folder_path=None, image_path=None, camera_params=None):
        if folder_path:
            # 从文件夹获取最新图片
            image_path = self.get_latest_image(folder_path)
        elif image_path:
            # 直接使用指定的图片路径
            pass
        else:
            raise ValueError("Either folder_path or image_path must be provided")

        # 加载图片
        self.load_image(image_path)

        # 设置相机参数
        if camera_params is None:
            camera_params = {
                'focal_length': 16,  # mm
                'sensor_height': 15.6,  # mm
                'aperture': 3.5
            }

        # 检测物体平面
        object_angle, line_points, detected_lines = self.detect_object_plane()

        # 计算倾斜角度
        tilt_angle = self.calculate_tilt_angle(object_angle, camera_params)

        # 计算摆动角度
        swing_angle = self.calculate_swing_angle(detected_lines)

        # 显示结果
        print("\n=== Scheimpflug Analysis Results ===")
        print(f"Detected object plane angle: {object_angle:.2f}°")
        print(f"Recommended lens tilt angle: {tilt_angle:.2f}°")
        print(f"Recommended lens swing angle: {swing_angle:.2f}°")
        print(f"\nCamera parameters used:")
        print(f"Focal length: {camera_params['focal_length']}mm")
        print(f"Sensor height: {camera_params['sensor_height']}mm")

        # 可视化
        self.visualize_analysis(object_angle, tilt_angle, swing_angle, line_points, detected_lines)

        return {
            'object_angle': object_angle,
            'tilt_angle': tilt_angle,
            'swing_angle': swing_angle,
            'camera_params': camera_params,
            'image_path': image_path
        }


def main():
    analyzer = ScheimpflugAnalyzer()

    try:
        # 从文件夹获取最新图片
        # 相机与电脑连接后图片存在E盘此处
        folder_path = r'E:\DCIM\100MSDCF'
        results = analyzer.analyze_and_calculate_tilt(folder_path=folder_path)

        print(f"\nInstructions:")
        print(f"1. Tilt the lens by {results['tilt_angle']:.2f}° around horizontal axis")
        print(f"2. Swing the lens by {results['swing_angle']:.2f}° around vertical axis")
        print(f"3. This will make the entire object plane sharp")
        print(f"4. Analyzed image: {Path(results['image_path']).name}")

        # 限制镜头旋转角度小于10度（镜头硬件限制）
        if abs(results['tilt_angle']) > 10:
            tilt_angle = 10 * (results['tilt_angle']/abs(results['tilt_angle']))
        else:
            tilt_angle = results['tilt_angle']

        # 将相关指令烧录到arduino
        robust_flash_workflow(tilt_angle)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
