import cv2
import numpy as np
import math
from pathlib import Path


class TiltShiftSimulator:
    def __init__(self):
        self.focal_length = None  # 焦距(mm)
        self.sensor_width = None  # 传感器宽度(mm)
        self.aperture = None  # 光圈值
        self.circle_of_confusion = 0.03  # 弥散圆直径(mm)，默认值

    def load_image(self, image_path):
        """从指定路径加载图片"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("无法读取图片文件")

        print(f"成功加载图片: {path.name}")
        print(f"图片尺寸: {image.shape[1]} x {image.shape[0]}")
        return image

    def set_camera_parameters(self, focal_length, sensor_width, aperture):
        """设置相机内参"""
        self.focal_length = focal_length  # mm
        self.sensor_width = sensor_width  # mm
        self.aperture = aperture
        print(f"相机参数设置: 焦距={focal_length}mm, 传感器宽度={sensor_width}mm, 光圈=f/{aperture}")

    def calculate_dof_parameters(self, focus_distance):
        """计算景深参数（传统镜头）"""
        # 超焦距公式
        hyperfocal = (self.focal_length ** 2) / (self.aperture * self.circle_of_confusion)

        # 近点和远点
        near_point = (hyperfocal * focus_distance) / (hyperfocal + focus_distance)
        far_point = (hyperfocal * focus_distance) / (hyperfocal - focus_distance)

        if focus_distance > hyperfocal:
            far_point = float('inf')

        total_dof = far_point - near_point

        return {
            'hyperfocal': hyperfocal,
            'near_point': near_point,
            'far_point': far_point,
            'total_dof': total_dof
        }

    def calculate_tilt_angle(self, desired_dof, focus_distance):
        """
        计算移轴镜头倾斜角度
        基于Scheimpflug原理的简化计算
        """
        if self.focal_length is None:
            raise ValueError("请先设置相机参数")

        # 计算传统镜头的景深
        standard_dof = self.calculate_dof_parameters(focus_distance)['total_dof']

        # 简化模型：倾斜角度与景深变化的关系
        # 实际应用中这个关系更复杂，这里使用近似公式
        dof_ratio = desired_dof / standard_dof if standard_dof > 0 else 1

        # 基础倾斜角度计算（简化模型）
        # 实际公式需要考虑Scheimpflug平面和镜头光学特性
        base_angle = math.degrees(math.atan(self.sensor_width / (2 * self.focal_length)))

        # 根据景深需求调整角度
        if dof_ratio > 1:
            # 需要增加景深，使用较小的倾斜角度
            tilt_angle = base_angle * (1 / math.sqrt(dof_ratio))
        else:
            # 需要减少景深或特殊效果，使用较大的倾斜角度
            tilt_angle = base_angle * (1 + (1 - dof_ratio))

        # 限制角度在合理范围内
        tilt_angle = max(0.5, min(8.0, tilt_angle))

        return tilt_angle

    def apply_tilt_shift_effect(self, image, tilt_angle, focus_distance, blur_strength=10):
        """应用移轴摄影效果（模拟）"""
        height, width = image.shape[:2]

        # 创建模糊掩模
        mask = np.zeros((height, width), dtype=np.float32)

        # 根据倾斜角度和焦距确定清晰区域
        # 简化模型：创建倾斜的清晰带
        center_y = height // 2
        clear_band_height = max(50, int(height * 0.3))  # 清晰带高度

        # 根据倾斜角度调整清晰带的角度
        tilt_rad = math.radians(tilt_angle)
        for y in range(height):
            for x in range(width):
                # 计算点到倾斜中心线的距离
                distance_to_center = abs((x - width // 2) * math.sin(tilt_rad) +
                                         (y - center_y) * math.cos(tilt_rad))

                if distance_to_center < clear_band_height // 2:
                    mask[y, x] = 0.0  # 清晰区域
                else:
                    # 根据距离增加模糊程度
                    normalized_dist = min(1.0, distance_to_center / (height // 2))
                    mask[y, x] = normalized_dist

        # 应用模糊
        blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

        # 混合原图和模糊图
        result = np.zeros_like(image, dtype=np.uint8)
        for i in range(3):  # 对每个通道处理
            result[:, :, i] = (image[:, :, i] * (1 - mask) + blurred[:, :, i] * mask).astype(np.uint8)

        return result


def main():
    simulator = TiltShiftSimulator()

    try:
        # 1. 加载图片
        image_path = "figures/DSC00020.JPG"
        image = simulator.load_image(image_path)

        # 2. 设置相机参数
        print("\n请输入相机参数:")
        focal_length = 16
        sensor_width = 15.6
        aperture = 3.5
        simulator.set_camera_parameters(focal_length, sensor_width, aperture)

        # 3. 获取用户输入的景深需求
        print("\n请输入拍摄参数:")
        focus_distance = float(input("对焦距离(m): "))
        desired_dof = float(input("期望景深(m): "))

        # 4. 计算倾斜角度
        tilt_angle = simulator.calculate_tilt_angle(desired_dof, focus_distance)

        print(f"\n计算结果:")
        print(f"推荐倾斜角度: {tilt_angle:.2f}°")

        # # 5. 应用移轴效果并显示
        # result_image = simulator.apply_tilt_shift_effect(image, tilt_angle, focus_distance)
        #
        # # 显示原图和结果
        # cv2.imshow('Original Image', image)
        # cv2.imshow('Tilt-Shift Effect', result_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # # 保存结果
        # save_path = "tilt_shift_result.jpg"
        # cv2.imwrite(save_path, result_image)
        # print(f"结果已保存为: {save_path}")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()