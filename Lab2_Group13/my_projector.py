import numpy as np
import cv2
import open3d as o3d
import os
import glob
from sklearn.neighbors import KDTree

class ImprovedMultiProjector:
    def __init__(self, calibration_file: str):
        """改进版多图像投影器"""
        self.data = np.load(calibration_file)
        self.image_params = self._load_image_params_from_arrays()

    def _load_image_params_from_arrays(self):
        """从数组加载所有图像的相机参数"""
        params = {}

        mtx = self.data['mtx']
        dist = self.data['dist']
        rvecs = self.data['rvecs']
        tvecs = self.data['tvecs']

        num_images = len(rvecs)
        print(f"找到 {num_images} 个图像的相机参数")

        for i in range(num_images):
            img_id = f"image_{i + 1}"
            params[img_id] = {
                'camera_matrix': mtx,
                'rvec': rvecs[i],
                'tvec': tvecs[i],
                'dist_coeffs': dist
            }

        return params

    def rodrigues_to_matrix(self, rvec):
        """旋转向量转旋转矩阵"""
        theta = np.linalg.norm(rvec)
        if theta < 1e-10:
            return np.eye(3)

        u = rvec.flatten() / theta
        u_cross = np.array([[0, -u[2], u[1]],
                            [u[2], 0, -u[0]],
                            [-u[1], u[0], 0]])

        R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(u, u) + np.sin(theta) * u_cross
        return R

    def depth_based_occlusion(self, points_2d, points_cam, image_shape, depth_threshold=0.05):
        """
        基于深度的简单遮挡处理
        在2D空间中，对于位置相近的点，只保留最近的点
        """
        h, w = image_shape[:2]

        # 创建深度缓冲区
        depth_buffer = np.full((h, w), np.inf)
        point_mask = np.ones(len(points_2d), dtype=bool)

        # 按深度排序（从近到远）
        depths = points_cam[:, 2]
        sorted_indices = np.argsort(depths)

        for idx in sorted_indices:
            x, y = int(points_2d[idx, 0]), int(points_2d[idx, 1])
            if 0 <= x < w and 0 <= y < h:
                current_depth = depths[idx]
                # 如果当前点比缓冲区中的点更近，则替换
                if current_depth < depth_buffer[y, x] + depth_threshold:
                    depth_buffer[y, x] = current_depth
                else:
                    # 如果被遮挡，标记为不可见
                    point_mask[idx] = False

        return point_mask

    def visibility_based_culling(self, points_3d, camera_position, max_angle=60):
        """
        基于可见性的剔除 - 更适合点云的方法
        移除与相机视线方向夹角过大的点
        """
        # 确保相机位置是正确形状 (3,)
        camera_position = camera_position.flatten()

        # 计算从相机到点的向量
        camera_to_point = points_3d - camera_position
        distances = np.linalg.norm(camera_to_point, axis=1)

        # 归一化
        camera_to_point_norm = camera_to_point / distances[:, np.newaxis]

        # 假设相机朝向Z轴负方向（OpenCV坐标系）
        camera_forward = np.array([0, 0, -1])

        # 计算夹角（度）
        angles = np.degrees(np.arccos(np.clip(
            np.dot(camera_to_point_norm, camera_forward), -1, 1
        )))

        # 保留夹角较小的点
        visible_mask = angles < max_angle

        print(f"可见性剔除: {np.sum(visible_mask)}/{len(points_3d)} 个点可见 (角度阈值: {max_angle}度)")
        return visible_mask

    def cluster_based_culling(self, points_3d, camera_position, cluster_threshold=0.1):
        """
        基于聚类的剔除 - 移除孤立的点
        """
        from sklearn.cluster import DBSCAN

        # 确保相机位置是正确形状 (3,)
        camera_position = camera_position.flatten()

        # 将点云投影到以相机为中心的球坐标系
        vectors = points_3d - camera_position
        distances = np.linalg.norm(vectors, axis=1)
        directions = vectors / distances[:, np.newaxis]

        # 使用方向向量进行聚类
        clustering = DBSCAN(eps=cluster_threshold, min_samples=5).fit(directions)

        # 找到最大的聚类（主要可见表面）
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

        if len(unique_labels) > 0:
            main_cluster = unique_labels[np.argmax(counts)]
            visible_mask = labels == main_cluster
        else:
            visible_mask = np.ones(len(points_3d), dtype=bool)

        print(f"聚类剔除: {np.sum(visible_mask)}/{len(points_3d)} 个点可见")
        return visible_mask

    def get_camera_position(self, rvec, tvec):
        """计算相机在世界坐标系中的位置"""
        R = self.rodrigues_to_matrix(rvec)
        # 相机位置 = -R^T * t
        camera_position = -R.T @ tvec
        return camera_position.flatten()

    def project_points(self, points_3d, colors, image_id, method="depth_occlusion", **kwargs):
        """投影3D点到指定图像，支持多种剔除方法"""
        params = self.image_params[image_id]
        cm = params['camera_matrix']
        rvec = params['rvec']
        tvec = params['tvec']
        dist = params['dist_coeffs']

        # 世界坐标 -> 相机坐标
        R = self.rodrigues_to_matrix(rvec)
        points_cam = (R @ points_3d.T + tvec).T

        # 基础剔除：相机后面的点
        basic_mask = points_cam[:, 2] > 0
        points_cam_basic = points_cam[basic_mask]
        colors_basic = colors[basic_mask]

        if len(points_cam_basic) == 0:
            return np.array([]), np.array([]), np.array([])

        # 应用选择的剔除方法
        if method == "depth_occlusion":
            # 先投影到2D，然后进行深度遮挡处理
            points_2d_temp, _, _ = self._project_to_2d(points_cam_basic, cm, dist)
            visible_mask = self.depth_based_occlusion(
                points_2d_temp, points_cam_basic,
                kwargs.get('image_shape', (1080, 1920, 3)),
                kwargs.get('depth_threshold', 0.05)
            )
        elif method == "visibility":
            camera_position = self.get_camera_position(rvec, tvec)
            visible_mask = self.visibility_based_culling(
                points_3d[basic_mask], camera_position,
                kwargs.get('max_angle', 60)
            )
        elif method == "cluster":
            camera_position = self.get_camera_position(rvec, tvec)
            visible_mask = self.cluster_based_culling(
                points_3d[basic_mask], camera_position,
                kwargs.get('cluster_threshold', 0.1)
            )
        else:
            # 不使用剔除
            visible_mask = np.ones(len(points_cam_basic), dtype=bool)

        points_cam_final = points_cam_basic[visible_mask]
        colors_final = colors_basic[visible_mask]
        final_indices = np.where(basic_mask)[0][visible_mask]

        if len(points_cam_final) == 0:
            return np.array([]), np.array([]), np.array([])

        # 最终投影
        points_2d, _, _ = self._project_to_2d(points_cam_final, cm, dist)

        return points_2d, colors_final, final_indices

    def _project_to_2d(self, points_cam, camera_matrix, dist_coeffs):
        """内部投影函数"""
        # 归一化坐标
        x = points_cam[:, 0] / points_cam[:, 2]
        y = points_cam[:, 1] / points_cam[:, 2]

        # 畸变校正
        if np.any(dist_coeffs != 0):
            dist = dist_coeffs.flatten()
            if len(dist) < 5:
                dist = np.pad(dist, (0, 5 - len(dist)))

            r2 = x ** 2 + y ** 2
            radial = 1 + dist[0] * r2 + dist[1] * r2 ** 2 + dist[4] * r2 ** 3
            x = x * radial + 2 * dist[2] * x * y + dist[3] * (r2 + 2 * x ** 2)
            y = y * radial + dist[2] * (r2 + 2 * y ** 2) + 2 * dist[3] * x * y

        # 应用内参
        u = camera_matrix[0, 0] * x + camera_matrix[0, 2]
        v = camera_matrix[1, 1] * y + camera_matrix[1, 2]

        return np.column_stack([u, v]), points_cam, np.arange(len(points_cam))


def load_ply(ply_path, max_points=100000):
	pcd = o3d.io.read_point_cloud(ply_path)
	pts = np.asarray(pcd.points)
	cols = np.asarray(pcd.colors)
	if cols.size == 0:
		# default to blue if no color in PLY
		cols = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(pts), 1))
		print("点云无颜色，使用默认蓝色")

	valid = ~np.isnan(pts).any(axis=1)
	pts = pts[valid]
	cols = cols[valid]

	if len(pts) > max_points:
		idx = np.random.choice(len(pts), max_points, replace=False)
		pts = pts[idx]
		cols = cols[idx]

	return pts, cols


def colors_to_bgr(colors):
    """颜色格式转换"""
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    bgr_colors = colors[:, [2, 1, 0]]
    return bgr_colors

def simple_scale_pointcloud(points_3d, colors, min_size=0.5, target_size=200):
    """缩放点云"""
    bbox_size = np.ptp(points_3d, axis=0)
    current_size = np.linalg.norm(bbox_size)

    print(f"当前点云尺寸: {current_size:.3f}")

    if current_size < min_size:
        center = np.mean(points_3d, axis=0)
        scale_factor = target_size / current_size
        scaled_points = (points_3d - center) * scale_factor + center
        new_size = current_size * scale_factor
        print(f"已缩放: {current_size:.3f} -> {new_size:.3f} (因子: {scale_factor:.2f})")
        return scaled_points, colors
    else:
        print("点云尺寸正常，无需缩放")
        return points_3d, colors

def map_images_to_ids(image_files, projector):
    """映射图像文件到图像ID"""
    image_mapping = {}
    available_ids = list(projector.image_params.keys())

    print(f"找到 {len(image_files)} 个图像文件")
    print(f"可用的图像ID: {available_ids}")

    if len(image_files) == len(available_ids):
        for i, img_path in enumerate(sorted(image_files)):
            img_id = available_ids[i]
            image_mapping[img_id] = img_path
            print(f"  映射: {img_id} -> {os.path.basename(img_path)}")
    else:
        for img_path in sorted(image_files):
            filename = os.path.basename(img_path)
            for img_id in available_ids:
                if img_id.replace('image_', '') in filename:
                    image_mapping[img_id] = img_path
                    print(f"  映射: {img_id} -> {filename}")
                    break
            else:
                if available_ids:
                    img_id = available_ids.pop(0)
                    image_mapping[img_id] = img_path
                    print(f"  映射: {img_id} -> {filename}")

    return image_mapping