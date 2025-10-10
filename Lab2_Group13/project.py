import argparse
import os
import glob
import cv2
import numpy as np
import open3d as o3d
from my_projector import ImprovedMultiProjector, load_ply, simple_scale_pointcloud, colors_to_bgr,map_images_to_ids


def load_calibration(npz_path):
	data = np.load(npz_path)
	# expect keys: mtx, dist, rvecs, tvecs
	mtx = data['mtx']
	dist = data['dist']
	rvecs = data.get('rvecs')
	tvecs = data.get('tvecs')
	return {'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}


def create_objp(cols, rows, square_size):
	objp = np.zeros((rows * cols, 3), np.float32)
	objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
	objp *= square_size
	return objp


def rvec_to_R(rvec):
	R, _ = cv2.Rodrigues(rvec)
	return R


def image_center_pixel(img):
	h, w = img.shape[:2]
	return np.array([w / 2.0, h / 2.0], dtype=float)


def chessboard_center_pixel_from_extrinsics(rvec, tvec, cols, rows, square_size, mtx, dist):
	"""
	Project the chessboard's object-space center into image pixels using rvec/tvec.
	"""
	objp = create_objp(cols, rows, square_size)
	center = objp.mean(axis=0).reshape(1, 3)
	pts2d = project_with_rvec_tvec(center, rvec, tvec, mtx, dist)
	return pts2d[0]


def shift_pointcloud_to_target(pts3d, rvec, tvec, mtx, target_pixel):
	"""
	Translate pts3d in world coordinates so that its centroid projects to target_pixel.
	"""
	centroid = pts3d.mean(axis=0)
	R = rvec_to_R(rvec)
	# centroid in camera coords
	Xc = (R @ centroid) + tvec.flatten()
	Zc = Xc[2]
	if Zc <= 0:
		# if centroid is behind camera, just return original
		return pts3d

	fx = mtx[0, 0]
	fy = mtx[1, 1]
	cx = mtx[0, 2]
	cy = mtx[1, 2]

	u_target, v_target = float(target_pixel[0]), float(target_pixel[1])
	x_new = (u_target - cx) / fx
	y_new = (v_target - cy) / fy

	Xc_new = np.array([x_new * Zc, y_new * Zc, Zc])
	delta_cam = Xc_new - Xc
	# convert to world shift
	delta_world = R.T @ delta_cam
	return pts3d + delta_world


def shift_pointcloud_using_projector(pts3d, rvec, tvec, mtx, target_pixel, projector: ImprovedMultiProjector):
	"""Like shift_pointcloud_to_target but use projector.rodrigues_to_matrix to match its Rodrigues implementation."""
	centroid = pts3d.mean(axis=0)
	R = projector.rodrigues_to_matrix(rvec)
	Xc = (R @ centroid) + tvec.flatten()
	Zc = Xc[2]
	if Zc <= 0:
		return pts3d

	fx = mtx[0, 0]
	fy = mtx[1, 1]
	cx = mtx[0, 2]
	cy = mtx[1, 2]

	u_target, v_target = float(target_pixel[0]), float(target_pixel[1])
	x_new = (u_target - cx) / fx
	y_new = (v_target - cy) / fy

	Xc_new = np.array([x_new * Zc, y_new * Zc, Zc])
	delta_cam = Xc_new - Xc
	delta_world = R.T @ delta_cam
	return pts3d + delta_world


def project_with_rvec_tvec(pts3d, rvec, tvec, mtx, dist):
	pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, mtx, dist)
	return pts2d.reshape(-1, 2)


def parse_color_option(colstr: str):
	"""Return BGR tuple for given color option. Supports 'auto', 'pink', and hex '#RRGGBB'."""
	if not colstr:
		return None
	s = colstr.strip().lower()
	if s == 'auto':
		return None
	if s == 'pink':
		# pastel pink RGB (255,192,203) -> BGR
		return (203, 192, 255)
	if s.startswith('#') and len(s) == 7:
		try:
			r = int(s[1:3], 16)
			g = int(s[3:5], 16)
			b = int(s[5:7], 16)
			return (b, g, r)
		except Exception:
			return None
	# fallback: try to parse comma separated
	parts = s.split(',')
	if len(parts) == 3:
		try:
			r, g, b = [int(p) for p in parts]
			return (b, g, r)
		except Exception:
			return None
	return None


def solvepnp_and_project(img, pts3d, mtx, dist, cols, rows, square_size):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
	if not found:
		return None
	term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
	objp = create_objp(cols, rows, square_size)
	ok, rvec, tvec = cv2.solvePnP(objp, corners_refined, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
	if not ok:
		return None
	pts2d = project_with_rvec_tvec(pts3d, rvec, tvec, mtx, dist)
	return pts2d


def map_images_to_saved_ids(image_files, rvecs):
	# simple mapping: if counts match, map in order; otherwise try filename pattern
	mapping = {}
	if rvecs is None:
		# no saved extrinsics
		for i, p in enumerate(sorted(image_files)):
			mapping[f'image_{i+1}'] = p
		return mapping

	if len(rvecs) == len(image_files):
		for i, p in enumerate(sorted(image_files)):
			mapping[f'image_{i+1}'] = p
		return mapping

	# fallback: try to find index substring in filename
	remaining_ids = [f'image_{i+1}' for i in range(len(rvecs))]
	for p in sorted(image_files):
		name = os.path.basename(p)
		assigned = False
		for img_id in remaining_ids:
			if img_id.replace('image_', '') in name:
				mapping[img_id] = p
				remaining_ids.remove(img_id)
				assigned = True
				break
		if not assigned and remaining_ids:
			mapping[remaining_ids.pop(0)] = p

	return mapping


def batch_project(args):
	# Use ImprovedMultiProjector to get same projection/culling behavior
	projector = ImprovedMultiProjector(args.calib)
	# check PLY exists first
	if not os.path.exists(args.ply):
		print(f"点云文件未找到: {args.ply}\n请确认路径或文件名是否正确（当前工作目录: {os.getcwd()})")
		return

	pts3d, colors = load_ply(args.ply)
	if pts3d is None or getattr(pts3d, 'size', 0) == 0:
		print(f"加载点云失败或点云为空: {args.ply}\n请确认 PLY 文件有效且包含点。")
		return
	pts3d, colors = simple_scale_pointcloud(pts3d, colors)
	pts3d[:, 2] = -pts3d[:, 2]
	bgr = colors_to_bgr(colors)

	image_files = sorted(glob.glob(os.path.join(args.images, '*.png')))
	if not image_files:
		print('未找到任何 PNG 图像')
		return

	image_mapping = map_images_to_ids(image_files, projector)
	os.makedirs(args.outdir, exist_ok=True)
	cols_corners, rows_corners = map(int, args.pattern.lower().split('x'))

	for img_id, img_path in image_mapping.items():
		print('处理', img_path)
		img = cv2.imread(img_path)
		if img is None:
			print('  无法读取，跳过')
			continue

		# alignment: if requested, shift the point cloud using projector's Rodrigues implementation
		pts_to_project = pts3d.copy()
		if args.align in ('image_center', 'chessboard_center'):
			params = projector.image_params.get(img_id)
			if params is not None:
				rvec = params['rvec']
				tvec = params['tvec']
				if args.align == 'image_center':
					target = image_center_pixel(img)
				else:
					target = chessboard_center_pixel_from_extrinsics(rvec, tvec, cols_corners, rows_corners, args.square_size, params['camera_matrix'], params['dist_coeffs'])
				pts_to_project = shift_pointcloud_using_projector(pts3d, rvec, tvec, params['camera_matrix'], target, projector)
			else:
				# fallback: try solvePnP for this image to get rvec/tvec
				tmp = solvepnp_and_project(img, pts3d[:1000], projector.data['mtx'], projector.data['dist'], cols_corners, rows_corners, args.square_size)
				if tmp is not None:
					# found chessboard -> compute rvec/tvec and shift
					gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					found, corners = cv2.findChessboardCorners(gray, (cols_corners, rows_corners), None)
					term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
					corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
					objp = create_objp(cols_corners, rows_corners, args.square_size)
					ok, rvec_sp, tvec_sp = cv2.solvePnP(objp, corners_refined, projector.data['mtx'], projector.data['dist'], flags=cv2.SOLVEPNP_ITERATIVE)
					if ok:
						if args.align == 'image_center':
							target = image_center_pixel(img)
						else:
							target = chessboard_center_pixel_from_extrinsics(rvec_sp, tvec_sp, cols_corners, rows_corners, args.square_size, projector.data['mtx'], projector.data['dist'])
						pts_to_project = shift_pointcloud_using_projector(pts3d, rvec_sp, tvec_sp, projector.data['mtx'], target, projector)

		# debug: print camera extrinsics and depth statistics to diagnose empty projections
		params_dbg = projector.image_params.get(img_id)
		if params_dbg is not None:
			rvec_dbg = params_dbg['rvec']
			tvec_dbg = params_dbg['tvec']
			cam_pos_dbg = projector.get_camera_position(rvec_dbg, tvec_dbg)
			R_dbg = projector.rodrigues_to_matrix(rvec_dbg)
			pts_cam_dbg = (R_dbg @ pts_to_project.T + tvec_dbg).T
			z_dbg = pts_cam_dbg[:, 2]
			print(f"  相机位置 (world): {cam_pos_dbg}")
			print(f"  深度统计 (cam Z): min={z_dbg.min():.3f}, max={z_dbg.max():.3f}, mean={z_dbg.mean():.3f}, >0 count={int((z_dbg>0).sum())}/{len(z_dbg)}")
		else:
			print("  警告: 没有为此图像找到保存的外参，将用内部回退/solvePnP 处理")

		# apply uniform scaling about centroid if requested
		if abs(args.scale - 1.0) > 1e-6:
			c = pts_to_project.mean(axis=0)
			pts_to_project = (pts_to_project - c) * float(args.scale) + c

		# now use projector.project_points which handles culling and camera intrinsics
		points_2d, visible_colors, valid_indices = projector.project_points(pts_to_project, colors, img_id, method=args.method, image_shape=img.shape)
		# if nothing projected and camera depths were all non-positive, try flipping Z convention on the point cloud
		if len(points_2d) == 0 and params_dbg is not None:
			if (z_dbg <= 0).all():
				print('  检测到所有点在相机后方，尝试翻转点云 Z 轴并重试')
				pts_to_project_flipped = pts_to_project.copy()
				pts_to_project_flipped[:, 2] *= -1
				points_2d, visible_colors, valid_indices = projector.project_points(pts_to_project_flipped, colors, img_id, method=args.method, image_shape=img.shape)
				if len(points_2d) > 0:
					print('  翻转 Z 后投影成功')
					pts_to_project = pts_to_project_flipped

		if len(points_2d) == 0:
			print('  无点被投影 (projector.project_points 返回空)')
			continue

		out = img.copy()
		bgr_colors = colors_to_bgr(colors)
		override_color = parse_color_option(args.color)
		for i, pt in enumerate(points_2d):
			x, y = int(round(pt[0])), int(round(pt[1]))
			if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
				if override_color is not None:
					color = override_color
				else:
					color = tuple(int(c) for c in bgr_colors[valid_indices[i]])
				cv2.circle(out, (x, y), args.point_size, color, -1)

		out_path = os.path.join(args.outdir, os.path.basename(img_path))
		cv2.imwrite(out_path, out)
		print('  保存:', out_path)


def live_project(args):
	calib = load_calibration(args.calib)
	pts3d, cols = load_ply(args.ply, max_points=args.max_points)
	pts3d, cols = np.array(pts3d), np.array(cols)
	pts3d[:, 2] = -pts3d[:, 2]
	bgr = colors_to_bgr(cols)

	cols_corners, rows_corners = map(int, args.pattern.lower().split('x'))
	objp = create_objp(cols_corners, rows_corners, args.square_size)

	cap = cv2.VideoCapture(args.camera)
	if not cap.isOpened():
		print('无法打开摄像头')
		return

	os.makedirs(args.outdir, exist_ok=True)
	frame_idx = 0
	print('按 q 或 Esc 退出，按 s 保存当前帧')

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		found, corners = cv2.findChessboardCorners(gray, (cols_corners, rows_corners), None)
		overlay = frame.copy()
		if found:
			term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
			corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
			ok, rvec, tvec = cv2.solvePnP(objp, corners_refined, calib['mtx'], calib['dist'], flags=cv2.SOLVEPNP_ITERATIVE)
			if ok:
				pts_to_proj = pts3d
				# alignment in live mode: if requested, shift point cloud so centroid projects to image center or chessboard center
				if args.align in ('image_center', 'chessboard_center'):
					if args.align == 'image_center':
						target = image_center_pixel(frame)
					else:
						target = chessboard_center_pixel_from_extrinsics(rvec, tvec, cols_corners, rows_corners, args.square_size, calib['mtx'], calib['dist'])
					pts_to_proj = shift_pointcloud_to_target(pts3d, rvec, tvec, calib['mtx'], target)

				# apply uniform scaling about centroid if requested
				if abs(args.scale - 1.0) > 1e-6:
					c = pts_to_proj.mean(axis=0)
					pts_to_proj = (pts_to_proj - c) * float(args.scale) + c

				pts2d = project_with_rvec_tvec(pts_to_proj, rvec, tvec, calib['mtx'], calib['dist'])
				override_color = parse_color_option(args.color)
				for i, p in enumerate(pts2d):
					x, y = int(round(p[0])), int(round(p[1]))
					if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
						if override_color is not None:
							color = override_color
						else:
							color = tuple(int(c) for c in bgr[i % len(bgr)])
						cv2.circle(overlay, (x, y), args.point_size, color, -1)

		blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
		cv2.imshow('Live Projection', blended)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q') or key == 27:
			break
		elif key == ord('s'):
			out_file = os.path.join(args.outdir, f'live_{frame_idx:04d}.jpg')
			cv2.imwrite(out_file, blended)
			print('保存:', out_file)
			frame_idx += 1

	cap.release()
	cv2.destroyAllWindows()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ply', default='cat.ply')
	parser.add_argument('--calib', default='camera_params_generated.npz')
	parser.add_argument('--images', default='chessboard_1_one')  # 更改为图片文件夹路径
	parser.add_argument('--outdir', default='results')
	parser.add_argument('--live', action='store_true')
	parser.add_argument('--camera', type=int, default=0)
	parser.add_argument('--pattern', default='11x8')
	parser.add_argument('--square_size', type=float, default=25.0)
	parser.add_argument('--max_points', type=int, default=50000)
	parser.add_argument('--point_size', type=int, default=1)
	parser.add_argument('--align', choices=['none', 'image_center', 'chessboard_center'], default='chessboard_center', help='align point cloud so centroid projects to target')
	parser.add_argument('--method', choices=['depth_occlusion', 'visibility', 'cluster'], default='depth_occlusion', help='projection method')
	parser.add_argument('--scale', type=float, default=3.0, help='uniform scale factor to apply to point cloud before projection (about its centroid)')
	parser.add_argument('--color', default='pink', help="override color: 'auto'|'pink'|'#RRGGBB' or 'R,G,B' (255-based)")
	args = parser.parse_args()

	if args.live:
		live_project(args)
	else:
		batch_project(args)


if __name__ == '__main__':
	main()

