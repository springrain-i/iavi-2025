import os
import argparse
import numpy as np
import cv2


def load_params(npz_path):
    data = np.load(npz_path)
    params = {k: data[k] for k in data.files}
    return params



def stereo_match_sgbm(left_gray, right_gray, num_disparities=128, block_size=5, min_disparity=0):
    """
    Compute disparity using StereoSGBM and return (disp_f32, disp8_uint8)
    - disp_f32: float32 disparity in pixels (invalid regions <= 0 or large values may appear)
    - disp8_uint8: uint8 visualization scaled to 0..255
    """
    h, w = left_gray.shape[:2]
    # ensure num_disparities is multiple of 16
    num_disp = num_disparities
    if num_disp % 16 != 0:
        num_disp = ((num_disp // 16) + 1) * 16
    # don't exceed image width
    max_allowed = max(16, (w // 16) * 16)
    num_disp = min(num_disp, max_allowed)

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,  # 降低从 10 到 5，获得更多点
        speckleWindowSize=150,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    raw = matcher.compute(left_gray, right_gray)
    # raw is int16 where value = disparity*16
    disp = raw.astype(np.float32) / 16.0

    # create an 8-bit visual: clip to [min_disp, min_disp+num_disp], normalize to 0..255
    valid_mask = (disp > min_disparity)
    disp_vis = np.zeros_like(disp, dtype=np.float32)
    if valid_mask.any():
        # clip to a reasonable range
        disp_clipped = np.clip(disp, min_disparity, float(min_disparity + num_disp))
        disp_vis = (disp_clipped - min_disparity) / float(max(1.0, num_disp))
    disp8 = (disp_vis * 255.0).astype(np.uint8)

    return disp, disp8


def creatDepthView(dispL, focal_length, baseline):
    """Create a colored depth visualization from disparity (dispL: float disparity in pixels).
    Returns a BGR uint8 colormap image.
    """
    # ensure float
    dispL = dispL.astype(np.float32)
    # mask positive disparities
    valid = dispL > 0
    if np.any(valid):
        dmin = np.percentile(dispL[valid], 2)
        dmax = np.percentile(dispL[valid], 98)
    else:
        dmin = 1.0
        dmax = dmin + 1.0
    disp_clamped = np.clip(dispL, dmin, dmax)
    depth_map = (baseline * focal_length) / (disp_clamped)
    # normalize to 0..255 and colormap
    depth_normalized = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_COOL)
    return depth_color


def write_ply(filename, verts, colors=None):
    """Write a PLY file with optional colors.
    verts: Nx3 array-like
    colors: Nx3 array-like (uint8) or None for grayscale (xyz only)
    """
    #verts = np.asarray(verts).reshape(-1, 3)
    # filter finite and positive Z
    valid = np.isfinite(verts).all(axis=1) & (verts[:, 2] > 0)
    verts = verts[valid]
    if colors is not None:
        colors = colors[valid]

    # ensure output directory exists
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        if colors is not None:
            for v, c in zip(verts, colors):
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for v in verts:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

def depth_to_grayscale_color(depth_values, vmin=None, vmax=None):
    """Convert depth values to grayscale colors (0=black, 255=white).
    depth_values: 1D array of depth/Z values
    Returns: Nx3 uint8 array (R, G, B) where R=G=B for grayscale
    """
    depth_values = np.asarray(depth_values, dtype=np.float32)
    finite_mask = np.isfinite(depth_values)
    
    if vmin is None:
        valid_depths = depth_values[finite_mask]
        if len(valid_depths) > 0:
            vmin = np.percentile(valid_depths, 2)
        else:
            vmin = 0.0
    
    if vmax is None:
        valid_depths = depth_values[finite_mask]
        if len(valid_depths) > 0:
            vmax = np.percentile(valid_depths, 98)
        else:
            vmax = vmin + 1.0
    
    # avoid zero range
    if vmax <= vmin:
        vmax = vmin + 1.0
    
    # normalize to 0-255
    normalized = (depth_values - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    gray_values = (normalized * 255).astype(np.uint8)
    
    # create RGB by duplicating grayscale value
    colors = np.stack([gray_values, gray_values, gray_values], axis=1)
    # set invalid to black (0, 0, 0)
    colors[~finite_mask] = 0
    
    return colors

def save_point_cloud_from_disp(disp, img, Q, filename, max_points=200000000, depth_range=None, use_colors=True):
    """Reproject disparity map to 3D and save an ASCII PLY with x y z vertices and optional depth-based colors.
    - disp: disparity map (float in pixels) or int16 raw from SGBM (will be divided by 16)
    - img: left image (unused here, kept for API compatibility)
    - Q: reprojection matrix
    - filename: output .ply path
    - max_points: limit number of vertices written (random subsample)
    - depth_range: optional (z_min, z_max) to filter points by Z
    - use_colors: if True, color points by depth (grayscale); if False, write xyz only
    Returns: number of points written
    """
    # handle integer scaled disparity (StereoSGBM raw is int16 scaled by 16)
    if np.issubdtype(disp.dtype, np.integer):
        disp_f = disp.astype(np.float32) / 16.0
        print("Converted integer disparity to float by dividing by 16")
    else:
        disp_f = disp.astype(np.float32)
    print(Q[2,3])
    # reproject to 3D
    pts3d = cv2.reprojectImageTo3D(disp_f, Q)
    print(pts3d.shape)
    # mask valid points: valid disparity and finite Z
    valid_mask = (disp_f > 0) & np.isfinite(pts3d[:, :, 2])
    pts = pts3d[valid_mask]
    
    # extract depth values for coloring
    if use_colors:
        depths = pts3d[:, :, 2][valid_mask]
    else:
        depths = None

    if pts.size == 0:
        # ensure output directory exists but write empty PLY
        write_ply(filename, np.zeros((0, 3)), colors=None)
        return 0

    # subsample if too many points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        if depths is not None:
            depths = depths[idx]
    
    # generate colors based on depth if requested
    if use_colors and depths is not None:
        colors = depth_to_grayscale_color(depths)
        write_ply(filename, pts, colors=colors)
    else:
        # write PLY without colors
        write_ply(filename, pts, colors=None)
    
    return len(pts)

def getRectifyTransform(h,w,params):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=params['cameraMatrix1'],
        distCoeffs1=params.get('distCoeffs1', None),
        cameraMatrix2=params['cameraMatrix2'],
        distCoeffs2=params.get('distCoeffs2', None),
        imageSize=(w, h),
        R=params['R'],
        T=params['T'],
        alpha=0
    )
    map1x, map1y = cv2.initUndistortRectifyMap(params['cameraMatrix1'], params.get('distCoeffs1', None), R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(params['cameraMatrix2'], params.get('distCoeffs2', None), R2, P2, (w, h), cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, Q, P1, P2

def rectifyImage(left,right,map1x, map1y, map2x, map2y):
    left_rect = cv2.remap(left, map1x, map1y, interpolation=cv2.INTER_AREA)
    right_rect = cv2.remap(right, map2x, map2y, interpolation=cv2.INTER_AREA)
    return left_rect, right_rect

def process_pair(cnt,left_path, right_path, params, num_disparities=128, block_size=5, downsample=1, max_points=5000000):
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        raise RuntimeError('Failed to read images')

    h, w = left.shape[:2]
    # rectify if we have R1/R2/P1/P2
    map1x, map1y, map2x, map2y, Q, P1, P2 = getRectifyTransform(h,w,params)
    left_rect, right_rect = rectifyImage(left,right,map1x, map1y, map2x, map2y)
    gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # compute disparity (SGBM) and a visual 8-bit disparity
    disp, disp8 = stereo_match_sgbm(gray_l, gray_r, num_disparities=num_disparities, block_size=block_size, min_disparity=0)


    # save disparity visualization next to PLY for debugging
    out_dir = os.path.normpath(os.path.join('..', 'data'))
    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'pointcloud'), exist_ok=True)
    try:
        disp_vis_path = os.path.join(out_dir, 'image', f"{cnt}_disp.png")
        cv2.imwrite(disp_vis_path, disp8)
    except Exception:
        pass

    # create colored depth visualization and save
    try:
        # use focal and baseline from params if available
        focal_px = float(params.get('cameraMatrix1', params.get('cam_matrix_l'))[0, 0])
        baseline_val = float(abs(params.get('T', params.get('baseline', np.array([[536.62]])))[0, 0]))
        depth_color = creatDepthView(disp, focal_length=focal_px, baseline=baseline_val)
        depth_vis_path = os.path.join(out_dir, 'image', f"{cnt}_depth.png")
        cv2.imwrite(depth_vis_path, depth_color)
    except Exception:
        pass

    ply_path = os.path.join(out_dir, 'pointcloud', f"{cnt}_cloud.ply")
    f = focal_px  # focal length in pixels (not 1/f)
    c_x = params['cameraMatrix1'][0,2]
    c_y = params['cameraMatrix1'][1,2]
    c_x_prime = float(P2[0,2])
    T_x = params['T'][0][0]
    # Standard Q matrix format for reprojection
    # X = (x - c_x) * Z / f
    # Y = (y - c_y) * Z / f
    # Z = f * baseline / disparity
    Q = np.float32([[1, 0, 0, -c_x],
                    [0, 1, 0, -c_y],
                    [0, 0, 0, f],
                    [0, 0, -1/T_x, (c_x - c_x_prime)/T_x]])
    
    n_written = save_point_cloud_from_disp(disp, left_rect, Q, ply_path, max_points=max_points, use_colors=True)
    print(f'Wrote {n_written} points to', ply_path)
    

def main():
    p = argparse.ArgumentParser(description='Generate colored point cloud from stereo pair(s) using stereo params')
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument('--folder', default="figures/use4ply", help='Folder with stereo pairs (default figures/object)')
    group.add_argument('--pair', nargs=2, metavar=('LEFT', 'RIGHT'), help='Single left/right pair')
    p.add_argument('--params', default='stereo_params.npz')
    p.add_argument('--num-disparities', type=int, default=128)
    p.add_argument('--block-size', type=int, default=5)
    p.add_argument('--downsample', type=int, default=1, help='Pixel stride downsampling before reprojecting')
    p.add_argument('--max-points', type=int, default=5000000, help='Maximum points to write to PLY (default 5M)')
    args = p.parse_args()

    params = load_params(args.params)

    pairs = []
    if args.pair:
        pairs = [(args.pair[0], args.pair[1])]
    else:
        folder = args.folder or os.path.join('figures', 'object')
        lefts = sorted([f for f in os.listdir(folder) if '_left' in f]) if os.path.isdir(folder) else []
        for l in lefts:
            r = l.replace('_left', '_right')
            if os.path.exists(os.path.join(folder, r)):
                pairs.append((os.path.join(folder, l), os.path.join(folder, r)))

    if len(pairs) == 0:
        print('No pairs found')
        return

    for i, (l, r) in enumerate(pairs):
        process_pair(i+1, l, r, params,num_disparities=args.num_disparities, block_size=args.block_size, downsample=args.downsample, max_points=args.max_points)


if __name__ == '__main__':
    main()
