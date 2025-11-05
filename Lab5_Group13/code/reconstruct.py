"""Stereo 3D reconstruction from Gray-code decoded maps (left/right cameras).

Algorithm:
  - Build a map from projector coords (x_p, y_p) -> list of right pixel locations.
  - For each valid left pixel, find right candidates with exact or tolerance match.
  - Aggregate candidate right pixels (median) and triangulate with stereo params.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path    
import numpy as np
import cv2
import os

def read_float_map(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == '.npy':
        arr = np.load(str(p))
        return arr.astype(np.float32)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Failed to read float map: {path}')
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def load_stereo_npz(npz_path: str, cam_id: int):
    npz = np.load(npz_path)
    K1 = npz['cameraMatrix1'].astype(np.float64)
    D1 = npz.get('distCoeffs1', np.zeros((1,5))).astype(np.float64)
    K2 = npz['cameraMatrix2'].astype(np.float64)
    D2 = npz.get('distCoeffs2', np.zeros((1,5))).astype(np.float64)
    R = npz['R'].astype(np.float64)
    T = npz['T'].astype(np.float64).reshape(3,1)
    if cam_id == 1:
        K_left, D_left, K_right, D_right = K1, D1, K2, D2
    else:
        K_left, D_left, K_right, D_right = K2, D2, K1, D1
    return K_left, D_left, K_right, D_right, R, T


def build_right_map(rx_map: np.ndarray, ry_map: np.ndarray, rmask: np.ndarray) -> dict[tuple[int,int], list[tuple[int,int]]]:
    h, w = rx_map.shape[:2]
    mp: dict[tuple[int,int], list[tuple[int,int]]] = defaultdict(list)
    for y in range(h):
        valid_row = rmask[y] > 0
        if not np.any(valid_row):
            continue
        xs = np.where(valid_row)[0]
        for x in xs:
            key = (int(rx_map[y, x]), int(ry_map[y, x]))
            mp[key].append((x, y))
    return mp


def find_candidates(key: tuple[int,int], right_map: dict, tol: int):
    if key in right_map:
        return right_map[key]
    if tol <= 0:
        return []
    px, py = key
    for r in range(1, tol+1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                k2 = (px+dx, py+dy)
                if k2 in right_map:
                    return right_map[k2]
    return []


def main():
    ap = argparse.ArgumentParser(description='Stereo 3D reconstruction using left/right Gray decoded maps')
    ap.add_argument('--left-x', default='img_decode/left_x.tiff')
    ap.add_argument('--left-y', default='img_decode/left_y.tiff')
    ap.add_argument('--left-mask', default='img_decode/left_mask.png')
    ap.add_argument('--right-x', default='img_decode/right_x.tiff')
    ap.add_argument('--right-y', default='img_decode/right_y.tiff')
    ap.add_argument('--right-mask', default='img_decode/right_mask.png')
    ap.add_argument('--cam-id', type=int, choices=[1,2], default=1, help='When using --npz, which matrix is the camera: 1 or 2 (default 1)')
    ap.add_argument('--npz', default="stereo_params.npz")
    ap.add_argument('--tol', type=int, default=0)
    ap.add_argument('--method', choices=['exact','tol'], default='tol')
    ap.add_argument('--texture', default='capture/1_left.png')
    ap.add_argument('--out', default='res/reconstructed_stereo.ply')
    # Depth map options
    ap.add_argument('--depth-out', default='res/depth.tiff', help='Optional path to save depth map. Supports .npy (float32, NaN holes), .tiff/.exr (float32), or .png (uint16 millimeters).')
    ap.add_argument('--depth-vis', default='res/depth.png', help='Optional path to save colorized depth visualization (PNG).')
    ap.add_argument('--min-z', type=float, default=0.0, help='Minimum valid Z (meters) when filtering depth.')
    ap.add_argument('--max-z', type=float, default=10.0, help='Maximum valid Z (meters) when filtering depth.')
    ap.add_argument('--vis-min', type=float, default=None, help='Depth visualization minimum (meters). If not set, uses robust percentile.')
    ap.add_argument('--vis-max', type=float, default=None, help='Depth visualization maximum (meters). If not set, uses robust percentile.')
    args = ap.parse_args()

    os.makedirs("res", exist_ok=True)
    lx = read_float_map(args.left_x)
    ly = read_float_map(args.left_y)
    rx = read_float_map(args.right_x)
    ry = read_float_map(args.right_y)
    lmask = cv2.imread(args.left_mask, cv2.IMREAD_GRAYSCALE)
    rmask = cv2.imread(args.right_mask, cv2.IMREAD_GRAYSCALE)
    if lmask is None or rmask is None:
        raise FileNotFoundError('Failed to read masks')
    if lx.shape != ly.shape or rx.shape != ry.shape or lx.shape != lmask.shape or rx.shape != rmask.shape:
        raise ValueError('Map/mask shapes must match per side')

    h, w = lx.shape[:2]
    K1, D1, K2, D2, R, T = load_stereo_npz(args.npz, args.cam_id)

    right_map = build_right_map(rx, ry, rmask)

    pts_left = []
    pts_right = []
    colors = []

    if args.texture:
        tex = cv2.imread(args.texture, cv2.IMREAD_COLOR)
    else:
        tex = None

    tol = int(args.tol)
    use_exact = (args.method == 'exact')

    for y in range(h):
        valid_row = lmask[y] > 0
        xs = np.where(valid_row)[0]
        for x in xs:
            key = (int(lx[y, x]), int(ly[y, x]))
            if use_exact:
                cand = right_map.get(key, [])
            else:
                cand = find_candidates(key, right_map, tol)
            if not cand:
                continue
            rx_med = int(np.median([c[0] for c in cand]))
            ry_med = int(np.median([c[1] for c in cand]))
            pts_left.append((x, y))
            pts_right.append((rx_med, ry_med))
            if tex is not None:
                colors.append(tex[y, x, ::-1])  # BGR->RGB

    if len(pts_left) < 6:
        raise SystemExit('Too few correspondences for triangulation')

    pts_left = np.array(pts_left, dtype=np.float32)
    pts_right = np.array(pts_right, dtype=np.float32)

    # Undistort to normalized coords
    pts1_ud = cv2.undistortPoints(pts_left.reshape(-1,1,2), K1, D1).reshape(-1,2)
    pts2_ud = cv2.undistortPoints(pts_right.reshape(-1,1,2), K2, D2).reshape(-1,2)

    P1 = np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = np.hstack([R, T])
    pts4d = cv2.triangulatePoints(P1, P2, pts1_ud.T, pts2_ud.T)
    X = (pts4d[:3, :] / pts4d[3, :]).T

    # Optional: build and save depth map (Z in left camera frame)
    if args.depth_out or args.depth_vis:
        depth = np.full((h, w), np.nan, dtype=np.float32)
        zz_raw = X[:, 2]

        # 1) If majority Z are negative, flip sign for depth map purposes
        med_z = np.nanmedian(zz_raw)
        if np.isfinite(med_z) and med_z < 0:
            zz = -zz_raw
        else:
            zz = zz_raw

        # 2) Infer unit: if typical depth magnitude is large (> 50), assume millimeters
        abs_valid = np.abs(zz[np.isfinite(zz)])
        scale_to_m = 1.0
        if abs_valid.size > 0:
            p50 = float(np.percentile(abs_valid, 50.0))
            if p50 > 50.0:  # likely millimeters
                scale_to_m = 1.0 / 1000.0
        z_m = zz * scale_to_m  # use meters internally for filtering/vis

        # 3) Filter by finite and Z range (in meters)
        finite_mask = np.isfinite(z_m)
        range_mask = (z_m > float(args.min_z)) & (z_m < float(args.max_z))
        valid = finite_mask & range_mask
        if np.any(valid):
            pl = pts_left.astype(np.int32)
            xv = pl[valid, 0]
            yv = pl[valid, 1]
            depth[yv, xv] = z_m[valid].astype(np.float32)  # store meters in float maps

        # Save raw depth map
        if args.depth_out:
            out_p = Path(args.depth_out)
            ext = out_p.suffix.lower()
            if ext == '.npy':
                np.save(str(out_p), depth)
            elif ext in ('.tif', '.tiff', '.exr'):
                # Write float32, keep NaNs (EXR/TIFF support NaN)
                cv2.imwrite(str(out_p), depth)
            elif ext == '.png':
                # Convert to uint16 millimeters, invalid -> 0
                depth_mm = np.zeros_like(depth, dtype=np.uint16)
                with np.errstate(invalid='ignore'):
                    mm = (depth * 1000.0)  # depth is meters -> mm
                mask = np.isfinite(mm) & (mm > 0)
                depth_mm[mask] = np.clip(mm[mask], 0, np.iinfo(np.uint16).max).astype(np.uint16)
                cv2.imwrite(str(out_p), depth_mm)
            else:
                print(f'Warning: unsupported depth extension: {ext}. Use .npy, .tiff/.tif, .exr or .png')

        # Save visualization
        if args.depth_vis:
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            valid_vals = depth[np.isfinite(depth) & (depth > 0)]  # meters
            if valid_vals.size > 0:
                if args.vis_min is not None and args.vis_max is not None and args.vis_max > args.vis_min:
                    vmin, vmax = float(args.vis_min), float(args.vis_max)  # meters
                else:
                    # Robust range
                    vmin = float(np.percentile(valid_vals, 2.0))
                    vmax = float(np.percentile(valid_vals, 98.0))
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                        vmin, vmax = float(np.min(valid_vals)), float(np.max(valid_vals))
                denom = (vmax - vmin) if (vmax > vmin) else 1.0
                norm = np.zeros_like(depth, dtype=np.float32)
                with np.errstate(invalid='ignore'):
                    norm = (depth - vmin) / denom
                norm = np.clip(norm, 0.0, 1.0)
                norm8 = (norm * 255.0).astype(np.uint8)
                cm = cv2.applyColorMap(norm8, cv2.COLORMAP_TURBO)
                # Set invalid to black
                inv = ~np.isfinite(depth) | (depth <= 0)
                cm[inv] = (0, 0, 0)
                vis = cm
            else:
                # No valid pixels after filtering – write a black image but also warn
                print('Warning: No valid depth values for visualization. Check min/max range and units.')
            cv2.imwrite(str(Path(args.depth_vis)), vis)

    # Write PLY
    with open(args.out, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {X.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        if colors:
            for i, p in enumerate(X):
                r, g, b = colors[i]
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(r)} {int(g)} {int(b)}\n')
        else:
            for p in X:
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')

    print(f'Reconstructed {X.shape[0]} points -> {args.out}')


if __name__ == '__main__':
    main()
