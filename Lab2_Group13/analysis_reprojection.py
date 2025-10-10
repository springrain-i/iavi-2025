"""Analyze factors affecting reprojection error.

Produces a CSV with per-image metrics and scatter plots showing correlation
between reprojection error and factors like number of detected corners,
coverage, mean depth, and tilt angle.

Usage example:
    python analysis_reprojection.py --images tmp --calib camera_calibration.npz --pattern 11x8 --square_size 25.0 --outdir analysis

"""
import os
import glob
import argparse
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt


def load_calib(npz_path):
    d = np.load(npz_path)
    return d


def parse_pattern(pattern):
    cols, rows = map(int, pattern.lower().split('x'))
    return cols, rows


def create_objp(cols, rows, square_size):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def detect_corners(img_path, pattern):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not found:
        return None, None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return img, corners_refined.reshape(-1, 2)


def map_images_to_ids(image_files, rvecs):
    mapping = {}
    if rvecs is None:
        for i, p in enumerate(sorted(image_files)):
            mapping[f'image_{i+1}'] = p
        return mapping
    if len(rvecs) == len(image_files):
        for i, p in enumerate(sorted(image_files)):
            mapping[f'image_{i+1}'] = p
        return mapping
    remaining = [f'image_{i+1}' for i in range(len(rvecs))]
    for p in sorted(image_files):
        name = os.path.basename(p)
        assigned = False
        for img_id in remaining:
            if img_id.replace('image_', '') in name:
                mapping[img_id] = p
                remaining.remove(img_id)
                assigned = True
                break
        if not assigned and remaining:
            mapping[remaining.pop(0)] = p
    return mapping


def polygon_area(points):
    # shoelace formula for polygon area
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def analyze(images_dir, calib_path, pattern, square_size, outdir):
    os.makedirs(outdir, exist_ok=True)
    cols, rows = parse_pattern(pattern)
    objp = create_objp(cols, rows, square_size)
    calib = load_calib(calib_path)
    mtx = calib['mtx']
    dist = calib['dist']
    rvecs = calib.get('rvecs')
    tvecs = calib.get('tvecs')

    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    mapping = map_images_to_ids(image_files, rvecs)

    rows_out = []

    for img_id, img_path in mapping.items():
        print('Processing', img_path)
        img, corners = detect_corners(img_path, (cols, rows))
        if img is None:
            print('  no corners found, skipping')
            continue

        # determine rvec/tvec: prefer saved
        rvec = None
        tvec = None
        if rvecs is not None and img_id in mapping:
            try:
                idx = int(img_id.replace('image_', '')) - 1
                rvec = rvecs[idx]
                tvec = tvecs[idx]
            except Exception:
                rvec = None

        if rvec is None:
            ok, rvec, tvec = cv2.solvePnP(objp, corners.reshape(-1, 1, 2), mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                print('  solvePnP failed for', img_path)
                continue

        # project object points using rvec/tvec
        proj_pts, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        proj_pts = proj_pts.reshape(-1, 2)

        # compute per-corner reprojection residuals vs detected corners
        # match order: assume detected corners correspond to objp ordering
        if corners.shape[0] != proj_pts.shape[0]:
            # if mismatch, skip
            print('  corner count mismatch, skipping')
            continue

        residuals = np.linalg.norm(proj_pts - corners, axis=1)
        mean_err = float(np.mean(residuals))
        std_err = float(np.std(residuals))

        # coverage: convex hull area over image area
        hull = cv2.convexHull(corners.astype(np.float32))
        hull = hull.reshape(-1, 2)
        area_hull = polygon_area(hull)
        img_area = img.shape[0] * img.shape[1]
        coverage = float(area_hull / img_area)

        # mean depth: compute object points in camera coords
        R, _ = cv2.Rodrigues(rvec)
        pts_cam = (R @ objp.T + tvec).T
        mean_depth = float(np.mean(pts_cam[:, 2]))

        # tilt angle: angle between chessboard normal and camera optical axis
        # chessboard normal in object coords = (0,0,1)
        normal_cam = R @ np.array([0.0, 0.0, 1.0])
        # camera optical axis is (0,0,1) in camera coords
        cosang = float(normal_cam[2] / (np.linalg.norm(normal_cam) + 1e-12))
        tilt_deg = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

        # center offset: distance from projected chessboard centroid to image center
        board_centroid = proj_pts.mean(axis=0)
        img_center = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0])
        center_offset = float(np.linalg.norm(board_centroid - img_center))

        rows_out.append({
            'image': os.path.basename(img_path),
            'mean_error': mean_err,
            'std_error': std_err,
            'n_corners': int(corners.shape[0]),
            'coverage': coverage,
            'mean_depth': mean_depth,
            'tilt_deg': tilt_deg,        #对应 view angle
            'center_offset': center_offset,
        })

    # save CSV with only requested fields
    import csv
    csv_path = os.path.join(outdir, 'reprojection_factors.csv')
    keys = ['image', 'mean_error', 'n_corners', 'coverage', 'tilt_deg']
    with open(csv_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r[k] for k in keys})

    print('Saved CSV to', csv_path)

    # prepare arrays
    arr = {k: np.array([r[k] for r in rows_out]) for k in keys if k != 'image'}

    # plot mean_error vs coverage and vs tilt_deg
    for factor in ['coverage', 'tilt_deg']:
        x = arr[factor]
        y = arr['mean_error']
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y)
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            plt.plot(xs, m * xs + b, color='red', linestyle='--')
            corr = np.corrcoef(x, y)[0, 1]
            plt.title(f'mean_error vs {factor} (corr={corr:.3f})')
        else:
            plt.title(f'mean_error vs {factor}')
        plt.xlabel(factor)
        plt.ylabel('mean reproj error (px)')
        plt.grid(True)
        plt.tight_layout()
        out_png = os.path.join(outdir, f'mean_error_vs_{factor}.png')
        plt.savefig(out_png)
        plt.close()
        print('Saved plot', out_png)

    # ANALYZE effect of number of images via bootstrap subsets
    # collect available image points for calibration
    img_points_map = {}
    for r in rows_out:
        img_name = r['image']
        # read detected corners again to build imagePoints
        img_path = os.path.join(images_dir, img_name)
        img, corners = detect_corners(img_path, (cols, rows))
        if img is None:
            continue
        img_points_map[img_name] = corners.reshape(-1, 1, 2).astype(np.float32)

    n_images = len(img_points_map)
    print(f'Number of usable images for calibration subsets: {n_images}')

    if n_images >= 1:
        # build corresponding object points dict
        obj_map = {name: objp.copy().reshape(-1, 1, 3).astype(np.float32) for name in img_points_map.keys()}

        def reproj_error_for_subset(names):
            objpoints = [obj_map[n] for n in names]
            imgpoints = [img_points_map[n] for n in names]
            # image size from first image
            first_img = cv2.imread(os.path.join(images_dir, names[0]))
            h, w = first_img.shape[:2]
            ret, mtx_new, dist_new, rvecs_new, tvecs_new = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
            # compute reprojection error manually
            tot_err = 0.0
            tot_pts = 0
            for i in range(len(objpoints)):
                imgp_proj, _ = cv2.projectPoints(objpoints[i], rvecs_new[i], tvecs_new[i], mtx_new, dist_new)
                imgp_proj = imgp_proj.reshape(-1, 2)
                imgp = imgpoints[i].reshape(-1, 2)
                tot_err += np.sum(np.linalg.norm(imgp_proj - imgp, axis=1))
                tot_pts += imgp.shape[0]
            return tot_err / tot_pts if tot_pts > 0 else np.nan

        max_k = n_images
        trials = min(30, 100)  # cap trials
        mean_errors_by_k = []
        std_errors_by_k = []
        ks = list(range(1, max_k + 1))
        import random
        for k in ks:
            errs = []
            for t in range(min(trials, 200)):
                names = random.sample(list(img_points_map.keys()), k)
                try:
                    e = reproj_error_for_subset(names)
                    if not np.isnan(e):
                        errs.append(e)
                except Exception:
                    continue
                # small speed: stop early if we've collected 'trials' errors
                if len(errs) >= trials:
                    break
            if len(errs) > 0:
                mean_errors_by_k.append(np.mean(errs))
                std_errors_by_k.append(np.std(errs))
            else:
                mean_errors_by_k.append(np.nan)
                std_errors_by_k.append(np.nan)

        # plot reprojection error vs number of images
        plt.figure(figsize=(6, 4))
        plt.errorbar(ks, mean_errors_by_k, yerr=std_errors_by_k, marker='o')
        plt.xlabel('# images (subset)')
        plt.ylabel('mean reprojection error (px)')
        plt.title('Reprojection error vs # images (bootstrap)')
        plt.grid(True)
        plt.tight_layout()
        out_png = os.path.join(outdir, 'reproj_vs_num_images.png')
        plt.savefig(out_png)
        plt.close()
        print('Saved plot', out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='chessboard_1_one')
    parser.add_argument('--calib', default='camera_params_generated.npz')
    parser.add_argument('--pattern', default='11x8')
    parser.add_argument('--square_size', type=float, default=25.0)
    parser.add_argument('--outdir', default='analysis')
    args = parser.parse_args()
    analyze(args.images, args.calib, args.pattern, args.square_size, args.outdir)


if __name__ == '__main__':
    main()
