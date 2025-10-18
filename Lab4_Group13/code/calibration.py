import os
import sys
import glob
import re
import argparse
import numpy as np
import cv2



def find_image_pairs(folder, left_tag='_left', right_tag='_right'):
    # find files like 1_left.png and corresponding 1_right.png
    imgs = glob.glob(os.path.join(folder, '*_left*'))
    pairs = []
    for left in imgs:
        right = os.path.join(folder, os.path.basename(left).replace(left_tag, right_tag))
        if os.path.exists(right):
            pairs.append((left, right))
    pairs.sort()
    print(pairs)
    return pairs


def make_object_points(nx, ny, square_size):
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= square_size
    return objp


def stereo_calibrate(folder, nx, ny, square_size, out_prefix, debug, free_intrinsics=False):
    pairs = find_image_pairs(folder)
    if len(pairs) < 1:
        print('No pairs found in', folder)
        return

    print(f'Found {len(pairs)} stereo pairs')

    objp = make_object_points(nx, ny, square_size)
    objpoints = []  # 3d points in real world space
    imgpoints_l = []  # 2d points in image plane.
    imgpoints_r = []

    valid_pairs = []
    cnt = 0 # use for debug image naming
    if debug:
        os.makedirs('figures/debug', exist_ok=True)
    for left_path, right_path in pairs:
        img_l = cv2.imread(left_path, cv2.IMREAD_COLOR)
        img_r = cv2.imread(right_path, cv2.IMREAD_COLOR)
        if img_l is None or img_r is None:
            continue
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, (nx, ny), None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (nx, ny), None)

        if ret_l and ret_r:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), term)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), term)
            # append a copy so each view has its own array (avoid sharing same buffer)
            objpoints.append(objp.copy())
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            valid_pairs.append((left_path, right_path))
            # Save debug image with drawn corners for the first valid pair
            if debug:

                cnt += 1
                disp_l = cv2.drawChessboardCorners(img_l.copy(), (nx, ny), corners_l, ret_l)
                disp_r = cv2.drawChessboardCorners(img_r.copy(), (nx, ny), corners_r, ret_r)
                debug_left = os.path.join(f'figures/debug/{cnt}_stereo_debug_corners_left.png')
                debug_right = os.path.join(f'figures/debug/{cnt}_stereo_debug_corners_right.png')
                cv2.imwrite(debug_left, disp_l)
                cv2.imwrite(debug_right, disp_r)
                print('Saved debug images:', debug_left, debug_right)

        else:
            print('Chessboard not found in pair:', os.path.basename(left_path))


    n_ok = len(objpoints)
    # if n_ok < 5:
    #     print('Not enough valid detections:', n_ok)
    #     return

    print(f'Using {n_ok} valid pairs for calibration')

    # Image size from first valid left
    sample = cv2.imread(valid_pairs[0][0], cv2.IMREAD_COLOR)
    h, w = sample.shape[:2]

    # Calibrate each camera individually
    print('Calibrating left camera...')
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, (w, h), None, None)
    # compute reprojection error for left
    tot_err_l = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_l[i], tvecs_l[i], mtx_l, dist_l)
        err = cv2.norm(imgpoints_l[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_err_l += err
    mean_err_l = tot_err_l / len(objpoints) if len(objpoints) else float('inf')
    print(f'Left camera mean reprojection error: {mean_err_l:.4f} pixels')
    # per-image reprojection errors for left
    per_err_l = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_l[i], tvecs_l[i], mtx_l, dist_l)
        err = np.sqrt(np.mean(np.sum((imgpoints_l[i].reshape(-1,2) - imgpoints2.reshape(-1,2))**2, axis=1)))
        per_err_l.append(err)
    print('Calibrating right camera...')
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, (w, h), None, None)
    # compute reprojection error for right
    tot_err_r = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
        err = cv2.norm(imgpoints_r[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_err_r += err
    mean_err_r = tot_err_r / len(objpoints) if len(objpoints) else float('inf')
    print(f'Right camera mean reprojection error: {mean_err_r:.4f} pixels')
    # per-image reprojection errors for right
    per_err_r = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
        err = np.sqrt(np.mean(np.sum((imgpoints_r[i].reshape(-1,2) - imgpoints2.reshape(-1,2))**2, axis=1)))
        per_err_r.append(err)

    # save overlays for worst views
    try:
        debug_dir = os.path.join('figures', 'debug_reproj')
        os.makedirs(debug_dir, exist_ok=True)
        # find worst indices by average of left/right per-view error
        per_mean = [(per_err_l[i] + per_err_r[i]) / 2.0 for i in range(len(per_err_l))]
        worst_idx = sorted(range(len(per_mean)), key=lambda i: per_mean[i], reverse=True)[:3]
        for rank, idx in enumerate(worst_idx, start=1):
            left_path, right_path = valid_pairs[idx]
            img_l = cv2.imread(left_path)
            img_r = cv2.imread(right_path)
            proj_l, _ = cv2.projectPoints(objpoints[idx], rvecs_l[idx], tvecs_l[idx], mtx_l, dist_l)
            proj_r, _ = cv2.projectPoints(objpoints[idx], rvecs_r[idx], tvecs_r[idx], mtx_r, dist_r)
            disp_l = img_l.copy()
            disp_r = img_r.copy()
            for p in proj_l.reshape(-1,2):
                cv2.circle(disp_l, (int(round(p[0])), int(round(p[1]))), 3, (0,0,255), -1)
            for p in proj_r.reshape(-1,2):
                cv2.circle(disp_r, (int(round(p[0])), int(round(p[1]))), 3, (0,0,255), -1)
            out_l = os.path.join(debug_dir, f'{idx+1}_stereo_reproj_left.png')
            out_r = os.path.join(debug_dir, f'{idx+1}_stereo_reproj_right.png')
            cv2.imwrite(out_l, disp_l)
            cv2.imwrite(out_r, disp_r)
        print('Saved reprojection debug overlays to', debug_dir)
    except Exception as e:
        print('Failed to save reprojection overlays:', e)

    # Stereo calibration
    if free_intrinsics:
        flags = 0
        print('Running stereoCalibrate with free intrinsics (not fixing single-camera intrinsics)')
    else:
        flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    print('Running stereoCalibrate...')
    ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r, (w, h), criteria=criteria, flags=flags)

    print('stereoCalibrate RMS:', ret)
    # debug printouts
    print('\n--- Debug info ---')
    print('left K:\n', mtx_l)
    print('left dist:', dist_l.ravel())
    print('right K:\n', mtx_r)
    print('right dist:', dist_r.ravel())
    print('stereo R:\n', R)
    print('stereo T (units of square_size):', T.ravel())
    try:
        baseline = np.linalg.norm(T.ravel()) * (1.0)
    except Exception:
        baseline = None
    print('baseline magnitude (in same units as square_size):', baseline)
    print('------------------\n')


    # Save parameters
    np.savez(out_prefix + '_params.npz',
             cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
             cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
             R=R, T=T, E=E, F=F)
    print('Saved stereo parameters to', out_prefix + '_params.npz')
    # Calibration only: parameters saved. Reconstruction is handled by separate script.
    print('Calibration finished. Use the saved params for reconstruction with reconstruct.py')



def main():
    p = argparse.ArgumentParser(description='Stereo calibration and point cloud generation')
    p.add_argument('--folder', default='figures/chessboard', help='Folder containing N_left.png / N_right.png pairs')
    p.add_argument('--nx', type=int, default=11, help='Chessboard inner corners per row')
    p.add_argument('--ny', type=int, default=8, help='Chessboard inner corners per column')
    p.add_argument('--square', type=float, default=25.0, help='Chessboard square size in chosen units（mm)')
    p.add_argument('--out', default='stereo', help='Output prefix for params and cloud')
    p.add_argument('--debug', default=False, action='store_true', help='Save debug images with detected corners')
    p.add_argument('--free-intrinsics', default=False, action='store_true', help='Allow stereoCalibrate to optimize intrinsics (do not fix)')
    args = p.parse_args()
    stereo_calibrate(args.folder, args.nx, args.ny, args.square, args.out, args.debug, free_intrinsics=args.free_intrinsics)


if __name__ == '__main__':
    main()
