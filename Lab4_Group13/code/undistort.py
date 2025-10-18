import os
import argparse
import numpy as np
import cv2


def load_params(npz_path):
    """Load stereo calibration parameters from npz file."""
    data = np.load(npz_path)
    params = {k: data[k] for k in data.files}
    return params


def undistort_and_rectify(left_img, right_img, params):
    """
    Undistort and rectify stereo image pair.
    
    Args:
        left_img: left image (BGR)
        right_img: right image (BGR)
        params: dictionary with calibration parameters
    
    Returns:
        left_rect: rectified and undistorted left image
        right_rect: rectified and undistorted right image
        Q: reprojection matrix for 3D reconstruction
    """
    h, w = left_img.shape[:2]
    
    # Extract parameters
    cameraMatrix1 = params['cameraMatrix1']
    distCoeffs1 = params.get('distCoeffs1', None)
    cameraMatrix2 = params['cameraMatrix2']
    distCoeffs2 = params.get('distCoeffs2', None)
    R = params['R']
    T = params['T']
    
    # Compute stereo rectification transforms
    # This computes R1, R2, P1, P2 and Q such that:
    # - Images are rectified (epipolar lines are horizontal)
    # - Images are undistorted
    # - Q is the reprojection matrix for cv2.reprojectImageTo3D
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=cameraMatrix1,
        distCoeffs1=distCoeffs1,
        cameraMatrix2=cameraMatrix2,
        distCoeffs2=distCoeffs2,
        imageSize=(w, h),
        R=R,
        T=T,
        alpha=0  # alpha=0: crop to valid region; alpha=1: retain all pixels
    )
    
    # Compute rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)
    
    # Apply rectification (undistortion + rectification)
    left_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)
    
    return left_rect, right_rect, Q


def draw_epipolar_lines(img, num_lines=10, color=(0, 255, 0), thickness=2):
    """
    Draw horizontal lines on image to visualize epipolar alignment.
    After rectification, epipolar lines should be horizontal and aligned between images.
    
    Args:
        img: image to draw on (uint8)
        num_lines: number of lines to draw
        color: line color in BGR
        thickness: line thickness
    
    Returns:
        img_with_lines: image with epipolar lines drawn
    """
    h, w = img.shape[:2]
    img_with_lines = img.copy()
    
    # Draw horizontal lines at regular intervals
    for i in range(1, num_lines):
        y = int(i * h / num_lines)
        cv2.line(img_with_lines, (0, y), (w, y), color, thickness)
    
    return img_with_lines


def create_comparison_image(left_rect, right_rect, with_lines=True):
    """
    Create a side-by-side comparison of rectified images with optional epipolar lines.
    
    Args:
        left_rect: rectified left image
        right_rect: rectified right image
        with_lines: if True, draw epipolar lines for alignment verification
    
    Returns:
        comparison: concatenated comparison image
    """
    if with_lines:
        left_display = draw_epipolar_lines(left_rect, num_lines=15)
        right_display = draw_epipolar_lines(right_rect, num_lines=15)
    else:
        left_display = left_rect
        right_display = right_rect
    
    # Make sure both images have the same height
    h_left, w_left = left_display.shape[:2]
    h_right, w_right = right_display.shape[:2]
    
    # Pad to same height if needed
    max_h = max(h_left, h_right)
    if h_left < max_h:
        pad = max_h - h_left
        left_display = cv2.copyMakeBorder(left_display, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    if h_right < max_h:
        pad = max_h - h_right
        right_display = cv2.copyMakeBorder(right_display, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    
    # Concatenate horizontally
    comparison = np.concatenate([left_display, right_display], axis=1)
    return comparison


def process_image_pair(left_path, right_path, params, output_dir):
    """
    Process a single stereo image pair: undistort and rectify.
    Save the processed images to output directory.
    
    Args:
        left_path: path to left image
        right_path: path to right image
        params: calibration parameters
        output_dir: output directory for results
    
    Returns:
        True if successful, False otherwise
    """
    # Read images
    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)
    
    if left is None or right is None:
        print(f'Failed to read images: {left_path}, {right_path}')
        return False
    
    # Undistort and rectify
    left_rect, right_rect, Q = undistort_and_rectify(left, right, params)
    
    # Create output filenames based on input
    base_name = os.path.splitext(os.path.basename(left_path))[0].replace('_left', '')
    
    # Save undistorted and rectified images
    left_out = os.path.join(output_dir, f'{base_name}_left_rectified.png')
    right_out = os.path.join(output_dir, f'{base_name}_right_rectified.png')
    
    cv2.imwrite(left_out, left_rect)
    cv2.imwrite(right_out, right_rect)
    print(f'Saved rectified images: {left_out}, {right_out}')
    
    # Create and save comparison image with epipolar lines
    comparison = create_comparison_image(left_rect, right_rect, with_lines=True)
    comparison_out = os.path.join(output_dir, f'{base_name}_comparison_epipolar.png')
    cv2.imwrite(comparison_out, comparison)
    print(f'Saved comparison image with epipolar lines: {comparison_out}')
    
    # Create and save comparison image without lines
    comparison_no_lines = create_comparison_image(left_rect, right_rect, with_lines=False)
    comparison_no_lines_out = os.path.join(output_dir, f'{base_name}_comparison.png')
    cv2.imwrite(comparison_no_lines_out, comparison_no_lines)
    print(f'Saved comparison image: {comparison_no_lines_out}')
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Undistort and rectify stereo image pairs. After rectification, epipolar lines are horizontal and aligned.'
    )
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--folder', default='figures/chessboard', 
                       help='Folder with stereo pairs (default: figures/chessboard)')
    group.add_argument('--pair', nargs=2, metavar=('LEFT', 'RIGHT'),
                       help='Single left/right image pair')
    
    parser.add_argument('--params', default='stereo_params.npz',
                        help='Path to stereo calibration parameters (default: stereo_params.npz)')
    parser.add_argument('--output', default='../data/image/rectified',
                        help='Output folder for rectified images (default: figures/rectified)')
    
    args = parser.parse_args()
    
    # Load calibration parameters
    if not os.path.exists(args.params):
        print(f'Error: Parameter file not found: {args.params}')
        return
    
    params = load_params(args.params)
    print(f'Loaded calibration parameters from {args.params}')
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find image pairs
    pairs = []
    if args.pair:
        pairs = [(args.pair[0], args.pair[1])]
    else:
        folder = args.folder
        if not os.path.isdir(folder):
            print(f'Error: Folder not found: {folder}')
            return
        
        lefts = sorted([f for f in os.listdir(folder) if '_left' in f])
        for l in lefts:
            r = l.replace('_left', '_right')
            left_path = os.path.join(folder, l)
            right_path = os.path.join(folder, r)
            if os.path.exists(right_path):
                pairs.append((left_path, right_path))
    
    if len(pairs) == 0:
        print('No image pairs found')
        return
    
    print(f'Found {len(pairs)} stereo pairs')
    
    # Process each pair
    success_count = 0
    for i, (left_path, right_path) in enumerate(pairs, start=1):
        print(f'\nProcessing pair {i}/{len(pairs)}: {os.path.basename(left_path)}')
        if process_image_pair(left_path, right_path, params, args.output):
            success_count += 1
    
    print(f'\n{success_count}/{len(pairs)} pairs processed successfully')
    print(f'All output images saved to: {args.output}')


if __name__ == '__main__':
    main()
