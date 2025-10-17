import cv2
import numpy as np
import os
import glob

calib_path = 'camera_params_generated.npz'
image_folder = 'chessboard_1_one'
output_folder = '../data/image'

os.makedirs(output_folder, exist_ok=True)

# 读取标定参数
calib = np.load(calib_path)
mtx = calib['mtx']
dist = calib['dist']

# 读取图片
images = glob.glob(os.path.join(image_folder, '*.png')) + glob.glob(os.path.join(image_folder, '*.jpg'))

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        continue
    undistorted = cv2.undistort(img, mtx, dist)
    out_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(out_path, undistorted)
    print(f"已保存: {out_path}")

print("去畸变处理完成。")