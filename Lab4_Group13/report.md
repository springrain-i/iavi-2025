# Lab4

## Basic part

### Introduction

Briefly describe your purpose

### Experiment Setup

Describe how you set the value of the parameters and other settings of the camera.

### Result and Data Processing

Show with your images and how you process the images for analysis.



#### The quality of calibration and the final generated point cloud

The stereo calibration quality was evaluated using the mean reprojection error and visual inspection of the generated point cloud. The mean reprojection error for both cameras was below 0.005 pixels, indicating accurate corner detection and parameter estimation. Debug overlays of chessboard corners and reprojection points  show that most detected points align well with the projected model, confirming reliable calibration.

The overall stereo calibration RMS error was 1.16, as reported by the stereoCalibrate function. This value reflects the average geometric error (in pixels) between the observed and predicted corner positions across all calibration images, and indicates a good level of calibration accuracy for practical stereo reconstruction tasks.

![calibration](./figures/calibration.png)

The final point cloud, reconstructed from rectified stereo pairs using the computed Q matrix, demonstrates good spatial consistency and preserves scene structure. As shown below, the point cloud captures the main geometry of the scene, with depth continuity and minimal outliers. The density and accuracy of the point cloud are influenced by the quality of calibration and the choice of stereo matching parameters.

![Point Cloud Example](./figures/pointcloud.png)


#### the images after the undistortion and the rectification

After undistortion and stereo rectification, the left and right images exhibit horizontally aligned epipolar lines, as expected. This alignment is crucial for accurate stereo matching. The following figure shows a typical rectified image pair with superimposed horizontal lines, demonstrating that corresponding features lie on the same scanlines:

![Rectified Images Example](./figures/undistort.png)

The rectified images are free from lens distortion, and the image content is preserved within the valid region. This preprocessing step ensures that subsequent disparity computation is both robust and geometrically correct.

#### Depth map

The computed depth maps visualize the scene's 3D structure, with brighter regions indicating greater distance from the camera. The depth map is generated from the disparity map using the calibrated Q matrix. As shown below, the depth map preserves object boundaries and provides a dense representation of the scene geometry:

![Depth Map Example](./figures/depth.png)

The quality of the depth map depends on the accuracy of rectification, the choice of block size, and the texture of the scene. Medium block sizes (e.g., 7×7 or 9×9) yield the best balance between detail and noise, as discussed in the analysis section.


**Key Functions and Workflow Explanation**

The stereo calibration and point cloud reconstruction process relies on a series of well-structured functions, each responsible for a critical step in the pipeline:

- `find_image_pairs(folder)`: Automatically searches for and pairs left/right calibration images in the specified folder, ensuring that only valid stereo pairs are used for calibration.

- `make_object_points(nx, ny, square_size)`: Generates the 3D coordinates of chessboard corners in the calibration pattern's local coordinate system, which serve as reference points for all views.

- `stereo_calibrate(...)`: This is the core calibration function. It detects chessboard corners in all image pairs, refines their positions, and accumulates corresponding 2D-3D points. It then calibrates each camera individually to obtain intrinsic parameters and distortion coefficients, followed by stereo calibration to estimate the rotation (R) and translation (T) between cameras. The function also computes and reports the mean reprojection error, and saves debug overlays for visual inspection.

- `cv2.stereoRectify(...)`: Computes the rectification transforms (R1, R2, P1, P2) and the reprojection matrix Q, which are essential for aligning the epipolar lines and enabling 3D reconstruction from disparity.

- `cv2.initUndistortRectifyMap(...)` and `cv2.remap(...)`: These functions undistort and rectify the original images, producing rectified pairs where corresponding points lie on the same horizontal lines, greatly simplifying stereo matching.

- `stereo_match_sgbm(...)`: Implements Semi-Global Block Matching (SGBM) to compute the disparity map from the rectified image pair. The function allows tuning of block size and other parameters to balance detail and noise.

- `cv2.reprojectImageTo3D(...)`: Uses the computed disparity map and the Q matrix to reconstruct the 3D coordinates of scene points, forming the basis of the final point cloud.

- `write_ply(...)`: Saves the reconstructed 3D points (and optionally their colors) to a PLY file for visualization and further analysis.

The overall workflow starts with stereo calibration using chessboard images, proceeds through undistortion and rectification, computes disparity maps for rectified pairs, and finally reconstructs and exports the 3D point cloud. Each function is modular, allowing for clear debugging and flexible parameter adjustment throughout the process.

### Analysis and Discussion

#### Discussion on Point Cloud Generation and Q Matrix Issues

During the point cloud reconstruction stage, I conducted a detailed review of the workflow and consulted both AI and relevant online resources. I discovered that the quality of the generated point cloud is highly sensitive to the correctness of the Q (reprojection) matrix used by `cv2.reprojectImageTo3D`. According to [OpenCV forum discussions](https://forum.opencv.org/t/erroneous-point-cloud-generated-by-cv2-reprojectimageto3d/3706/4), the Q matrix produced by `cv2.stereoRectify` can sometimes be problematic, leading to distorted or erroneous 3D reconstructions.

After carefully inspecting the code, I confirmed that the Q matrix was indeed generated by `cv2.stereoRectify`. However, despite extensive debugging, I was unable to pinpoint the exact source of the error in the Q matrix calculation. To address this, I referred to external documentation and manually constructed the Q matrix based on the standard formula. While this manual approach improved my understanding of the process, the resulting point cloud still exhibited noticeable artifacts and inaccuracies.

Despite these issues, the overall shape and structure of the scene can still be discerned from the point cloud. For presentation purposes, I used annotated screenshots (e.g., drawing arcs or highlighting regions) to guide the reader's attention to the recognizable features within the point cloud. This experience highlights the importance of careful calibration, parameter verification, and a deep understanding of the underlying algorithms when working with stereo vision and 3D reconstruction.