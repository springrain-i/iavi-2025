### 2.1 Introduction

The purpose of this analysis is to investigate the relationship between the number of chessboard images used in camera calibration and the resulting reprojection error. Reprojection error serves as a key metric for evaluating the accuracy and reliability of camera calibration parameters. 

### 2.2 Experiment Setup

**Image Collection Strategy:**

- Multiple image sets were created with varying quantities: 1, 5, 10, 15, and 20 images
- All images were captured with the camera focused on the same chessboard pattern
- Smaller image sets are subsets of larger sets (e.g., 10-image set contains all images from 5-image set)
- Care was taken to select images with the chessboard facing directly toward the camera to minimize perspective distortion effects

### 2.3 Result and Data Processing

#### Data Collection Results:

| Image Set    | Total Images | Valid Images | Reprojection Error (pixels) | Status  |
| ------------ | ------------ | ------------ | --------------------------- | ------- |
| image_set_1  | 1            | 1            | 0.004903                    | Success |
| image_set_5  | 5            | 5            | 0.006725                    | Success |
| image_set_10 | 10           | 10           | 0.016291                    | Success |
| image_set_15 | 15           | 14           | 0.014551                    | Success |
| image_set_20 | 20           | 19           | 0.013566                    | Success |

#### Image Processing Pipeline:

1. **Corner Detection**: Automatic detection of chessboard corners using OpenCV's `findChessboardCorners`
2. **Sub-pixel Refinement**: Precision enhancement of corner locations using `cornerSubPix` with termination criteria (30 iterations, 0.001 accuracy)
3. **Calibration Execution**: Camera parameter estimation using all valid detected points
4. **Error Calculation**: Reprojection error computed as the RMS distance between detected and projected points

### 2.4 Analysis and Discussion

#### Chart of the Result

![](./reprojection_error_analysis.png)

#### Key Observations:

1. **Minimum Error with Single Image**: Surprisingly, the single-image calibration achieved the lowest reprojection error (0.004903 pixels). This counterintuitive result suggests that with optimal positioning and minimal perspective distortion, a single well-captured image can provide excellent calibration accuracy.

2. **Error Increase with Additional Images**: As more images were added (from 1 to 10 images), the reprojection error increased significantly, reaching a maximum of 0.016291 pixels with 10 images. This indicates that incorporating images with varying perspectives and potential distortions can initially degrade calibration accuracy.

3. **Error Stabilization**: Beyond 10 images, the reprojection error began to decrease and stabilize (0.014551 pixels with 15 images, 0.013566 pixels with 20 images), suggesting that a sufficiently large and diverse dataset helps the calibration algorithm converge to more robust parameters.

#### Limitations:

- **Variable Control**: Despite efforts to maintain consistent capture angles, unavoidable variations in perspective, lighting, and chessboard coverage area existed across images
- **Subset Relationship**: The nested nature of image sets (smaller sets being subsets of larger ones) means that error progression reflects cumulative effects rather than independent sampling
- **Sensitivity to Small Distortion Coefficients**: The overall distortion coefficients in this experiment were relatively small, which means that minor numerical variations could lead to proportionally significant changes in the reprojection error. This heightened sensitivity may amplify the impact of other interfering factors on the analysis results, potentially obscuring the underlying relationship between image count and calibration accuracy.

### 2.5 Conclusion

The conclusions we draw from the result of image-number-experiment are as follows:

1. **Optimal Single Image Performance**: For applications where a perfectly frontal chessboard image can be obtained, single-image calibration can yield exceptional accuracy (0.004903 pixels).

2. **Robust Multi-image Calibration**: When diverse perspectives are necessary or perfect frontal capture is not feasible, 15-20 images provide a good balance between accuracy and robustness, with errors stabilizing around 0.013-0.015 pixels.
