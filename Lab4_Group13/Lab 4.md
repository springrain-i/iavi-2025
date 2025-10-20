### Analysis of Patch Size Impact on Stereo Matching

#### I. Experimental Results  
The experimental data, derived from disparity maps and quantitative metrics, reveals distinct patterns across patch sizes (3×3 to 13×13):  

1. **Visual Characteristics of Disparity Maps**  

![Disparity](experiment_results/block_size_disparity_maps.png)

   - **Small windows (3×3, 5×5)**: Retain rich details with sharp object edges but suffer from scattered noise due to high sensitivity to pixel-level fluctuations.  
   - **Medium windows (7×7, 9×9)**: Balance detail retention and noise reduction, with clear edges and minimal artifacts—representing an optimal trade-off.  
   - **Large windows (11×11, 13×13)**: Produce smoother results with reduced noise but blur edges and lose small-scale details, as extensive averaging merges adjacent depth regions.  

2. **Quantitative Trends**  

![trend](experiment_results/block_size_analysis.png)

   - **Valid disparity ratio**: Increases with larger windows, as their smoothing effect "forces" more matches (including potential false matches) by mitigating noise.  
   - **Average disparity**: Rises steadily with window size, reflecting greater stability from averaging but reduced accuracy at depth discontinuities.  
   - **Disparity variance**: First decreases (3×3 to 9×9) due to noise suppression, then increases (11×11 to 13×13) due to artificial fluctuations from averaging across distinct depth regions.  

#### II. Analysis

The observed trends, particularly the rising valid ratio and late-increasing variance with larger windows, deviate from typical expectations. Key reasons include:  

1. **Image Noise and Low Texture**  
   Small windows fail to match in noisy or low-texture regions due to unreliable pixel similarities, lowering their valid ratio. Larger windows average out noise, inflating valid matches but introducing inaccuracies.  

2. **Scene Complexity**  
   Large windows spanning multiple depth regions (e.g., foreground and background) generate intermediate disparity values, creating artificial variance. This contrasts with the expected trend of decreasing variance from smoothing.  

3. **Suboptimal Parameters or Calibration**  
   Misconfigured SGBM parameters or poor rectification can amplify noise in small windows or mask errors in large ones, distorting trends.  

4. **Narrow Depth Range**  
   Scenes with limited depth variation reduce small windows’ advantage in capturing fine details, making larger windows appear more effective than they are.  


#### III. Conclusion

- **Medium windows (7×7, 9×9)** remain the most reliable, balancing detail and noise for general scenarios.  
- **Small windows** suit noise-free, high-texture environments (e.g., industrial inspection), while **large windows** are only viable for low-detail, noisy scenes.  

Discrepancies from expected patterns highlight the critical role of image quality, scene complexity, and parameter tuning in stereo matching. Addressing these factors aligns results with the core trade-off: smaller windows preserve details (but amplify noise), while larger windows smooth noise (but lose details).