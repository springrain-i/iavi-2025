def quick_generate_depth():
    """快速生成深度图"""
    try:
        depth, disparity, pointcloud, mask = generate_depth_map()
        print("深度图生成成功！")
        
        # 显示基本信息
        valid_depths = depth[mask.astype(bool)]
        print(f"深度统计:")
        print(f"  最小值: {valid_depths.min():.2f} mm")
        print(f"  最大值: {valid_depths.max():.2f} mm") 
        print(f"  平均值: {valid_depths.mean():.2f} mm")
        print(f"  中位数: {np.median(valid_depths):.2f} mm")
        
        # 可视化
        visualize_results(depth, disparity, pointcloud, mask)
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查：")
        print("1. stereo_params.npz 文件是否存在")
        print("2. img_decode 目录是否有解码结果")
        print("3. 文件路径是否正确")

# 直接运行
quick_generate_depth()