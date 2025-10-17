# lab2

`grap.py` 是图片抓取脚本

首先运行`CameraCenter_Chessboard.py`生成相机中心和棋盘的点云以及会保存相机的内参

`analysis_reprojection.py`是重投影分析脚本，运行后会生成重投影误差的统计图

`my_projector.py`里是自定义投影器，没有使用cv2里现成的投影器，增加了背面剔除功能，解决了投影中前后视觉混乱的问题。

`project.py`是主投影脚本,`python project.py --live`可以实时投影, 使用投影功能时，记得使用自己相机拍摄的图片和内参

`results`里是生成的小猫投影图片




