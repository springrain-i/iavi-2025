import os
import sys
import time
import json
import csv
from datetime import datetime

try:
    import PySpin
except Exception:
    try:
        import pyspin as PySpin  # fallback name
    except Exception as e:
        print("Unable to import PySpin/pyspin:", e)
        sys.exit(1)


import numpy as np
try:
    import cv2
except Exception:
    cv2 = None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_image_from_spy(image, filename):
    """Save a PySpin imageResult to filename. Try NDArray -> OpenCV if available,
    else fall back to image.Save()."""
    try:
        arr = image.GetNDArray()
        if cv2 is not None:
            # PySpin NDArray is usually BGR; verify in your camera
            cv2.imwrite(filename, arr)
        else:
            # No OpenCV; use PySpin save
            image.Save(filename)
    except Exception:
        # If GetNDArray not available, try Save directly
        try:
            image.Save(filename)
        except Exception as e:
            print("Failed to save image:", e)

def main():
    out_dir = "chessboard_1"
    ensure_dir(out_dir)

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    print(f"Detected {num_cameras} cameras")
    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print("No cameras detected. Exiting.")
        return

    cam = cam_list.GetByIndex(0)
    try:
        cam.Init()
        try:
            if hasattr(cam, 'PixelFormat'):
                cam.PixelFormat.SetValue('RGB8')
        except Exception:
            pass

        idx = 1
        print("按空格拍照，按q退出...")
        if cv2 is not None:
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        while True:
            # 采集一帧用于显示
            try:
                #print("Acquiring image...")
                cam.BeginAcquisition()
            except Exception:
                try:
                    cam.StartAcquisition()
                except Exception as e:
                    print('Failed to start acquisition:', e)
                    break
            try:
                #print("Getting image...")
                image = cam.GetNextImage(5000)
            except Exception:
                try:
                    image = cam.GetNextImageEx(5000)
                except Exception as e:
                    print('Failed to get image:', e)
                    break
            if image.IsIncomplete():
                print('Incomplete image:', image.GetImageStatus())
                image.Release()
                continue
            arr = None
            try:
                arr = image.GetNDArray()
            except Exception:
                pass
            if cv2 is not None and arr is not None:
                cv2.imshow("Camera", arr)
                key = cv2.waitKey(1) & 0xFF
            else:
                print("按y拍照，按q退出（无预览）...")
                key = ord(input("输入y拍照，q退出: ").strip() or ' ')
            # 按空格或y拍照
            if key == 32 or key == ord('y'):
                filename = os.path.join(out_dir, f"{idx}.png")
                save_image_from_spy(image, filename)
                print(f"Saved {filename}")
                idx += 1
            # 按q退出
            if key == ord('q'):
                image.Release()
                break
            image.Release()
            try:
                cam.EndAcquisition()
            except Exception:
                try:
                    cam.StopAcquisition()
                except Exception:
                    pass
        if cv2 is not None:
            cv2.destroyAllWindows()
    finally:
        try:
            cam.DeInit()
        except Exception:
            pass
        del cam
        cam_list.Clear()
        system.ReleaseInstance()

if __name__ == '__main__':
    main()

