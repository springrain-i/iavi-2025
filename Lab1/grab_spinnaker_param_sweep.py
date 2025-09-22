"""
Parameter sweep capture script for Spinnaker (PySpin).

This script tries to import PySpin (or pyspin) and performs sweeps over
exposure, gain, and gamma. For each parameter combination it captures
`images_per_setting` images, saves them to a timestamped folder and
writes `metadata.csv` and `metadata.json` with the parameter values.

Usage:
    python grab_spinnaker_param_sweep.py

Notes:
- Ensure the Spinnaker SDK and Python bindings are installed and that
  the Python interpreter bitness matches the SDK (usually 64-bit).
- Adjust the parameter lists below to fit your camera capabilities.
"""
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
    # adjustable parameters - change these to match your camera's valid ranges
    exposures = [500.0, 1000.0, 5000.0,10000.0,15000.0,20000.0]  # microseconds
    gains = [0.0, 5.0, 10.0, 13.0, 15.0, 17.0, 20.0]  # units depend on camera (could be dB or gain value)
    gammas = [0.8, 1.0, 1.2]
    images_per_setting = 5

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(os.getcwd(), f"param_sweep_{timestamp}")
    ensure_dir(out_dir)

    metadata = []

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

        # Try to set pixel format to a known color format if available
        try:
            if hasattr(cam, 'PixelFormat'):
                # common format name; adjust if your camera uses different names
                cam.PixelFormat.SetValue('RGB8')
        except Exception:
            pass

        # Iterate parameter combinations
        idx = 0
        for exp in exposures:
            # ExposureTime may be in microseconds; some cameras use different node names
            try:
                if hasattr(cam, 'ExposureTime'):
                    cam.ExposureAuto.SetValue('Off')
                    cam.ExposureTime.SetValue(float(exp))
                else:
                    print('Camera has no ExposureTime node')
            except Exception as e:
                print('Failed to set exposure:', e)

            for gain in gains:
                try:
                    if hasattr(cam, 'Gain'):
                        cam.GainAuto.SetValue('Off')
                        cam.Gain.SetValue(float(gain))
                    else:
                        print('Camera has no Gain node')
                except Exception as e:
                    # some cameras expose Gain as Gain.Value
                    try:
                        cam.Gain.Value = float(gain)
                    except Exception:
                        print('Failed to set gain:', e)

                for gamma in gammas:
                    try:
                        if hasattr(cam, 'Gamma'):
                            cam.Gamma.SetValue(float(gamma))
                        else:
                            print('Camera has no Gamma node')
                    except Exception as e:
                        print('Failed to set gamma:', e)

                    # give camera time to apply settings
                    time.sleep(0.2)

                    # Start acquisition for this setting
                    try:
                        cam.BeginAcquisition()
                    except Exception:
                        # fallback to older API
                        try:
                            cam.StartAcquisition()
                        except Exception as e:
                            print('Failed to start acquisition:', e)
                            continue

                    for i in range(images_per_setting):
                        try:
                            # Wait for and retrieve the next image
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

                        filename = os.path.join(out_dir, f"img_{idx:06}_exp{exp}_gain{gain}_gamma{gamma}_#{i}.png")
                        save_image_from_spy(image, filename)

                        meta = {
                            'filename': filename,
                            'index': idx,
                            'exposure_us': exp,
                            'gain': gain,
                            'gamma': gamma,
                            'timestamp': datetime.now().isoformat(),
                        }
                        metadata.append(meta)
                        idx += 1

                        image.Release()

                    # stop acquisition for this setting
                    try:
                        cam.EndAcquisition()
                    except Exception:
                        try:
                            cam.StopAcquisition()
                        except Exception:
                            pass

        # write metadata files
        csv_path = os.path.join(out_dir, 'metadata.csv')
        json_path = os.path.join(out_dir, 'metadata.json')

        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=['filename', 'index', 'exposure_us', 'gain', 'gamma', 'timestamp'])
            writer.writeheader()
            for row in metadata:
                writer.writerow(row)

        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(metadata, jf, indent=2, ensure_ascii=False)

        print('Capture complete. Output directory:', out_dir)
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
