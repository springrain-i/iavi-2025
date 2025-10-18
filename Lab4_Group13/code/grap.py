import os
import sys
import time
from datetime import datetime

try:
	import PySpin
except Exception:
	try:
		import pyspin as PySpin
	except Exception as e:
		print("Unable to import PySpin/pyspin:", e)
		sys.exit(1)

import numpy as np
try:
	import cv2
except Exception:
	cv2 = None




def capture_stereo_pairs(out_root='figures', left_id=0, right_id=1):
	"""Capture synchronized stereo pairs from two cameras and save them.

	Saves images directly under `out_root` with names like `1_left.png` and
	`1_right.png`. Use SPACE (or 'y') to capture a pair, 'q' to quit.
	"""
	os.makedirs(out_root, exist_ok=True)

	system = PySpin.System.GetInstance()
	cam_list = system.GetCameras()
	print(cam_list)
	num_cams = cam_list.GetSize()
	print(f"Detected {num_cams} cameras")
	if num_cams < 2:
		print('Need at least 2 cameras for stereo capture')
		cam_list.Clear()
		system.ReleaseInstance()
		return

	left_cam = cam_list.GetByIndex(left_id)
	right_cam = cam_list.GetByIndex(right_id)

	try:
		left_cam.Init()
		right_cam.Init()
	except Exception as e:
		print('Camera init failed:', e)
		cam_list.Clear()
		system.ReleaseInstance()
		return

	try:
		if cv2 is not None:
			cv2.namedWindow('Left', cv2.WINDOW_NORMAL)
			cv2.namedWindow('Right', cv2.WINDOW_NORMAL)

		left_idx = 1
		print('Stereo capture: press SPACE (or y) to capture a pair, q to quit')

		# Start acquisition on both cameras
		try:
			left_cam.BeginAcquisition()
		except Exception:
			try:
				left_cam.StartAcquisition()
			except Exception as e:
				print('Failed to start left acquisition:', e)
				return
		try:
			right_cam.BeginAcquisition()
		except Exception:
			try:
				right_cam.StartAcquisition()
			except Exception as e:
				print('Failed to start right acquisition:', e)
				return

		while True:
			# Acquire left frame
			try:
				left_img = left_cam.GetNextImage(2000)
			except Exception:
				try:
					left_img = left_cam.GetNextImageEx(2000)
				except Exception as e:
					print('Failed to get left image:', e)
					break
			# Acquire right frame
			try:
				right_img = right_cam.GetNextImage(2000)
			except Exception:
				try:
					right_img = right_cam.GetNextImageEx(2000)
				except Exception as e:
					print('Failed to get right image:', e)
					left_img.Release()
					break

			if left_img.IsIncomplete() or right_img.IsIncomplete():
				print('Incomplete image, skipping')
				if not left_img.IsIncomplete():
					left_img.Release()
				if not right_img.IsIncomplete():
					right_img.Release()
				continue

			left_arr = None
			right_arr = None
			try:
				left_arr = left_img.GetNDArray()
			except Exception:
				pass
			try:
				right_arr = right_img.GetNDArray()
			except Exception:
				pass

			if cv2 is not None and left_arr is not None and right_arr is not None:
				cv2.imshow('Left', left_arr)
				cv2.imshow('Right', right_arr)
				key = cv2.waitKey(1) & 0xFF
			else:
				print('Preview not available, press y to capture, q to quit')
				key = ord(input('y=save, q=quit: ').strip() or ' ')

			if key == 32 or key == ord('y'):
				left_name = f'{left_idx}_left.png'
				right_name = f'{left_idx}_right.png'
				left_path = os.path.join(out_root, left_name)
				right_path = os.path.join(out_root, right_name)

				# save using NDArray if available, else use PySpin Save
				if left_arr is not None and cv2 is not None:
					cv2.imwrite(left_path, left_arr)
				else:
					try:
						left_img.Save(left_path)
					except Exception as e:
						print('Failed to save left image:', e)

				if right_arr is not None and cv2 is not None:
					cv2.imwrite(right_path, right_arr)
				else:
					try:
						right_img.Save(right_path)
					except Exception as e:
						print('Failed to save right image:', e)

				print(f'Saved pair: {left_name} , {right_name}')
				left_idx += 1

			if key == ord('q'):
				left_img.Release()
				right_img.Release()
				break

			# release images each loop
			left_img.Release()
			right_img.Release()

		# end loop
		# stop acquisition
		try:
			left_cam.EndAcquisition()
		except Exception:
			try:
				left_cam.StopAcquisition()
			except Exception:
				pass
		try:
			right_cam.EndAcquisition()
		except Exception:
			try:
				right_cam.StopAcquisition()
			except Exception:
				pass

		# No CSV log requested; images are saved directly under out_root

		if cv2 is not None:
			cv2.destroyAllWindows()

	finally:
		try:
			left_cam.DeInit()
		except Exception:
			pass
		try:
			right_cam.DeInit()
		except Exception:
			pass
		del left_cam
		del right_cam
		cam_list.Clear()
		system.ReleaseInstance()


def main():
	# basic CLI: allow output root and camera indices
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--out', default='figures/use4ply')
	parser.add_argument('--left', type=int, default=1)
	parser.add_argument('--right', type=int, default=0)
	args = parser.parse_args()
	capture_stereo_pairs(out_root=args.out, left_id=args.left, right_id=args.right)


if __name__ == '__main__':
	main()

