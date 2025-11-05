import os
import sys
import time
import argparse
import numpy as np
import cv2
import platform

if platform.system() == 'Windows':
    import ctypes
    from ctypes import wintypes


def get_monitor_rects():
    rects = []
    if platform.system() != 'Windows':
        return rects
    try:
        user32 = ctypes.windll.user32
        MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(wintypes.RECT), ctypes.c_double)

        def _callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
            r = lprcMonitor.contents
            rects.append((r.left, r.top, r.right, r.bottom))
            return 1

        enum_cb = MonitorEnumProc(_callback)
        user32.EnumDisplayMonitors(0, 0, enum_cb, 0)
    except Exception:
        pass
    return rects


def show_image_fullscreen(img, window_name='Projector', monitor_rect=None):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if monitor_rect is not None:
        left, top, right, bottom = monitor_rect
        w = right - left
        h = bottom - top
        try:
            cv2.moveWindow(window_name, int(left), int(top))
            cv2.resizeWindow(window_name, int(w), int(h))
        except Exception:
            pass
    cv2.imshow(window_name, img)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)


def project_sequence(pattern_dir, proj_width, proj_height, monitor_index=None, delay=0.5, manual=False):
    # ensure patterns exist
    from graycode import generate_graycode_patterns
    saved = generate_graycode_patterns(proj_width, proj_height, pattern_dir)
    vertical = saved['vertical']
    horizontal = saved['horizontal']
    white = saved['white']
    black = saved['black']

    sequence = [white, black] + vertical + horizontal

    monitor_rect = None
    if monitor_index is not None:
        rects = get_monitor_rects()
        if rects and 0 <= monitor_index < len(rects):
            monitor_rect = rects[monitor_index]
            print('Using monitor', monitor_index, 'rect', monitor_rect)
        else:
            print('Monitor index not found or out of range; using default display')

    win = 'Projector'
    for p in sequence:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('Failed to load', p)
            continue
        show_image_fullscreen(img, window_name=win, monitor_rect=monitor_rect)

        if manual:
            # Support both GUI keypress (when projector window has focus) and
            # console Enter key (when terminal has focus). Some systems do not
            # deliver key events to the fullscreen window unless it has focus,
            # so we allow pressing Enter in the terminal as fallback.
            print(f'Pattern: {os.path.basename(p)} displayed.')
            print('Controls: [SPACE]=advance  [s]=skip  [q]=quit  [a]=auto-run remaining')
            # start a background thread to detect ENTER in console as a fallback
            import threading

            console_pressed = {'enter': False}

            def wait_enter():
                try:
                    input()
                    console_pressed['enter'] = True
                except Exception:
                    pass

            t = threading.Thread(target=wait_enter, daemon=True)
            t.start()

            while True:
                k = cv2.waitKey(100)
                if k == -1:
                    # check console fallback
                    if console_pressed['enter']:
                        break
                    continue
                k = k & 0xFF
                if k == 32:  # SPACE
                    break
                if k == ord('s'):
                    break
                if k == ord('q'):
                    print('User requested quit')
                    cv2.destroyWindow(win)
                    return
                if k == ord('a'):
                    manual = False
                    break
        else:
            time.sleep(delay)

    cv2.destroyWindow(win)


def main():
    p = argparse.ArgumentParser(description='Project Gray-code patterns to a selected monitor')
    p.add_argument('--proj-width', type=int, default=1280, help='Projector pixel width')
    p.add_argument('--proj-height', type=int, default=720, help='Projector pixel height')
    p.add_argument('--patterns-dir', default='gray_patterns', help='Folder to save/load patterns')
    p.add_argument('--monitor-index', type=int, default=1, help='Monitor index to display patterns (Windows)')
    p.add_argument('--delay', type=float, default=0.5, help='Delay between patterns in auto mode (s)')
    p.add_argument('--manual', action='store_true', help='Wait for manual confirmation (SPACE) before advancing each pattern')
    args = p.parse_args()

    project_sequence(args.patterns_dir, args.proj_width, args.proj_height, monitor_index=args.monitor_index, delay=args.delay, manual=args.manual)


if __name__ == '__main__':
    main()
