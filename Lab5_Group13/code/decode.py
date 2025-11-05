from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
import os

def needed_bits(n: int) -> int:
    b = 0
    while (1 << b) < n:
        b += 1
    return b


def gray_to_binary_arr(gray: np.ndarray, nbits: int) -> np.ndarray:
    """Vectorized Gray->binary conversion for non-negative integers.

    gray: int array with values in [0, 2^nbits)
    returns same-shape int array of binary code.
    """
    b = np.zeros_like(gray)
    g = gray.copy()
    for _ in range(nbits):
        b ^= g
        g >>= 1
    return b


def compute_shadow_mask(black: np.ndarray, white: np.ndarray, threshold: int) -> np.ndarray:
    """Mimic C++ computeShadowMask: white > black + threshold -> 255 else 0 (uint8)."""
    mask = (white.astype(np.int32) > (black.astype(np.int32) + int(threshold))).astype(np.uint8) * 255
    return mask


def decode_axis(patterns: list[np.ndarray], white: np.ndarray, black: np.ndarray,
                axis_bits: int, white_thresh: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode one axis from a sequence of bit images (MSB..LSB).

    Returns (index_img, reliable_mask) where index_img is int32 with -1 for invalid.
    A pixel is considered unreliable if any bit is too close to mid (|img-mid|<=white_thresh).
    """
    h, w = white.shape[:2]
    if len(patterns) < axis_bits:
        raise ValueError(f"Not enough pattern images for axis: have {len(patterns)} need {axis_bits}")

    # Per-pixel mid
    mid = ((white.astype(np.int32) + black.astype(np.int32)) // 2).astype(np.int32)
    gray_vals = np.zeros((h, w), dtype=np.int32)
    reliable = np.ones((h, w), dtype=bool)

    # Expect MSB..LSB order in 'patterns'
    for bit_i in range(axis_bits):
        img = patterns[bit_i].astype(np.int32)
        diff = np.abs(img - mid)
        reliable &= (diff > int(white_thresh))
        bit_mask = (img > mid).astype(np.int32)
        shift = (axis_bits - 1 - bit_i)
        gray_vals |= (bit_mask << shift)

    # Convert Gray to binary index
    idx = gray_to_binary_arr(gray_vals, axis_bits)
    return idx, reliable

def decode_for_suffix(suffix: str, out_prefix: str, args: argparse.Namespace, out_dir: Path) -> tuple[int, int]:
    proj_width = args.proj_width
    proj_height = args.proj_height
    in_dir = Path(args.in_dir)
    def p(num: int) -> Path:
        return in_dir / f"{num}{suffix}"

    white_path = p(1)
    black_path = p(2)
    if not white_path.exists() or not black_path.exists():
        print(f"[skip] Missing refs for {out_prefix}: {white_path if not white_path.exists() else ''} {black_path if not black_path.exists() else ''}")
        return 0, 0
    # Discovery rule (user's convention):
    #   1_<cam>.png = reference white
    #   2_<cam>.png = reference black
    #   3..(3+nbits_x-1) = vertical MSB->LSB
    #   next nbits_y images = horizontal MSB->LSB
    nbits_x = needed_bits(int(proj_width))
    nbits_y = needed_bits(int(proj_height))
    vert_paths = [p(i) for i in range(3, 3 + nbits_x)]
    horz_paths = [p(i) for i in range(3 + nbits_x, 3 + nbits_x + nbits_y)]

    # Verify all required pattern files exist
    missing = [str(pp) for pp in (vert_paths + horz_paths) if not pp.exists()]
    if missing:
        print(f"[skip] Missing pattern images for {out_prefix} ({len(missing)}): e.g., {missing[:5]}")
        return 0, 0

    # Load references and patterns
    white_img = cv2.imread(str(white_path), cv2.IMREAD_GRAYSCALE)
    black_img = cv2.imread(str(black_path), cv2.IMREAD_GRAYSCALE)
    if white_img is None or black_img is None:
        print(f"[skip] Failed to read white/black refs for {out_prefix}")
        return 0, 0
    patt_x = [cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE) for pp in vert_paths]
    patt_y = [cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE) for pp in horz_paths]
    if any(im is None for im in patt_x + patt_y):
        print(f"[skip] Failed to read some pattern images for {out_prefix}")
        return 0, 0

    h, w = white_img.shape[:2]
    for im in patt_x + patt_y:
        if im.shape[:2] != (h, w):
            print(f"[skip] Size mismatch among images for {out_prefix}")
            return 0, 0

    shadow_mask = compute_shadow_mask(black_img, white_img, args.black_thresh).astype(bool)
    x_idx, rel_x = decode_axis(patt_x, white_img, black_img, nbits_x, args.white_thresh)
    y_idx, rel_y = decode_axis(patt_y, white_img, black_img, nbits_y, args.white_thresh)

    valid = shadow_mask & rel_x & rel_y
    valid &= (x_idx >= 0) & (x_idx < int(proj_width)) & (y_idx >= 0) & (y_idx < int(proj_height))

    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)
    x_map[valid] = x_idx[valid].astype(np.float32)
    y_map[valid] = y_idx[valid].astype(np.float32)

    x_vis = np.zeros((h, w), dtype=np.uint8)
    y_vis = np.zeros((h, w), dtype=np.uint8)
    if int(proj_width) > 0:
        x_vis[valid] = np.clip((x_map[valid] * 255.0 / float(int(proj_width))).round(), 0, 255).astype(np.uint8)
    if int(proj_height) > 0:
        y_vis[valid] = np.clip((y_map[valid] * 255.0 / float(int(proj_height))).round(), 0, 255).astype(np.uint8)

    # Save outputs under standardized names (directory created by caller)
    x_png = out_dir / f"{out_prefix}_x.png"
    y_png = out_dir / f"{out_prefix}_y.png"
    mask_png = out_dir / f"{out_prefix}_mask.png"
    x_float = out_dir / f"{out_prefix}_x.tiff"
    y_float = out_dir / f"{out_prefix}_y.tiff"

    cv2.imwrite(str(x_png), x_vis)
    cv2.imwrite(str(y_png), y_vis)

    def _save_float(path: Path, img: np.ndarray):
        try:
            ok = cv2.imwrite(str(path), img)
            if ok:
                return str(path)
        except cv2.error:
            pass
        # Fallback: write TIFF
        fallback = path.with_suffix('.tiff')
        try:
            ok2 = cv2.imwrite(str(fallback), img)
            if ok2:
                print(f"[warn] Failed to write {path.name}; wrote {fallback.name} instead (float32)")
                return str(fallback)
        except cv2.error:
            pass
        # Final fallback: numpy npy
        npy = path.with_suffix('.npy')
        np.save(str(npy), img)
        print(f"[warn] Failed to write {path.name}; saved numpy array {npy.name} instead")
        return str(npy)

    x_float_written = _save_float(x_float, x_map)
    y_float_written = _save_float(y_float, y_map)
    cv2.imwrite(str(mask_png), (valid.astype(np.uint8) * 255))

    total = h * w
    n_valid = int(valid.sum())
    print(f"{out_prefix}: Decoded pixels {n_valid}/{total} ({n_valid/total:.1%})")
    print('Wrote:', x_png.name, y_png.name, Path(x_float_written).name, Path(y_float_written).name, mask_png.name)
    return n_valid, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proj_width', type=int, default=1280, help='Override projector width')
    ap.add_argument('--proj_height', type=int, default=720, help='Override projector height')
    ap.add_argument('--in-dir', type=str, default='capture', help='Directory containing captured PNGs')
    ap.add_argument('--out-dir', type=str, default='img_decode', help='Subdirectory (inside in-dir) to write decoded outputs')
    ap.add_argument('--cam-suffix', type=str, default='both', help="Which camera to decode: '_left.png', '_right.png', or 'both' (default)")
    ap.add_argument('--left-suffix', type=str, default='_left.png', help='Left camera filename suffix')
    ap.add_argument('--right-suffix', type=str, default='_right.png', help='Right camera filename suffix')
    ap.add_argument('--white-thresh', type=int, default=5, help='White threshold (binarization ambiguity tolerance)')
    ap.add_argument('--black-thresh', type=int, default=40, help='Black threshold (shadow mask)')
    # Legacy args removed from auto-discovery flow
    args = ap.parse_args()

    # Resolve parameters
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    # Decide which cameras to decode
    decoded = []
    decoded.append(decode_for_suffix(args.left_suffix, 'left', args, out_dir))
    decoded.append(decode_for_suffix(args.right_suffix, 'right', args, out_dir))

    # Summary
    dec_ok = [n for (n, t) in decoded if t > 0]
    if dec_ok:
        total_pixels = sum(t for (_, t) in decoded if t > 0)
        total_valid = sum(n for (n, t) in decoded if t > 0)
        print(f"Total decoded valid pixels: {total_valid}/{total_pixels} ({(total_valid/max(total_pixels,1)):.1%})")
    else:
        raise SystemExit("No decoded outputs written. Check input files and suffixes (see [skip] messages above).")


if __name__ == '__main__':
    main()
