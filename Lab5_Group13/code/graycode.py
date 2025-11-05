"""Gray code pattern generator

Generates vertical (column) and horizontal (row) Gray-code bit patterns
for projector-based structured light systems.

Functions
- generate_graycode_patterns(width, height, nbits, out_dir)
  -> saves PNG images for vertical and horizontal bit patterns and
     complementary black/white reference images.
"""
import os
import numpy as np
import cv2


def int_to_gray(n):
    return n ^ (n >> 1)


def gray_to_binary(g):
    # Convert Gray code integer to binary integer
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b


def needed_bits(n):
    b = 0
    while (1 << b) < n:
        b += 1
    return b


def generate_graycode_patterns(width, height, out_dir, nbits=None, nbits_x=None, nbits_y=None, extra_bits=0):
    """Generate and save Gray code patterns.

    Creates vertical (encode x / columns) and horizontal (encode y / rows)
    patterns. Each bit is saved as a PNG (0 or 255). Also saves full
    white and full black reference images.

    Args:
        width, height: projector resolution in pixels
        out_dir: directory to save images
        nbits: optional number of bits; if None it will be auto-computed
    Returns:
        dict with file lists: {'vertical': [...], 'horizontal': [...], 'white': path, 'black': path}
    """
    os.makedirs(out_dir, exist_ok=True)

    # Allow caller to specify nbits for x/y independently or a single nbits for both.
    if nbits is not None:
        # legacy: single value for both axes
        nbits_x = nbits_y = int(nbits)
    if nbits_x is None:
        nbits_x = needed_bits(width)
    if nbits_y is None:
        nbits_y = needed_bits(height)

    # allow adding extra bits (generate more patterns than strictly needed)
    nbits_x = int(nbits_x) + int(extra_bits)
    nbits_y = int(nbits_y) + int(extra_bits)

    saved = {'vertical': [], 'horizontal': []}

    # Save black and white
    black = np.zeros((height, width), dtype=np.uint8)
    white = np.full((height, width), 255, dtype=np.uint8)
    black_p = os.path.join(out_dir, 'ref_black.png')
    white_p = os.path.join(out_dir, 'ref_white.png')
    cv2.imwrite(black_p, black)
    cv2.imwrite(white_p, white)
    saved['black'] = black_p
    saved['white'] = white_p

    # Vertical patterns (columns encode x coordinate)
    for bit in range(nbits_x - 1, -1, -1):
        patt = np.zeros((height, width), dtype=np.uint8)
        # For each column, compute Gray code and set columns where bit==1
        cols = np.arange(width, dtype=np.int32)
        gray = cols ^ (cols >> 1)
        # If bit index exceeds native gray width, the shift will yield 0 for all cols
        mask = ((gray >> bit) & 1).astype(np.uint8) * 255
        patt[:, :] = mask[np.newaxis, :]
        path = os.path.join(out_dir, f'vert_bit_{bit:02d}.png')
        cv2.imwrite(path, patt)
        saved['vertical'].append(path)

    # Horizontal patterns (rows encode y coordinate)
    for bit in range(nbits_y - 1, -1, -1):
        patt = np.zeros((height, width), dtype=np.uint8)
        rows = np.arange(height, dtype=np.int32)
        gray = rows ^ (rows >> 1)
        mask = ((gray >> bit) & 1).astype(np.uint8) * 255
        patt[:, :] = mask[:, np.newaxis]
        path = os.path.join(out_dir, f'hor_bit_{bit:02d}.png')
        cv2.imwrite(path, patt)
        saved['horizontal'].append(path)

    return saved


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--out', default='gray_patterns')
    p.add_argument('--nbits', type=int, default=None, help='(legacy) number of bits for both axes')
    p.add_argument('--nbits-x', type=int, default=None, help='number of vertical bits (columns) to generate')
    p.add_argument('--nbits-y', type=int, default=None, help='number of horizontal bits (rows) to generate')
    p.add_argument('--extra-bits', type=int, default=0, help='add extra bits on top of minimum needed')
    args = p.parse_args()
    res = generate_graycode_patterns(args.width, args.height, args.out, nbits=args.nbits, nbits_x=args.nbits_x, nbits_y=args.nbits_y, extra_bits=args.extra_bits)
    print('Saved patterns to', args.out)