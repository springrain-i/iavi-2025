from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import os

def imread_color(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def list_images(img_dir: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> list[Path]:
    files = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    return files


def try_stitcher(images: list[np.ndarray]) -> tuple[int, np.ndarray | None]:
    """Use OpenCV Stitcher API. Returns (status, pano)."""
    # Use SCANS mode for translations; PANORAMA for rotations. Try both.
    for mode in (cv2.Stitcher_SCANS, cv2.Stitcher_PANORAMA):
        stitcher = cv2.Stitcher_create(mode)
        status, pano = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            return status, pano
    return status, None


def find_homography(img1: np.ndarray, img2: np.ndarray, feature: str = "ORB", ratio: float = 0.75) -> tuple[np.ndarray | None, int]:
    """Find H that maps img2 -> img1. Returns (H, num_inliers)."""
    if feature.upper() == "SIFT":
        try:
            det = cv2.SIFT_create()
            norm = cv2.NORM_L2
        except Exception:
            det = cv2.ORB_create(nfeatures=4000)
            norm = cv2.NORM_HAMMING
    else:
        det = cv2.ORB_create(nfeatures=4000)
        norm = cv2.NORM_HAMMING

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    k1, d1 = det.detectAndCompute(gray1, None)
    k2, d2 = det.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = matcher.knnMatch(d2, d1, k=2)  # query: img2 -> train: img1
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, 0

    pts2 = np.float32([k2[m.queryIdx].pt for m in good])
    pts1 = np.float32([k1[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def accumulate_homographies(images: list[np.ndarray], feature: str = "ORB") -> list[np.ndarray] | None:
    """Accumulate H[i]: img i -> img 0 (base)."""
    Hs = [np.eye(3, dtype=np.float64)]
    for i in range(1, len(images)):
        H, inl = find_homography(images[i-1], images[i], feature)
        if H is None:
            print(f"[warn] Homography {i-1}->{i} failed (inliers={inl}).")
            return None
        Hs.append(Hs[-1] @ H)  # compose to base 0
    return Hs


def warp_and_blend(images: list[np.ndarray], Hs: list[np.ndarray]) -> np.ndarray:
    # compute overall canvas
    corners = []
    for img, H in zip(images, Hs):
        h, w = img.shape[:2]
        cs = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
        wc = H @ cs
        wc /= wc[2]
        corners.append(wc[:2].T)
    corners = np.vstack(corners)
    min_xy = np.floor(corners.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(corners.max(axis=0)).astype(np.int32)
    offset = (-min_xy[0], -min_xy[1])
    W = int(max(1, max_xy[0] - min_xy[0]))
    Hh = int(max(1, max_xy[1] - min_xy[1]))

    T_off = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]], dtype=np.float64)
    pano = np.zeros((Hh, W, 3), dtype=np.uint8)
    weight = np.zeros((Hh, W), dtype=np.float32)

    for img, H in zip(images, Hs):
        M = T_off @ H
        warped = cv2.warpPerspective(img, M, (W, Hh))
        mask = (warped.sum(axis=2) > 0).astype(np.float32)
        # simple feather blending
        pano = (pano.astype(np.float32) * weight[..., None] + warped.astype(np.float32) * mask[..., None]) / np.maximum(weight + mask, 1e-6)[..., None]
        weight += mask
        pano = pano.astype(np.uint8)
    return pano


def downscale_images(images: list[np.ndarray], scale: float) -> list[np.ndarray]:
    if scale == 1.0:
        return images
    out = []
    for im in images:
        h, w = im.shape[:2]
        out.append(cv2.resize(im, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA))
    return out


def main():
    ap = argparse.ArgumentParser(description="Axis-shift image stitching (test phase)")
    ap.add_argument("--images-dir", default="./figures", help="Folder containing input images")
    ap.add_argument("--out", default="./out/panorama.png", help="Output panorama file path or directory (if directory, will save as panorama.png)")
    ap.add_argument("--feature", choices=["ORB", "SIFT"], default="SIFT", help="Feature for fallback pipeline")
    ap.add_argument("--scale", type=float, default=1.0, help="Resize factor (e.g., 0.5 for speed)")
    ap.add_argument("--no-stitcher", action="store_true", help="Skip cv2.Stitcher and use manual pipeline only")
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    # do not blindly create this path; we'll resolve file vs directory below
    files = list_images(img_dir)
    if len(files) < 2:
        print(f"No enough images in {img_dir} (found {len(files)}). Need >=2.")
        sys.exit(1)

    print(f"Found {len(files)} images. Loading...")
    images = [imread_color(p) for p in files]
    images = [im for im in images if im is not None]
    if len(images) < 2:
        print("Failed to load enough images.")
        sys.exit(1)

    if args.scale != 1.0:
        images_small = downscale_images(images, args.scale)
    else:
        images_small = images

    out_arg = Path(args.out)
    # If user passed a directory or no extension, treat as directory and write panorama.png inside
    if out_arg.is_dir() or out_arg.suffix.lower() == "":
        out_dir = out_arg
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "panorama.png"
    else:
        out_arg.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_arg

    pano = None
    if not args.no_stitcher:
        print("Trying OpenCV Stitcher...")
        status, pano_try = try_stitcher(images_small)
        if status == cv2.Stitcher_OK and pano_try is not None:
            print("Stitcher succeeded.")
            pano = pano_try
        else:
            print(f"Stitcher failed with status={status}. Falling back to manual pipeline.")

    if pano is None:
        print("Estimating pairwise homographies (manual pipeline)...")
        Hs = accumulate_homographies(images_small, feature=args.feature)
        if Hs is None:
            print("Failed to estimate homographies across images.")
            sys.exit(2)
        pano = warp_and_blend(images_small, Hs)

    # If we worked at reduced scale, upscale pano to approx original height
    if args.scale != 1.0 and pano is not None:
        scale_back = 1.0 / float(args.scale)
        Hh, Ww = pano.shape[:2]
        pano = cv2.resize(pano, (int(Ww*scale_back), int(Hh*scale_back)), interpolation=cv2.INTER_CUBIC)

    # Robust save: ensure uint8 and fallback extension if needed
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        arr = img.astype(np.float32)
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.clip(arr, 0, 255).astype(np.uint8)
        # If looks like [0,1] range, scale to [0,255]
        if vmax <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    pano_u8 = _to_uint8(pano)
    try:
        ok = cv2.imwrite(str(out_path), pano_u8)
        if not ok:
            raise cv2.error("imwrite returned False")
        print(f"Saved panorama -> {out_path}")
    except Exception as e:
        # Fallback to PNG
        alt = out_path.with_suffix('.png')
        try:
            ok2 = cv2.imwrite(str(alt), pano_u8)
            if ok2:
                print(f"[warn] Failed to write {out_path.name} ({e}); wrote {alt.name} instead.")
            else:
                raise cv2.error("imwrite PNG returned False")
        except Exception as e2:
            print(f"[error] Could not save panorama to {out_path} or {alt}: {e2}")
            sys.exit(3)


if __name__ == "__main__":
    main()