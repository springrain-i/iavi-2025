from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import json
from typing import Tuple, Dict, Any


def imread_gray(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def tenengrad(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy  # gradient energy per-pixel


def grid_sharpness(gray: np.ndarray, grid: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cell sharpness map using Tenengrad. Returns (S, Xc, Yc)
    S: (gy, gx) mean sharpness per cell
    Xc, Yc: cell center coordinates (pixel)
    """
    H, W = gray.shape
    gh = max(1, H // grid)
    gw = max(1, W // grid)
    g2 = tenengrad(gray)
    S = np.zeros((grid, grid), dtype=np.float32)
    Xc = np.zeros_like(S, dtype=np.float32)
    Yc = np.zeros_like(S, dtype=np.float32)
    for iy in range(grid):
        y0 = iy * gh
        y1 = H if iy == grid - 1 else (iy + 1) * gh
        for ix in range(grid):
            x0 = ix * gw
            x1 = W if ix == grid - 1 else (ix + 1) * gw
            patch = g2[y0:y1, x0:x1]
            val = float(np.mean(patch)) if patch.size else 0.0
            S[iy, ix] = val
            Xc[iy, ix] = 0.5 * (x0 + x1)
            Yc[iy, ix] = 0.5 * (y0 + y1)
    return S, Xc, Yc


def fit_plane_least_squares(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[float, float, float]:
    """Fit plane Z = a*x + b*y + c by least squares."""
    A = np.stack([X.ravel(), Y.ravel(), np.ones_like(X.ravel())], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    a, b, c = coef.tolist()
    return a, b, c


def direction_from_gradient(a: float, b: float) -> Dict[str, Any]:
    # gradient points toward increasing sharpness; we want to tilt toward low-sharpness side → opposite to gradient
    dx = -a
    dy = -b
    mag = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx)))  # degrees, image coords: +x right, +y down
    # Map to cardinal directions primarily
    dirs = []
    if abs(dx) >= abs(dy):
        dirs.append('left' if dx < 0 else 'right')  # because dx negative means suggest more to left
        if abs(dy) > 0.3 * abs(dx):
            dirs.append('up' if dy < 0 else 'down')
    else:
        dirs.append('up' if dy < 0 else 'down')
        if abs(dx) > 0.3 * abs(dy):
            dirs.append('left' if dx < 0 else 'right')
    return {"dx": dx, "dy": dy, "mag": mag, "angle_deg": angle, "dirs": dirs}


def analyze_image(gray: np.ndarray, grid: int = 20) -> Dict[str, Any]:
    H, W = gray.shape
    S, Xc, Yc = grid_sharpness(gray, grid=grid)
    # Normalize coordinates to [-1,1] for scale invariance
    Xn = (Xc - W * 0.5) / max(W * 0.5, 1.0)
    Yn = (Yc - H * 0.5) / max(H * 0.5, 1.0)
    # Log-transform sharpness to reduce dynamic range
    Sz = np.log1p(S)
    a, b, c = fit_plane_least_squares(Xn, Yn, Sz)
    grad = direction_from_gradient(a, b)

    # Global sharpness stats
    s_mean = float(S.mean())
    s_med = float(np.median(S))
    s_std = float(S.std())

    # Exposure/contrast quick checks
    mean_int = float(gray.mean())
    std_int = float(gray.std())
    # clipping ratios
    clip_dark = float((gray <= 2).mean())
    clip_bright = float((gray >= 253).mean())

    # Left/Right/Top/Bottom averages
    left_mean = float(S[:, : S.shape[1] // 2].mean())
    right_mean = float(S[:, S.shape[1] // 2 :].mean())
    top_mean = float(S[: S.shape[0] // 2, :].mean())
    bot_mean = float(S[S.shape[0] // 2 :, :].mean())

    # Normalize gradient magnitude to relative scale
    rel_mag = grad["mag"] / max(abs(a) + abs(b) + 1e-6, 1e-6)
    strength = 'small'
    if rel_mag > 0.6:
        strength = 'strong'
    elif rel_mag > 0.3:
        strength = 'moderate'

    # Simple quality verdict
    notes = []
    verdict = "PASS"
    # thresholds are empirical; Tenengrad patch median depends on image scale; use relative stats
    if s_med < 200.0:  # quite soft
        verdict = "WARN"
        notes.append("图像整体偏虚（锐度较低），请检查对焦或增加光照/提高快门")
    if s_med < 80.0:
        verdict = "FAIL"
    if clip_bright > 0.05:
        verdict = "WARN" if verdict == "PASS" else verdict
        notes.append(f"高光溢出比例 {clip_bright*100:.1f}% 较高，建议降低曝光/缩小光圈")
    if clip_dark > 0.10:
        verdict = "WARN" if verdict == "PASS" else verdict
        notes.append(f"暗部死黑比例 {clip_dark*100:.1f}% 较高，建议增加曝光/补光")

    return {
        "plane_coef": {"a": a, "b": b, "c": c},
        "grid": grid,
        "sharp_stats": {
            "mean": s_mean,
            "median": s_med,
            "std": s_std,
            "left_mean": left_mean,
            "right_mean": right_mean,
            "top_mean": top_mean,
            "bottom_mean": bot_mean,
        },
        "exposure_stats": {
            "mean": mean_int,
            "std": std_int,
            "clip_dark_ratio": clip_dark,
            "clip_bright_ratio": clip_bright,
        },
        "quality": {"verdict": verdict, "notes": notes},
        "gradient": grad,
        "strength": strength,
    }


def heatmap_png(gray: np.ndarray, S: np.ndarray, out_path: Path):
    H, W = gray.shape
    gh = H // S.shape[0]
    gw = W // S.shape[1]
    # Normalize S to [0,255]
    s = S.astype(np.float32)
    s -= s.min() if np.isfinite(s.min()) else 0.0
    d = (s.max() - s.min()) if np.isfinite(s.max()) else 1.0
    s = (s / max(d, 1e-6) * 255.0).astype(np.uint8)
    vis = cv2.resize(s, (W, H), interpolation=cv2.INTER_NEAREST)
    cm = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    # blend with gray image for context
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray3, 0.4, cm, 0.6, 0)
    cv2.imwrite(str(out_path), overlay)


def main():
    ap = argparse.ArgumentParser(description="Tilt/shift adviser based on depth-of-field/sharpness distribution")
    ap.add_argument("--images-dir", default="./figures", help="Folder of images (analyze the latest by name)")
    ap.add_argument("--image", default="", help="Analyze a single image path (overrides --images-dir)")
    ap.add_argument("--grid", type=int, default=20, help="Grid cells per side for sharpness map")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "adviser_out"), help="Output directory for debug overlays and report")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        img_path = Path(args.image)
    else:
        img_dir = Path(args.images_dir)
        files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")])
        if not files:
            print(f"No images found in {img_dir}")
            sys.exit(1)
        img_path = files[-1]  # latest by name

    gray = imread_gray(img_path)
    if gray is None:
        print(f"Failed to read image: {img_path}")
        sys.exit(2)

    # Analyze
    S, Xc, Yc = grid_sharpness(gray, grid=args.grid)
    analysis = analyze_image(gray, grid=args.grid)

    # Save heatmap
    heatmap_path = out_dir / (img_path.stem + "_sharpness.png")
    heatmap_png(gray, S, heatmap_path)

    # Build recommendation text
    dirs = analysis["gradient"]["dirs"]
    strength = analysis["strength"]
    # Direction mapping for user: recommend tilting lens toward low-sharpness side
    if dirs:
        primary = dirs[0]
        if primary in ("up", "down"):
            axis = "水平轴（绕x轴）"
        else:
            axis = "垂直轴（绕y轴）"
        rec = f"建议沿{axis}向‘{primary}’方向微倾（幅度：{strength}）。理由：画面Sharpness在该方向较弱，需要将合焦面朝该侧旋转。"
    else:
        rec = "Sharpness分布近似均匀；优先检查对焦与光圈，或仅做小幅倾斜试探。"

    # Compose report
    report = {
        "image": str(img_path),
        "grid": args.grid,
        "analysis": analysis,
        "recommendation": rec,
        "heatmap": str(heatmap_path),
    }

    out_json = out_dir / (img_path.stem + "_advice.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("==== Tilt/Shift Adviser ====")
    print("Image:", img_path.name)
    print("Quality:", analysis.get("quality", {}).get("verdict", "-"))
    notes = analysis.get("quality", {}).get("notes", [])
    if notes:
        for n in notes:
            print(" -", n)
    print("Recommendation:", rec)
    print("Debug heatmap:", heatmap_path)
    print("Report JSON:", out_json)


if __name__ == "__main__":
    main()
