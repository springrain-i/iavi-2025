from __future__ import annotations
import argparse
from pathlib import Path
import json
import csv
import sys
import cv2
import numpy as np
from typing import Dict, Any, List


def imread_gray(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def list_images(img_dir: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> list[Path]:
    return [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]


def downscale(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)


def laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)
    return float(lap.var())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g2 = gx*gx + gy*gy
    return float(np.mean(g2))


def rms_contrast(gray: np.ndarray) -> float:
    return float(gray.std())


def exposure_stats(gray: np.ndarray) -> Dict[str, float]:
    mean = float(gray.mean())
    under = float((gray <= 5).sum()) / gray.size
    over = float((gray >= 250).sum()) / gray.size
    return {"mean": mean, "under_pct": under, "over_pct": over}


def entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    nz = hist[hist > 0]
    return float(-(nz * np.log2(nz)).sum())


def noise_sigma_mad_lap(gray: np.ndarray) -> float:
    # Estimate noise via MAD on Laplacian
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    sigma = 1.4826 * mad
    return float(sigma)


def keypoint_richness(gray: np.ndarray, feature: str = "ORB") -> Dict[str, float]:
    if feature.upper() == "SIFT":
        try:
            det = cv2.SIFT_create()
        except Exception:
            det = cv2.ORB_create(nfeatures=4000)
    else:
        det = cv2.ORB_create(nfeatures=4000)
    kps = det.detect(gray, None)
    cnt = len(kps)
    resp = float(np.mean([kp.response for kp in kps])) if cnt > 0 else 0.0
    return {"kp_count": float(cnt), "kp_response": resp}


def compute_metrics(gray: np.ndarray, feature: str = "ORB") -> Dict[str, float]:
    lpv = laplacian_variance(gray)
    tng = tenengrad(gray)
    con = rms_contrast(gray)
    exp = exposure_stats(gray)
    ent = entropy(gray)
    noi = noise_sigma_mad_lap(gray)
    kpr = keypoint_richness(gray, feature=feature)
    m: Dict[str, float] = {
        "lap_var": lpv,
        "tenengrad": tng,
        "contrast": con,
        "mean": exp["mean"],
        "under_pct": exp["under_pct"],
        "over_pct": exp["over_pct"],
        "entropy": ent,
        "noise_sigma": noi,
        "kp_count": kpr["kp_count"],
        "kp_response": kpr["kp_response"],
    }
    return m


def robust_min_max(values: np.ndarray, lo=10.0, hi=90.0) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(values, lo))
    vmax = float(np.percentile(values, hi))
    if vmax <= vmin:
        vmin, vmax = float(values.min()), float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def quality_scores(all_metrics: List[Dict[str, float]]) -> List[float]:
    # Convert to arrays
    keys = list(all_metrics[0].keys()) if all_metrics else []
    arr = {k: np.array([m[k] for m in all_metrics], dtype=np.float64) for k in keys}

    # Define direction (True: higher better)
    higher_better = {
        "lap_var": True,
        "tenengrad": True,
        "contrast": True,
        "entropy": True,
        "kp_count": True,
        "kp_response": True,
        "noise_sigma": False,
        "under_pct": False,
        "over_pct": False,
        # mean: penalize deviation from mid grayscale (~127.5)
    }

    # Normalize metrics into [0,1] using robust percentiles
    norm: Dict[str, np.ndarray] = {}
    for k, v in arr.items():
        if k == "mean":
            # score higher if close to mid tone
            score = 1.0 - np.minimum(1.0, np.abs(v - 127.5) / 127.5)
            norm[k] = score
            continue
        vmin, vmax = robust_min_max(v)
        if higher_better.get(k, True):
            score = (v - vmin) / max(vmax - vmin, 1e-6)
        else:
            score = (vmax - v) / max(vmax - vmin, 1e-6)
        norm[k] = np.clip(score, 0.0, 1.0)

    # Weights
    weights = {
        "lap_var": 0.25,
        "tenengrad": 0.20,
        "contrast": 0.10,
        "entropy": 0.10,
        "kp_count": 0.15,
        "kp_response": 0.05,
        "noise_sigma": 0.10,
        "under_pct": 0.025,
        "over_pct": 0.025,
        "mean": 0.10,
    }

    total_w = sum(weights.values())
    scores = []
    for i in range(len(all_metrics)):
        s = 0.0
        for k, w in weights.items():
            s += float(norm[k][i]) * w
        scores.append(s / total_w)
    return scores


def verdict_and_notes(m: Dict[str, float], score: float) -> tuple[str, str]:
    issues = []
    if m["lap_var"] < 20 and m["tenengrad"] < 100:
        issues.append("Blur/low sharpness")
    if m["under_pct"] > 0.05:
        issues.append("Underexposed areas")
    if m["over_pct"] > 0.02:
        issues.append("Overexposed areas")
    if m["contrast"] < 15:
        issues.append("Low contrast")
    if m["kp_count"] < 200:
        issues.append("Few features (low texture)")
    if m["entropy"] < 4.0:
        issues.append("Low information content")

    if score >= 0.6 and not issues:
        return "PASS", "OK"
    if score >= 0.5:
        note = ", ".join(issues[:2]) if issues else "Marginal but usable"
        return "WARN", note
    return "FAIL", ", ".join(issues[:3]) if issues else "Low quality"


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Image quality analysis for axis-shift capture")
    ap.add_argument("--images-dir", default="figures", help="Folder of input images")
    ap.add_argument("--out-csv", default="quality.csv", help="Path to write CSV report")
    ap.add_argument("--out-json", default="", help="Optional path to write JSON report")
    ap.add_argument("--feature", choices=["ORB", "SIFT"], default="ORB", help="Feature type for richness metric")
    ap.add_argument("--scale", type=float, default=1.0, help="Resize factor for analysis (speed)")
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    files = list_images(img_dir)
    if len(files) == 0:
        print(f"No images found in {img_dir}")
        sys.exit(1)

    print(f"Analyzing {len(files)} images from {img_dir} ...")
    metrics_list: List[Dict[str, float]] = []
    names: List[str] = []

    for p in files:
        gray = imread_gray(p)
        if gray is None:
            print(f"[skip] Failed to read {p}")
            continue
        gray = downscale(gray, args.scale)
        m = compute_metrics(gray, feature=args.feature)
        metrics_list.append(m)
        names.append(p.name)

    if not metrics_list:
        print("No valid images for analysis.")
        sys.exit(2)

    scores = quality_scores(metrics_list)
    rows: List[Dict[str, Any]] = []
    for name, m, s in zip(names, metrics_list, scores):
        verdict, note = verdict_and_notes(m, s)
        row = {"image": name, **{k: round(v, 6) for k, v in m.items()}, "score": round(float(s), 4), "verdict": verdict, "note": note}
        rows.append(row)

    # Sort by score descending
    rows.sort(key=lambda r: r["score"], reverse=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, rows)
    print(f"Wrote CSV -> {out_csv}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON -> {out_json}")

    # Print top/bottom summary
    top = rows[0]
    bot = rows[-1]
    print("Best:", top["image"], "score=", top["score"], "verdict=", top["verdict"], "note=", top["note"])
    print("Worst:", bot["image"], "score=", bot["score"], "verdict=", bot["verdict"], "note=", bot["note"])


if __name__ == "__main__":
    main()
