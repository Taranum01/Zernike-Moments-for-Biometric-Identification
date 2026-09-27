"""
Build the strongest possible ZM-based verification system.

Feature stack:
- ZM-36: 36 Zernike moment magnitudes up to order 8 (currently uses 12 / order 5)
- LBP: 64-bin LBP histogram
- Gabor: 24-orientation Gabor filter bank (3 scales x 8 orientations)
- HOG: 144-d HOG features (8x8 cells, 9 orientations)
- BSIF: 256-d BSIF features (8x8 filters)
- Image statistics: mean, std, median, brightness quantiles (10-d)

Matcher strategies tested:
- Euclidean distance on raw features
- L2-normalized cosine distance
- Per-dimension z-score + Euclidean
- LDA-projected + Euclidean (learned Mahalanobis approx)
- PCA-reduced + Euclidean

Then evaluate on the same 2000-genuine + 2000-impostor pair protocol
across Altered-Easy / Medium / Hard.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------- Feature extractors ----------------
def zm_features(img, order=8):
    """ZM up to given order. Returns magnitudes as a 1-D vector.
    Total length: count of valid (n,m) with (n-|m|) even, n=1..order.
    For order 8 this is 44 (not 36)."""
    import math
    H, W = img.shape
    cy, cx = H / 2, W / 2
    r_max = min(cy, cx)
    yy = np.arange(H).reshape(-1, 1) - cy
    xx = np.arange(W).reshape(1, -1) - cx
    rho = np.sqrt(xx ** 2 + yy ** 2) / r_max
    rho = np.minimum(rho, 1.0)
    theta = np.arctan2(yy, xx)
    mask = rho <= 1.0
    img_norm = img.astype(np.float64)
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max() * 255
    zms = []
    for n in range(1, order + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
                # Radial polynomial R_n^m
                R = np.zeros_like(rho)
                for k in range(int((n - abs(m)) / 2) + 1):
                    R += (((-1) ** k) * math.factorial(n - k) /
                          (math.factorial(k) *
                           math.factorial(int((n + abs(m)) / 2) - k) *
                           math.factorial(int((n - abs(m)) / 2) - k))) * \
                          (rho ** (n - 2 * k))
                Z = R * np.exp(1j * m * theta)
                A = (img_norm * Z * mask).sum()
                zms.append(abs(A))
    return np.array(zms, dtype=np.float32)


def zm_total_length(order):
    return sum(
        1 for n in range(1, order + 1)
        for m in range(-n, n + 1)
        if (n - abs(m)) % 2 == 0
    )


def lbp_histogram(img, n_bins=64):
    """LBP histogram (uniform LBP) — vectorized."""
    img_r = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    # 8-neighbor comparison vectorized via shifts
    center = img_r[1:-1, 1:-1]
    shifts = [(-1, -1), (-1, 0), (-1, 1),
              (0, 1), (1, 1), (1, 0),
              (1, -1), (0, -1)]
    code = np.zeros_like(center, dtype=np.uint8)
    for k, (di, dj) in enumerate(shifts):
        nb = img_r[1+di:img_r.shape[0]-1+di, 1+dj:img_r.shape[1]-1+dj]
        code |= ((nb >= center).astype(np.uint8) << k)
    hist, _ = np.histogram(code.ravel(), bins=n_bins, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    return hist.astype(np.float32)


def gabor_features(img, n_orient=8, n_scales=3):
    img_r = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float64)
    img_norm = (img_r - img_r.mean()) / (img_r.std() + 1e-8)
    feats = []
    for scale in range(n_scales):
        freq = 0.1 * (2 ** scale)
        for o in range(n_orient):
            theta = np.pi * o / n_orient
            kernel = cv2.getGaborKernel((15, 15), 4.0, theta, 1.0 / freq, 0.5, 0, ktype=cv2.CV_64F)
            response = cv2.filter2D(img_norm, -1, kernel)
            feats.append(response.mean())
            feats.append(response.std())
    return np.array(feats, dtype=np.float32)


def hog_features(img):
    """Vectorized HOG: 4x4 cell grid with 9-bin gradient histograms -> 144 features."""
    img_r = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
    gx = cv2.Sobel(img_r, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_r, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx) % np.pi  # 0..pi
    nbins = 9
    cell = 24  # 96/24 = 4 cells; 4*4*9 = 144-D
    n_cells_h = 96 // cell
    n_cells_w = 96 // cell
    feats = np.zeros(n_cells_h * n_cells_w * nbins, dtype=np.float32)
    bin_idx = np.minimum((ang / np.pi * nbins).astype(np.int32), nbins - 1)
    # Vectorized histogram per cell
    for ci in range(n_cells_h):
        for cj in range(n_cells_w):
            i0, j0 = ci * cell, cj * cell
            mag_cell = mag[i0:i0+cell, j0:j0+cell].ravel()
            bin_cell = bin_idx[i0:i0+cell, j0:j0+cell].ravel()
            hist = np.bincount(bin_cell, weights=mag_cell, minlength=nbins)
            flat_idx = ci * n_cells_w * nbins + cj * nbins
            feats[flat_idx:flat_idx + nbins] = hist
    return feats


def image_stats(img):
    img_r = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
    feats = [
        img_r.mean(), img_r.std(), np.median(img_r),
        np.quantile(img_r, 0.1), np.quantile(img_r, 0.25),
        np.quantile(img_r, 0.75), np.quantile(img_r, 0.9),
    ]
    gx = cv2.Sobel(img_r, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_r, cv2.CV_64F, 0, 1, ksize=3)
    feats += [np.abs(gx).mean(), np.abs(gy).mean(), np.sqrt((gx ** 2 + gy ** 2).mean())]
    return np.array(feats, dtype=np.float32)


def bsif_features(img):
    """Texture histogram (BSIF-like; we use a simple LBP-like filter bank as proxy)."""
    img_r = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    feats = []
    # 5x5 filter bank: 16 random filters from local neighborhoods
    rng = np.random.default_rng(0)
    for _ in range(16):
        f = rng.normal(0, 1, (5, 5))
        f = f / (np.abs(f).sum() + 1e-8)
        resp = cv2.filter2D(img_r.astype(np.float64), -1, f)
        feats.append(resp.mean())
        feats.append(resp.std())
        feats.append(np.median(resp))
    return np.array(feats, dtype=np.float32)


# ---------------- Matching ----------------
def build_pairs(real_subjects, real_hands, real_fingers,
                alt_subjects, alt_hands, alt_fingers,
                n_genuine=2000, n_impostor=2000, seed=42):
    rng = np.random.default_rng(seed)
    real_key_to_idx = {}
    for i, (s, h, f) in enumerate(zip(real_subjects, real_hands, real_fingers)):
        real_key_to_idx[(s, h, f)] = i
    alt_key_to_idx = {}
    for j, (s, h, f) in enumerate(zip(alt_subjects, alt_hands, alt_fingers)):
        if (s, h, f) not in alt_key_to_idx:
            alt_key_to_idx[(s, h, f)] = j
    genuine_pairs = []
    for (s, h, f), real_idx in real_key_to_idx.items():
        if (s, h, f) in alt_key_to_idx:
            genuine_pairs.append((real_idx, alt_key_to_idx[(s, h, f)]))
    rng.shuffle(genuine_pairs)
    n_real = len(real_subjects)
    impostor_pairs = []
    seen = set()
    while len(impostor_pairs) < max(n_impostor, len(genuine_pairs)):
        i = rng.integers(0, n_real)
        j = rng.integers(0, n_real)
        if i == j: continue
        if (real_subjects[i], real_hands[i], real_fingers[i]) == (real_subjects[j], real_hands[j], real_fingers[j]): continue
        if real_subjects[i] == real_subjects[j]: continue
        pair = (i, j)
        if pair in seen: continue
        seen.add(pair)
        impostor_pairs.append(pair)
    rng.shuffle(impostor_pairs)
    n = min(len(genuine_pairs), len(impostor_pairs), n_genuine, n_impostor)
    return genuine_pairs[:n], impostor_pairs[:n]


def compute_metrics(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    fnr = 1 - tpr
    idx_eer = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2)
    best_acc = 0
    for t in thr:
        pred = (scores > t).astype(int)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc = acc
    return roc_auc, eer, best_acc


def make_score_features(real_feats, alt_feats, pairs):
    """Compute pairwise distances. real_feats: (N_real, D); alt_feats: (N_alt, D)."""
    fa = real_feats[[p[0] for p in pairs]]
    fb = alt_feats[[p[1] for p in pairs]]
    return fa, fb


# ---------------- File loading ----------------
def parse_real_filename(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    subject = int(parts[0])
    rest = parts[1]
    tokens = rest.split("_")
    return subject, tokens[0], tokens[1], tokens[2]


def parse_altered_filename(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    subject = int(parts[0])
    rest = parts[1]
    tokens = rest.split("_")
    return subject, tokens[0], tokens[1], tokens[2], tokens[-1]


REAL_DIR = "Real"
DIRS = {
    "Easy": "SOCOFing/Altered/Altered-Easy",
    "Medium": "SOCOFing/Altered/Altered-Medium",
    "Hard": "SOCOFing/Altered/Altered-Hard",
}
OUT_DIR = "results_strong_zm"
os.makedirs(OUT_DIR, exist_ok=True)


def load_imgs_from_dir(dir_path):
    files = sorted(os.listdir(dir_path))
    imgs = []
    return files, imgs


def main():
    # Load real images
    print("Loading Real images...", flush=True)
    real_files = sorted(os.listdir(REAL_DIR))
    real_imgs = []
    real_subjects, real_hands, real_fingers = [], [], []
    for fn in real_files:
        img = cv2.imread(os.path.join(REAL_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((96, 96), dtype=np.uint8)
        else:
            img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        real_imgs.append(img)
        s, _, h, f = parse_real_filename(fn)
        real_subjects.append(s); real_hands.append(h); real_fingers.append(f)
    real_imgs = np.array(real_imgs)
    real_subjects = np.array(real_subjects); real_hands = np.array(real_hands); real_fingers = np.array(real_fingers)
    print(f"  Real: {real_imgs.shape}", flush=True)

    # Compute real features (cache)
    cache_real = os.path.join(OUT_DIR, "real_features.npz")
    zm_dim = zm_total_length(8)  # 44 for order 8
    print(f"ZM order=8 total length = {zm_dim}", flush=True)
    if os.path.exists(cache_real):
        print("Loading cached real features...", flush=True)
        data = np.load(cache_real)
        real_zm36 = data["zm44"]
        real_lbp = data["lbp"]
        real_gab = data["gab"]
        real_hog = data["hog"]
        real_stats = data["stats"]
    else:
        print("Computing real features (this takes a while)...", flush=True)
        real_zm36 = np.zeros((len(real_imgs), zm_dim), dtype=np.float32)
        real_lbp = np.zeros((len(real_imgs), 64), dtype=np.float32)
        real_gab = np.zeros((len(real_imgs), 48), dtype=np.float32)
        real_hog = np.zeros((len(real_imgs), 144), dtype=np.float32)  # approximate size for HOG
        real_stats = np.zeros((len(real_imgs), 10), dtype=np.float32)
        for i, img in enumerate(tqdm(real_imgs)):
            real_zm36[i] = zm_features(img, order=8)
            real_lbp[i] = lbp_histogram(img)
            real_gab[i] = gabor_features(img)
            real_hog[i] = hog_features(img)[:144]  # truncate if larger
            real_stats[i] = image_stats(img)
        np.savez_compressed(cache_real, zm44=real_zm36, lbp=real_lbp, gab=real_gab, hog=real_hog, stats=real_stats)
        print(f"Cached real features to {cache_real}", flush=True)

    print(f"  ZM-36 shape: {real_zm36.shape}")
    print(f"  LBP shape:   {real_lbp.shape}")
    print(f"  Gabor shape: {real_gab.shape}")
    print(f"  HOG shape:   {real_hog.shape}")
    print(f"  Stats shape: {real_stats.shape}")

    # Process each altered level
    results = {}
    for level_name, alt_dir in DIRS.items():
        print(f"\n===== Altered-{level_name} =====", flush=True)
        cache_alt = os.path.join(OUT_DIR, f"altered_features_{level_name}.npz")
        if os.path.exists(cache_alt):
            print("  Loading cached altered features...", flush=True)
            data = np.load(cache_alt)
            alt_zm36 = data["zm44"]
            alt_lbp = data["lbp"]
            alt_gab = data["gab"]
            alt_hog = data["hog"]
            alt_stats = data["stats"]
            alt_subjects = data["subjects"]
            alt_hands = data["hands"]
            alt_fingers = data["fingers"]
        else:
            print("  Loading altered images...", flush=True)
            alt_files = sorted(os.listdir(alt_dir))
            alt_imgs = []
            alt_subjects, alt_hands, alt_fingers = [], [], []
            for fn in alt_files:
                img = cv2.imread(os.path.join(alt_dir, fn), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = np.zeros((96, 96), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
                alt_imgs.append(img)
                s, _, h, f, _ = parse_altered_filename(fn)
                alt_subjects.append(s); alt_hands.append(h); alt_fingers.append(f)
            alt_imgs = np.array(alt_imgs)
            alt_subjects = np.array(alt_subjects); alt_hands = np.array(alt_hands); alt_fingers = np.array(alt_fingers)
            print(f"  Altered-{level_name}: {alt_imgs.shape}", flush=True)

            print("  Computing altered features...", flush=True)
            alt_zm36 = np.zeros((len(alt_imgs), zm_dim), dtype=np.float32)
            alt_lbp = np.zeros((len(alt_imgs), 64), dtype=np.float32)
            alt_gab = np.zeros((len(alt_imgs), 48), dtype=np.float32)
            alt_hog = np.zeros((len(alt_imgs), 144), dtype=np.float32)
            alt_stats = np.zeros((len(alt_imgs), 10), dtype=np.float32)
            for i, img in enumerate(tqdm(alt_imgs)):
                alt_zm36[i] = zm_features(img, order=8)
                alt_lbp[i] = lbp_histogram(img)
                alt_gab[i] = gabor_features(img)
                alt_hog[i] = hog_features(img)[:144]
                alt_stats[i] = image_stats(img)
            np.savez_compressed(cache_alt,
                zm44=alt_zm36, lbp=alt_lbp, gab=alt_gab, hog=alt_hog, stats=alt_stats,
                subjects=alt_subjects, hands=alt_hands, fingers=alt_fingers)

        # ----- Now build feature combinations -----
        feature_combos = {
            "ZM-12 (order 5)": [real_zm36[:, :12], alt_zm36[:, :12]],
            f"ZM-{zm_dim} (order 8)": [real_zm36, alt_zm36],
            f"ZM-{zm_dim} + LBP + Gabor + HOG + Stats":
                [np.hstack([real_zm36, real_lbp, real_gab, real_hog, real_stats]),
                 np.hstack([alt_zm36, alt_lbp, alt_gab, alt_hog, alt_stats])],
            f"ZM-{zm_dim} + HOG":
                [np.hstack([real_zm36, real_hog]),
                 np.hstack([alt_zm36, alt_hog])],
        }

        # Build pairs
        genuine, impostor = build_pairs(
            real_subjects, real_hands, real_fingers,
            alt_subjects, alt_hands, alt_fingers,
            n_genuine=2000, n_impostor=2000, seed=42,
        )
        all_pairs = genuine + impostor
        labels = np.array([1] * len(genuine) + [0] * len(impostor))

        level_results = {}
        for combo_name, (rf, af) in feature_combos.items():
            # Euclidean distance (with optional normalization)
            fa = rf[[p[0] for p in all_pairs]]
            fb = af[[p[1] for p in all_pairs]]

            # Standardize per dimension on real-only
            scaler = StandardScaler().fit(rf)
            fa_s = scaler.transform(fa)
            fb_s = scaler.transform(fb)

            # PCA-reduced then standardize
            for n_pca in [None, 64, 32]:
                if n_pca is None:
                    fa_used, fb_used = fa_s, fb_s
                    tag = "zscore"
                else:
                    n = min(n_pca, fa_s.shape[1], fa_s.shape[0])
                    pca = PCA(n_components=n).fit(fa_s)
                    fa_used = pca.transform(fa_s)
                    fb_used = pca.transform(fb_s)
                    tag = f"PCA-{n}"

                # Raw euclidean distance (higher = less similar -> use -d as score)
                d = np.linalg.norm(fa_used - fb_used, axis=1)
                scores = -d
                auc_v, eer_v, acc_v = compute_metrics(scores, labels)
                key = f"{combo_name} | {tag}"
                level_results[key] = {"AUC": auc_v, "EER": eer_v, "BestAcc": acc_v, "Dim": int(fa.shape[1])}
                print(f"    {key:60s} | dim={fa.shape[1]:4d} | AUC={auc_v:.4f}  EER={eer_v:.4f}  BestAcc={acc_v:.4f}", flush=True)

        results[level_name] = level_results

    # Save
    with open(os.path.join(OUT_DIR, "strong_zm_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n\n========= SUMMARY =========")
    for level, lr in results.items():
        print(f"\n  Altered-{level}:")
        for k, v in sorted(lr.items(), key=lambda kv: -kv[1]["AUC"]):
            print(f"    AUC={v['AUC']:.4f}  EER={v['EER']:.4f}  BestAcc={v['BestAcc']:.4f}  | {k}")


if __name__ == "__main__":
    main()