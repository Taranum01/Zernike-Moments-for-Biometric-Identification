"""
Enhanced same-finger verification: ZM-12 + augmented features, per-alteration type.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

sys.path.insert(0, ".")
from compute_zernike_batch import compute_zernike_moments

REAL_DIR = "Real"
ALTERED_EASY = "SOCOFing/Altered/Altered-Easy"
ALTERED_MED = "SOCOFing/Altered/Altered-Medium"
ALTERED_HARD = "SOCOFing/Altered/Altered-Hard"
OUT_DIR = "results_verification"
os.makedirs(OUT_DIR, exist_ok=True)

ZM = ["Z00","Z11","Z20","Z22","Z31","Z33","Z40","Z42","Z44","Z51","Z53","Z55"]


def parse_altered_filename(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    subject = int(parts[0])
    rest = parts[1]
    tokens = rest.split("_")
    sex = tokens[0]
    hand = tokens[1]
    finger = tokens[2]
    alteration = tokens[-1]
    return subject, sex, hand, finger, alteration


def parse_real_filename(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    subject = int(parts[0])
    rest = parts[1]
    tokens = rest.split("_")
    sex = tokens[0]
    hand = tokens[1]
    finger = tokens[2]
    return subject, sex, hand, finger


def compute_zm(img, order=5):
    zm = compute_zernike_moments(img, order=order)
    zm = zm + [0.0] * (12 - len(zm))
    return zm


def lbp_histogram(img, n_bins=256):
    """Standard LBP histogram."""
    h, w = img.shape
    neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                 (1, 0), (1, -1), (0, -1), (-1, -1)]
    patterns = []
    for dy, dx in neighbors:
        shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        patterns.append((shifted >= img).astype(np.uint8))
    code = np.zeros_like(img, dtype=np.uint8)
    for k, p in enumerate(patterns):
        code |= (p << k)
    hist, _ = np.histogram(code, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def gabor_features(img, n_orient=8, n_scales=3):
    """Multi-orientation, multi-scale Gabor filter responses (mean, std)."""
    feats = []
    for scale in range(n_scales):
        for orient in range(n_orient):
            theta = np.pi * orient / n_orient
            sigma = 1.0 + scale * 0.5
            lam = 4.0 + scale * 2.0
            kernel = cv2.getGaborKernel((15, 15), sigma, theta, lam, 0.5, 0)
            resp = cv2.filter2D(img, cv2.CV_32F, kernel)
            feats.append(resp.mean())
            feats.append(resp.std())
    return np.array(feats)


def compute_features(img):
    """Combined feature: ZM(12) + LBP(64) + Gabor(48) = 124 dimensions."""
    zm = compute_zm(img, order=5)
    # Resize to 96x96 for consistent LBP/Gabor
    img_r = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    lbp = lbp_histogram(img_r, n_bins=64)
    gab = gabor_features(img_r, n_orient=8, n_scales=3)
    return np.concatenate([zm, lbp, gab])


def build_pairs(real_features, real_subjects, real_hands, real_fingers,
                alt_features, alt_subjects, alt_hands, alt_fingers, alt_alts,
                n_genuine=2000, n_impostor=2000, seed=42,
                alt_filter=None, exclude_same_subject=True):
    """Build genuine and impostor pairs with optional alteration-type filter."""
    rng = np.random.default_rng(seed)

    # Group real by (sub, hand, finger)
    real_key_to_idx = {}
    for i, (s, h, f) in enumerate(zip(real_subjects, real_hands, real_fingers)):
        real_key_to_idx[(s, h, f)] = i

    # Group altered: include alteration type
    alt_key_to_alt = {}  # (sub, hand, finger) -> list of (idx, alt)
    for j, (s, h, f, a) in enumerate(zip(alt_subjects, alt_hands, alt_fingers, alt_alts)):
        if alt_filter is not None and a not in alt_filter:
            continue
        alt_key_to_alt.setdefault((s, h, f), []).append((j, a))

    # Genuine pairs: pick first altered of each (sub, hand, finger)
    genuine_pairs = []
    for (s, h, f), real_idx in real_key_to_idx.items():
        if (s, h, f) in alt_key_to_alt:
            j, _ = alt_key_to_alt[(s, h, f)][0]
            genuine_pairs.append((real_idx, j))
    rng.shuffle(genuine_pairs)
    genuine_pairs = genuine_pairs[:n_genuine]

    # Impostor: real vs real (different (sub, hand, finger))
    n_real = len(real_subjects)
    impostor_pairs = []
    seen = set()
    while len(impostor_pairs) < n_impostor:
        i = rng.integers(0, n_real)
        j = rng.integers(0, n_real)
        if i == j: continue
        if (real_subjects[i], real_hands[i], real_fingers[i]) == (real_subjects[j], real_hands[j], real_fingers[j]): continue
        # Optionally exclude same subject (different finger)
        if exclude_same_subject and real_subjects[i] == real_subjects[j]: continue
        pair = (i, j)
        if pair in seen: continue
        seen.add(pair)
        impostor_pairs.append(pair)

    return genuine_pairs, impostor_pairs


def evaluate(feat_real, feat_alt, real_subjects, real_hands, real_fingers,
             alt_subjects, alt_hands, alt_fingers, alt_alts,
             name, n_genuine=2000, n_impostor=2000, seed=42,
             alt_filter=None, exclude_same_subject=True):
    """Run verification evaluation."""
    genuine_pairs, impostor_pairs = build_pairs(
        feat_real, real_subjects, real_hands, real_fingers,
        feat_alt, alt_subjects, alt_hands, alt_fingers, alt_alts,
        n_genuine, n_impostor, seed, alt_filter, exclude_same_subject,
    )
    if not genuine_pairs or not impostor_pairs:
        return None

    n_g = len(genuine_pairs)
    n_i = len(impostor_pairs)
    gen_d = np.array([np.linalg.norm(feat_real[i] - feat_alt[j]) for i, j in genuine_pairs])
    imp_d = np.array([np.linalg.norm(feat_real[i] - feat_real[j]) for i, j in impostor_pairs])

    labels = np.concatenate([np.ones(n_g), np.zeros(n_i)])
    scores = -np.concatenate([gen_d, imp_d])
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
    idx_low = np.argmin(np.abs(fpr - 0.01))
    acc_at_low = (tpr[idx_low] * n_g + (1 - 0.01) * n_i) / (n_g + n_i)

    return {
        "name": name,
        "n_genuine": n_g,
        "n_impostor": n_i,
        "genuine_mean": float(np.mean(gen_d)),
        "impostor_mean": float(np.mean(imp_d)),
        "auc": roc_auc,
        "eer": eer,
        "best_acc": best_acc,
        "acc_at_fpr_0.01": float(acc_at_low),
    }


def main():
    print("Loading Real features...")
    df = pd.read_csv("zernike_moments.csv")
    parsed = df["filename"].apply(parse_real_filename)
    df["subject"] = parsed.apply(lambda x: x[0])
    df["hand"] = parsed.apply(lambda x: x[2])
    df["finger"] = parsed.apply(lambda x: x[3])
    real_subjects = df["subject"].values
    real_hands = df["hand"].values
    real_fingers = df["finger"].values
    real_features_zm = df[ZM].values
    print(f"Real: {len(real_subjects)} images, {len(np.unique(real_subjects))} subjects")

    # Load altered ZM
    data = np.load("zm_cache_Altered-Easy.npz", allow_pickle=True)
    altered_easy_zm = data["features"]; altered_easy_sub = data["subjects"]
    altered_easy_hand = data["hands"]; altered_easy_fing = data["fingers"]
    data = np.load("zm_cache_Altered-Medium.npz", allow_pickle=True)
    altered_med_zm = data["features"]; altered_med_sub = data["subjects"]
    altered_med_hand = data["hands"]; altered_med_fing = data["fingers"]
    data = np.load("zm_cache_Altered-Hard.npz", allow_pickle=True)
    altered_hard_zm = data["features"]; altered_hard_sub = data["subjects"]
    altered_hard_hand = data["hands"]; altered_hard_fing = data["fingers"]

    # Parse alteration types from filenames (alphabetical order, all stored)
    def get_alts(dir_path):
        files = sorted(os.listdir(dir_path))
        alts = []
        for fn in files:
            try:
                _, _, _, _, a = parse_altered_filename(fn)
                alts.append(a)
            except:
                alts.append("?")
        return np.array(alts)

    altered_easy_alts = get_alts(ALTERED_EASY)
    altered_med_alts = get_alts(ALTERED_MED)
    altered_hard_alts = get_alts(ALTERED_HARD)
    print(f"Altered-Easy alterations: {set(altered_easy_alts.tolist())}")
    print(f"Altered-Medium alterations: {set(altered_med_alts.tolist())}")
    print(f"Altered-Hard alterations: {set(altered_hard_alts.tolist())}")

    # Build augmented features for Real (ZM + LBP + Gabor) - reuse images
    print("\nComputing augmented features for Real...")
    real_features_aug = []
    for fn in tqdm(df["filename"].values):
        img = cv2.imread(os.path.join(REAL_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            real_features_aug.append(np.zeros(12 + 64 + 48))
            continue
        real_features_aug.append(compute_features(img))
    real_features_aug = np.array(real_features_aug)
    print(f"Real augmented features: {real_features_aug.shape}")

    # Build augmented features for Altered-Easy (sample 5000 for speed)
    print("\nComputing augmented features for Altered-Easy...")
    alt_easy_aug = []
    files_easy = sorted(os.listdir(ALTERED_EASY))
    for j, fn in enumerate(tqdm(files_easy)):
        img = cv2.imread(os.path.join(ALTERED_EASY, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            alt_easy_aug.append(np.zeros(12 + 64 + 48))
            continue
        alt_easy_aug.append(compute_features(img))
    alt_easy_aug = np.array(alt_easy_aug)
    print(f"Altered-Easy augmented: {alt_easy_aug.shape}")

    # === Experiments ===
    results = []

    print("\n" + "="*70)
    print("EXPERIMENT 1: ZM-12 only (per difficulty)")
    print("="*70)
    for level_name, alt_zm, alt_sub, alt_hand, alt_fing, alt_alts in [
        ("Easy", altered_easy_zm, altered_easy_sub, altered_easy_hand, altered_easy_fing, altered_easy_alts),
        ("Medium", altered_med_zm, altered_med_sub, altered_med_hand, altered_med_fing, altered_med_alts),
        ("Hard", altered_hard_zm, altered_hard_sub, altered_hard_hand, altered_hard_fing, altered_hard_alts),
    ]:
        res = evaluate(
            real_features_zm, alt_zm,
            real_subjects, real_hands, real_fingers,
            alt_sub, alt_hand, alt_fing, alt_alts,
            name=f"ZM-12 / Altered-{level_name}",
        )
        results.append({"exp": "ZM-12", **res})
        for k, v in res.items():
            print(f"  {k}: {v}")

    print("\n" + "="*70)
    print("EXPERIMENT 2: ZM-12 + LBP + Gabor (Easy only)")
    print("="*70)
    res = evaluate(
        real_features_aug, alt_easy_aug,
        real_subjects, real_hands, real_fingers,
        altered_easy_sub, altered_easy_hand, altered_easy_fing, altered_easy_alts,
        name="ZM+LBP+Gabor / Altered-Easy",
    )
    results.append({"exp": "Augmented", **res})
    for k, v in res.items():
        print(f"  {k}: {v}")

    print("\n" + "="*70)
    print("EXPERIMENT 3: ZM-12 only per alteration type (Easy)")
    print("="*70)
    for alt_type in ["CR", "Obl", "Zcut"]:
        res = evaluate(
            real_features_zm, altered_easy_zm,
            real_subjects, real_hands, real_fingers,
            altered_easy_sub, altered_easy_hand, altered_easy_fing, altered_easy_alts,
            name=f"ZM-12 / Altered-Easy-{alt_type}",
            alt_filter={alt_type},
        )
        results.append({"exp": "Per-alt-type", **res})
        for k, v in res.items():
            print(f"  {k}: {v}")

    # Save results
    with open(os.path.join(OUT_DIR, "verification_v2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # === Plot ROC for all difficulty levels (ZM-12 only) ===
    fig, ax = plt.subplots(figsize=(7, 6))
    for level_name, alt_zm, alt_sub, alt_hand, alt_fing, alt_alts in [
        ("Easy", altered_easy_zm, altered_easy_sub, altered_easy_hand, altered_easy_fing, altered_easy_alts),
        ("Medium", altered_med_zm, altered_med_sub, altered_med_hand, altered_med_fing, altered_med_alts),
        ("Hard", altered_hard_zm, altered_hard_sub, altered_hard_hand, altered_hard_fing, altered_hard_alts),
    ]:
        genuine_pairs, impostor_pairs = build_pairs(
            real_features_zm, real_subjects, real_hands, real_fingers,
            alt_zm, alt_sub, alt_hand, alt_fing, alt_alts,
            n_genuine=2000, n_impostor=2000, seed=42,
            exclude_same_subject=True,
        )
        gen_d = np.array([np.linalg.norm(real_features_zm[i] - alt_zm[j]) for i, j in genuine_pairs])
        imp_d = np.array([np.linalg.norm(real_features_zm[i] - real_features_zm[j]) for i, j in impostor_pairs])
        labels = np.concatenate([np.ones_like(gen_d), np.zeros_like(imp_d)])
        scores = -np.concatenate([gen_d, imp_d])
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"Altered-{level_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Same-finger verification ROC (ZM-12)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_verification_roc_v2.png"), dpi=200)
    plt.close()

    # === Plot distance distributions for Altered-Easy ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (level_name, alt_zm, alt_sub, alt_hand, alt_fing, alt_alts) in zip(axes, [
        ("Easy", altered_easy_zm, altered_easy_sub, altered_easy_hand, altered_easy_fing, altered_easy_alts),
        ("Medium", altered_med_zm, altered_med_sub, altered_med_hand, altered_med_fing, altered_med_alts),
        ("Hard", altered_hard_zm, altered_hard_sub, altered_hard_hand, altered_hard_fing, altered_hard_alts),
    ]):
        genuine_pairs, impostor_pairs = build_pairs(
            real_features_zm, real_subjects, real_hands, real_fingers,
            alt_zm, alt_sub, alt_hand, alt_fing, alt_alts,
            n_genuine=2000, n_impostor=2000, seed=42, exclude_same_subject=True,
        )
        gen_d = np.array([np.linalg.norm(real_features_zm[i] - alt_zm[j]) for i, j in genuine_pairs])
        imp_d = np.array([np.linalg.norm(real_features_zm[i] - real_features_zm[j]) for i, j in impostor_pairs])
        ax.hist(gen_d, bins=50, alpha=0.6, label=f"Genuine (mean={gen_d.mean():.1f})", color="green")
        ax.hist(imp_d, bins=50, alpha=0.6, label=f"Impostor (mean={imp_d.mean():.1f})", color="red")
        ax.set_xlabel("Euclidean distance")
        ax.set_ylabel("Count")
        ax.set_title(f"Altered-{level_name}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("Same-finger verification distance distributions (ZM-12)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_distance_distributions.png"), dpi=200)
    plt.close()

    print(f"\nResults saved to {OUT_DIR}/verification_v2_results.json")


if __name__ == "__main__":
    main()