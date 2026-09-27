"""
Robustness experiment for §IV.F of camera-ready.

Tests ZM-44 same-finger verification under:
  (a) Additive Gaussian noise with sigma in {10, 20, 30} (uint8 scale 0..255).
  (b) Random binary mask removing 50% of foreground pixels.
Uses the same 2000-gen / 2000-imp pair protocol on Altered-Easy.
"""
import os, json
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

REAL_DIR = "Real"
ALT_DIR = "SOCOFing/Altered/Altered-Easy"


def parse_real(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    s = int(parts[0]); tokens = parts[1].split("_")
    return s, tokens[0], tokens[1], tokens[2]


def parse_alt(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    s = int(parts[0]); tokens = parts[1].split("_")
    return s, tokens[0], tokens[1], tokens[2], tokens[-1]


def zm_features(img, order=8):
    """Same as run_strong_zm.zm_features."""
    import math
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
    H, W = img.shape
    cx = (W - 1) / 2.0; cy = (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    xx = xx - cx
    yy = yy - cy
    rho = np.sqrt(xx ** 2 + yy ** 2)
    r_max = max(cx, cy)
    rho = rho / r_max
    mask = (rho <= 1.0)
    rho[~mask] = 0
    xxn = np.zeros_like(xx); yyn = np.zeros_like(yy)
    xxn[mask] = xx[mask] / r_max; yyn[mask] = yy[mask] / r_max
    theta = np.arctan2(yyn, xxn)
    img_norm = img * mask
    zms = [abs(img_norm.sum())]  # Z00
    for n in range(1, order + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
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


def degrade(img, sigma=None, mask_frac=None):
    img = img.copy()
    if sigma is not None:
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if mask_frac is not None:
        m = np.random.rand(*img.shape) < mask_frac
        img[m] = 0
    return img


def build_pairs(real_s, real_h, real_f, a_s, a_h, a_f, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    real_key = {(int(s), str(h), str(f)): i for i, (s, h, f) in enumerate(zip(real_s, real_h, real_f))}
    alt_first = {}
    for j, (s, h, f) in enumerate(zip(a_s, a_h, a_f)):
        k = (int(s), str(h), str(f))
        if k not in alt_first: alt_first[k] = j
    genuine = [(real_key[k], alt_first[k]) for k in real_key if k in alt_first]
    rng.shuffle(genuine); genuine = genuine[:n]
    impostor = []; seen = set(); nr = len(real_s)
    while len(impostor) < n and len(impostor) < nr * (nr - 1):
        i = int(rng.integers(0, nr)); j = int(rng.integers(0, nr))
        if i == j: continue
        if real_s[i] == real_s[j]: continue
        if (str(real_h[i]), str(real_f[i])) == (str(real_h[j]), str(real_f[j])): continue
        p = (i, j)
        if p in seen: continue
        seen.add(p); impostor.append(p)
    return genuine, impostor[:n]


def evaluate(real_imgs, alt_imgs, genuine_pairs, impostor_pairs, sigma=None, mask_frac=None):
    rng = np.random.default_rng(7)
    # Extract features on degraded images
    real_zm_list = [zm_features(degrade(img, sigma=sigma, mask_frac=mask_frac)) for img in real_imgs]
    alt_zm_list = [zm_features(degrade(img, sigma=sigma, mask_frac=mask_frac)) for img in alt_imgs]
    real_zm = np.array(real_zm_list); alt_zm = np.array(alt_zm_list)
    scaler = StandardScaler().fit(real_zm)
    real_z = scaler.transform(real_zm); alt_z = scaler.transform(alt_zm)
    gen_d = np.array([np.linalg.norm(real_z[r] - alt_z[a]) for r, a in genuine_pairs])
    imp_d = np.array([np.linalg.norm(real_z[i] - real_z[j]) for i, j in impostor_pairs])
    labels = np.concatenate([np.ones(len(gen_d)), np.zeros(len(imp_d))])
    scores = -np.concatenate([gen_d, imp_d])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    return roc_auc


def main():
    print("Loading images...")
    real_files = sorted(os.listdir(REAL_DIR))
    real_imgs = []
    real_s = []; real_h = []; real_f = []
    for fn in real_files:
        img = cv2.imread(os.path.join(REAL_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None: img = np.zeros((96, 96), np.uint8)
        real_imgs.append(img)
        s, _, h, f = parse_real(fn)
        real_s.append(s); real_h.append(h); real_f.append(f)
    real_s=np.array(real_s,dtype=int); real_h=np.array(real_h,dtype=str); real_f=np.array(real_f,dtype=str)

    alt_files = sorted(os.listdir(ALT_DIR))
    alt_imgs = []
    a_s = []; a_h = []; a_f = []
    for fn in alt_files:
        img = cv2.imread(os.path.join(ALT_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None: img = np.zeros((96, 96), np.uint8)
        alt_imgs.append(img)
        s, _, h, f, _ = parse_alt(fn)
        a_s.append(s); a_h.append(h); a_f.append(f)
    a_s=np.array(a_s,dtype=int); a_h=np.array(a_h,dtype=str); a_f=np.array(a_f,dtype=str)

    print("Building pairs...")
    genuine, impostor = build_pairs(real_s, real_h, real_f, a_s, a_h, a_f, n=2000, seed=42)

    results = {}
    print("\n=== Baseline (clean) ===")
    auc_clean = evaluate(real_imgs, alt_imgs, genuine, impostor)
    results["clean"] = auc_clean
    print(f"  Clean AUC: {auc_clean:.4f}")

    for sigma in [10, 20, 30]:
        print(f"\n=== Gaussian noise sigma={sigma} ===")
        rng = np.random.default_rng(123)
        auc_v = evaluate(real_imgs, alt_imgs, genuine, impostor, sigma=sigma)
        results[f"noise_sigma_{sigma}"] = auc_v
        print(f"  AUC: {auc_v:.4f}")

    for frac in [0.25, 0.50]:
        print(f"\n=== {int(frac*100)}% binary mask ===")
        rng = np.random.default_rng(456)
        auc_v = evaluate(real_imgs, alt_imgs, genuine, impostor, mask_frac=frac)
        results[f"mask_{int(frac*100)}"] = auc_v
        print(f"  AUC: {auc_v:.4f}")

    with open("results_strong_zm/robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results_strong_zm/robustness_results.json")


if __name__ == "__main__":
    main()