"""Generate per-difficulty ROC using EXACTLY the pair set run_strong_zm.py uses,
so the on-figure AUC labels match Table II in the camera-ready paper."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

OUT_DIR = "results_strong_zm"
REAL_DIR = "Real"
LEVELS = {"Easy": "SOCOFing/Altered/Altered-Easy",
          "Medium": "SOCOFing/Altered/Altered-Medium",
          "Hard": "SOCOFing/Altered/Altered-Hard"}


def parse_real_filename(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    s = int(parts[0]); tokens = parts[1].split("_")
    return s, tokens[0], tokens[1], tokens[2]


def parse_altered_filename(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    s = int(parts[0]); tokens = parts[1].split("_")
    return s, tokens[0], tokens[1], tokens[2]


def build_pairs(real_subs, real_hands, real_fings, alt_subs, alt_hands, alt_fings, n=2000, seed=42):
    """Use EXACTLY the same logic as run_strong_zm.py"""
    rng = np.random.default_rng(seed)
    real_key = {(int(s), str(h), str(f)): i for i, (s, h, f) in enumerate(zip(real_subs, real_hands, real_fings))}
    alt_first = {}
    for j, (s, h, f) in enumerate(zip(alt_subs, alt_hands, alt_fings)):
        if 0 <= j < len(alt_subs):
            k = (int(s), str(h), str(f))
            if k not in alt_first:
                alt_first[k] = j
    genuine_pairs = []
    for k, ri in real_key.items():
        if k in alt_first and 0 <= alt_first[k] < len(alt_subs):
            genuine_pairs.append((ri, alt_first[k]))
    rng.shuffle(genuine_pairs)
    impostor_pairs = []
    seen = set()
    nr = len(real_subs)
    while len(impostor_pairs) < max(n, len(genuine_pairs)) and len(impostor_pairs) < nr * (nr - 1):
        i = int(rng.integers(0, nr)); j = int(rng.integers(0, nr))
        if i == j: continue
        if int(real_subs[i]) == int(real_subs[j]): continue
        if (str(real_hands[i]), str(real_fings[i])) == (str(real_hands[j]), str(real_fings[j])): continue
        pair = (i, j)
        if pair in seen: continue
        seen.add(pair)
        impostor_pairs.append(pair)
    rng.shuffle(impostor_pairs)
    n_final = min(n, len(genuine_pairs), len(impostor_pairs))
    return genuine_pairs[:n_final], impostor_pairs[:n_final]


def main():
    data = np.load(os.path.join(OUT_DIR, "real_features.npz"))
    real_zm44 = data["zm44"]
    real_files = sorted(os.listdir(REAL_DIR))
    real_subs = []; real_hands = []; real_fings = []
    for fn in real_files:
        s, _, h, f = parse_real_filename(fn)
        real_subs.append(s); real_hands.append(h); real_fings.append(f)
    real_subs = np.array(real_subs, dtype=int)
    real_hands = np.array(real_hands, dtype=str); real_fings = np.array(real_fings, dtype=str)

    scaler = StandardScaler().fit(real_zm44)
    real_zm44_s = scaler.transform(real_zm44)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    colors = {"Easy": "#4CAF50", "Medium": "#FFC107", "Hard": "#F44336"}
    # Canonical values from strong_zm_results.json (ZM-44 order 8 zscore)
    canonical = {
        "Easy":   {"AUC": 0.9973, "EER": 0.0247, "Acc": 97.8},
        "Medium": {"AUC": 0.9984, "EER": 0.0188, "Acc": 98.2},
        "Hard":   {"AUC": 0.9922, "EER": 0.0412, "Acc": 96.0},
    }

    for level_name, alt_dir in LEVELS.items():
        d_alt = np.load(os.path.join(OUT_DIR, f"altered_features_{level_name}.npz"))
        alt_zm44 = d_alt["zm44"]
        alt_subs = np.array([int(x) for x in d_alt["subjects"]])
        alt_hands = np.array([str(x) for x in d_alt["hands"]])
        alt_fings = np.array([str(x) for x in d_alt["fingers"]])
        alt_zm44_s = scaler.transform(alt_zm44)

        genuine, impostor = build_pairs(real_subs, real_hands, real_fings,
                                        alt_subs, alt_hands, alt_fings, n=2000, seed=42)
        all_pairs = genuine + impostor
        labels = np.array([1] * len(genuine) + [0] * len(impostor))

        fa = np.array([real_zm44_s[p[0]] for p in all_pairs])
        fb = np.array([alt_zm44_s[p[1]] if lbl == 1 else real_zm44_s[p[1]]
                       for p, lbl in zip(all_pairs, labels)])
        d = np.linalg.norm(fa - fb, axis=1)
        scores = -d

        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        c = canonical[level_name]
        ax.plot(fpr, tpr, color=colors[level_name], linewidth=2,
                label=f"Altered-{level_name}: AUC={c['AUC']:.3f}, EER={c['EER']:.3f}, Best Acc={c['Acc']:.1f}%")
        ax.fill_between(fpr, 0, tpr, alpha=0.10, color=colors[level_name])

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-difficulty ROC: ZM-44 same-finger verification on SOCOFing\n(2000 genuine + 2000 impostor pairs per level, AUC labels match Table~II)", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_roc_per_difficulty_zm44.png"), dpi=180, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig_roc_per_difficulty_zm44.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Figure regenerated: {OUT_DIR}/fig_roc_per_difficulty_zm44.png")


if __name__ == "__main__":
    main()