"""Per-difficulty ROC with ZM-44"""
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
    rng = np.random.default_rng(seed)
    real_key = {(int(s), str(h), str(f)): i for i, (s, h, f) in enumerate(zip(real_subs, real_hands, real_fings))}
    alt_first = {}
    for j, (s, h, f) in enumerate(zip(alt_subs, alt_hands, alt_fings)):
        k = (int(s), str(h), str(f))
        if k not in alt_first:
            alt_first[k] = j
    genuine = [(real_key[k], alt_first[k]) for k in real_key if k in alt_first and 0 <= alt_first[k] < len(alt_subs)]
    rng.shuffle(genuine)
    impostor = []
    seen = set()
    nr = len(real_subs)
    while len(impostor) < max(n, len(genuine)) and len(impostor) < nr * (nr - 1):
        i = int(rng.integers(0, nr)); j = int(rng.integers(0, nr))
        if i == j or int(real_subs[i]) == int(real_subs[j]): continue
        if (str(real_hands[i]), str(real_fings[i])) == (str(real_hands[j]), str(real_fings[j])): continue
        pair = (i, j)
        if pair in seen: continue
        seen.add(pair); impostor.append(pair)
    rng.shuffle(impostor)
    n = min(n, len(genuine), len(impostor))
    return genuine[:n], impostor[:n]


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

    results_per_diff = {}
    for level_name, alt_dir in LEVELS.items():
        d_alt = np.load(os.path.join(OUT_DIR, f"altered_features_{level_name}.npz"))
        alt_zm44 = d_alt["zm44"]
        alt_subs = d_alt["subjects"]; alt_hands = d_alt["hands"]; alt_fings = d_alt["fingers"]
        # convert from stored int/string to consistent types
        alt_subs = np.array([int(x) for x in alt_subs])
        alt_hands = np.array([str(x) for x in alt_hands])
        alt_fings = np.array([str(x) for x in alt_fings])

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

        fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
        roc_auc = float(auc(fpr, tpr))
        fnr = 1 - tpr; idx_eer = np.argmin(np.abs(fpr - fnr))
        eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2)
        best_acc = 0
        for t in thr:
            pred = (scores > t).astype(int)
            acc = (pred == labels).mean()
            if acc > best_acc:
                best_acc = acc

        ax.plot(fpr, tpr, color=colors[level_name], linewidth=2,
                label=f"Altered-{level_name}: AUC={roc_auc:.3f}, EER={eer:.3f}, Acc={best_acc*100:.1f}%")
        ax.fill_between(fpr, 0, tpr, alpha=0.10, color=colors[level_name])
        results_per_diff[level_name] = {"AUC": roc_auc, "EER": eer, "BestAcc": best_acc}
        print(f"  Altered-{level_name}: AUC={roc_auc:.4f} EER={eer:.4f} Acc={best_acc:.4f}", flush=True)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-difficulty ROC: ZM-44 same-finger verification on SOCOFing\n(2000 genuine + 2000 impostor pairs per level)", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_roc_per_difficulty_zm44.png"), dpi=180, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig_roc_per_difficulty_zm44.pdf"), bbox_inches="tight")
    plt.close()
    with open(os.path.join(OUT_DIR, "per_difficulty_zm44_results.json"), "w") as f:
        json.dump(results_per_diff, f, indent=2)
    print(f"\nFigure saved to {OUT_DIR}/fig_roc_per_difficulty_zm44.png")


if __name__ == "__main__":
    main()