"""
Generate per-alteration ROC curves on Altered-Easy.

Clean approach: for each alteration type, build genuine (real vs alt of same key)
pairs whose alt index is guaranteed < len(sub_zm).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

OUT_DIR = "results_strong_zm"
REAL_DIR = "Real"
ALTERED_EASY = "SOCOFing/Altered/Altered-Easy"
ALT_TYPES = {"CR": "central rotation", "Obl": "obliteration", "Zcut": "z-cut"}


def parse_real_filename(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    subject = int(parts[0])
    tokens = parts[1].split("_")
    return subject, tokens[0], tokens[1], tokens[2]


def parse_altered_filename(fn):
    base = fn.replace(".BMP", "").replace(".bmp", "")
    parts = base.split("__")
    subject = int(parts[0])
    tokens = parts[1].split("_")
    return subject, tokens[0], tokens[1], tokens[2], tokens[-1]


def to_int_subjects(arr_or_list):
    """Convert an array of subject strings to int array"""
    return np.array([int(x) if isinstance(x, str) else int(x) for x in arr_or_list])


def build_pairs(real_subs, real_hands, real_fings,
                alt_subs, alt_hands, alt_fings, n=2000, seed=42):
    """real_subs/alt_subs are np.int arrays. Returns (genuine, impostor)
    with all alt indices guaranteed < len(alt_subs)."""
    rng = np.random.default_rng(seed)
    real_key = {(int(s), str(h), str(f)): i for i, (s, h, f) in enumerate(zip(real_subs, real_hands, real_fings))}
    alt_first = {}
    for j, (s, h, f) in enumerate(zip(alt_subs, alt_hands, alt_fings)):
        if 0 <= j < len(alt_subs):
            k = (int(s), str(h), str(f))
            if k not in alt_first:
                alt_first[k] = j
    # Genuine pairs: real_idx -> alt_idx
    genuine = []
    for k, ri in real_key.items():
        if k in alt_first:
            ai = alt_first[k]
            if 0 <= ai < len(alt_subs):
                genuine.append((ri, ai))
    rng.shuffle(genuine)
    # Impostor pairs: real vs real, diff subject
    impostor = []
    seen = set()
    nr = len(real_subs)
    while len(impostor) < max(n, len(genuine)) and len(impostor) < nr * (nr - 1):
        i = int(rng.integers(0, nr))
        j = int(rng.integers(0, nr))
        if i == j: continue
        if int(real_subs[i]) == int(real_subs[j]): continue
        if (str(real_hands[i]), str(real_fings[i])) == (str(real_hands[j]), str(real_fings[j])): continue
        pair = (i, j)
        if pair in seen: continue
        seen.add(pair)
        impostor.append(pair)
    rng.shuffle(impostor)
    n_final = min(n, len(genuine), len(impostor))
    return genuine[:n_final], impostor[:n_final]


def main():
    print("Loading features...", flush=True)
    data = np.load(os.path.join(OUT_DIR, "real_features.npz"))
    real_zm44 = data["zm44"]

    real_files = sorted(os.listdir(REAL_DIR))
    real_subs_i = []; real_hands = []; real_fings = []
    for fn in real_files:
        s, _, h, f = parse_real_filename(fn)
        real_subs_i.append(s); real_hands.append(h); real_fings.append(f)
    real_subs = np.array(real_subs_i, dtype=int)
    real_hands = np.array(real_hands, dtype=str)
    real_fings = np.array(real_fings, dtype=str)

    alt_files = sorted(os.listdir(ALTERED_EASY))
    alt_subs_i = []; alt_hands = []; alt_fings = []; alt_types = []
    for fn in alt_files:
        s, _, h, f, t = parse_altered_filename(fn)
        alt_subs_i.append(s); alt_hands.append(h); alt_fings.append(f); alt_types.append(t)
    alt_subs_all = np.array(alt_subs_i, dtype=int)
    alt_hands_all = np.array(alt_hands, dtype=str)
    alt_fings_all = np.array(alt_fings, dtype=str)
    alt_types_all = np.array(alt_types)
    n_alt = len(alt_subs_all)
    print(f"  Total Altered-Easy: {n_alt}", flush=True)
    for tname in ["CR", "Obl", "Zcut"]:
        print(f"    {tname}: {(alt_types_all == tname).sum()}", flush=True)

    data_e = np.load(os.path.join(OUT_DIR, "altered_features_Easy.npz"))
    alt_zm44 = data_e["zm44"]
    assert len(alt_zm44) == n_alt, f"cache mismatch: {len(alt_zm44)} vs {n_alt}"

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    colors = {"CR": "#FF9800", "Obl": "#2196F3", "Zcut": "#E91E63"}

    scaler = StandardScaler().fit(real_zm44)
    real_zm44_s = scaler.transform(real_zm44)
    alt_zm44_s = scaler.transform(alt_zm44)

    results_per_alt = {}

    for alt_type in ["CR", "Obl", "Zcut"]:
        mask = alt_types_all == alt_type
        idx = np.where(mask)[0]   # into alt_*
        sub_subs = alt_subs_all[idx]
        sub_hands = alt_hands_all[idx]
        sub_fings = alt_fings_all[idx]
        sub_zm_s = alt_zm44_s[idx]
        print(f"\n  --- {alt_type}: subset size {len(sub_zm_s)} ---", flush=True)

        genuine, impostor = build_pairs(real_subs, real_hands, real_fings,
                                        sub_subs, sub_hands, sub_fings, n=2000, seed=42)

        # Re-check that all alt indices are valid
        n_sub = len(sub_zm_s)
        genuine = [(ri, ai) for ri, ai in genuine if 0 <= ai < n_sub]
        n_real_s = len(real_zm44_s)
        # Impostor pairs are real-vs-real so always valid
        impostor = [(ri, rj) for ri, rj in impostor if 0 <= ri < n_real_s and 0 <= rj < n_real_s]
        print(f"    after validation: genuine={len(genuine)}, impostor={len(impostor)}", flush=True)

        # Compose final lists
        n_final = min(2000, len(genuine), len(impostor))
        genuine = genuine[:n_final]
        impostor = impostor[:n_final]
        all_pairs = genuine + impostor
        labels = np.array([1] * len(genuine) + [0] * len(impostor))

        fa = np.array([real_zm44_s[p[0]] for p in all_pairs])
        # Genuine pairs: second idx is into sub_zm_s. Impostor: second idx is into real_zm44_s
        fb = np.array([sub_zm_s[p[1]] if lbl == 1 else real_zm44_s[p[1]]
                       for p, lbl in zip(all_pairs, labels)])
        d = np.linalg.norm(fa - fb, axis=1)
        scores = -d

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

        ax.plot(fpr, tpr, color=colors[alt_type], linewidth=2,
                label=f"{alt_type} ({ALT_TYPES[alt_type]}) — AUC={roc_auc:.3f}, EER={eer:.3f}, Acc={best_acc*100:.1f}%")
        ax.fill_between(fpr, 0, tpr, alpha=0.10, color=colors[alt_type])
        results_per_alt[alt_type] = {"AUC": roc_auc, "EER": eer, "BestAcc": best_acc}
        print(f"  {alt_type}: AUC={roc_auc:.4f} EER={eer:.4f} Acc={best_acc:.4f}", flush=True)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-alteration ROC: ZM-44 same-finger verification on Altered-Easy\n(2000 genuine + 2000 impostor pairs per alteration type)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_roc_per_alteration.png"), dpi=180, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig_roc_per_alteration.pdf"), bbox_inches="tight")
    plt.close()

    with open(os.path.join(OUT_DIR, "per_alteration_roc_results.json"), "w") as f:
        json.dump(results_per_alt, f, indent=2)
    print(f"\nFigure saved to {OUT_DIR}/fig_roc_per_alteration.png")


if __name__ == "__main__":
    main()