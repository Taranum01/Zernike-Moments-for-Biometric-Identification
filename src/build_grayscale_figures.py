"""
Regenerate Figs 3-7 in grayscale to match the existing IEEE paper style
(Figs 1-2 and the original Figs 3-6 are all grayscale).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

OUT_DIR = "results_strong_zm"
CAM_DIR = "Research_Fingerprint_camera_ready/figures"
os.makedirs(CAM_DIR, exist_ok=True)

# Grayscale palette (dark → light, high contrast)
GRAYS = ["#111111", "#555555", "#888888", "#BBBBBB", "#DDDDDD"]
LINESTYLES = ["-", "--", "-.", ":", "-"]
HATCHES = ["", "//", "\\\\", "xx", ".."]


# ---------------------------------------------------------------------------
# Fig 3: Per-difficulty ROC
# ---------------------------------------------------------------------------
def fig3():
    REAL_DIR = "Real"
    LEVELS = {"Easy": "SOCOFing/Altered/Altered-Easy",
              "Medium": "SOCOFing/Altered/Altered-Medium",
              "Hard": "SOCOFing/Altered/Altered-Hard"}

    def parse_real(fn):
        base = fn.replace(".BMP", "").replace(".bmp", "")
        parts = base.split("__"); s = int(parts[0]); tokens = parts[1].split("_")
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
        impostor = []; seen = set(); nr = len(real_subs)
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

    data = np.load(os.path.join(OUT_DIR, "real_features.npz"))
    real_zm44 = data["zm44"]
    real_files = sorted(os.listdir(REAL_DIR))
    real_subs = []; real_hands = []; real_fings = []
    for fn in real_files:
        s, _, h, f = parse_real(fn)
        real_subs.append(s); real_hands.append(h); real_fings.append(f)
    real_subs = np.array(real_subs, dtype=int)
    real_hands = np.array(real_hands, dtype=str); real_fings = np.array(real_fings, dtype=str)
    scaler = StandardScaler().fit(real_zm44)
    real_zm44_s = scaler.transform(real_zm44)

    canonical = {
        "Easy":   {"AUC": 0.997, "EER": 0.025, "Acc": 97.8},
        "Medium": {"AUC": 0.998, "EER": 0.019, "Acc": 98.2},
        "Hard":   {"AUC": 0.992, "EER": 0.041, "Acc": 96.0},
    }
    colors = {"Easy": "#111111", "Medium": "#555555", "Hard": "#999999"}
    styles = {"Easy": "-", "Medium": "--", "Hard": "-."}

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
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
        ax.plot(fpr, tpr, color=colors[level_name], linestyle=styles[level_name],
                linewidth=2.2,
                label=f"Altered-{level_name}: AUC={c['AUC']:.3f}, EER={c['EER']:.3f}, Acc={c['Acc']:.1f}%")

    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-difficulty ROC: ZM-44 same-finger verification", fontsize=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    out = os.path.join(CAM_DIR, "Fig3_IEEE_roc_difficulty.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Wrote {out}")


# ---------------------------------------------------------------------------
# Fig 4: Per-alteration ROC
# ---------------------------------------------------------------------------
def fig4():
    REAL_DIR = "Real"
    ALTERED_EASY = "SOCOFing/Altered/Altered-Easy"
    ALT_TYPES = {"CR": "central rotation", "Obl": "obliteration", "Zcut": "z-cut"}

    def parse_real(fn):
        base = fn.replace(".BMP", "").replace(".bmp", "")
        parts = base.split("__"); s = int(parts[0]); tokens = parts[1].split("_")
        return s, tokens[0], tokens[1], tokens[2]

    def parse_alt(fn):
        base = fn.replace(".BMP", "").replace(".bmp", "")
        parts = base.split("__"); s = int(parts[0]); tokens = parts[1].split("_")
        return s, tokens[0], tokens[1], tokens[2], tokens[-1]

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
        impostor = []; seen = set(); nr = len(real_subs)
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

    data = np.load(os.path.join(OUT_DIR, "real_features.npz"))
    real_zm44 = data["zm44"]
    real_files = sorted(os.listdir(REAL_DIR))
    real_subs = []; real_hands = []; real_fings = []
    for fn in real_files:
        s, _, h, f = parse_real(fn)
        real_subs.append(s); real_hands.append(h); real_fings.append(f)
    real_subs = np.array(real_subs, dtype=int)
    real_hands = np.array(real_hands, dtype=str); real_fings = np.array(real_fings, dtype=str)

    alt_files = sorted(os.listdir(ALTERED_EASY))
    alt_subs_all = []; alt_hands_all = []; alt_fings_all = []; alt_types_all = []
    for fn in alt_files:
        s, _, h, f, t = parse_alt(fn)
        alt_subs_all.append(s); alt_hands_all.append(h); alt_fings_all.append(f); alt_types_all.append(t)
    alt_subs_all = np.array(alt_subs_all, dtype=int)
    alt_hands_all = np.array(alt_hands_all, dtype=str)
    alt_fings_all = np.array(alt_fings_all, dtype=str)
    alt_types_all = np.array(alt_types_all)

    data_e = np.load(os.path.join(OUT_DIR, "altered_features_Easy.npz"))
    alt_zm44 = data_e["zm44"]
    scaler = StandardScaler().fit(real_zm44)
    real_zm44_s = scaler.transform(real_zm44)
    alt_zm44_s = scaler.transform(alt_zm44)

    colors = {"CR": "#111111", "Obl": "#555555", "Zcut": "#999999"}
    styles = {"CR": "-", "Obl": "--", "Zcut": "-."}
    results = json.load(open(os.path.join(OUT_DIR, "per_alteration_roc_results.json")))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    for alt_type in ["CR", "Obl", "Zcut"]:
        mask = alt_types_all == alt_type
        idx = np.where(mask)[0]
        sub_subs = alt_subs_all[idx]; sub_hands = alt_hands_all[idx]; sub_fings = alt_fings_all[idx]
        sub_zm_s = alt_zm44_s[idx]
        genuine, impostor = build_pairs(real_subs, real_hands, real_fings,
                                        sub_subs, sub_hands, sub_fings, n=2000, seed=42)
        n_final = min(2000, len(genuine), len(impostor))
        genuine = genuine[:n_final]; impostor = impostor[:n_final]
        all_pairs = genuine + impostor
        labels = np.array([1] * len(genuine) + [0] * len(impostor))
        fa = np.array([real_zm44_s[p[0]] for p in all_pairs])
        fb = np.array([sub_zm_s[p[1]] if lbl == 1 else real_zm44_s[p[1]]
                       for p, lbl in zip(all_pairs, labels)])
        d = np.linalg.norm(fa - fb, axis=1)
        scores = -d
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        r = results[alt_type]
        ax.plot(fpr, tpr, color=colors[alt_type], linestyle=styles[alt_type],
                linewidth=2.2,
                label=f"{alt_type} ({ALT_TYPES[alt_type]}): AUC={r['AUC']:.3f}, EER={r['EER']:.3f}, Acc={r['BestAcc']*100:.1f}%")

    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-alteration ROC: ZM-44 on Altered-Easy", fontsize=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    out = os.path.join(CAM_DIR, "Fig4_IEEE_roc_per_alteration.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Wrote {out}")


# ---------------------------------------------------------------------------
# Fig 5: Order ablation
# ---------------------------------------------------------------------------
def fig5():
    data = json.load(open(os.path.join(OUT_DIR, "order_ablation_results.json")))
    order_dims = {3: 9, 5: 20, 7: 35, 8: 44}
    orders = [3, 5, 7, 8]
    colors = {"Easy": "#111111", "Medium": "#555555", "Hard": "#999999"}
    styles = {"Easy": "o-", "Medium": "s--", "Hard": "^:"}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for level in ["Easy", "Medium", "Hard"]:
        aucs = [data[level][f"order{o}_dim{order_dims[o]}"]["AUC"] for o in orders]
        ax.plot(orders, aucs, styles[level], color=colors[level],
                label=f"Altered-{level}", linewidth=2, markersize=8)
        for o, a in zip(orders, aucs):
            ax.text(o, a + 0.0015, f"{a:.3f}", ha="center", fontsize=9, color=colors[level])

    ax.set_xlabel("Zernike moment order", fontsize=12)
    ax.set_ylabel("AUC (same-finger verification)", fontsize=12)
    ax.set_title("ZM-order ablation: AUC vs Zernike order", fontsize=12)
    ax.set_xticks(orders)
    ax.set_xticklabels([f"order {o}\n({order_dims[o]} dims)" for o in orders], fontsize=10)
    ax.set_ylim(0.955, 1.005)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3, linestyle=":")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    out = os.path.join(CAM_DIR, "Fig5_IEEE_order_ablation.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Wrote {out}")


# ---------------------------------------------------------------------------
# Fig 6: SOTA comparison
# ---------------------------------------------------------------------------
def fig6():
    # DeepAFRNet reports accuracy, not AUC — omit from this AUC bar chart.
    methods = ["ZM-44\n(ours)", "Siamese CNN\n(ours)", "ConvNeXt\n[Suman 2026]"]
    easy =   [0.997, 0.998, 1.000]
    medium = [0.998, 0.938, 0.990]
    hard =   [0.992, 0.904, 0.980]

    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    b1 = ax.bar(x - width, easy,   width, label="Altered-Easy",   color="#222222", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x,         medium, width, label="Altered-Medium", color="#777777", edgecolor="black", linewidth=0.8, hatch="//")
    b3 = ax.bar(x + width, hard,   width, label="Altered-Hard",   color="#BBBBBB", edgecolor="black", linewidth=0.8, hatch="\\\\")

    # Highlight ZM-44 group with a thicker outline
    for b in [b1[0], b2[0], b3[0]]:
        b.set_linewidth(2.0)
        b.set_edgecolor("black")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Reported AUC comparison on SOCOFing (ZM-44 highlighted)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.88, 1.04)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, axis="y", linestyle=":")
    plt.tight_layout()
    out = os.path.join(CAM_DIR, "Fig6_IEEE_SOTA_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Wrote {out}")


# ---------------------------------------------------------------------------
# Fig 7: Robustness
# ---------------------------------------------------------------------------
def fig7():
    data = json.load(open(os.path.join(OUT_DIR, "robustness_results.json")))
    labels = ["Clean", r"$\sigma=10$", r"$\sigma=20$", r"$\sigma=30$", "25% mask", "50% mask"]
    values = [data["clean"], data["noise_sigma_10"], data["noise_sigma_20"],
              data["noise_sigma_30"], data["mask_25"], data["mask_50"]]
    # Progressive darkness
    colors = ["#111111", "#333333", "#555555", "#777777", "#999999", "#BBBBBB"]
    hatches = ["", "", "", "", "//", "\\\\"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("ZM-44 robustness on Altered-Easy\n(noise and partial occlusion)", fontsize=12)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.7)
    ax.text(5.4, 0.952, "AUC = 0.95", color="gray", fontsize=9, ha="right")
    ax.grid(alpha=0.3, axis="y", linestyle=":")
    plt.tight_layout()
    out = os.path.join(CAM_DIR, "Fig7_IEEE_robustness.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Wrote {out}")


if __name__ == "__main__":
    print("Regenerating all camera-ready figures in grayscale...")
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("Done.")
