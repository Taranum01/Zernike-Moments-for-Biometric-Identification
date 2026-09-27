"""
Generate the ZM-order ablation figure and table.

Uses ZM-44 (order 8) features and takes prefixes corresponding to lower orders:
- order 3: 4 magnitudes  (Z00, Z11, Z20, Z22, Z31, Z33) -> actually 6; we take first 4
- order 5: 12 magnitudes (subset of order 8)
- order 7: 24 magnitudes
- order 8: 44 magnitudes (full)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

OUT_DIR = "results_strong_zm"
os.makedirs(OUT_DIR, exist_ok=True)

REAL_DIR = "Real"
LEVELS = {
    "Easy": "SOCOFing/Altered/Altered-Easy",
    "Medium": "SOCOFing/Altered/Altered-Medium",
    "Hard": "SOCOFing/Altered/Altered-Hard",
}


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


def zm_total_length(order):
    return sum(1 for n in range(1, order + 1) for m in range(-n, n + 1) if (n - abs(m)) % 2 == 0)


def compute_metrics(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    best_acc = 0
    for t in thr:
        pred = (scores > t).astype(int)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc = acc
    return roc_auc, eer, best_acc


def build_pairs(real_subjects, real_hands, real_fingers,
                alt_subjects, alt_hands, alt_fingers, seed=42, n=2000):
    rng = np.random.default_rng(seed)
    real_key = {(s, h, f): i for i, (s, h, f) in enumerate(zip(real_subjects, real_hands, real_fingers))}
    alt_first = {}
    for j, (s, h, f) in enumerate(zip(alt_subjects, alt_hands, alt_fingers)):
        if (s, h, f) not in alt_first:
            alt_first[(s, h, f)] = j
    genuine = [(real_key[k], alt_first[k]) for k in real_key if k in alt_first]
    rng.shuffle(genuine)
    impostor = []
    seen = set()
    nr = len(real_subjects)
    while len(impostor) < max(n, len(genuine)):
        i = rng.integers(0, nr); j = rng.integers(0, nr)
        if i == j: continue
        if real_subjects[i] == real_subjects[j]: continue
        if (real_subjects[i], real_hands[i], real_fingers[i]) == (real_subjects[j], real_hands[j], real_fingers[j]): continue
        pair = (i, j)
        if pair in seen: continue
        seen.add(pair); impostor.append(pair)
    rng.shuffle(impostor)
    n = min(n, len(genuine), len(impostor))
    return genuine[:n], impostor[:n]


def main():
    print("Loading cached ZM-44 features...", flush=True)
    data = np.load(os.path.join(OUT_DIR, "real_features.npz"))
    real_zm44 = data["zm44"]
    N_real, D44 = real_zm44.shape
    print(f"  ZM-44 real: {real_zm44.shape}", flush=True)

    # Order prefix sizes for ZM magnitudes up to order n
    order_dims = {3: zm_total_length(3), 5: zm_total_length(5), 7: zm_total_length(7), 8: zm_total_length(8)}
    print(f"  Order lengths: {order_dims}", flush=True)

    real_files = sorted(os.listdir(REAL_DIR))
    real_subjects = []; real_hands = []; real_fingers = []
    for fn in real_files:
        s, _, h, f = parse_real_filename(fn)
        real_subjects.append(s); real_hands.append(h); real_fingers.append(f)
    real_subjects = np.array(real_subjects); real_hands = np.array(real_hands); real_fingers = np.array(real_fingers)

    results = {"Easy": {}, "Medium": {}, "Hard": {}}

    for level_name, alt_dir in LEVELS.items():
        print(f"\n===== Altered-{level_name} =====", flush=True)
        data = np.load(os.path.join(OUT_DIR, f"altered_features_{level_name}.npz"))
        alt_zm44 = data["zm44"]
        alt_subjects = data["subjects"]; alt_hands = data["hands"]; alt_fingers = data["fingers"]

        genuine, impostor = build_pairs(real_subjects, real_hands, real_fingers,
                                         alt_subjects, alt_hands, alt_fingers, seed=42, n=2000)
        all_pairs = genuine + impostor
        labels = np.array([1] * len(genuine) + [0] * len(impostor))

        for order in [3, 5, 7, 8]:
            d = order_dims[order]
            r_sub = real_zm44[:, :d]
            a_sub = alt_zm44[:, :d]
            fa = np.array([r_sub[p[0]] for p in all_pairs])
            fb = np.array([a_sub[p[1]] for p in all_pairs])
            scaler = StandardScaler().fit(r_sub)
            fa_s = scaler.transform(fa); fb_s = scaler.transform(fb)
            d_euc = np.linalg.norm(fa_s - fb_s, axis=1)
            scores = -d_euc
            roc_auc, eer, acc = compute_metrics(scores, labels)
            print(f"    Order {order:2d} (dim={d:2d}): AUC={roc_auc:.4f}  EER={eer:.4f}  BestAcc={acc:.4f}", flush=True)
            results[level_name][f"order{order}_dim{d}"] = {"AUC": roc_auc, "EER": eer, "BestAcc": acc, "Dim": d, "Order": order}

    with open(os.path.join(OUT_DIR, "order_ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- Figure: AUC vs ZM order, three lines (Easy/Medium/Hard) ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    orders = [3, 5, 7, 8]
    colors = {"Easy": "#4CAF50", "Medium": "#FFC107", "Hard": "#F44336"}
    for level in ["Easy", "Medium", "Hard"]:
        aucs = [results[level][f"order{o}_dim{order_dims[o]}"]["AUC"] for o in orders]
        ax.plot(orders, aucs, "o-", color=colors[level], label=f"Altered-{level}", linewidth=2, markersize=10)
        for o, a in zip(orders, aucs):
            ax.text(o, a + 0.002, f"{a:.3f}", ha="center", fontsize=9)

    ax.set_xlabel("Zernike moment order", fontsize=12)
    ax.set_ylabel("AUC (same-finger verification)", fontsize=12)
    ax.set_title("ZM-order ablation: AUC vs Zernike order\n(300-d pool × 3 difficulty levels, 2000/2000 pair protocol)", fontsize=12)
    ax.set_xticks(orders)
    ax.set_xticklabels([f"order {o}\n({order_dims[o]} dims)" for o in orders], fontsize=10)
    ax.set_ylim(0.97, 1.0)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_zm_order_ablation.png"), dpi=180, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig_zm_order_ablation.pdf"), bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {OUT_DIR}/fig_zm_order_ablation.png")

    # Print the table
    print("\n\nOrder-ablation table:")
    print("| Order | Dim | Easy AUC | Med AUC | Hard AUC |")
    print("|---|---|---|---|---|")
    for o in orders:
        d = order_dims[o]
        e = results["Easy"][f"order{o}_dim{d}"]["AUC"]
        m = results["Medium"][f"order{o}_dim{d}"]["AUC"]
        h = results["Hard"][f"order{o}_dim{d}"]["AUC"]
        print(f"| {o} | {d} | {e:.3f} | {m:.3f} | {h:.3f} |")


if __name__ == "__main__":
    main()