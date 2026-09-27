"""Generate the ZM-44 robustness figure."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results_strong_zm"
data = json.load(open(os.path.join(OUT_DIR, "robustness_results.json")))

fig, ax = plt.subplots(figsize=(9, 5.5))
labels = ["Clean", r"$\sigma=10$", r"$\sigma=20$", r"$\sigma=30$", "25\% mask", "50\% mask"]
values = [data["clean"], data["noise_sigma_10"], data["noise_sigma_20"],
          data["noise_sigma_30"], data["mask_25"], data["mask_50"]]
colors = ["#4CAF50", "#FFB74D", "#FF9800", "#F44336", "#7986CB", "#3F51B5"]
bars = ax.bar(labels, values, color=colors)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylim(0.85, 1.0)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("ZM-44 robustness on Altered-Easy\n(2000-genuine + 2000-impostor protocol under each perturbation)", fontsize=12)
ax.axhline(0.95, color="gray", linestyle="--", linewidth=0.7)
ax.text(0.02, 0.952, "AUC = 0.95", color="gray", fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_robustness.png"), dpi=180, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "fig_robustness.pdf"), bbox_inches="tight")
plt.close()
print(f"Figure saved to {OUT_DIR}/fig_robustness.png")