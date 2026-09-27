"""
Regenerate the system pipeline figure (Fig. 1) for the camera-ready paper.
Same structure as the old figure but with verification-appropriate labels
instead of "Person Identified" / "Identity Matching".
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

# Sample fingerprint image: load one from the Real folder
from PIL import Image
real_dir = "Real"
real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(".BMP")])
fp_img = np.array(Image.open(os.path.join(real_dir, real_files[0])).convert("L"))

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Fingerprint Verification Pipeline Using Zernike Moments",
             fontsize=16, fontweight="bold", pad=14)


def box(x, y, w, h, text, fontsize=10, fontweight="normal", fill="white"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                       linewidth=1.4, edgecolor="black", facecolor=fill)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.25, text, ha="center", va="top",
            fontsize=fontsize, fontweight=fontweight)


def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=18,
                        linewidth=1.4, color="black")
    ax.add_patch(a)


# Row 1: Input -> Preprocessing -> ZM Extraction -> Feature Vector -> Verification
box(0.2, 5.5, 2.0, 1.6, "Input\nFingerprint", fontsize=11, fontweight="bold")
# insert fingerprint thumbnail
ax.imshow(fp_img, cmap="gray", extent=(0.4, 1.8, 5.7, 7.0))

box(2.6, 5.5, 2.2, 1.6, "Zernike Moment\nExtraction",
    fontsize=10, fontweight="bold")
ax.text(3.7, 5.85, r"$Z_{pq} = \frac{p+1}{\pi}\sum f(x,y)V_{pq}^*$",
        ha="center", va="center", fontsize=9)

box(5.2, 5.5, 2.2, 1.6, "Feature Vector\n44 magnitudes\n$|Z_{nm}|$",
    fontsize=10, fontweight="bold")
# mini feature vector
for i, lab in enumerate(["Z00", "Z11", "Z1-1", "Z20", "...", "Z88"]):
    ax.text(5.4 + i * 0.32, 5.95, lab, ha="center", va="center",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="#f0f0f0",
                      edgecolor="gray", linewidth=0.5))

box(7.8, 5.5, 2.4, 1.6, "Euclidean Distance\n(normalised ZM)",
    fontsize=10, fontweight="bold")
ax.text(9.0, 5.95, r"$d = \sqrt{\sum (x_i - y_i)^2}$",
        ha="center", va="center", fontsize=11)

box(10.6, 5.5, 3.2, 1.6, "Verification Decision\nGenuine / Impostor",
    fontsize=11, fontweight="bold", fill="#e8f4ea")
ax.text(12.2, 5.95, "Threshold $\\tau$\n(EER or max-accuracy)",
        ha="center", va="center", fontsize=9)

# Row 1 arrows
arrow(2.2, 6.3, 2.6, 6.3)
arrow(4.8, 6.3, 5.2, 6.3)
arrow(7.4, 6.3, 7.8, 6.3)
arrow(10.2, 6.3, 10.6, 6.3)

# Row 2: Output Results and Advantages
box(0.2, 1.4, 5.2, 3.2, "Output Results", fontsize=12, fontweight="bold")
ax.text(0.5, 3.2, "• AUC: 0.997 / 0.998 / 0.992", fontsize=10)
ax.text(0.5, 2.7, "• EER: 0.025 / 0.019 / 0.041", fontsize=10)
ax.text(0.5, 2.2, "• Best accuracy: 97.8 / 98.2 / 96.0%", fontsize=10)
ax.text(0.5, 1.7, "• Per alteration: AUC ≥ 0.998 (Z-cut: 1.000)", fontsize=10)

box(5.8, 1.4, 8.0, 3.2, "Key Properties", fontsize=12, fontweight="bold")
ax.text(6.1, 3.2, "• Theoretical rotation invariance (|Z_{nm}|)",
        fontsize=10)
ax.text(6.1, 2.7, "• Zero trainable parameters", fontsize=10)
ax.text(6.1, 2.2, "• Robust to noise and partial occlusion", fontsize=10)
ax.text(6.1, 1.7, "• CPU-only, ~1.2 µs per pair, 44 floats per template",
        fontsize=10)

# Bottom row: Use cases / deployment
box(0.2, 0.2, 13.6, 1.0, "")
ax.text(7.0, 0.7, "Suitable for resource-constrained biometric verification; "
                   "combine with liveness detection and cancellable templates "
                   "(BioHashing) before deployment.",
        ha="center", va="center", fontsize=10, style="italic")

plt.tight_layout()
out = "Research_Fingerprint_camera_ready/figures/fingerprint_pipeline.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
