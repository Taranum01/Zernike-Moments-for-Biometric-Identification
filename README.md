# Fingerprint Identification Using Zernike Moments

Code for the paper **"Fingerprint Identification Using Zernike Moments"** (ICOICI).

We use orthogonal **Zernike Moments (ZM)** of orders 1–8 — a **44-dimensional,
rotation-invariant, training-free** descriptor — as a fixed-length biometric
template, and evaluate same-finger verification on the public
[SOCOFing](https://arxiv.org/abs/1807.10609) dataset. The 44-number descriptor
is competitive with deep models while using **zero trainable parameters** and
about **1.2 µs** of CPU matching per pair.

## Headline results (Table II)

| Difficulty | AUC   | EER   | Accuracy |
|------------|-------|-------|----------|
| Easy       | 0.997 | 0.025 | 97.8%    |
| Medium     | 0.998 | 0.019 | 98.2%    |
| Hard       | 0.992 | 0.041 | 96.0%    |

Genuine vs impostor distance separation: Cohen's *d* ≈ 2.98, Welch *t* = 94.2,
*p* < 0.001. Per-alteration AUC: central rotation 0.998, obliteration 0.999,
z-cut 1.000.

## Repository layout

```
src/
  compute_zernike_batch.py            # Zernike moment feature extraction
  run_strong_zm.py                    # headline ZM-44 verification  (Table II)
  run_same_finger_v2.py               # same-finger verification, per-alteration
  run_noise_partial_eval.py           # robustness: noise + occlusion (Table V)
  build_order_ablation.py             # ZM-order ablation            (Table IV)
  build_per_alt_roc.py                # per-alteration ROC           (Fig. 4)
  build_per_difficulty_roc.py         # per-difficulty ROC           (Fig. 3)
  build_per_difficulty_roc_canonical.py
  build_sota_comparison.py            # comparison table + figure    (Table VII, Fig. 7)
  build_pipeline_figure.py            # pipeline diagram
  build_robustness_figure.py          # robustness figure
  build_grayscale_figures.py          # sample/illustration figures
  run_cnn_baseline.py                 # in-house Siamese CNN baseline (Table VII)
  eval_cnn_cross_difficulty.py        # CNN cross-difficulty evaluation
```

> The SOCOFing dataset, computed feature caches, virtualenv, and result
> figures are **not** included in this repository (see `.gitignore`). Download
> the dataset separately and place it as described below.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download SOCOFing (e.g. from Kaggle / arXiv:1807.10609) and arrange it at the
repository root so the scripts find it via these relative paths:

```
Real/                                 # 6,000 real fingerprints
SOCOFing/Altered/Altered-Easy/        # 17,931 altered
SOCOFing/Altered/Altered-Medium/      # 17,067 altered
SOCOFing/Altered/Altered-Hard/        # 14,272 altered
```

Filenames follow SOCOFing's convention, e.g.
`1__M_Left_index_finger.BMP` (real) and `1__M_Left_index_finger_CR.BMP`
(altered; suffix `CR`/`Obl`/`Zcut`).

## Reproducing the results

Run from the repository root (scripts use relative paths):

```bash
python src/compute_zernike_batch.py Real      # extract ZM features
python src/run_strong_zm.py                    # headline verification (Table II)
python src/build_order_ablation.py             # order ablation        (Table IV)
python src/run_noise_partial_eval.py           # robustness            (Table V)
python src/build_per_alt_roc.py                # per-alteration ROC    (Fig. 4)
python src/build_sota_comparison.py            # comparison            (Table VII)
python src/run_cnn_baseline.py                 # CNN baseline (needs PyTorch)
```

## Method summary

1. **Normalize** — map the fingerprint onto the unit disk (coordinates → [−1, 1]).
2. **Extract** — compute Zernike moment magnitudes for orders 1–8 → 44 numbers.
3. **Match** — Euclidean distance on z-score-normalised descriptors; accept a
   pair as genuine if the distance is below a threshold.

Each template is 44 floats (~180 bytes); matching is CPU-only, no GPU required.

## Citation

```bibtex
@inproceedings{wasu_zernike_fingerprint,
  title     = {Fingerprint Identification Using Zernike Moments},
  author    = {Wasu, Taranum and Kaur, Harinder and Pannu, H. S.},
  booktitle = {International Conference on Optimization, Intelligent and Computing Innovations (ICOICI)},
  year      = {2026}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

The SOCOFing dataset is distributed by its original authors under its own
license and is not included here.
