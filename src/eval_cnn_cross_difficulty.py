"""
Evaluate the trained Siamese CNN on Altered-Easy, Medium, and Hard
to test for data leakage / generalization.
"""
import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from run_cnn_baseline import EmbeddingNet, PairDataset, parse_real_filename, parse_altered_filename, build_pair_indices, evaluate_siamese

REAL_DIR = "Real"
ALTERED_EASY = "SOCOFing/Altered/Altered-Easy"
ALTERED_MED = "SOCOFing/Altered/Altered-Medium"
ALTERED_HARD = "SOCOFing/Altered/Altered-Hard"


def load_imgs(dir_path):
    files = sorted(os.listdir(dir_path))
    imgs = []
    metas = []
    for fn in files:
        img = cv2.imread(os.path.join(dir_path, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((96, 96), dtype=np.uint8)
        else:
            img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        imgs.append(img)
        metas.append(fn)
    return np.array(imgs), metas


def main():
    device = "cpu"

    print("Loading Real images...")
    real_files = sorted(os.listdir(REAL_DIR))
    real_imgs = []
    real_subjects, real_hands, real_fingers = [], [], []
    for fn in real_files:
        img = cv2.imread(os.path.join(REAL_DIR, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((96, 96), dtype=np.uint8)
        else:
            img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        real_imgs.append(img)
        s, _, h, f = parse_real_filename(fn)
        real_subjects.append(s); real_hands.append(h); real_fingers.append(f)
    real_imgs = np.array(real_imgs)
    real_subjects = np.array(real_subjects); real_hands = np.array(real_hands); real_fingers = np.array(real_fingers)

    # Load CNN
    model = EmbeddingNet(emb_dim=128).to(device)
    model.load_state_dict(torch.load("results_cnn_baseline/siamese_best.pt"))
    model.eval()

    results = {}
    for level_name, alt_dir in [("Easy", ALTERED_EASY), ("Medium", ALTERED_MED), ("Hard", ALTERED_HARD)]:
        print(f"\nEvaluating on Altered-{level_name}...")
        alt_imgs = []
        alt_subjects, alt_hands, alt_fingers = [], [], []
        for fn in sorted(os.listdir(alt_dir)):
            img = cv2.imread(os.path.join(alt_dir, fn), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((96, 96), dtype=np.uint8)
            else:
                img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
            alt_imgs.append(img)
            s, _, h, f, _ = parse_altered_filename(fn)
            alt_subjects.append(s); alt_hands.append(h); alt_fingers.append(f)
        alt_imgs = np.array(alt_imgs)
        alt_subjects = np.array(alt_subjects); alt_hands = np.array(alt_hands); alt_fingers = np.array(alt_fingers)

        # Build pairs (using same seed)
        genuine_pairs, impostor_pairs = build_pair_indices(
            real_subjects, real_hands, real_fingers,
            alt_subjects, alt_hands, alt_fingers,
            n_genuine=2000, n_impostor=2000, seed=42,
        )
        pairs = genuine_pairs + impostor_pairs
        labels = np.array([1] * len(genuine_pairs) + [0] * len(impostor_pairs))
        ds = PairDataset(real_imgs, alt_imgs, pairs, labels)
        loader = DataLoader(ds, batch_size=128, shuffle=False)

        auc_v, eer_v, acc_v = evaluate_siamese(model, loader, device)
        print(f"  Altered-{level_name}: AUC={auc_v:.4f}, EER={eer_v:.4f}, BestAcc={acc_v:.4f}")
        results[level_name] = {"AUC": auc_v, "EER": eer_v, "BestAcc": acc_v}

    with open("results_cnn_baseline/cnn_cross_difficulty_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results_cnn_baseline/cnn_cross_difficulty_results.json")


if __name__ == "__main__":
    main()