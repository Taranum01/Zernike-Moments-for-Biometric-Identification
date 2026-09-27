#!/usr/bin/env python3
"""
compute_zernike_batch.py
Port of the provided MATLAB Zernike moment code.

Output: zernike_moments.xlsx (and .csv) in the same folder where script is run,
with columns: filename, Z00, Z11, Z20, Z22, Z31, Z33, Z40, Z42, Z44, Z51, Z53, Z55

Usage:
    python compute_zernike_batch.py /path/to/image_folder /path/to/output.xlsx

If output path omitted, creates zernike_moments.xlsx in current folder.
"""

import os
import sys
import math
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- Config ----
ORDER = 5  # matches MATLAB code
# expected sequence of (p,q) pairs per MATLAB code: (0,0),(1,1),(2,0),(2,2),(3,1),(3,3),(4,0),(4,2),(4,4),(5,1),(5,3),(5,5)

def radialpoly(r, n, m):
    """Compute radial polynomial R_n^m(r) for given r (numpy array)."""
    rad = np.zeros_like(r, dtype=np.float64)
    # s ranges from 0 to (n-|m|)/2 inclusive; ensure integer steps
    maxs = (n - abs(m)) // 2
    for s in range(maxs + 1):
        num = ((-1)**s) * math.factorial(n - s)
        denom = (math.factorial(s) *
                 math.factorial((n + abs(m))//2 - s) *
                 math.factorial((n - abs(m))//2 - s))
        c = num / denom
        rad = rad + c * (r ** (n - 2*s))
    return rad

def compute_zernike_moments(img, order=5):
    """
    img: 2D numpy array (grayscale), dtype can be uint8 or float.
    Returns list of magnitudes A in the MATLAB ordering (12 moments for order=5).
    """
    # Ensure square: center-crop or pad to square (we will center and pad with zeros if needed)
    h, w = img.shape
    N = max(h, w)
    if (h != w):
        # place image into center of NxN array
        sq = np.zeros((N, N), dtype=img.dtype)
        y0 = (N - h) // 2
        x0 = (N - w) // 2
        sq[y0:y0+h, x0:x0+w] = img
        image = sq
    else:
        image = img.copy()

    # Convert to float
    f = image.astype(np.float64)

    # Prepare coordinate mapping same as MATLAB:
    # x = 1:N; y = 1:N;
    # xi = (2.*X+1-N)/(N*sqrt(2)); yj = (2.*Y+1-N)/(N*sqrt(2));
    Nf = float(N)
    coords = np.arange(1, N+1)
    X, Y = np.meshgrid(coords, coords)
    X = X - 1
    Y = Y - 1
    xi = (2.0*X + 1.0 - Nf) / (Nf * np.sqrt(2.0))
    yj = (2.0*Y + 1.0 - Nf) / (Nf * np.sqrt(2.0))

    R = np.sqrt(xi**2 + yj**2)
    mask = (R <= 1.0)
    R = R * mask  # zeros outside circle
    Theta = np.arctan2(yj, xi)
    # MATLAB Theta=(Theta<0)*2*pi+Theta;  i.e. map to [0,2pi)
    Theta = np.where(Theta < 0, Theta + 2*np.pi, Theta)

    # compute moments according to order scanning & parity condition
    A = []
    # iterate p and q in same way as MATLAB code
    for p in range(0, order+1):
        for q in range(0, p+1):
            if ((p - abs(q)) % 2) != 0:
                continue
            # radial polynomial
            Rad = radialpoly(R, p, q)
            # product = f .* Rad .* exp(-i*q*Theta)
            Product = f * Rad * np.exp(-1j * q * Theta)
            Z = Product.sum()  # sum over all pixels
            Z = ((2*(p+1)) / (np.pi * (Nf**2))) * Z
            A.append(np.abs(Z))
    return A

def process_folder(folder_path, output_xlsx):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                files.append(os.path.join(root, fname))
    files = sorted(files)
    if len(files) == 0:
        print("No image files found in folder:", folder_path)
        return

    rows = []
    header = ['filename', 'Z00','Z11','Z20','Z22','Z31','Z33','Z40','Z42','Z44','Z51','Z53','Z55']

    for filepath in tqdm(files, desc="Processing images"):
        # read grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Warning: could not read", filepath)
            continue
        zm = compute_zernike_moments(img, order=ORDER)
        # If image smaller or non-square, algorithm still produces 12 magnitudes
        # Ensure we have 12 values (order=5). If not, pad with zeros (should not happen).
        if len(zm) != 12:
            zm = zm + [0.0] * (12 - len(zm))
        row = [os.path.relpath(filepath, folder_path)] + zm
        rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    # write Excel and CSV
    df.to_excel(output_xlsx, index=False)
    csv_out = os.path.splitext(output_xlsx)[0] + '.csv'
    df.to_csv(csv_out, index=False)
    print(f"Saved {len(df)} rows to {output_xlsx} and {csv_out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_zernike_batch.py /path/to/image_folder [output.xlsx]")
        sys.exit(1)
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else "zernike_moments.xlsx"
    process_folder(folder, out)
