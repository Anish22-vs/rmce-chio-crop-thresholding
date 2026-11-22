"""
Real pipeline demo: CIE Lab conversion + Gaussian/Guided-style filtering + 
Recursive Minimum Cross Entropy (R-MCE) thresholding, run on actual uploaded crop/leaf images.
This mirrors the MATLAB equivalent given separately to the team.
"""
import numpy as np
from skimage import io, color, filters
import matplotlib.pyplot as plt
import glob
import os
import time

INPUT_DIR = "."   # put WHEAT.jpg and BANANA_LEAVES.jpg in the same folder as this script
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def minimum_cross_entropy_threshold(gray, n_thresholds=2):
    """
    Simple recursive-style minimum cross entropy thresholding.
    Finds n_thresholds that minimize cross entropy between original
    and thresholded (quantized) image histograms.
    """
    hist, bin_edges = np.histogram(gray, bins=256, range=(0, 1), density=False)
    hist = hist.astype(np.float64)
    hist[hist == 0] = 1e-12  # avoid log(0)
    levels = np.arange(256) / 255.0

    def cross_entropy_for_thresholds(thresholds):
        thresholds = sorted(thresholds)
        bounds = [0] + thresholds + [256]
        total_ce = 0.0
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            seg_hist = hist[lo:hi]
            seg_levels = levels[lo:hi]
            seg_sum = seg_hist.sum()
            if seg_sum <= 0:
                continue
            mean_level = (seg_hist * seg_levels).sum() / seg_sum
            if mean_level <= 0:
                continue
            total_ce += (seg_hist * seg_levels * np.log(seg_levels / mean_level + 1e-12)).sum()
        return total_ce

    # brute-force small search since n_thresholds is small (recursive idea: solve incrementally)
    best_thresholds = None
    best_ce = np.inf
    candidates = np.linspace(10, 245, 24).astype(int)
    if n_thresholds == 1:
        for t in candidates:
            ce = cross_entropy_for_thresholds([t])
            if ce < best_ce:
                best_ce, best_thresholds = ce, [t]
    else:
        for i, t1 in enumerate(candidates):
            for t2 in candidates[i + 1:]:
                ce = cross_entropy_for_thresholds([t1, t2])
                if ce < best_ce:
                    best_ce, best_thresholds = ce, [t1, t2]

    return sorted(best_thresholds), best_ce


def process_image(path, idx):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n=== Processing {name} ===")

    rgb = io.imread(path)
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    rgb = rgb / 255.0

    # Step 1: CIE Lab conversion
    lab = color.rgb2lab(rgb)
    L = lab[:, :, 0] / 100.0  # normalize L channel to 0-1

    # Step 2: Gaussian filter (smooth flat regions)
    gaussian_filtered = filters.gaussian(L, sigma=2)

    # Step 3: Guided-filter-style edge-preserving smoothing (approximate using bilateral-like approach)
    from skimage.restoration import denoise_bilateral
    guided_filtered = denoise_bilateral(gaussian_filtered, sigma_color=0.1, sigma_spatial=5)

    # Step 4: R-MCE thresholding
    start = time.time()
    thresholds, ce_value = minimum_cross_entropy_threshold(guided_filtered, n_thresholds=2)
    elapsed = time.time() - start

    # Build segmented image using found thresholds
    segmented = np.zeros_like(guided_filtered)
    bounds = [0] + thresholds + [256]
    scaled = (guided_filtered * 255).astype(int)
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (scaled >= lo) & (scaled < hi)
        segmented[mask] = i / (len(bounds) - 2)

    # Metrics
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    psnr_val = psnr(L, segmented, data_range=1.0)
    ssim_val = ssim(L, segmented, data_range=1.0)
    rmse_val = np.sqrt(np.mean((L - segmented) ** 2))

    # Save figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Crop/Leaf Image")
    axes[1].imshow(L, cmap="gray")
    axes[1].set_title("CIE Lab - L channel")
    axes[2].imshow(guided_filtered, cmap="gray")
    axes[2].set_title("After Gaussian + Guided Filtering")
    axes[3].imshow(segmented, cmap="gray")
    axes[3].set_title(f"Segmented (R-MCE thresholds)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/result_{name}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()

    print(f"Thresholds found: {thresholds} (pixel levels out of 255)")
    print(f"Cross-entropy value: {ce_value:.6f}")
    print(f"PSNR: {psnr_val:.4f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"CPU time: {elapsed:.4f} sec")
    print(f"Saved: {out_path}")

    return {
        "name": name,
        "thresholds": thresholds,
        "cross_entropy": ce_value,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "rmse": rmse_val,
        "cpu_time": elapsed,
        "figure": out_path,
    }


if __name__ == "__main__":
    # Only process these two specific files (avoids picking up unrelated images in the folder)
    target_files = ["WHEAT.jpg", "BANANA_LEAVES.jpg"]
    images = [f for f in target_files if os.path.exists(os.path.join(INPUT_DIR, f))]
    if not images:
        print("ERROR: WHEAT.jpg and BANANA_LEAVES.jpg not found in this folder.")
        print("Place both images in the same folder as this script and run again.")
        exit(1)
    print(f"Found {len(images)} images: {images}")
    results = []
    for i, path in enumerate(images):
        results.append(process_image(path, i))

    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Image':<20}{'Thresholds':<15}{'PSNR':<10}{'SSIM':<10}{'RMSE':<10}{'CPU(s)':<10}")
    for r in results:
        print(f"{r['name']:<20}{str(r['thresholds']):<15}{r['psnr']:<10.3f}{r['ssim']:<10.3f}{r['rmse']:<10.4f}{r['cpu_time']:<10.4f}")
