import numpy as np
import rasterio
from sklearn.decomposition import IncrementalPCA
from rasterio.windows import Window

# ---------- 1. Read pre and post images ----------
pre_path = "C:\\Users\\dt536\\Documents\\optical_high_res\\pre_satellite.tiff"   # pre-event RGB image
post_path = "C:\\Users\\dt536\\Documents\\optical_high_res\\post_satellite.tiff"  # post-event RGB image

# ----------------- 1. Open rasters & basic info -----------------
with rasterio.open(pre_path) as src_pre, rasterio.open(post_path) as src_post:
    assert src_pre.width == src_post.width
    assert src_pre.height == src_post.height
    assert src_pre.count >= 4 and src_post.count >= 4, "Need 4 bands (B,G,R,NIR)"
    
    width = src_pre.width
    height = src_pre.height
    profile = src_pre.profile

print("Image size:", width, "x", height)

# We'll use 4 bands from each: B,G,R,NIR -> 8 features total
num_bands_total = 8

# ----------------- 2. Set up Incremental PCA -----------------
n_components = 3
ipca = IncrementalPCA(n_components=n_components)

# Choose a block size that fits in RAM
block_size = 512  # you can try 512 or 1024

# ----------------- 3. PASS 1: Fit PCA in chunks -----------------
print("Pass 1: fitting IncrementalPCA on blocks...")

with rasterio.open(pre_path) as src_pre, rasterio.open(post_path) as src_post:
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            h = min(block_size, height - row)
            w = min(block_size, width - col)
            win = Window(col, row, w, h)

            # Read 4 bands from each (B,G,R,NIR) as float32
            pre_block = src_pre.read([1, 2, 3, 4], window=win).astype(np.float32)
            post_block = src_post.read([1, 2, 3, 4], window=win).astype(np.float32)

            # Stack: (8, h, w)
            stack_block = np.concatenate([pre_block, post_block], axis=0)

            nb, bh, bw = stack_block.shape  # nb should be 8
            X_block = stack_block.reshape(nb, -1).T  # (pixels_in_block, 8)

            # Optionally skip blocks with all zeros/nodata
            if np.all(X_block == 0):
                continue

            # Incremental fit on this block
            ipca.partial_fit(X_block)

print("PCA components learned.")

# ----------------- 4. Prepare output rasters -----------------
# We’ll output one PCA component image + one change mask
# (you can also save all components if you want)

pc_profile = profile.copy()
pc_profile.update(dtype=rasterio.float32, count=1, nodata=None)

mask_profile = profile.copy()
mask_profile.update(dtype=rasterio.uint8, count=1, nodata=0)

pc_component_path = "pca_component.tif"
change_mask_path = "change_mask.tif"

# Choose which PCA component emphasises change (start with 2nd, index 1)
pc_index = 1

# ----------------- 5. PASS 2: Transform blocks & write outputs -----------------
print("Pass 2: applying PCA and writing outputs...")

with rasterio.open(pre_path) as src_pre, rasterio.open(post_path) as src_post, \
     rasterio.open(pc_component_path, "w", **pc_profile) as dst_pc, \
     rasterio.open(change_mask_path, "w", **mask_profile) as dst_mask:

    # we need some global stats to compute z-scores; you can:
    # (a) do a quick pass to estimate mean/std on a subset OR
    # (b) compute them from all pixels progressively.
    # For simplicity here, we’ll store block means/stds and then normalise per-block
    # relative to the global mean/std approximated from first pass of transform.

    # First, collect rough stats on the chosen component
    comp_values = []

    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            h = min(block_size, height - row)
            w = min(block_size, width - col)
            win = Window(col, row, w, h)

            pre_block = src_pre.read([1, 2, 3, 4], window=win).astype(np.float32)
            post_block = src_post.read([1, 2, 3, 4], window=win).astype(np.float32)
            stack_block = np.concatenate([pre_block, post_block], axis=0)
            nb, bh, bw = stack_block.shape
            X_block = stack_block.reshape(nb, -1).T

            if X_block.size == 0 or np.all(X_block == 0):
                continue

            X_pca_block = ipca.transform(X_block)  # (pixels_in_block, n_components)
            pc_block = X_pca_block[:, pc_index]
            comp_values.append(pc_block)

    # Concatenate sample and get global mean/std
    all_vals = np.concatenate(comp_values)
    global_mean = all_vals.mean()
    global_std = all_vals.std()
    del comp_values, all_vals  # free memory

    print("Estimated global mean/std for component:",
          float(global_mean), float(global_std))

    # Now second transform pass: write normalised component + change mask
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            h = min(block_size, height - row)
            w = min(block_size, width - col)
            win = Window(col, row, w, h)

            pre_block = src_pre.read([1, 2, 3, 4], window=win).astype(np.float32)
            post_block = src_post.read([1, 2, 3, 4], window=win).astype(np.float32)
            stack_block = np.concatenate([pre_block, post_block], axis=0)
            nb, bh, bw = stack_block.shape
            X_block = stack_block.reshape(nb, -1).T

            if X_block.size == 0 or np.all(X_block == 0):
                # write zeros
                pc_img_block = np.zeros((h, w), dtype=np.float32)
                mask_block = np.zeros((h, w), dtype=np.uint8)
            else:
                X_pca_block = ipca.transform(X_block)
                pc_block = X_pca_block[:, pc_index]  # 1D
                pc_img_block = pc_block.reshape(bh, bw)

                # normalise to z-score using global stats
                pc_norm_block = (pc_img_block - global_mean) / (global_std + 1e-6)

                # threshold: |z| > 2 -> change
                threshold = 2.0
                mask_block = (np.abs(pc_norm_block) > threshold).astype(np.uint8)

            # Write this window to the output rasters
            dst_pc.write(pc_img_block.astype(np.float32), 1, window=win)
            dst_mask.write(mask_block, 1, window=win)

print("Done. Wrote", pc_component_path, "and", change_mask_path)