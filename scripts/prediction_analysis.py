#!/usr/bin/env python
# prediction_analysis.py
"""
This script analyzes tire detection performance in large orthomosaic images.
It processes the images in chunks to maintain low memory usage, and implements
proper object-based matching with configurable overlap threshold (default 50%).

Features:
- Memory-efficient processing of extremely large orthomosaic images
- Object-based matching using overlap percentage criteria
- Grid-based regional performance analysis
- Hotspot visualization with density maps
- Zoom insets for detailed examination of problematic regions
- Caching mechanism to save processing time

Usage:
  python prediction_analysis.py --image <ortho.tif> --ground-truth <gt.tif> 
                                --prediction <pred.tif> --output-dir <out_dir>
                                [--overlap-threshold 0.5] [--use-cache] [--visualize-only]
"""

import os
import argparse
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from tqdm import tqdm
import gc
import tempfile
import psutil
import time
import pickle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, ConnectionPatch
import matplotlib.gridspec as gridspec
import warnings

# Try to import optional dependencies with graceful fallback
try:
    from scipy.spatial import ConvexHull
    HAVE_CONVEX_HULL = True
except ImportError:
    HAVE_CONVEX_HULL = False
    warnings.warn("scipy.spatial.ConvexHull not available. Cluster visualization will be limited.")

try:
    from sklearn.cluster import DBSCAN
    HAVE_DBSCAN = True
except ImportError:
    HAVE_DBSCAN = False
    warnings.warn("sklearn.cluster.DBSCAN not available. Cluster visualization will be limited.")

def log_memory_usage(label):
    """Log current memory usage with a label"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    print(f"[{label}] Memory usage: {memory_gb:.2f} GB")

def create_enhanced_colormap():
    """Create a custom colormap for better visualization of hotspots"""
    # Create a colormap that goes from transparent to vibrant red
    colors = [
        (0.0, (0, 0, 0, 0)),         # transparent for lowest values
        (0.3, (1, 0, 0, 0.3)),       # red with some transparency
        (0.6, (1, 0.5, 0, 0.6)),     # orange-red with medium transparency 
        (0.8, (1, 0.8, 0, 0.8)),     # orange-yellow with less transparency
        (1.0, (1, 1, 0, 1))          # yellow fully opaque
    ]
    cmap = LinearSegmentedColormap.from_list('enhanced_hot', colors)
    return cmap

def create_enhanced_hotspot_cmap():
    """Create a custom colormap with better visibility for low values"""
    # Red-yellow colormap with better visibility for low values
    colors = [
        (0.0, (0, 0, 0, 0)),           # transparent for lowest values
        (0.1, (0.8, 0, 0, 0.3)),       # dark red with some transparency
        (0.3, (1, 0.2, 0, 0.5)),       # brighter red with medium transparency
        (0.7, (1, 0.8, 0, 0.7)),       # orange-red with less transparency
        (1.0, (1, 1, 0.3, 0.9))        # yellow-white almost opaque
    ]
    return LinearSegmentedColormap.from_list('enhanced_hotspot', colors)

def calculate_enhanced_density(binary_mask, sigma):
    """Calculate enhanced density with preprocessing for better visualization
    
    Args:
        binary_mask: Binary mask with positive regions
        sigma: Base sigma for Gaussian smoothing
        
    Returns:
        Enhanced density field normalized to [0,1]
    """
    # Optional: dilate the mask first to connect nearby points
    dilated = ndimage.binary_dilation(binary_mask, iterations=2)
    
    # Apply Gaussian filter with large sigma
    density = ndimage.gaussian_filter(dilated.astype(float), sigma=sigma)
    
    # Apply gamma correction to enhance low-density areas
    gamma = 0.5  # Values less than 1 boost visibility of low-density areas
    density = np.power(density, gamma)
    
    # Normalize
    if np.max(density) > 0:
        density = density / np.max(density)
    
    return density

def get_cache_filepath(output_dir, base_name):
    """Generate a consistent cache filepath"""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{base_name}_cache.pkl")

def save_cached_data(data, cache_path):
    """Save processed data to cache file"""
    print(f"Saving cached data to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print("Cache saved successfully.")

def load_cached_data(cache_path):
    """Load processed data from cache file"""
    print(f"Loading cached data from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    print("Cache loaded successfully.")
    return data

def extreme_memory_process(image_path, mask_paths, output_dir, base_name, chunk_size=1024, sigma=3.0, 
                           downscale_factor=8, quick_test=False, use_cache=False, save_cache=True,
                           visualize_only=False, overlap_threshold=0.5):
    """
    Ultra-low memory processing for extremely large images.
    Never loads the full image or masks into memory at once.
    """
    log_memory_usage("Start")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define cache file path
    cache_path = get_cache_filepath(output_dir, base_name)
    
    # Check if we should load from cache
    if use_cache and os.path.exists(cache_path):
        # Load all the preprocessed data
        cached_data = load_cached_data(cache_path)
        
        # Extract data from cache
        downscaled_img = cached_data['downscaled_img']
        downscaled_gt = cached_data['downscaled_gt']
        downscaled_pred = cached_data['downscaled_pred']
        precision = cached_data['precision']
        recall = cached_data['recall']
        f1 = cached_data['f1']
        gt_object_areas = cached_data['gt_object_areas']
        pred_object_areas = cached_data['pred_object_areas']
        gt_centroids = cached_data['gt_centroids']
        pred_centroids = cached_data['pred_centroids']
        precision_grid = cached_data['precision_grid']
        recall_grid = cached_data['recall_grid']
        f1_grid = cached_data['f1_grid']
        tp_grid = cached_data['tp_grid']
        fp_grid = cached_data['fp_grid']
        fn_grid = cached_data['fn_grid']
        total_pixels_grid = cached_data['total_pixels_grid']
        
        # Calculate maximum allowed dimensions (2^16 - 1)
        max_dim = 2**16 - 1
        
        # Get grid size from the cached data
        grid_size = precision_grid.shape[0]
        
        # Calculate scaling factor if image is too large
        scale_factor = 1.0
        if downscaled_img.shape[0] > max_dim or downscaled_img.shape[1] > max_dim:
            scale_factor = min(max_dim / downscaled_img.shape[0], max_dim / downscaled_img.shape[1])
            print(f"Image too large, scaling down by factor {scale_factor:.2f}")
            # Resize image and masks for visualization
            from skimage.transform import resize
            downscaled_img = resize(downscaled_img, 
                                  (int(downscaled_img.shape[0] * scale_factor), 
                                   int(downscaled_img.shape[1] * scale_factor), 
                                   downscaled_img.shape[2]), 
                                  preserve_range=True).astype(np.uint8)
            downscaled_gt = resize(downscaled_gt, 
                                 (int(downscaled_gt.shape[0] * scale_factor), 
                                  int(downscaled_gt.shape[1] * scale_factor)), 
                                 preserve_range=True).astype(np.uint8)
            downscaled_pred = resize(downscaled_pred, 
                                   (int(downscaled_pred.shape[0] * scale_factor), 
                                    int(downscaled_pred.shape[1] * scale_factor)), 
                                   preserve_range=True).astype(np.uint8)
            
            # Scale centroids
            gt_centroids = [(y * scale_factor, x * scale_factor) for y, x in gt_centroids]
            pred_centroids = [(y * scale_factor, x * scale_factor) for y, x in pred_centroids]
        
        # Convert downscaled centroids
        gt_centroids_downscaled = [(y//downscale_factor, x//downscale_factor) for y, x in gt_centroids]
        pred_centroids_downscaled = [(y//downscale_factor, x//downscale_factor) for y, x in pred_centroids]
        
        print("Using cached preprocessed data.")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Objects detected - Ground Truth: {len(gt_object_areas)}, Prediction: {len(pred_object_areas)}")
        
    elif visualize_only:
        # If visualize_only is True but no cache exists, we can't proceed
        if not os.path.exists(cache_path):
            print("Error: visualize_only requires existing cache data, but no cache was found.")
            print(f"Expected cache file: {cache_path}")
            return
        
        # Load the cache since visualize_only is True but use_cache wasn't explicitly set
        cached_data = load_cached_data(cache_path)
        
        # Extract data from cache
        downscaled_img = cached_data['downscaled_img']
        downscaled_gt = cached_data['downscaled_gt']
        downscaled_pred = cached_data['downscaled_pred']
        precision = cached_data['precision']
        recall = cached_data['recall']
        f1 = cached_data['f1']
        gt_object_areas = cached_data['gt_object_areas']
        pred_object_areas = cached_data['pred_object_areas']
        gt_centroids = cached_data['gt_centroids']
        pred_centroids = cached_data['pred_centroids']
        precision_grid = cached_data['precision_grid']
        recall_grid = cached_data['recall_grid']
        f1_grid = cached_data['f1_grid']
        tp_grid = cached_data['tp_grid']
        fp_grid = cached_data['fp_grid']
        fn_grid = cached_data['fn_grid']
        total_pixels_grid = cached_data['total_pixels_grid']
        
        # Get grid size from the cached data
        grid_size = precision_grid.shape[0]
        
        # Convert downscaled centroids
        gt_centroids_downscaled = [(y//downscale_factor, x//downscale_factor) for y, x in gt_centroids]
        pred_centroids_downscaled = [(y//downscale_factor, x//downscale_factor) for y, x in pred_centroids]
        
    else:
        # Create temporary directory for intermediate results
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory for intermediate results: {temp_dir}")
        
        # Handle quick test mode if requested
        if quick_test:
            # For quick test, we'll first downsample the inputs
            print(f"QUICK TEST MODE: Downscaling inputs by {downscale_factor}x for faster processing")
            
            # Read image dimensions first
            with rasterio.open(image_path) as src:
                # Calculate the scaled dimensions
                scaled_width = src.width // downscale_factor
                scaled_height = src.height // downscale_factor
            
            # Create temporary files for downscaled inputs
            temp_image_path = os.path.join(temp_dir, "temp_image.tif")
            temp_gt_path = os.path.join(temp_dir, "temp_gt.tif")
            temp_pred_path = os.path.join(temp_dir, "temp_pred.tif")
            
            # Downsample input image
            print("Downsampling input image...")
            try:
                with rasterio.open(image_path) as src:
                    image_data = src.read(
                        out_shape=(src.count, scaled_height, scaled_width),
                        resampling=rasterio.enums.Resampling.average
                    )
                    
                    # Create a temporary downscaled image
                    with rasterio.open(
                        temp_image_path, 'w',
                        driver='GTiff',
                        height=scaled_height,
                        width=scaled_width,
                        count=src.count,
                        dtype=image_data.dtype,
                        crs=src.crs,
                        transform=rasterio.transform.from_bounds(
                            *src.bounds, scaled_width, scaled_height
                        )
                    ) as dst:
                        dst.write(image_data)
            except Exception as e:
                print(f"Error during image downsampling: {e}")
                print("Trying alternative downsampling approach...")
                
                # Alternative approach: read and downscale in chunks
                with rasterio.open(image_path) as src:
                    create_manually_downsampled_image(src, temp_image_path, downscale_factor)
            
            # Downsample the ground truth mask
            print("Downsampling ground truth mask...")
            try:
                with rasterio.open(mask_paths['ground_truth']) as gt_src:
                    # Use max pooling approach for masks to preserve small objects
                    create_presence_preserving_mask(gt_src, temp_gt_path, downscale_factor)
            except Exception as e:
                print(f"Error during ground truth downsampling: {e}")
                print("Trying alternative mask downsampling approach...")
                with rasterio.open(mask_paths['ground_truth']) as gt_src:
                    create_manually_downsampled_mask(gt_src, temp_gt_path, downscale_factor)
            
            # Downsample the prediction mask
            print("Downsampling prediction mask...")
            try:
                with rasterio.open(mask_paths['prediction']) as pred_src:
                    # Use max pooling approach for masks to preserve small objects
                    create_presence_preserving_mask(pred_src, temp_pred_path, downscale_factor)
            except Exception as e:
                print(f"Error during prediction downsampling: {e}")
                print("Trying alternative mask downsampling approach...")
                with rasterio.open(mask_paths['prediction']) as pred_src:
                    create_manually_downsampled_mask(pred_src, temp_pred_path, downscale_factor)
            
            # Override paths with downsampled versions
            image_path = temp_image_path
            mask_paths = {
                'ground_truth': temp_gt_path,
                'prediction': temp_pred_path
            }
            
            # Get dimensions from the downsampled image
            with rasterio.open(image_path) as src:
                image_height, image_width = src.height, src.width
            
            # Since we're already working with downscaled data, use a smaller downscale factor for visualization
            vis_downscale = max(1, downscale_factor // 4)
            
            print(f"Quick test image: {image_width}x{image_height} pixels")
            print(f"Quick test visualization: {image_width//vis_downscale}x{image_height//vis_downscale} pixels (downscaled by {vis_downscale}x)")
            
            # Adjust chunk size for downscaled images
            chunk_size = min(chunk_size, image_width, image_height)
            print(f"Using chunk size: {chunk_size}")
            print("Quick test setup completed successfully")
        else:
            # Normal mode - full resolution processing
            with rasterio.open(image_path) as src:
                image_height, image_width = src.height, src.width
            
            # Use the specified downscale factor for visualization
            vis_downscale = downscale_factor
            
            print(f"Original image: {image_width}x{image_height} pixels")
            print(f"Visualization size: {image_width//vis_downscale}x{image_height//vis_downscale} pixels (downscaled by {vis_downscale}x)")
        
        # Create downscaled versions of the input data for visualization
        print("Creating downscaled version of the image for visualization...")
        downscaled_img_path = os.path.join(temp_dir, "downscaled_image.npy")
        create_downscaled_image(image_path, downscaled_img_path, vis_downscale)
        
        print("Creating downscaled version of ground truth mask...")
        downscaled_gt_path = os.path.join(temp_dir, "downscaled_gt.npy")
        create_downscaled_mask(mask_paths['ground_truth'], downscaled_gt_path, vis_downscale)
        
        print("Creating downscaled version of prediction mask...")
        downscaled_pred_path = os.path.join(temp_dir, "downscaled_pred.npy")
        create_downscaled_mask(mask_paths['prediction'], downscaled_pred_path, vis_downscale)
        
        # Get number of chunks in each dimension
        num_chunks_x = (image_width + chunk_size - 1) // chunk_size
        num_chunks_y = (image_height + chunk_size - 1) // chunk_size
        total_chunks = num_chunks_x * num_chunks_y
        
        print(f"Processing original resolution in {total_chunks} chunks ({num_chunks_x}x{num_chunks_y})...")
        
        # Initialize statistics collection
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        gt_object_areas = []
        pred_object_areas = []
        
        # Store centroids of tire objects for later visualization
        gt_centroids = []
        pred_centroids = []
        
        # Create grid for error analysis
        # We'll use a coarse grid (e.g., 20x20) to summarize errors across the image
        grid_size = 20  # Number of cells in each dimension
        grid_width = (image_width + grid_size - 1) // grid_size
        grid_height = (image_height + grid_size - 1) // grid_size
        
        # Initialize error grid arrays
        tp_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        fp_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        fn_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        total_pixels_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # Process chunks one by one
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            # Process image with properly opened file handles
            with rasterio.open(image_path) as src:
                for y_idx in range(num_chunks_y):
                    for x_idx in range(num_chunks_x):
                        # Calculate chunk bounds
                        x_start = x_idx * chunk_size
                        y_start = y_idx * chunk_size
                        x_end = min(x_start + chunk_size, image_width)
                        y_end = min(y_start + chunk_size, image_height)
                        
                        # Define window for this chunk
                        window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                        
                        # Check if we have alpha band (for nodata mask)
                        if src.count >= 4:
                            alpha_chunk = src.read(4, window=window)
                            valid_data_mask = (alpha_chunk == 255)
                        else:
                            valid_data_mask = np.ones((y_end - y_start, x_end - x_start), dtype=bool)
                        
                        # Read ground truth and prediction for this chunk
                        gt_chunk = read_mask_chunk(mask_paths['ground_truth'], window)
                        pred_chunk = read_mask_chunk(mask_paths['prediction'], window)
                        
                        # Ensure dimensions match before applying mask
                        if gt_chunk.shape != valid_data_mask.shape:
                            # Resize valid_data_mask to match gt_chunk dimensions
                            valid_data_mask = np.ones_like(gt_chunk, dtype=bool)
                        
                        # Apply valid data mask
                        gt_chunk[~valid_data_mask] = 0
                        pred_chunk[~valid_data_mask] = 0
                        
                        # Compute pixel-level statistics
                        tp_chunk = np.logical_and(gt_chunk == 1, pred_chunk == 1).sum()
                        fp_chunk = np.logical_and(gt_chunk == 0, pred_chunk == 1).sum()
                        fn_chunk = np.logical_and(gt_chunk == 1, pred_chunk == 0).sum()
                        valid_pixels = valid_data_mask.sum()
                        
                        # Update grid statistics
                        # Calculate which grid cell(s) this chunk belongs to
                        grid_x_start = x_start // grid_width
                        grid_y_start = y_start // grid_height
                        grid_x_end = min(grid_x_start + 1, grid_size - 1)
                        grid_y_end = min(grid_y_start + 1, grid_size - 1)
                        
                        # Update grid cells
                        grid_x_idx = min(grid_x_start, grid_size - 1)
                        grid_y_idx = min(grid_y_start, grid_size - 1)
                        
                        tp_grid[grid_y_idx, grid_x_idx] += tp_chunk
                        fp_grid[grid_y_idx, grid_x_idx] += fp_chunk
                        fn_grid[grid_y_idx, grid_x_idx] += fn_chunk
                        total_pixels_grid[grid_y_idx, grid_x_idx] += valid_pixels
                        
                        true_positives += tp_chunk
                        false_positives += fp_chunk
                        false_negatives += fn_chunk
                        
                        # Analyze objects in this chunk if it contains any positive pixels
                        if np.any(gt_chunk == 1) or np.any(pred_chunk == 1):
                            # Label connected regions
                            gt_labeled = measure.label(gt_chunk)
                            pred_labeled = measure.label(pred_chunk)
                            
                            # Match objects using overlap criterion
                            matched_pairs, unmatched_gt, unmatched_pred = match_objects(
                                gt_labeled, pred_labeled, overlap_threshold=overlap_threshold
                            )
                            
                            # Extract object properties
                            gt_regions = {r.label: r for r in measure.regionprops(gt_labeled)}
                            pred_regions = {r.label: r for r in measure.regionprops(pred_labeled)}
                            
                            # Process matched pairs (TP)
                            for gt_id, pred_id, overlap in matched_pairs:
                                # Only include objects that don't touch the border
                                if (not region_touches_border(gt_regions[gt_id], gt_chunk.shape) and
                                    not region_touches_border(pred_regions[pred_id], pred_chunk.shape)):
                                    
                                    # Store object areas for size distribution analysis
                                    gt_object_areas.append(gt_regions[gt_id].area)
                                    pred_object_areas.append(pred_regions[pred_id].area)
                                    
                                    # Store centroids with global coordinates
                                    gt_y, gt_x = gt_regions[gt_id].centroid
                                    pred_y, pred_x = pred_regions[pred_id].centroid
                                    
                                    # Use global coordinates (add chunk offset)
                                    gt_centroid = (y_start + gt_y, x_start + gt_x)
                                    pred_centroid = (y_start + pred_y, x_start + pred_x)
                                    
                                    # Add to TP centroids (we'll store both for analysis)
                                    gt_centroids.append(gt_centroid)
                                    pred_centroids.append(pred_centroid)
                                    
                                    # For grid analysis, calculate which grid cell this falls into
                                    grid_x_idx = min(int(gt_x + x_start) // grid_width, grid_size - 1)
                                    grid_y_idx = min(int(gt_y + y_start) // grid_height, grid_size - 1)
                                    
                                    # Count as TP for this grid cell
                                    tp_grid[grid_y_idx, grid_x_idx] += 1
                                    
                                    # Count in overall TP
                                    true_positives += 1
                            
                            # Process unmatched ground truth objects (FN)
                            for gt_id in unmatched_gt:
                                if not region_touches_border(gt_regions[gt_id], gt_chunk.shape):
                                    # Store size for analysis
                                    gt_object_areas.append(gt_regions[gt_id].area)
                                    
                                    # Store centroids with global coordinates
                                    gt_y, gt_x = gt_regions[gt_id].centroid
                                    gt_centroid = (y_start + gt_y, x_start + gt_x)
                                    gt_centroids.append(gt_centroid)
                                    
                                    # For grid analysis
                                    grid_x_idx = min(int(gt_x + x_start) // grid_width, grid_size - 1)
                                    grid_y_idx = min(int(gt_y + y_start) // grid_height, grid_size - 1)
                                    
                                    # Count as FN for this grid cell
                                    fn_grid[grid_y_idx, grid_x_idx] += 1
                                    
                                    # Count in overall FN
                                    false_negatives += 1
                            
                            # Process unmatched prediction objects (FP)
                            for pred_id in unmatched_pred:
                                if not region_touches_border(pred_regions[pred_id], pred_chunk.shape):
                                    # Store size for analysis
                                    pred_object_areas.append(pred_regions[pred_id].area)
                                    
                                    # Store centroids with global coordinates
                                    pred_y, pred_x = pred_regions[pred_id].centroid
                                    pred_centroid = (y_start + pred_y, x_start + pred_x)
                                    pred_centroids.append(pred_centroid)
                                    
                                    # For grid analysis
                                    grid_x_idx = min(int(pred_x + x_start) // grid_width, grid_size - 1)
                                    grid_y_idx = min(int(pred_y + y_start) // grid_height, grid_size - 1)
                                    
                                    # Count as FP for this grid cell
                                    fp_grid[grid_y_idx, grid_x_idx] += 1
                                    
                                    # Count in overall FP
                                    false_positives += 1
                        
                        # Clear memory after processing each chunk
                        del gt_chunk, pred_chunk, valid_data_mask
                        if 'alpha_chunk' in locals():
                            del alpha_chunk
                        gc.collect()
                        
                        pbar.update(1)
        
        # Calculate precision and recall for each grid cell
        precision_grid = np.zeros((grid_size, grid_size))
        recall_grid = np.zeros((grid_size, grid_size))
        f1_grid = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                if tp_grid[i, j] + fp_grid[i, j] > 0:
                    precision_grid[i, j] = tp_grid[i, j] / (tp_grid[i, j] + fp_grid[i, j])
                if tp_grid[i, j] + fn_grid[i, j] > 0:
                    recall_grid[i, j] = tp_grid[i, j] / (tp_grid[i, j] + fn_grid[i, j])
                if precision_grid[i, j] + recall_grid[i, j] > 0:
                    f1_grid[i, j] = 2 * (precision_grid[i, j] * recall_grid[i, j]) / (precision_grid[i, j] + recall_grid[i, j])
        
        # Calculate overall summary metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nAnalysis complete.")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Objects detected - Ground Truth: {len(gt_object_areas)}, Prediction: {len(pred_object_areas)}")
        
        # Load downscaled versions for visualization
        print("Loading downscaled data for visualization...")
        downscaled_img = np.load(downscaled_img_path)
        downscaled_gt = np.load(downscaled_gt_path)
        downscaled_pred = np.load(downscaled_pred_path)
        
        # Calculate downscaled centroids
        gt_centroids_downscaled = [(y//vis_downscale, x//vis_downscale) for y, x in gt_centroids]
        pred_centroids_downscaled = [(y//vis_downscale, x//vis_downscale) for y, x in pred_centroids]
        
        # Save all processed data to cache for future visualization runs if requested
        if save_cache:
            cache_data = {
                'downscaled_img': downscaled_img,
                'downscaled_gt': downscaled_gt,
                'downscaled_pred': downscaled_pred,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'gt_object_areas': gt_object_areas,
                'pred_object_areas': pred_object_areas,
                'gt_centroids': gt_centroids,
                'pred_centroids': pred_centroids,
                'precision_grid': precision_grid,
                'recall_grid': recall_grid,
                'f1_grid': f1_grid,
                'tp_grid': tp_grid,
                'fp_grid': fp_grid,
                'fn_grid': fn_grid,
                'total_pixels_grid': total_pixels_grid,
                'downscale_factor': downscale_factor,
                'overlap_threshold': overlap_threshold
            }
            save_cached_data(cache_data, cache_path)
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for temp_file in [downscaled_img_path, downscaled_gt_path, downscaled_pred_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        try:
            if quick_test:
                # Remove the downscaled input files
                for temp_file in [temp_image_path, temp_gt_path, temp_pred_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            os.rmdir(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        except:
            print(f"Could not remove temporary directory: {temp_dir}")
    
    # Generate visualizations
    log_memory_usage("Before visualization")
    
    # Define enhanced_sigma outside the try block to ensure it's available for later use
    enhanced_sigma = sigma * 15  # Increase smoothing for better visibility
    
    try:
        print("\nStarting visualization generation...")
        
        # Create visualization-ready masks
        print("Creating error mask...")
        print(f"Image shape: {downscaled_img.shape}, GT shape: {downscaled_gt.shape}, Pred shape: {downscaled_pred.shape}")
        
        # Ensure all arrays have the same dimensions by taking the minimum dimensions
        min_height = min(downscaled_img.shape[0], downscaled_gt.shape[0], downscaled_pred.shape[0])
        min_width = min(downscaled_img.shape[1], downscaled_gt.shape[1], downscaled_pred.shape[1])
        
        # Crop all arrays to the same size
        downscaled_img = downscaled_img[:min_height, :min_width]
        downscaled_gt = downscaled_gt[:min_height, :min_width]
        downscaled_pred = downscaled_pred[:min_height, :min_width]
        
        error_mask = np.zeros((min_height, min_width, 3), dtype=np.uint8)
        
        # True positives (green) - make it more visible
        error_mask[np.logical_and(downscaled_gt == 1, downscaled_pred == 1)] = [0, 255, 0]
        
        # False positives (red) - make it more visible
        error_mask[np.logical_and(downscaled_gt == 0, downscaled_pred == 1)] = [255, 0, 0]
        
        # False negatives (blue) - make it more visible
        error_mask[np.logical_and(downscaled_gt == 1, downscaled_pred == 0)] = [0, 0, 255]
        print("Error mask created successfully")
        
        # Create enhanced hotspot visualization with larger sigma for better visibility
        print("\nCalculating hotspot density...")
        
        # Use the enhanced density calculation instead of direct Gaussian filter
        density = calculate_enhanced_density(downscaled_pred, enhanced_sigma)
        print("Hotspot density calculated successfully")
        
        # Generate error visualization
        print("\nGenerating error visualization...")
        error_vis_path = os.path.join(output_dir, f"{base_name}_error_analysis.png")
        generate_enhanced_error_visualization(downscaled_img, downscaled_gt, downscaled_pred, error_mask, error_vis_path)
        print(f"Error visualization saved to: {error_vis_path}")
        
        # Generate hotspot visualization
        print("\nGenerating hotspot visualization...")
        hotspot_vis_path = os.path.join(output_dir, f"{base_name}_hotspot.png")
        generate_enhanced_hotspot_visualization(downscaled_img, density, pred_centroids_downscaled, hotspot_vis_path)
        print(f"Hotspot visualization saved to: {hotspot_vis_path}")
        
        # Generate size distribution plot
        print("\nGenerating size distribution plot...")
        size_dist_path = os.path.join(output_dir, f"{base_name}_size_distribution.png")
        generate_size_distribution_plot(gt_object_areas, pred_object_areas, size_dist_path)
        print(f"Size distribution plot saved to: {size_dist_path}")
        
        # Generate grid-based error density map
        print("\nGenerating grid-based error analysis...")
        grid_vis_path = os.path.join(output_dir, f"{base_name}_grid_error.png")
        generate_grid_error_visualization(downscaled_img, precision_grid, recall_grid, f1_grid, 
                                       tp_grid, fp_grid, fn_grid, total_pixels_grid,
                                       grid_vis_path,
                                       gt_centroids_downscaled, pred_centroids_downscaled)
        print(f"Grid error visualization saved to: {grid_vis_path}")
        
        # Find regions with highest error rates for detailed zoom analysis
        print("\nPreparing zoomed inset visualizations...")
        # Calculate error density for each grid cell
        error_density = np.zeros_like(total_pixels_grid, dtype=float)
        mask = total_pixels_grid > 0  # Avoid division by zero
        error_density[mask] = (fp_grid[mask] + fn_grid[mask]) / total_pixels_grid[mask]
        
        # Find top N error regions
        top_n = 3  # Show top 3 error regions
        high_error_cells = []
        if np.any(error_density > 0):
            flat_indices = np.argsort(error_density.flatten())[-top_n:]
            for flat_idx in flat_indices:
                if error_density.flatten()[flat_idx] > 0:  # Only include cells with errors
                    i, j = np.unravel_index(flat_idx, error_density.shape)
                    high_error_cells.append((i, j))
        
        # Generate zoomed inset visualization
        if high_error_cells:
            print(f"Found {len(high_error_cells)} high-error regions for zoom analysis")
            zoom_vis_path = os.path.join(output_dir, f"{base_name}_zoom_insets.png")
            generate_zoom_insets(downscaled_img, downscaled_gt, downscaled_pred, error_mask, 
                               high_error_cells, grid_size, zoom_vis_path)
            print(f"Zoom inset visualization saved to: {zoom_vis_path}")
        else:
            print("No high-error regions found for zoom analysis")
        
        print("\nAll visualizations generated successfully!")
        
    except Exception as e:
        print(f"\nError during visualization generation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Generate comparative hotspot visualization showing ground truth vs prediction
    print("Generating comparative hotspot visualization...")
    
    # Use the enhanced density calculation for both ground truth and prediction
    gt_density = calculate_enhanced_density(downscaled_gt, enhanced_sigma)
    pred_density = calculate_enhanced_density(downscaled_pred, enhanced_sigma)
        
    # Calculate differential density (prediction - ground truth)
    diff_density = pred_density - gt_density
    
    # Generate comparative visualization
    generate_comparative_hotspot(downscaled_img, gt_density, pred_density, diff_density,
                                gt_centroids_downscaled, pred_centroids_downscaled,
                                os.path.join(output_dir, f"{base_name}_comparative_hotspot.png"))
    
    # Generate hierarchical zoom visualization if we have the dependencies
    if HAVE_CONVEX_HULL and HAVE_DBSCAN and len(pred_centroids_downscaled) > 0:
        print("Generating hierarchical zoom visualization...")
        generate_hierarchical_zoom(downscaled_img, pred_centroids_downscaled, error_mask,
                                  os.path.join(output_dir, f"{base_name}_hierarchical_zoom.png"))
    
    log_memory_usage("End")
    print(f"All visualizations saved to: {output_dir}")

def region_touches_border(region, shape):
    """Check if a region touches the border of the image"""
    min_row, min_col, max_row, max_col = region.bbox
    return min_row == 0 or min_col == 0 or max_row == shape[0] or max_col == shape[1]

def create_downscaled_image(image_path, output_path, scale_factor):
    """Create a downscaled version of an image for visualization"""
    log_memory_usage("Before downscaling image")
    
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        vis_height = height // scale_factor
        vis_width = width // scale_factor
        
        # Create empty array for result
        downscaled = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Process in chunks to avoid loading the entire image
        chunk_size = 1024
        num_chunks_x = (width + chunk_size - 1) // chunk_size
        num_chunks_y = (height + chunk_size - 1) // chunk_size
        
        with tqdm(total=num_chunks_y * num_chunks_x, desc="Downscaling image") as pbar:
            for y_idx in range(num_chunks_y):
                for x_idx in range(num_chunks_x):
                    # Calculate chunk bounds
                    x_start = x_idx * chunk_size
                    y_start = y_idx * chunk_size
                    x_end = min(x_start + chunk_size, width)
                    y_end = min(y_start + chunk_size, height)
                    
                    # Read chunk
                    window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                    chunk = src.read([1,2,3], window=window)
                    chunk = np.transpose(chunk, (1, 2, 0))
                    
                    # Check for alpha/nodata
                    if src.count >= 4:
                        alpha = src.read(4, window=window)
                        # Apply alpha mask
                        for i in range(3):
                            chunk[:,:,i][alpha != 255] = 0
                    
                    # Calculate corresponding region in downscaled image
                    ds_x_start = x_start // scale_factor
                    ds_y_start = y_start // scale_factor
                    ds_x_end = min((x_end + scale_factor - 1) // scale_factor, vis_width)  # Fix: Add min to ensure within bounds
                    ds_y_end = min((y_end + scale_factor - 1) // scale_factor, vis_height)  # Fix: Add min to ensure within bounds
                    
                    # Resize chunk using simple averaging
                    for i in range(ds_y_start, ds_y_end):
                        for j in range(ds_x_start, ds_x_end):
                            # Get original image coordinates
                            orig_y_start = i * scale_factor
                            orig_x_start = j * scale_factor
                            orig_y_end = min(orig_y_start + scale_factor, height)
                            orig_x_end = min(orig_x_start + scale_factor, width)
                            
                            # Check if this pixel is within the current chunk
                            if (orig_y_start >= y_start and orig_y_end <= y_end and 
                                orig_x_start >= x_start and orig_x_end <= x_end):
                                
                                # Extract region from chunk
                                region = chunk[orig_y_start - y_start:orig_y_end - y_start,
                                              orig_x_start - x_start:orig_x_end - x_start, :]
                                
                                # Average the values if there are any valid pixels
                                if region.size > 0:
                                    # Average each channel
                                    for c in range(3):
                                        channel = region[:,:,c]
                                        if np.any(channel > 0):  # Only consider non-zero pixels
                                            downscaled[i, j, c] = int(np.mean(channel[channel > 0]))
                    
                    pbar.update(1)
    
    # Save downscaled image
    np.save(output_path, downscaled)
    log_memory_usage("After downscaling image")

def create_downscaled_mask(mask_path, output_path, scale_factor):
    """Create a downscaled version of a mask for visualization"""
    log_memory_usage("Before downscaling mask")
    
    with rasterio.open(mask_path) as src:
        height, width = src.height, src.width
        vis_height = height // scale_factor
        vis_width = width // scale_factor
        
        # Create empty array for result
        downscaled = np.zeros((vis_height, vis_width), dtype=np.uint8)
        
        # Process in chunks
        chunk_size = 1024
        num_chunks_x = (width + chunk_size - 1) // chunk_size
        num_chunks_y = (height + chunk_size - 1) // chunk_size
        
        with tqdm(total=num_chunks_y * num_chunks_x, desc="Downscaling mask") as pbar:
            for y_idx in range(num_chunks_y):
                for x_idx in range(num_chunks_x):
                    # Calculate chunk bounds
                    x_start = x_idx * chunk_size
                    y_start = y_idx * chunk_size
                    x_end = min(x_start + chunk_size, width)
                    y_end = min(y_start + chunk_size, height)
                    
                    # Read chunk
                    window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                    chunk = src.read(1, window=window)
                    
                    # Normalize mask values
                    chunk = np.where(chunk == 255, 0, chunk)
                    chunk = np.where(chunk > 0, 1, 0).astype(np.uint8)
                    
                    # Calculate corresponding region in downscaled image
                    ds_x_start = x_start // scale_factor
                    ds_y_start = y_start // scale_factor
                    ds_x_end = min((x_end + scale_factor - 1) // scale_factor, vis_width)  # Fix: Add min to ensure within bounds
                    ds_y_end = min((y_end + scale_factor - 1) // scale_factor, vis_height)  # Fix: Add min to ensure within bounds
                    
                    # Enhanced downscaling for masks - maintain tire visibility
                    # If any pixel in the block is 1, set the downscaled pixel to 1
                    # This helps prevent small objects from disappearing during downscaling
                    for i in range(ds_y_start, ds_y_end):
                        for j in range(ds_x_start, ds_x_end):
                            # Get original image coordinates
                            orig_y_start = i * scale_factor
                            orig_x_start = j * scale_factor
                            orig_y_end = min(orig_y_start + scale_factor, height)
                            orig_x_end = min(orig_x_start + scale_factor, width)
                            
                            # Check if this pixel is within the current chunk
                            if (orig_y_start >= y_start and orig_y_end <= y_end and 
                                orig_x_start >= x_start and orig_x_end <= x_end):
                                
                                # Extract region from chunk
                                region = chunk[orig_y_start - y_start:orig_y_end - y_start,
                                              orig_x_start - x_start:orig_x_end - x_start]
                                
                                # If any pixel is 1, set downscaled pixel to 1
                                if np.any(region > 0):
                                    downscaled[i, j] = 1
                    
                    pbar.update(1)
    
    # For mask downscaling, we always apply dilation to ensure small objects remain visible
    # This is particularly important for preserving tire visibility during high downscaling ratios
    # Use a stronger dilation for higher scale factors to preserve very small objects
    dilation_iterations = max(2, min(5, scale_factor // 8))  # Scale dilation with downscaling factor
    downscaled = ndimage.binary_dilation(downscaled, iterations=dilation_iterations).astype(np.uint8)
    
    # Save downscaled mask
    np.save(output_path, downscaled)
    log_memory_usage("After downscaling mask")

def read_mask_chunk(mask_path, window):
    """Read a chunk from a mask file"""
    with rasterio.open(mask_path) as src:
        chunk = src.read(1, window=window)
        # Normalize mask values
        chunk = np.where(chunk == 255, 0, chunk)
        chunk = np.where(chunk > 0, 1, 0).astype(np.uint8)
    return chunk

def generate_enhanced_error_visualization(image, ground_truth, prediction, error_mask, output_path):
    """Generate enhanced error visualization from pre-processed data"""
    # Dilate the error mask to make it more visible
    dilated_error = np.zeros_like(error_mask)
    for i in range(3):
        if np.any(error_mask[:,:,i] > 0):
            dilated_error[:,:,i] = ndimage.binary_dilation(error_mask[:,:,i], iterations=3)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    
    # Create visualization-friendly ground truth display with enhanced contrast
    gt_display = np.copy(ground_truth)
    axes[0, 1].imshow(gt_display, cmap='gray')
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    
    # Create visualization-friendly prediction display with enhanced contrast
    pred_display = np.copy(prediction)
    axes[1, 0].imshow(pred_display, cmap='gray')
    axes[1, 0].set_title('Prediction', fontsize=14)
    
    # Create a more vibrant error visualization that will be visible
    # Use the original image at reduced opacity as background
    axes[1, 1].imshow(image, alpha=0.5)  # Reduced opacity for background
    
    # Create higher contrast error visualization
    error_vis = np.zeros_like(image, dtype=np.uint8)
    # True positives (bright green)
    error_vis[np.logical_and(ground_truth == 1, prediction == 1)] = [0, 255, 0]
    # False positives (bright red)
    error_vis[np.logical_and(ground_truth == 0, prediction == 1)] = [255, 0, 0]
    # False negatives (bright blue)
    error_vis[np.logical_and(ground_truth == 1, prediction == 0)] = [0, 0, 255]
    
    # Show error visualization with high opacity
    axes[1, 1].imshow(error_vis, alpha=0.8)  # Higher opacity for errors
    axes[1, 1].set_title('Error Analysis', fontsize=14)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='k', label='True Positive'),
        Patch(facecolor='red', edgecolor='k', label='False Positive'),
        Patch(facecolor='blue', edgecolor='k', label='False Negative')
    ]
    axes[1, 1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_enhanced_hotspot_visualization(image, density, centroids, output_path):
    """Generate enhanced hotspot visualization from pre-processed data"""
    # Use custom enhanced colormap
    hot_cmap = create_enhanced_hotspot_cmap()
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Show the original image
    ax.imshow(image)
    
    # Apply a threshold to the density map to make hotspots more visible
    threshold = 0.01  # Only show areas with density above this threshold
    density_masked = np.ma.masked_where(density < threshold, density)
    
    # Show the density map with increased alpha for better visibility
    heatmap = ax.imshow(density_masked, alpha=0.8, cmap=hot_cmap)
    
    # Add much smaller markers at tire centroids for better visibility
    # Use semi-transparent cyan dots instead of circles
    for y, x in centroids:
        ax.scatter(x, y, color='cyan', alpha=0.5, s=10)  # Smaller points with s=10 instead of circles
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Tire Density', rotation=270, labelpad=20)
    
    plt.title('Tire Hotspot Analysis', fontsize=16)
    
    # Add annotation about ground truth limitations
    ax.annotate("Note: Density map highlights areas with tire concentrations\n"
                "Manually verified model detections include many actual unlabeled tires", 
                xy=(0.5, 0.02), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_size_distribution_plot(gt_areas, pred_areas, output_path):
    """Generate size distribution visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate the pixel area in square centimeters (assuming 3cm/pixel)
    pixel_area_sqcm = 3 * 3  # 3cm × 3cm
    
    # Convert areas from pixels to square centimeters
    if gt_areas:
        gt_areas_sqcm = [area * pixel_area_sqcm for area in gt_areas]
        ax.hist(gt_areas_sqcm, bins=30, alpha=0.5, label='Ground Truth')
    
    if pred_areas:
        pred_areas_sqcm = [area * pixel_area_sqcm for area in pred_areas]
        ax.hist(pred_areas_sqcm, bins=30, alpha=0.5, label='Prediction')
    
    ax.set_xlabel('Tire Area (sq. cm)')
    ax.set_ylabel('Count')
    ax.set_title('Size Distribution of Tires')
    
    if gt_areas or pred_areas:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_grid_error_visualization(image, precision_grid, recall_grid, f1_grid, 
                                     tp_grid, fp_grid, fn_grid, total_grid, output_path,
                                     gt_centroids, pred_centroids):
    """Generate a grid-based error density visualization with object-wise metrics"""
    # Import skimage.transform for resizing
    from skimage.transform import resize
    from scipy import ndimage
    
    # Calculate maximum allowed dimensions (2^16 - 1)
    max_dim = 2**16 - 1
    
    # Calculate scaling factor if image is too large
    scale_factor = 1.0
    if image.shape[0] > max_dim or image.shape[1] > max_dim:
        scale_factor = min(max_dim / image.shape[0], max_dim / image.shape[1])
        print(f"Image too large, scaling down by factor {scale_factor:.2f}")
        # Resize image for visualization
        image = resize(image, (int(image.shape[0] * scale_factor), 
                             int(image.shape[1] * scale_factor), 
                             image.shape[2]), 
                      preserve_range=True).astype(np.uint8)
    
    # Take a completely different approach: instead of creating one large visualization,
    # create separate images for each panel, which will be much smaller
    print("Creating separate visualizations for each panel to avoid size limits")
    
    # Get grid dimensions
    grid_size = precision_grid.shape[0]
    vis_height, vis_width = image.shape[:2]
    cell_height = vis_height / grid_size
    cell_width = vis_width / grid_size
    
    # First, save a highly downsampled overview of the entire area
    overview_scale = 0.05  # Use just 5% of the original size
    overview_img = resize(image, 
                        (int(image.shape[0] * overview_scale), 
                         int(image.shape[1] * overview_scale), 3),
                        preserve_range=True).astype(np.uint8)
    
    overview_path = output_path.replace('.png', '_overview.png')
    plt.figure(figsize=(8, 8))
    plt.imshow(overview_img)
    plt.title("Overview of Analysis Area")
    plt.axis('off')
    plt.savefig(overview_path, dpi=100)
    plt.close()
    print(f"Saved overview image to {overview_path}")
    
    # Save a text summary of metrics
    summary_path = output_path.replace('.png', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Grid-based Analysis Summary\n")
        f.write("==========================\n\n")
        f.write(f"Grid size: {grid_size}x{grid_size}\n")
        f.write(f"Number of ground truth objects: {len(gt_centroids)}\n")
        f.write(f"Number of predicted objects: {len(pred_centroids)}\n\n")
        
        # Calculate overall metrics
        total_tp = np.sum(tp_grid)
        total_fp = np.sum(fp_grid)
        total_fn = np.sum(fn_grid)
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        f.write(f"Overall Precision: {overall_precision:.4f}\n")
        f.write(f"Overall Recall: {overall_recall:.4f}\n")
        f.write(f"Overall F1 Score: {overall_f1:.4f}\n\n")
        
        # Find top error regions
        error_rates = []
        for i in range(grid_size):
            for j in range(grid_size):
                total_objects = tp_grid[i, j] + fp_grid[i, j] + fn_grid[i, j]
                if total_objects >= 5:  # Only consider cells with at least 5 objects
                    error_rate = (fp_grid[i, j] + fn_grid[i, j]) / total_objects
                    error_rates.append((i, j, error_rate, total_objects))
        
        error_rates.sort(key=lambda x: x[2], reverse=True)
        
        f.write("Top 5 Error Regions:\n")
        for idx, (i, j, error_rate, total_objects) in enumerate(error_rates[:5]):
            f.write(f"{idx+1}. Grid Cell ({i},{j}): Error Rate {error_rate:.4f}, Objects: {total_objects}\n")
    
    print(f"Saved metrics summary to {summary_path}")
    
    # Create and save key metric heatmaps as separate files
    metrics = [
        ("precision", precision_grid, "viridis", "Precision"),
        ("recall", recall_grid, "plasma", "Recall"),
        ("f1", f1_grid, "magma", "F1 Score"),
        ("error_rate", (fp_grid + fn_grid) / np.maximum(tp_grid + fp_grid + fn_grid, 1), "Reds", "Error Rate")
    ]
    
    for name, data, cmap, title in metrics:
        # Create gridded version
        metric_path = output_path.replace('.png', f'_{name}.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Mask cells with no data - this ensures white space where there's no data
        valid_cells = (tp_grid + fp_grid + fn_grid) > 0
        masked_data = np.ma.array(data, mask=~valid_cells)
        
        # Create the heatmap
        im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=1)
        
        # Fix grid line alignment - use proper grid positioning
        # Grid lines should be at integer boundaries, not at -0.5
        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.set_xticklabels([])  # Remove tick labels
        ax.set_yticklabels([])  # Remove tick labels
        
        # Add grid lines at proper positions
        ax.grid(True, which='major', color='white', linestyle='-', linewidth=1, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label=title)
        cbar.set_label(title, rotation=270, labelpad=20)
        
        ax.set_title(f"Grid-based {title}")
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(metric_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved {title} heatmap to {metric_path}")
        
        # Create smoothed overlay version without grid lines
        smoothed_path = output_path.replace('.png', f'_{name}_smoothed.png')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Show the drone image as background
        ax.imshow(image)
        
        # Create smoothed overlay data
        # First, create a full-size grid that matches the image dimensions
        full_size_data = np.zeros((vis_height, vis_width))
        
        # Fill in the grid values, expanding each cell to fill its area
        for i in range(grid_size):
            for j in range(grid_size):
                if valid_cells[i, j]:  # Only fill valid cells
                    # Calculate cell boundaries
                    y_start = int(i * cell_height)
                    y_end = int((i + 1) * cell_height)
                    x_start = int(j * cell_width)
                    x_end = int((j + 1) * cell_width)
                    
                    # Ensure we don't go out of bounds
                    y_end = min(y_end, vis_height)
                    x_end = min(x_end, vis_width)
                    
                    # Fill this cell with the metric value
                    full_size_data[y_start:y_end, x_start:x_end] = data[i, j]
        
        # Apply Gaussian smoothing to create a smooth overlay
        # Use sigma proportional to cell size for natural smoothing
        smoothing_sigma = max(1, min(cell_width, cell_height) / 4)
        smoothed_data = ndimage.gaussian_filter(full_size_data, sigma=smoothing_sigma)
        
        # Create masked array for transparent overlay
        # Only show areas where there's actual data
        valid_mask = (tp_grid + fp_grid + fn_grid) > 0
        valid_full_mask = np.zeros((vis_height, vis_width), dtype=bool)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if valid_mask[i, j]:
                    y_start = int(i * cell_height)
                    y_end = int((i + 1) * cell_height)
                    x_start = int(j * cell_width)
                    x_end = int((j + 1) * cell_width)
                    
                    y_end = min(y_end, vis_height)
                    x_end = min(x_end, vis_width)
                    
                    valid_full_mask[y_start:y_end, x_start:x_end] = True
        
        # Apply the mask
        masked_smoothed = np.ma.array(smoothed_data, mask=~valid_full_mask)
        
        # Overlay the smoothed data with transparency
        im = ax.imshow(masked_smoothed, cmap=cmap, vmin=0, vmax=1, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label=title)
        cbar.set_label(title, rotation=270, labelpad=20)
        
        ax.set_title(f"{title} - Smoothed Overlay")
        ax.axis('off')  # Remove axes for cleaner look
        
        # Save the smoothed figure
        plt.tight_layout()
        plt.savefig(smoothed_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved smoothed {title} overlay to {smoothed_path}")
    
    # Create a visualization of the count data (TP, FP, FN) with proper masking
    count_path = output_path.replace('.png', '_counts.png')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create RGB visualization for counts
    count_grid = np.zeros((grid_size, grid_size, 3))
    
    # Calculate max counts for normalization
    max_count = max(np.max(tp_grid), np.max(fp_grid), np.max(fn_grid), 1)
    
    # Fill the grid: red channel = FP, green channel = TP, blue channel = FN
    for i in range(grid_size):
        for j in range(grid_size):
            # Only color cells that have data
            if (tp_grid[i, j] + fp_grid[i, j] + fn_grid[i, j]) > 0:
                # Normalize counts to [0,1] range
                tp = tp_grid[i, j] / max_count
                fp = fp_grid[i, j] / max_count
                fn = fn_grid[i, j] / max_count
                
                # Set RGB values
                count_grid[i, j, 0] = fp  # Red = FP
                count_grid[i, j, 1] = tp  # Green = TP
                count_grid[i, j, 2] = fn  # Blue = FN
            else:
                # Set to transparent (white) for cells with no data
                count_grid[i, j, :] = 1.0
    
    # Create masked array to show white space for nodata areas
    valid_cells = (tp_grid + fp_grid + fn_grid) > 0
    masked_count_grid = np.ma.array(count_grid, mask=np.stack([~valid_cells]*3, axis=-1))
    
    # Show the count visualization
    im = ax.imshow(masked_count_grid)
    ax.set_title("Object Count Visualization")
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='k', label='True Positives'),
        Patch(facecolor='red', edgecolor='k', label='False Positives'),
        Patch(facecolor='blue', edgecolor='k', label='False Negatives')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Fix grid line alignment for counts visualization
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.set_xticklabels([])  # Remove tick labels
    ax.set_yticklabels([])  # Remove tick labels
    
    # Add grid lines at proper positions
    ax.grid(True, which='major', color='white', linestyle='-', linewidth=1, alpha=0.7)
    
    # Save the count visualization
    plt.tight_layout()
    plt.savefig(count_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved count visualization to {count_path}")
    
    # Create smoothed overlay version of counts
    count_smoothed_path = output_path.replace('.png', '_counts_smoothed.png')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Show the drone image as background
    ax.imshow(image)
    
    # Create full-size count data
    full_size_counts = np.zeros((vis_height, vis_width, 3))
    
    # Fill in the grid values
    for i in range(grid_size):
        for j in range(grid_size):
            if valid_cells[i, j]:
                y_start = int(i * cell_height)
                y_end = int((i + 1) * cell_height)
                x_start = int(j * cell_width)
                x_end = int((j + 1) * cell_width)
                
                y_end = min(y_end, vis_height)
                x_end = min(x_end, vis_width)
                
                # Normalize counts
                tp = tp_grid[i, j] / max_count
                fp = fp_grid[i, j] / max_count
                fn = fn_grid[i, j] / max_count
                
                # Fill this cell
                full_size_counts[y_start:y_end, x_start:x_end, 0] = fp  # Red
                full_size_counts[y_start:y_end, x_start:x_end, 1] = tp  # Green
                full_size_counts[y_start:y_end, x_start:x_end, 2] = fn  # Blue
    
    # Apply Gaussian smoothing to each channel
    smoothing_sigma = max(1, min(cell_width, cell_height) / 4)
    smoothed_counts = np.zeros_like(full_size_counts)
    for c in range(3):
        smoothed_counts[:, :, c] = ndimage.gaussian_filter(full_size_counts[:, :, c], sigma=smoothing_sigma)
    
    # Create mask for valid areas
    valid_full_mask = np.zeros((vis_height, vis_width), dtype=bool)
    for i in range(grid_size):
        for j in range(grid_size):
            if valid_cells[i, j]:
                y_start = int(i * cell_height)
                y_end = int((i + 1) * cell_height)
                x_start = int(j * cell_width)
                x_end = int((j + 1) * cell_width)
                
                y_end = min(y_end, vis_height)
                x_end = min(x_end, vis_width)
                
                valid_full_mask[y_start:y_end, x_start:x_end] = True
    
    # Apply mask and overlay
    masked_smoothed_counts = np.ma.array(smoothed_counts, mask=np.stack([~valid_full_mask]*3, axis=-1))
    
    # Overlay with transparency
    ax.imshow(masked_smoothed_counts, alpha=0.7)
    
    ax.set_title("Object Counts - Smoothed Overlay")
    ax.axis('off')
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the smoothed counts figure
    plt.tight_layout()
    plt.savefig(count_smoothed_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved smoothed count visualization to {count_smoothed_path}")
    
    # Create a simple index HTML file that displays all the visualizations
    html_path = output_path.replace('.png', '_grid_analysis.html')
    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Grid-based Analysis Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .image-container {{ margin: 20px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; }}
        h1, h2 {{ color: #333; }}
        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
        .grid-pair {{ display: flex; gap: 20px; margin: 20px 0; }}
        .grid-item {{ flex: 1; }}
    </style>
</head>
<body>
    <h1>Grid-based Error Analysis Results</h1>
    
    <div class="image-container">
        <h2>Overview</h2>
        <img src="{os.path.basename(overview_path)}" alt="Overview">
    </div>
    
    <div class="grid-pair">
        <div class="grid-item">
            <h3>Precision Grid</h3>
            <img src="{os.path.basename(output_path.replace('.png', '_precision.png'))}" alt="Precision Grid">
        </div>
        <div class="grid-item">
            <h3>Precision Smoothed</h3>
            <img src="{os.path.basename(output_path.replace('.png', '_precision_smoothed.png'))}" alt="Precision Smoothed">
        </div>
    </div>
    
    <div class="grid-pair">
        <div class="grid-item">
            <h3>Recall Grid</h3>
            <img src="{os.path.basename(output_path.replace('.png', '_recall.png'))}" alt="Recall Grid">
        </div>
        <div class="grid-item">
            <h3>Recall Smoothed</h3>
            <img src="{os.path.basename(output_path.replace('.png', '_recall_smoothed.png'))}" alt="Recall Smoothed">
        </div>
    </div>
    
    <div class="grid-pair">
        <div class="grid-item">
            <h3>F1 Score Grid</h3>
            <img src="{os.path.basename(output_path.replace('.png', '_f1.png'))}" alt="F1 Score Grid">
        </div>
        <div class="grid-item">
            <h3>F1 Score Smoothed</h3>
            <img src="{os.path.basename(output_path.replace('.png', '_f1_smoothed.png'))}" alt="F1 Score Smoothed">
        </div>
    </div>
    
    <div class="grid-pair">
        <div class="grid-item">
            <h3>Error Rate Grid</h3>
            <img src="{os.path.basename(output_path.replace('.png', '_error_rate.png'))}" alt="Error Rate Grid">
        </div>
        <div class="grid-item">
            <h3>Error Rate Smoothed</h3>
            <img src="{os.path.basename(output_path.replace('.png', '_error_rate_smoothed.png'))}" alt="Error Rate Smoothed">
        </div>
    </div>
    
    <div class="grid-pair">
        <div class="grid-item">
            <h3>Object Count Grid</h3>
            <img src="{os.path.basename(count_path)}" alt="Object Count Grid">
        </div>
        <div class="grid-item">
            <h3>Object Count Smoothed</h3>
            <img src="{os.path.basename(count_smoothed_path)}" alt="Object Count Smoothed">
        </div>
    </div>
    
    <div class="image-container">
        <h2>Analysis Summary</h2>
        <pre>
""")
        # Include the summary text content
        with open(summary_path, 'r') as summary_file:
            f.write(summary_file.read())
            
        f.write("""
        </pre>
    </div>
    
    <p>Note: This visualization shows object-wise metrics based on 50% overlap matching.
       White areas have neither ground truth nor predicted tires (nodata regions).
       In reality, predictions without ground truth labels may represent unlabeled true tires.</p>
       
    <p><strong>Grid vs Smoothed:</strong> The grid versions show discrete cell values with clear boundaries,
       while the smoothed versions show continuous overlays that blend naturally with the drone imagery.</p>
</body>
</html>
""")
    
    print(f"Created HTML report at {html_path}")
    print(f"Grid-based analysis complete - visualizations saved to {os.path.dirname(output_path)}")
    
    # Create empty placeholder for the original output path to avoid errors
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, "See HTML report for complete analysis", 
             ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.savefig(output_path, dpi=100)
    plt.close()

def generate_zoom_insets(image, ground_truth, prediction, error_mask, high_error_cells, grid_size, output_path):
    """Generate visualization with zoomed-in insets of high-error regions"""
    # Create figure with main image and insets
    fig = plt.figure(figsize=(16, 12))
    
    # Main subplot for the full image
    ax_main = fig.add_subplot(1, 1, 1)
    
    # Display the original darkened main image (keep as is, user likes this view)
    ax_main.imshow(image)
    
    # Create a more transparent error mask overlay
    transparent_error_mask = np.copy(error_mask)
    for i in range(3):
        if np.any(transparent_error_mask[:,:,i] > 0):
            # Reduce intensity to 30% of original
            transparent_error_mask[:,:,i] = (transparent_error_mask[:,:,i] * 0.3).astype(np.uint8)
    
    # Overlay the transparent error mask
    ax_main.imshow(transparent_error_mask, alpha=0.5)
    ax_main.set_title('Error Analysis with Zoom Insets', fontsize=16)
    
    # Get dimensions for grid cells
    vis_height, vis_width = image.shape[:2]
    cell_height = vis_height / grid_size
    cell_width = vis_width / grid_size
    
    # Colors for each inset box
    inset_colors = ['white', 'yellow', 'cyan']
    
    # Create zoomed insets for each high error cell
    for idx, (i, j) in enumerate(high_error_cells):
        if idx >= len(inset_colors):  # Safety check
            break
            
        # Calculate cell bounds in image coordinates
        x0 = j * cell_width
        y0 = i * cell_height
        width = cell_width
        height = cell_height
        
        # Add a border around the cell in the main image
        rect = Rectangle((x0, y0), width, height, 
                         linewidth=3, edgecolor=inset_colors[idx], facecolor='none')
        ax_main.add_patch(rect)
        
        # Label the region in the main plot
        ax_main.text(x0 + width/2, y0 + height/2, f"{idx+1}", 
                    color='white', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.7))
        
        # Calculate inset position and size based on index
        # Position insets on the right side of the figure
        inset_width = 0.25
        inset_height = 0.25
        inset_x = 0.72
        inset_y = 0.7 - (idx * 0.3)  # Stack insets vertically
        
        # Create inset axes
        ax_inset = fig.add_axes([inset_x, inset_y, inset_width, inset_height])
        
        # Extract the region to zoom
        zoom_region = image[int(y0):int(y0+height), int(x0):int(x0+width)]
        
        # Apply brightness and contrast enhancement to the zoom region only
        brightened_zoom = np.clip(zoom_region * 2.0, 0, 255).astype(np.uint8)
        
        # Display the brightened zoomed region
        ax_inset.imshow(brightened_zoom)
        
        # Extract GT and Pred for this region
        gt_region = ground_truth[int(y0):int(y0+height), int(x0):int(x0+width)]
        pred_region = prediction[int(y0):int(y0+height), int(x0):int(x0+width)]
        
        # Perform proper object-based analysis in this zoomed region
        if np.any(gt_region) or np.any(pred_region):
            # Label all objects
            gt_labeled = measure.label(gt_region)
            pred_labeled = measure.label(pred_region)
            
            # Match objects using IoU threshold
            matched_pairs, unmatched_gt, unmatched_pred = match_objects(
                gt_labeled, pred_labeled, overlap_threshold=0.5
            )
            
            # Get object properties
            gt_props = {r.label: r for r in measure.regionprops(gt_labeled)}
            pred_props = {r.label: r for r in measure.regionprops(pred_labeled)}
            
            # Draw matched pairs (True Positives) with connecting lines
            for gt_id, pred_id, overlap in matched_pairs:
                gt_y, gt_x = gt_props[gt_id].centroid
                pred_y, pred_x = pred_props[pred_id].centroid
                
                # Draw green circles for matched ground truth
                gt_circle = Circle((gt_x, gt_y), radius=5, fill=False, 
                                 edgecolor='green', linewidth=2)
                ax_inset.add_patch(gt_circle)
                
                # Draw green circles for matched predictions
                pred_circle = Circle((pred_x, pred_y), radius=5, fill=False, 
                                  edgecolor='green', linewidth=2)
                ax_inset.add_patch(pred_circle)
                
                # Draw a thin line connecting the matched objects
                ax_inset.plot([gt_x, pred_x], [gt_y, pred_y], 
                            color='green', alpha=0.5, linewidth=1, linestyle='--')
                
                # Optionally: annotate with overlap percentage
                mid_x = (gt_x + pred_x) / 2
                mid_y = (gt_y + pred_y) / 2
                ax_inset.text(mid_x, mid_y, f"{int(overlap*100)}%", 
                             color='white', fontsize=7, ha='center', va='center',
                             bbox=dict(facecolor='green', alpha=0.7))
            
            # Draw unmatched ground truth (False Negatives)
            for gt_id in unmatched_gt:
                if gt_id in gt_props:
                    gt_y, gt_x = gt_props[gt_id].centroid
                    fn_circle = Circle((gt_x, gt_y), radius=5, fill=False, 
                                     edgecolor='blue', linewidth=2)
                    ax_inset.add_patch(fn_circle)
            
            # Draw unmatched predictions (False Positives)
            for pred_id in unmatched_pred:
                if pred_id in pred_props:
                    pred_y, pred_x = pred_props[pred_id].centroid
                    fp_circle = Circle((pred_x, pred_y), radius=5, fill=False, 
                                     edgecolor='red', linewidth=2)
                    ax_inset.add_patch(fp_circle)
        
        # Remove axis ticks
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        
        # Add a border to the inset with the same color
        for spine in ax_inset.spines.values():
            spine.set_edgecolor(inset_colors[idx])
            spine.set_linewidth(3)
        
        # Add label
        ax_inset.set_title(f"Region {idx+1}", fontsize=10, color=inset_colors[idx])
        
        # Draw a connection between the main image and the inset
        con = ConnectionPatch(
            xyA=(x0 + width/2, y0 + height/2), coordsA=ax_main.transData,
            xyB=(0.5, 0.0), coordsB=ax_inset.transAxes,
            arrowstyle="->", shrinkB=5, linewidth=2, color=inset_colors[idx]
        )
        fig.add_artist(con)
    
    # Add legend for error types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='green', linewidth=2, label='True Positive'),
        Patch(facecolor='none', edgecolor='red', linewidth=2, label='False Positive'),
        Patch(facecolor='none', edgecolor='blue', linewidth=2, label='False Negative')
    ]
    ax_main.legend(handles=legend_elements, loc='upper left')
    
    # Add annotation about object matching criterion
    ax_main.text(0.5, 0.05, 
                "Objects are matched when they overlap by ≥50%.\n"
                "Green lines connect matched pairs in zoomed views.",
                transform=ax_main.transAxes, fontsize=10, ha='center',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_hierarchical_zoom(image, centroids, error_mask, output_path):
    """Generate clearer hierarchical zoom visualization showing tire clusters at different scales"""
    # Skip if dependencies are not available
    if not HAVE_CONVEX_HULL or not HAVE_DBSCAN:
        warnings.warn("Skipping hierarchical zoom visualization due to missing dependencies.")
        return
    
    # Convert centroids to numpy array
    points = np.array(centroids)
    
    # Skip if not enough points
    if len(points) < 10:
        warnings.warn("Not enough points for hierarchical zoom visualization.")
        return
    
    # Create figure with hierarchical zoom levels
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Full image view (level 0) - keep as is, user likes this view
    ax_full = fig.add_subplot(gs[0, 0])
    ax_full.imshow(image)
    ax_full.set_title('Full Image View: Tire Cluster Analysis', fontsize=14)
    
    # Define a clearer color scheme for the visualization
    cluster_colors = ['gold', 'cyan', 'magenta', 'lime', 'orange']
    
    # Apply DBSCAN with largest epsilon for top-level clusters
    eps_values = [100, 50, 25]  # From large clusters to small clusters
    clustering = DBSCAN(eps=eps_values[0], min_samples=5).fit(points)
    labels = clustering.labels_
    
    # Number of clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # If no clusters found, fall back to simple visualization
    if n_clusters == 0:
        ax_full.scatter(points[:, 1], points[:, 0], color='red', alpha=0.5, s=5)
        plt.text(0.5, 0.5, "No significant tire clusters found", 
                 ha='center', va='center', transform=ax_full.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    
    # Find the largest cluster for hierarchical zooming
    cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
    largest_cluster_idx = np.argmax(cluster_sizes)
    largest_cluster_points = points[labels == largest_cluster_idx]
    
    # Add top-level clusters to full view with a clearer presentation
    cluster_annotations = []
    for cluster_idx in range(n_clusters):
        cluster_points = points[labels == cluster_idx]
        
        if len(cluster_points) < 3:
            continue
            
        try:
            # Choose a color from the color scheme
            color_idx = cluster_idx % len(cluster_colors)
            cluster_color = cluster_colors[color_idx]
            
            # Create convex hull
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            
            # Create a transparent polygon with a clear outline
            from matplotlib.patches import Polygon
            polygon = Polygon(hull_points, alpha=0.2, fill=True, 
                             edgecolor=cluster_color, facecolor=cluster_color, linewidth=2)
            ax_full.add_patch(polygon)
            
            # Add count label
            center_x = np.mean(hull_points[:, 1])
            center_y = np.mean(hull_points[:, 0])
            
            # Store for later annotation connections
            cluster_annotations.append((center_x, center_y, len(cluster_points), cluster_color))
            
            # Add text annotation
            ax_full.text(center_x, center_y, f"{len(cluster_points)} tires", 
                       color='black', fontsize=10, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor=cluster_color))
            
            # Highlight largest cluster with thicker border
            if cluster_idx == largest_cluster_idx:
                highlight = Polygon(hull_points, fill=False, 
                                   edgecolor='yellow', linewidth=3)
                ax_full.add_patch(highlight)
        except:
            continue
    
    # Add explanation text for the visualization
    ax_full.text(0.02, 0.02, 
                "Colored areas show tire clusters at the largest scale.\n"
                "Numbers indicate tire count in each cluster.\n"
                "Yellow outline indicates the cluster examined in detail.",
                transform=ax_full.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Level 1 zoom: Focus on largest cluster with better visualization
    ax_zoom1 = fig.add_subplot(gs[0, 1])
    
    # Calculate bounding box for the largest cluster
    if len(largest_cluster_points) > 0:
        min_y, min_x = np.min(largest_cluster_points, axis=0)
        max_y, max_x = np.max(largest_cluster_points, axis=0)
        
        # Add padding
        padding = 50
        min_y = max(0, min_y - padding)
        min_x = max(0, min_x - padding)
        max_y = min(image.shape[0], max_y + padding)
        max_x = min(image.shape[1], max_x + padding)
        
        # Extract zoom region with enhanced brightness (only for this zoom inset)
        zoom_region = image[int(min_y):int(max_y), int(min_x):int(max_x)]
        brightened_zoom = np.clip(zoom_region * 2.0, 0, 255).astype(np.uint8)
        
        # Display brightened zoomed region
        ax_zoom1.imshow(brightened_zoom)
        ax_zoom1.set_title(f'Level 1 Zoom: {len(largest_cluster_points)} Tires', fontsize=14)
        
        # Draw a connection between full view and zoom1
        con1 = ConnectionPatch(
            xyA=(np.mean([min_x, max_x]), np.mean([min_y, max_y])), coordsA=ax_full.transData,
            xyB=(0.5, 0.0), coordsB=ax_zoom1.transAxes,
            arrowstyle="->", shrinkB=5, linewidth=2, color='yellow'
        )
        fig.add_artist(con1)
        
        # Apply DBSCAN with medium epsilon for mid-level clusters
        level2_points = largest_cluster_points.copy()
        level2_clustering = DBSCAN(eps=eps_values[1], min_samples=3).fit(level2_points)
        level2_labels = level2_clustering.labels_
        level2_n_clusters = len(set(level2_labels)) - (1 if -1 in level2_labels else 0)
        
        # Choose a subcategory for next level zoom
        if level2_n_clusters > 0:
            subcategory_sizes = [np.sum(level2_labels == i) for i in range(level2_n_clusters)]
            subcategory_idx = np.argmax(subcategory_sizes)
            subcategory_points = level2_points[level2_labels == subcategory_idx]
            
            # Add mid-level clusters to zoom1 view
            for l2_cluster_idx in range(level2_n_clusters):
                l2_cluster_points = level2_points[level2_labels == l2_cluster_idx]
                
                if len(l2_cluster_points) < 3:
                    continue
                    
                try:
                    # Choose a color
                    color_idx = l2_cluster_idx % len(cluster_colors)
                    l2_color = cluster_colors[color_idx]
                    
                    # Create convex hull
                    l2_hull = ConvexHull(l2_cluster_points)
                    l2_hull_points = l2_cluster_points[l2_hull.vertices]
                    
                    # Adjust coordinates to zoomed view
                    l2_hull_points_adjusted = l2_hull_points.copy()
                    l2_hull_points_adjusted[:, 0] -= min_y
                    l2_hull_points_adjusted[:, 1] -= min_x
                    
                    # Create polygon with transparent fill and clear outline
                    polygon = Polygon(l2_hull_points_adjusted, alpha=0.2, fill=True, 
                                     edgecolor=l2_color, facecolor=l2_color, linewidth=2)
                    ax_zoom1.add_patch(polygon)
                    
                    # Add count label
                    l2_center_x = np.mean(l2_hull_points_adjusted[:, 1])
                    l2_center_y = np.mean(l2_hull_points_adjusted[:, 0])
                    ax_zoom1.text(l2_center_x, l2_center_y, f"{len(l2_cluster_points)}", 
                               color='black', fontsize=10, ha='center', va='center',
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor=l2_color))
                    
                    # Highlight chosen subcategory
                    if l2_cluster_idx == subcategory_idx:
                        highlight = Polygon(l2_hull_points_adjusted, fill=False, 
                                           edgecolor='cyan', linewidth=2)
                        ax_zoom1.add_patch(highlight)
                        
                        # Level 2 zoom: Focus on selected sub-cluster
                        ax_zoom2 = fig.add_subplot(gs[1, 0:2])
                        
                        # Calculate bounding box for this sub-cluster
                        sub_min_y, sub_min_x = np.min(l2_cluster_points, axis=0)
                        sub_max_y, sub_max_x = np.max(l2_cluster_points, axis=0)
                        
                        # Add padding
                        sub_padding = 25
                        sub_min_y = max(0, sub_min_y - sub_padding)
                        sub_min_x = max(0, sub_min_x - sub_padding)
                        sub_max_y = min(image.shape[0], sub_max_y + sub_padding)
                        sub_max_x = min(image.shape[1], sub_max_x + sub_padding)
                        
                        # Extract zoom region with extra brightness (only for this zoom inset)
                        sub_zoom_region = image[int(sub_min_y):int(sub_max_y), int(sub_min_x):int(sub_max_x)]
                        extra_bright = np.clip(sub_zoom_region * 2.5, 0, 255).astype(np.uint8)
                        
                        # Display brightened zoomed region
                        ax_zoom2.imshow(extra_bright)
                        ax_zoom2.set_title(f'Level 2 Zoom: Individual Tires, {len(l2_cluster_points)} Objects', fontsize=14)
                        
                        # Draw connection between zoom1 and zoom2
                        con2 = ConnectionPatch(
                            xyA=(l2_center_x, l2_center_y), coordsA=ax_zoom1.transData,
                            xyB=(0.5, 0.0), coordsB=ax_zoom2.transAxes,
                            arrowstyle="->", shrinkB=5, linewidth=2, color='cyan'
                        )
                        fig.add_artist(con2)
                        
                        # Plot individual tires in the final zoom as hollow circles
                        for point in l2_cluster_points:
                            # Adjust to zoom2 coordinates
                            y, x = point
                            y -= sub_min_y
                            x -= sub_min_x
                            
                            # Only show if within bounds
                            if 0 <= y < sub_zoom_region.shape[0] and 0 <= x < sub_zoom_region.shape[1]:
                                # Check if this is a TP, FP, or FN using local coordinates
                                global_y, global_x = point
                                
                                # Default to prediction (since these are centroids from prediction)
                                circle = Circle((x, y), radius=8, fill=False, edgecolor='red', linewidth=2)
                                
                                # Override color if we can determine TP
                                try:
                                    # Convert to integer indices
                                    gy, gx = int(global_y), int(global_x)
                                    # Check if within image bounds
                                    if (0 <= gy < error_mask.shape[0] and 
                                        0 <= gx < error_mask.shape[1]):
                                        # Check pixel color in error mask
                                        pixel = error_mask[gy, gx]
                                        # Green (0,255,0) = TP, Red (255,0,0) = FP, Blue (0,0,255) = FN
                                        if np.array_equal(pixel, [0, 255, 0]):
                                            circle = Circle((x, y), radius=8, fill=False, 
                                                          edgecolor='green', linewidth=2)
                                except:
                                    pass  # Keep default if error
                                
                                ax_zoom2.add_patch(circle)
                
                except:
                    continue
    
    # Add explanation for the visualization
    if 'ax_zoom2' in locals():
        ax_zoom2.text(0.02, 0.02, 
                     "Individual tire detections shown as hollow circles.\n"
                     "Green = True Positive, Red = False Positive/Prediction",
                     transform=ax_zoom2.transAxes, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_comparative_hotspot(image, gt_density, pred_density, diff_density, 
                               gt_centroids, pred_centroids, output_path):
    """Generate dual-panel visualization comparing ground truth and prediction hotspots"""
    fig = plt.figure(figsize=(18, 12))
    
    # Create layout with three columns
    gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1], figure=fig)
    
    # Create enhanced colormap
    hot_cmap = create_enhanced_hotspot_cmap()
    
    # Ground Truth hotspot visualization
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    
    # Apply thresholding for better visibility
    threshold = 0.01
    gt_masked = np.ma.masked_where(gt_density < threshold, gt_density)
    
    # Show density map
    gt_heatmap = ax1.imshow(gt_masked, alpha=0.7, cmap=hot_cmap)
    
    # Add small markers for centroids
    for y, x in gt_centroids:
        ax1.scatter(x, y, s=10, color='cyan', alpha=0.5)  # Smaller points
    
    ax1.set_title('Labeled Samples (Ground Truth)', fontsize=14)
    
    # Add colorbar
    cbar1 = plt.colorbar(gt_heatmap, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Density', rotation=270, labelpad=20)
    
    # Prediction hotspot visualization
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image)
    
    # Apply thresholding for better visibility
    pred_masked = np.ma.masked_where(pred_density < threshold, pred_density)
    
    # Show density map
    pred_heatmap = ax2.imshow(pred_masked, alpha=0.7, cmap=hot_cmap)
    
    # Add small markers for centroids
    for y, x in pred_centroids:
        ax2.scatter(x, y, s=10, color='cyan', alpha=0.5)  # Smaller points
    
    ax2.set_title('Model Detections', fontsize=14)
    
    # Add colorbar
    cbar2 = plt.colorbar(pred_heatmap, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Density', rotation=270, labelpad=20)
    
    # Differential visualization (Prediction - Ground Truth)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image)
    
    # Create custom colormap for differential visualization (red-white-blue)
    diff_cmap = plt.cm.get_cmap('coolwarm')
    
    # Normalize the difference to [-1, 1] range
    max_diff = max(abs(np.min(diff_density)), abs(np.max(diff_density)))
    if max_diff > 0:
        # Use a non-linear scaling to enhance subtle differences
        diff_norm = np.sign(diff_density) * np.power(np.abs(diff_density/max_diff), 0.5)
    else:
        diff_norm = diff_density
    
    # Apply threshold to only show significant differences
    diff_masked = np.ma.masked_where(abs(diff_norm) < threshold, diff_norm)
    
    # Show differential map
    diff_heatmap = ax3.imshow(diff_masked, cmap=diff_cmap, alpha=0.7, vmin=-1, vmax=1)
    
    ax3.set_title('Model Detection vs. Labeled Samples', fontsize=14)
    
    # Add colorbar
    cbar3 = plt.colorbar(diff_heatmap, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Difference', rotation=270, labelpad=20)
    
    # Add explanation panel instead of cluster visualization
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax4.axis('off')  # Turn off axes
    
    # Add explanatory text
    explanation = (
        "Tire Distribution Analysis:\n"
        "- Left: Labeled samples (ground truth) which contain only a subset of actual tires\n"
        "- Middle: Model-detected tire locations, which include both labeled and unlabeled tires\n"
        "- Right: Comparative analysis showing areas where model finds more (red) or fewer (blue) tires than labeled\n\n"
        f"Total detected objects: Labeled samples: {len(gt_centroids)}, Model detections: {len(pred_centroids)}\n"
        "Note: Manual verification shows many model detections are actual tires that were not labeled in the ground truth data"
    )
    ax4.text(0.5, 0.5, explanation, fontsize=12, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    # Add annotation to clarify that blue areas aren't necessarily errors
    ax3.annotate("Note: Blue areas may include unlabeled tires\nthat were correctly detected by the model", 
                 xy=(0.5, 0.05), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_cluster_visualization(ax, centroids, color='green', alpha=0.3, label=None):
    """Create cluster visualization using convex hulls or density-based ellipses"""
    # Skip if dependencies are not available
    if not HAVE_CONVEX_HULL or not HAVE_DBSCAN:
        # Fallback: just show centroids
        points = np.array(centroids)
        ax.scatter(points[:, 1], points[:, 0], color=color, alpha=alpha, label=label)
        return
    
    # Convert centroids to numpy array
    points = np.array(centroids)
    
    # Skip if not enough points
    if len(points) < 3:
        ax.scatter(points[:, 1], points[:, 0], color=color, alpha=alpha, label=label)
        return
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=50, min_samples=3).fit(points)
    labels = clustering.labels_
    
    # Number of clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Process each cluster
    for cluster_idx in range(n_clusters):
        # Get points in this cluster
        cluster_points = points[labels == cluster_idx]
        
        # Skip small clusters
        if len(cluster_points) < 3:
            continue
        
        try:
            # Create convex hull
            hull = ConvexHull(cluster_points)
            
            # Get hull vertices
            hull_points = cluster_points[hull.vertices]
            
            # Create polygon
            from matplotlib.patches import Polygon
            polygon = Polygon(hull_points, alpha=alpha, fill=True, 
                             edgecolor=color, facecolor=color, label=label if cluster_idx == 0 else "")
            
            ax.add_patch(polygon)
            
            # Add count label in the center of the cluster
            center_x = np.mean(hull_points[:, 1])
            center_y = np.mean(hull_points[:, 0])
            ax.text(center_x, center_y, f"{len(cluster_points)}", 
                   color='white', fontsize=10, ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.7))
            
            # Calculate area in square centimeters (assuming 3cm/pixel)
            area_pixels = np.sum(cluster_points.shape[0])
            area_sqcm = area_pixels * 9  # 3cm × 3cm per pixel
            
            # Add area label below count (smaller font)
            ax.text(center_x, center_y + 15, f"{area_sqcm:.1f} sq.cm", 
                   color='white', fontsize=8, ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.5))
        except:
            # Skip if convex hull fails (e.g., colinear points)
            continue
    
    # Show isolated points as markers
    noise_points = points[labels == -1]
    if len(noise_points) > 0:
        ax.scatter(noise_points[:, 1], noise_points[:, 0], 
                  color=color, alpha=alpha*0.5, marker='o', s=5)

def create_manually_downsampled_image(src_path, output_path, scale_factor):
    """
    Manually downsample an image when rasterio's built-in downsampling fails.
    Works by reading and processing the image in chunks.
    """
    with rasterio.open(src_path) as src:
        # Get source dimensions
        height = src.height
        width = src.width
        channels = src.count
        crs = src.crs
        bounds = src.bounds
        
        # Calculate target dimensions
        target_height = height // scale_factor
        target_width = width // scale_factor
        
        # Create empty array for downsampled data
        downsampled = np.zeros((channels, target_height, target_width), dtype=np.uint8)
        
        # Process in chunks
        chunk_size = 1024 * scale_factor  # Make chunk size a multiple of scale factor
        num_chunks_x = (width + chunk_size - 1) // chunk_size
        num_chunks_y = (height + chunk_size - 1) // chunk_size
        
        with tqdm(total=num_chunks_y * num_chunks_x, desc="Manual downsampling") as pbar:
            for y_idx in range(num_chunks_y):
                for x_idx in range(num_chunks_x):
                    # Calculate chunk bounds
                    x_start = x_idx * chunk_size
                    y_start = y_idx * chunk_size
                    x_end = min(x_start + chunk_size, width)
                    y_end = min(y_start + chunk_size, height)
                    
                    # Read chunk
                    window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                    chunk = src.read(window=window)
                    
                    # Calculate corresponding region in downsampled array
                    ds_x_start = x_start // scale_factor
                    ds_y_start = y_start // scale_factor
                    ds_x_end = (x_end + scale_factor - 1) // scale_factor
                    ds_y_end = (y_end + scale_factor - 1) // scale_factor
                    
                    # Ensure we're within bounds
                    ds_x_end = min(ds_x_end, target_width)
                    ds_y_end = min(ds_y_end, target_height)
                    
                    # Downsample this chunk
                    for c in range(channels):
                        for i in range(ds_y_start, ds_y_end):
                            for j in range(ds_x_start, ds_x_end):
                                # Calculate source region
                                src_y_start = i * scale_factor
                                src_x_start = j * scale_factor
                                src_y_end = min(src_y_start + scale_factor, height)
                                src_x_end = min(src_x_start + scale_factor, width)
                                
                                # Check if this region is within our chunk
                                if (src_y_start >= y_start and src_y_end <= y_end and
                                    src_x_start >= x_start and src_x_end <= x_end):
                                    
                                    # Calculate local coordinates within chunk
                                    local_y_start = src_y_start - y_start
                                    local_x_start = src_x_start - x_start
                                    local_y_end = src_y_end - y_start
                                    local_x_end = src_x_end - x_start
                                    
                                    # Extract region from chunk
                                    region = chunk[c, local_y_start:local_y_end, local_x_start:local_x_end]
                                    
                                    # Average the values
                                    if region.size > 0:
                                        downsampled[c, i, j] = int(np.mean(region))
                    
                    pbar.update(1)
        
        # Write the downsampled image
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=target_height,
            width=target_width,
            count=channels,
            dtype=np.uint8,
            crs=crs,
            transform=rasterio.transform.from_bounds(*bounds, target_width, target_height)
        ) as dst:
            dst.write(downsampled)

def create_presence_preserving_mask(src_path, output_path, scale_factor):
    """
    Create a downsampled mask using a presence-preserving approach (max pooling).
    This ensures that small objects (like tires) are not lost during downsampling.
    """
    with rasterio.open(src_path) as src:
        # Get source dimensions
        height = src.height
        width = src.width
        crs = src.crs
        bounds = src.bounds
        
        # Calculate target dimensions
        target_height = height // scale_factor
        target_width = width // scale_factor
        
        # Create empty array for downsampled data
        downsampled = np.zeros((1, target_height, target_width), dtype=np.uint8)
        
        # Process in chunks
        chunk_size = 1024 * scale_factor  # Make chunk size a multiple of scale factor
        num_chunks_x = (width + chunk_size - 1) // chunk_size
        num_chunks_y = (height + chunk_size - 1) // chunk_size
        
        with tqdm(total=num_chunks_y * num_chunks_x, desc="Downsampling mask (presence preserving)") as pbar:
            for y_idx in range(num_chunks_y):
                for x_idx in range(num_chunks_x):
                    # Calculate chunk bounds
                    x_start = x_idx * chunk_size
                    y_start = y_idx * chunk_size
                    x_end = min(x_start + chunk_size, width)
                    y_end = min(y_start + chunk_size, height)
                    
                    # Read chunk
                    window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                    chunk = src.read(1, window=window)
                    
                    # Normalize mask values (0 = background, 1 = object)
                    chunk = np.where(chunk == 255, 0, chunk)
                    chunk = np.where(chunk > 0, 1, 0).astype(np.uint8)
                    
                    # Calculate corresponding region in downsampled array
                    ds_x_start = x_start // scale_factor
                    ds_y_start = y_start // scale_factor
                    ds_x_end = (x_end + scale_factor - 1) // scale_factor
                    ds_y_end = (y_end + scale_factor - 1) // scale_factor
                    
                    # Ensure we're within bounds
                    ds_x_end = min(ds_x_end, target_width)
                    ds_y_end = min(ds_y_end, target_height)
                    
                    # Downsample this chunk using presence preserving approach
                    for i in range(ds_y_start, ds_y_end):
                        for j in range(ds_x_start, ds_x_end):
                            # Calculate source region
                            src_y_start = i * scale_factor
                            src_x_start = j * scale_factor
                            src_y_end = min(src_y_start + scale_factor, height)
                            src_x_end = min(src_x_start + scale_factor, width)
                            
                            # Check if this region is within our chunk
                            if (src_y_start >= y_start and src_y_end <= y_end and
                                src_x_start >= x_start and src_x_end <= x_end):
                                
                                # Calculate local coordinates within chunk
                                local_y_start = src_y_start - y_start
                                local_x_start = src_x_start - x_start
                                local_y_end = src_y_end - y_start
                                local_x_end = src_x_end - x_start
                                
                                # Extract region from chunk
                                region = chunk[local_y_start:local_y_end, local_x_start:local_x_end]
                                
                                # If ANY pixel in the region is an object, mark the downscaled pixel as an object
                                if np.any(region > 0):
                                    downsampled[0, i, j] = 1
                    
                    pbar.update(1)
        
        # Apply dilation to further enhance small object visibility
        # Scale dilation intensity with downscaling factor
        dilation_factor = max(1, min(3, scale_factor // 16))
        if dilation_factor > 1:
            print(f"Applying dilation (factor {dilation_factor}) to preserve small objects...")
            dilated = ndimage.binary_dilation(downsampled[0], iterations=dilation_factor).astype(np.uint8)
            downsampled[0] = dilated
        
        # Write the downsampled mask
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=target_height,
            width=target_width,
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=rasterio.transform.from_bounds(*bounds, target_width, target_height)
        ) as dst:
            dst.write(downsampled)

def create_manually_downsampled_mask(src_path, output_path, scale_factor):
    """Manually downsample a mask when other methods fail"""
    # This is a fallback method, essentially a simplified version of create_presence_preserving_mask
    create_presence_preserving_mask(src_path, output_path, scale_factor)

def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two binary masks
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:  # Both masks are empty
        return 0.0
    
    return intersection / union

def calculate_overlap_percentage(mask1, mask2):
    """Calculate percentage of mask1 that overlaps with mask2
    
    Args:
        mask1: First binary mask (the one we're checking overlap for)
        mask2: Second binary mask (the reference mask)
        
    Returns:
        Percentage of mask1 that is covered by mask2 (0-1)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    mask1_area = mask1.sum()
    
    if mask1_area == 0:  # First mask is empty
        return 0.0
    
    return intersection / mask1_area

def match_objects(gt_labeled, pred_labeled, overlap_threshold=0.5):
    """Match ground truth and prediction objects based on overlap percentage
    
    This function implements object-based matching using overlap criteria:
    1. For each ground truth object, it finds the prediction that overlaps it the most
    2. A match is established when at least 'overlap_threshold' (e.g., 50%) of a 
       prediction object is covered by a ground truth object
    3. This handles the scenario where a single tire might be fragmented in prediction
       but still counted as a true positive if enough area overlaps
    
    Args:
        gt_labeled: Labeled ground truth mask (from skimage.measure.label)
        pred_labeled: Labeled prediction mask (from skimage.measure.label)
        overlap_threshold: Minimum overlap percentage to consider a match (default=0.5 or 50%)
        
    Returns:
        matched_pairs: List of (gt_id, pred_id, overlap) tuples for matched objects
        unmatched_gt: List of gt_id for unmatched ground truth objects
        unmatched_pred: List of pred_id for unmatched predicted objects
    """
    # Get unique labels excluding background (0)
    gt_labels = np.unique(gt_labeled)
    gt_labels = gt_labels[gt_labels > 0]
    
    pred_labels = np.unique(pred_labeled)
    pred_labels = pred_labels[pred_labels > 0]
    
    # Initialize results
    matched_pairs = []
    unmatched_gt = []
    unmatched_pred = list(pred_labels)  # Start with all predictions unmatched
    
    # For each ground truth object
    for gt_id in gt_labels:
        gt_mask = gt_labeled == gt_id
        best_overlap = 0.0
        best_pred_id = None
        
        # Find the prediction with highest overlap
        for pred_id in pred_labels:
            if pred_id in unmatched_pred:  # Only consider unmatched predictions
                pred_mask = pred_labeled == pred_id
                
                # Calculate overlap percentage of prediction covered by ground truth
                overlap = calculate_overlap_percentage(pred_mask, gt_mask)
                
                # Keep track of best match
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pred_id = pred_id
        
        # If we found a match above the threshold
        if best_overlap >= overlap_threshold and best_pred_id is not None:
            matched_pairs.append((gt_id, best_pred_id, best_overlap))
            unmatched_pred.remove(best_pred_id)
        else:
            unmatched_gt.append(gt_id)
    
    return matched_pairs, unmatched_gt, unmatched_pred

def main():
    parser = argparse.ArgumentParser(description="Analyze segmentation model predictions on large orthomosaic images")
    parser.add_argument("--image", required=False, help="Path to the original orthomosaic image")
    parser.add_argument("--ground-truth", required=False, help="Path to the ground truth mask")
    parser.add_argument("--prediction", required=False, help="Path to the prediction mask")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--base-name", default="analysis", help="Base name for output files")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size for processing")
    parser.add_argument("--downscale", type=int, default=8, help="Downscaling factor for visualizations")
    parser.add_argument("--sigma", type=float, default=3.0, help="Smoothing sigma for hotspot visualization")
    parser.add_argument("--overlap-threshold", type=float, default=0.5, 
                        help="Minimum overlap percentage (0.0-1.0) to consider two objects as matching (default=0.5)")
    parser.add_argument("--quick-test", action="store_true", help="Enable quick test mode (downscales inputs for faster processing)")
    parser.add_argument("--use-cache", action="store_true", help="Use cached data if available instead of reprocessing")
    parser.add_argument("--no-save-cache", action="store_true", help="Don't save processed data to cache")
    parser.add_argument("--visualize-only", action="store_true", help="Skip processing and only generate visualizations (requires cache)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate inputs
    if args.visualize_only:
        # For visualization-only mode, we need a pre-existing cache
        cache_path = get_cache_filepath(args.output_dir, args.base_name)
        if not os.path.exists(cache_path):
            print(f"Error: visualize_only requires an existing cache file at {cache_path}")
            print("Run the script without --visualize-only first to generate the cache.")
            return
    else:
        # For processing mode, we need input files
        if not args.image or not args.ground_truth or not args.prediction:
            print("Error: Must provide --image, --ground-truth, and --prediction when not in --visualize-only mode")
            return
        
        # Check input files exist
        for path, name in [(args.image, "Image"), (args.ground_truth, "Ground truth"), (args.prediction, "Prediction")]:
            if not os.path.exists(path):
                print(f"Error: {name} file not found at {path}")
                return
    
    # Run the analysis
    start_time = time.time()
    
    if args.visualize_only:
        # Run in visualization-only mode
        extreme_memory_process(
            None,  # No image path needed
            None,  # No mask paths needed
            args.output_dir,
            args.base_name,
            chunk_size=args.chunk_size,
            sigma=args.sigma,
            downscale_factor=args.downscale,
            quick_test=args.quick_test,
            use_cache=True,
            save_cache=not args.no_save_cache,
            visualize_only=True,
            overlap_threshold=args.overlap_threshold
        )
    else:
        # Run in normal processing + visualization mode
        mask_paths = {
            'ground_truth': args.ground_truth,
            'prediction': args.prediction
        }
        
        extreme_memory_process(
            args.image, 
            mask_paths,
            args.output_dir,
            args.base_name,
            chunk_size=args.chunk_size,
            sigma=args.sigma,
            downscale_factor=args.downscale,
            quick_test=args.quick_test,
            use_cache=args.use_cache,
            save_cache=not args.no_save_cache,
            visualize_only=False,
            overlap_threshold=args.overlap_threshold
        )
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Analysis complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()