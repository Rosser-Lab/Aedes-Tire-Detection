import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.plot import show
from rasterio.windows import Window
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import libpysal
from esda.getisord import G_Local
from esda.moran import Moran_Local
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as ctx
from shapely.geometry import Point, box
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pickle
import hashlib
import time
from pathlib import Path
import argparse
import gc

# Set the style for the plots
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # Newer matplotlib versions
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')  # Older matplotlib versions
    except OSError:
        plt.style.use('default')  # Fallback to default style
        print("Warning: Could not load seaborn style, using default matplotlib style")

sns.set_context("paper", font_scale=1.5)

def clear_cache():
    """Clear all cached files"""
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(cache_dir, file))
        print(f"Cleared cache directory: {cache_dir}")

def get_memory_usage():
    """Get current memory usage (rough estimate)"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0  # psutil not available

def check_memory_limit():
    """Check if memory usage exceeds the limit"""
    current_memory = get_memory_usage()
    if current_memory > args.memory_limit:
        print(f"WARNING: Memory usage ({current_memory:.1f} MB) exceeds limit ({args.memory_limit} MB)")
        return False
    return True

def optimize_memory():
    """Force garbage collection and memory optimization"""
    gc.collect()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Spatial analysis of tire detection with caching')
parser.add_argument('--clear-cache', action='store_true', help='Clear all cached data before running')
parser.add_argument('--chunk-size', type=int, default=1024, help='Chunk size for raster reading (default: 1024)')
parser.add_argument('--memory-limit', type=int, default=65536, help='Memory limit in MB (default: 65536)')
args = parser.parse_args()

if args.clear_cache:
    clear_cache()

# File paths - update these to your file locations
original_tires_path = "C:/Users/achamb/Documents/GitHub/drone_dengue/data/idn/tires/spatial_analysis/tires_534_points.shp"
refined_tires_path = "C:/Users/achamb/Documents/GitHub/drone_dengue/data/idn/tires/spatial_analysis/tires_1300_points.shp"
orthomosaic_path = "C:/Users/achamb/Documents/GitHub/drone_dengue/data/idn/tires/spatial_analysis/Makassar_Tallo_DroneMap_June2024.tif"

# Output directory
output_dir = "spatial_analysis_results"
cache_dir = "spatial_analysis_cache"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

def get_file_hash(filepath):
    """Generate a hash of the file to use as cache key"""
    if not os.path.exists(filepath):
        return None
    
    # Get file modification time and size for a quick hash
    stat = os.stat(filepath)
    return f"{filepath}_{stat.st_mtime}_{stat.st_size}"

def get_cache_path(filepath, suffix=""):
    """Generate cache file path"""
    file_hash = get_file_hash(filepath)
    if file_hash is None:
        return None
    
    # Create a safe filename from the hash
    safe_hash = hashlib.md5(file_hash.encode()).hexdigest()
    cache_filename = f"cached_{Path(filepath).stem}_{suffix}_{safe_hash}.pkl"
    return os.path.join(cache_dir, cache_filename)

def load_from_cache(filepath, suffix=""):
    """Load data from cache if it exists and is valid"""
    cache_path = get_cache_path(filepath, suffix)
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                print(f"Loading cached data from {cache_path}")
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            print("Cache file corrupted, will regenerate")
            return None
    return None

def save_to_cache(data, filepath, suffix=""):
    """Save data to cache"""
    cache_path = get_cache_path(filepath, suffix)
    if cache_path:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved data to cache: {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

def read_raster_in_chunks(filepath, chunk_size=1024):
    """
    Read large raster files in chunks to manage memory usage.
    
    Parameters:
    -----------
    filepath : str
        Path to the raster file
    chunk_size : int
        Size of chunks to read (in pixels)
    
    Returns:
    --------
    dict : Dictionary containing processed raster data
    """
    print(f"Reading raster file in chunks: {filepath}")
    start_time = time.time()
    
    with rasterio.open(filepath) as src:
        # Get basic info
        height = src.height
        width = src.width
        count = src.count
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        
        print(f"Raster dimensions: {width}x{height}, {count} bands")
        print(f"Chunk size: {chunk_size}x{chunk_size}")
        
        # Calculate number of chunks
        chunks_x = (width + chunk_size - 1) // chunk_size
        chunks_y = (height + chunk_size - 1) // chunk_size
        total_chunks = chunks_x * chunks_y
        
        print(f"Processing {total_chunks} chunks...")
        
        # Initialize arrays for the full image
        if count >= 4:
            # Read alpha band for mask
            mask = np.zeros((height, width), dtype=bool)
        else:
            mask = np.ones((height, width), dtype=bool)
        
        # Process chunks
        for chunk_y in range(chunks_y):
            for chunk_x in range(chunks_x):
                # Calculate window bounds
                x_start = chunk_x * chunk_size
                y_start = chunk_y * chunk_size
                x_end = min(x_start + chunk_size, width)
                y_end = min(y_start + chunk_size, height)
                
                # Create window
                window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
                
                # Read chunk
                chunk_data = src.read(window=window)
                
                # Process alpha band for mask if available
                if count >= 4:
                    alpha_chunk = chunk_data[3]  # 4th band (0-indexed)
                    mask[y_start:y_end, x_start:x_end] = alpha_chunk > 0
                
                # Progress indicator
                chunk_num = chunk_y * chunks_x + chunk_x + 1
                if chunk_num % 10 == 0 or chunk_num == total_chunks:
                    print(f"Processed chunk {chunk_num}/{total_chunks}")
        
        # Create result dictionary
        result = {
            'crs': crs,
            'transform': transform,
            'bounds': bounds,
            'mask': mask,
            'height': height,
            'width': width,
            'count': count
        }
        
        elapsed_time = time.time() - start_time
        print(f"Raster processing completed in {elapsed_time:.2f} seconds")
        
        return result

# Load the data with caching
print("Loading tire datasets...")

# Try to load cached tire data first
cached_tires = load_from_cache(original_tires_path, "tire_data")

if cached_tires is None:
    print("No cached tire data found, loading from files...")
    original_tires = gpd.read_file(original_tires_path)
    refined_tires = gpd.read_file(refined_tires_path)
    
    # Cache the loaded tire data
    tire_data = {
        'original_tires': original_tires,
        'refined_tires': refined_tires,
        'original_path': original_tires_path,
        'refined_path': refined_tires_path
    }
    save_to_cache(tire_data, original_tires_path, "tire_data")
else:
    print("Using cached tire data")
    original_tires = cached_tires['original_tires']
    refined_tires = cached_tires['refined_tires']

# Load orthomosaic with caching
print("Loading orthomosaic...")
cached_orthomosaic = load_from_cache(orthomosaic_path, "orthomosaic_data")

if cached_orthomosaic is None:
    # Read in chunks and cache the result
    print("No cached data found, processing orthomosaic in chunks...")
    orthomosaic_data = read_raster_in_chunks(orthomosaic_path, chunk_size=args.chunk_size)
    save_to_cache(orthomosaic_data, orthomosaic_path, "orthomosaic_data")
else:
    print("Using cached orthomosaic data")
    orthomosaic_data = cached_orthomosaic

# Check memory usage (only warn if significantly over limit)
if get_memory_usage() > args.memory_limit * 0.9:
    print("WARNING: High memory usage detected. Consider clearing cache with --clear-cache")

# Extract data from the loaded orthomosaic
crs = orthomosaic_data['crs']
transform = orthomosaic_data['transform']
mask = orthomosaic_data['mask']
bounds = box(*orthomosaic_data['bounds'])

# Clean up orthomosaic data to free memory
orthomosaic_data = None
optimize_memory()

# Check if we need to reproject to a projected CRS for spatial operations
def get_appropriate_projected_crs(geodf, target_crs=None):
    """
    Get an appropriate projected CRS for the data.
    If target_crs is provided and is projected, use it.
    Otherwise, find the best projected CRS for the data's location.
    """
    if target_crs and target_crs.is_projected:
        return target_crs
    
    # Get the centroid of the data to determine appropriate UTM zone
    centroid = geodf.union_all().centroid
    lon, lat = centroid.x, centroid.y
    
    # Find appropriate UTM zone
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
    
    return f"EPSG:{epsg_code}"

# Determine if we need to reproject
original_crs = original_tires.crs
refined_crs = refined_tires.crs
orthomosaic_crs = crs

print(f"Original tires CRS: {original_crs}")
print(f"Refined tires CRS: {refined_crs}")
print(f"Orthomosaic CRS: {orthomosaic_crs}")

# Check if any CRS is geographic (WGS84-like)
needs_reprojection = any(crs.is_geographic for crs in [original_crs, refined_crs, orthomosaic_crs])

if needs_reprojection:
    print("Geographic CRS detected. Reprojecting to projected CRS for spatial operations...")
    
    # Get appropriate projected CRS
    projected_crs = get_appropriate_projected_crs(original_tires, orthomosaic_crs)
    print(f"Target projected CRS: {projected_crs}")
    
    # Reproject all datasets
    original_tires = original_tires.to_crs(projected_crs)
    refined_tires = refined_tires.to_crs(projected_crs)
    
    # Update the working CRS
    working_crs = projected_crs
    print("Data reprojected successfully")
else:
    print("All CRS are already projected. No reprojection needed.")
    working_crs = crs

# Convert working_crs to CRS object if it's a string
if isinstance(working_crs, str):
    working_crs = rasterio.crs.CRS.from_string(working_crs)

# Ensure both datasets have the same CRS
original_tires = original_tires.to_crs(working_crs)
refined_tires = refined_tires.to_crs(working_crs)

# Create a study area from the actual tire data bounds (not orthomosaic bounds)
# This ensures we have a meaningful study area that contains all our data
all_tire_points = pd.concat([original_tires, refined_tires])
data_bounds = all_tire_points.total_bounds

print(f"Raw data bounds: {data_bounds}")
print(f"Data bounds type: {type(data_bounds)}")
print(f"Data bounds shape: {data_bounds.shape if hasattr(data_bounds, 'shape') else 'no shape'}")

# Check if bounds are valid
if np.any(np.isnan(data_bounds)) or np.any(np.isinf(data_bounds)):
    print("Warning: Invalid bounds detected, using fallback bounds")
    # Use a simple bounding box around the data
    x_coords = np.concatenate([original_tires.geometry.x, refined_tires.geometry.x])
    y_coords = np.concatenate([original_tires.geometry.y, refined_tires.geometry.y])
    data_bounds = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
    print(f"Fallback bounds: {data_bounds}")

# Add some padding to the bounds for better visualization
padding = max(data_bounds[2] - data_bounds[0], data_bounds[3] - data_bounds[1]) * 0.1
padded_bounds = [
    data_bounds[0] - padding,  # minx
    data_bounds[1] - padding,  # miny
    data_bounds[2] + padding,  # maxx
    data_bounds[3] + padding   # maxy
]

# Create study area from padded bounds
study_area_bounds = box(*padded_bounds)
study_area = gpd.GeoDataFrame(geometry=[study_area_bounds], crs=working_crs)

print(f"Study area bounds: {data_bounds}")
print(f"Study area with padding: {padded_bounds}")
print(f"Study area CRS: {study_area.crs}")
print(f"Study area geometry type: {type(study_area_bounds)}")



# Add a unique ID to each point for later reference
original_tires['dataset'] = 'original'
original_tires['point_id'] = range(1, len(original_tires) + 1)
refined_tires['dataset'] = 'refined'
refined_tires['point_id'] = range(1, len(refined_tires) + 1)

# Identify new points in the refined dataset
# This requires finding points in refined that don't match original points
# Using a spatial join with a small buffer to account for slight differences in point placement

# Buffer distance in CRS units (adjust based on your data precision)
if working_crs.is_projected:
    buffer_distance = 0.5  # meters
    print(f"Using buffer distance: {buffer_distance} meters")
else:
    buffer_distance = 0.00001  # degrees (very small for geographic CRS)
    print(f"Using buffer distance: {buffer_distance} degrees")

# Buffer the original points
original_buffered = original_tires.copy()
original_buffered.geometry = original_tires.geometry.buffer(buffer_distance).values

# Spatial join to find refined points that intersect with buffered original points
joined = gpd.sjoin(refined_tires, original_buffered, how='left', predicate='intersects')

# Points in refined dataset that don't intersect with any original points are new
joined_valid = joined.dropna(subset=['index_right'])
new_tires_idx = ~refined_tires.index.isin(joined_valid.index)
new_tires = refined_tires.loc[new_tires_idx].copy()
new_tires['dataset'] = 'new'

print(f"Original tires: {len(original_tires)}")
print(f"Refined tires: {len(refined_tires)}")
print(f"New tires (found by model): {len(new_tires)}")

# Combine all datasets for some analyses
all_tires = pd.concat([original_tires, new_tires])

# Create a figure with multiple subplots
fig = plt.figure(figsize=(24, 30))  # Increased figure size for better spacing
fig.suptitle('Spatial Analysis of Tire Detection\n(Scale bars in meters, coordinates in UTM Zone 50S)', fontsize=24, y=0.98)

# 1. SPATIAL DENSITY COMPARISON MAPS
# ----------------------------------------

# Helper function for KDE maps
def create_kde_map(points, bounds, grid_size=100):
    """
    Create kernel density estimation map with improved bandwidth selection
    and normalization for better visualization.
    """
    if len(points) < 2:
        # Return empty grid if insufficient points
        minx, miny, maxx, maxy = bounds
        xx, yy = np.mgrid[minx:maxx:grid_size*1j, miny:maxy:grid_size*1j]
        z = np.zeros_like(xx)
        return xx, yy, z
    
    minx, miny, maxx, maxy = bounds
    xx, yy = np.mgrid[minx:maxx:grid_size*1j, miny:maxy:grid_size*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    values = np.vstack([points.geometry.x, points.geometry.y])
    
    # Use Scott's rule for bandwidth selection (more robust)
    try:
        # Calculate optimal bandwidth using Scott's rule
        n_points = values.shape[1]
        d = values.shape[0]  # dimensions (2 for x,y)
        bandwidth = n_points ** (-1. / (d + 4))
        
        # Apply bandwidth scaling to the data
        scaled_values = values / bandwidth
        
        kernel = stats.gaussian_kde(scaled_values)
        z = np.reshape(kernel(positions / bandwidth), xx.shape)
        
        # Normalize to 0-1 range for better visualization
        if z.max() > z.min():
            z = (z - z.min()) / (z.max() - z.min())
        
        return xx, yy, z
    except Exception as e:
        print(f"Warning: KDE calculation failed: {e}")
        # Fallback: return uniform distribution
        z = np.ones_like(xx) * 0.5
        return xx, yy, z

# Create KDE maps
print(f"Creating KDE maps with study area bounds: {study_area.total_bounds}")
x_min, y_min, x_max, y_max = study_area.total_bounds

# Check if bounds are valid for KDE
if x_max <= x_min or y_max <= y_min:
    print("Warning: Invalid bounds for KDE, using data bounds instead")
    x_min, y_min, x_max, y_max = data_bounds
    print(f"Using data bounds for KDE: {data_bounds}")

xx, yy, z_original = create_kde_map(original_tires, [x_min, y_min, x_max, y_max])
_, _, z_refined = create_kde_map(refined_tires, [x_min, y_min, x_max, y_max])
z_diff = z_refined - z_original

print(f"KDE results - Original range: {z_original.min():.6f} to {z_original.max():.6f}")
print(f"KDE results - Refined range: {z_refined.min():.6f} to {z_refined.max():.6f}")
print(f"KDE results - Difference range: {z_diff.min():.6f} to {z_diff.max():.6f}")



# Define a custom diverging colormap for the difference map
# Use a more robust normalization that handles edge cases
z_diff_max = np.max(np.abs(z_diff))
if z_diff_max > 0:
    divnorm = plt.Normalize(vmin=-z_diff_max*0.8, vmax=z_diff_max*0.8)
else:
    divnorm = plt.Normalize(vmin=-0.1, vmax=0.1)

div_cmap = LinearSegmentedColormap.from_list('custom_div_cmap', 
                                             ['#3b4cc0', 'white', '#b40426'], 
                                             N=256)

# Plot the KDE maps
# Original Density
ax1 = fig.add_subplot(321)
im1 = ax1.pcolormesh(xx, yy, z_original, cmap='viridis', shading='auto')
ax1.set_title(f'A) Original Tire Annotation Density (n={len(original_tires)})\nShows spatial clustering of human-annotated tires\n(Density values normalized 0-1 for visualization)')
ax1.set_aspect('equal')
plt.colorbar(im1, ax=ax1, label='Normalized Density')
ax1.add_artist(ScaleBar(1000, dimension='si-length', units='m'))  # 1000m = 1km

ax1.set_axis_off()

# Refined Density
ax2 = fig.add_subplot(322)
im2 = ax2.pcolormesh(xx, yy, z_refined, cmap='viridis', shading='auto')
ax2.set_title(f'B) Refined Tire Annotation Density (n={len(refined_tires)})\nShows spatial clustering including model-detected additional tires\n(Density values normalized 0-1 for visualization)')
ax2.set_aspect('equal')
plt.colorbar(im2, ax=ax2, label='Normalized Density')
ax2.add_artist(ScaleBar(1000, dimension='si-length', units='m'))  # 1000m = 1km

ax2.set_axis_off()

# Difference Map
ax3 = fig.add_subplot(323)
im3 = ax3.pcolormesh(xx, yy, z_diff, cmap=div_cmap, norm=divnorm, shading='auto')
ax3.set_title(f'C) Difference in Density (Refined - Original)\nRed areas: model found more tires, Blue areas: original had higher density')
ax3.set_aspect('equal')
plt.colorbar(im3, ax=ax3, label='Density Difference')
ax3.add_artist(ScaleBar(1000, dimension='si-length', units='m'))  # 1000m = 1km

ax3.set_axis_off()

# Add summary statistics as text overlay
total_diff = np.sum(z_diff)
mean_diff = np.mean(z_diff)
ax3.text(0.02, 0.90, f'Total diff: {total_diff:.3f}\nMean diff: {mean_diff:.3f}\nNote: Values represent relative\ndensity changes after normalization', 
         transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# 2. NEAREST NEIGHBOR ANALYSIS
# ----------------------------------------

# Calculate nearest neighbor distances
def calculate_nn_distances(points):
    if len(points) <= 1:
        return np.array([])
    
    coords = np.vstack((points.geometry.x, points.geometry.y)).T
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    # Second column contains distance to the nearest neighbor
    # (first column is 0, distance to self)
    return distances[:, 1]

original_nn_distances = calculate_nn_distances(original_tires)
refined_nn_distances = calculate_nn_distances(refined_tires)

# Plot nearest neighbor distance distributions
ax4 = fig.add_subplot(324)
sns.histplot(original_nn_distances, kde=True, stat='density', 
             color='blue', label='Original Annotations', alpha=0.5, ax=ax4)
sns.histplot(refined_nn_distances, kde=True, stat='density', 
             color='red', label='Refined Annotations', alpha=0.5, ax=ax4)
ax4.set_xlabel('Distance to Nearest Neighbor (m)')
ax4.set_ylabel('Density')
ax4.set_title(f'D) Nearest Neighbor Distance Distribution\nShows clustering patterns - peaks at low distances indicate tire clusters')
ax4.legend()

# Run KS test to compare distributions
ks_stat, ks_pval = stats.ks_2samp(original_nn_distances, refined_nn_distances)

# Calculate additional statistics
orig_mean = np.mean(original_nn_distances)
orig_std = np.std(original_nn_distances)
ref_mean = np.mean(refined_nn_distances)
ref_std = np.std(refined_nn_distances)

# Add comprehensive statistics
stats_text = f'KS test: p={ks_pval:.4f}\n'
stats_text += f'Original: μ={orig_mean:.1f}m, σ={orig_std:.1f}m\n'
stats_text += f'Refined: μ={ref_mean:.1f}m, σ={ref_std:.1f}m'

ax4.annotate(stats_text, xy=(0.05, 0.90), xycoords='axes fraction', 
             fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# 3. QUADRAT-BASED ENRICHMENT ANALYSIS
# ----------------------------------------

# Create a grid for quadrat analysis
def create_grid(bounds, size):
    minx, miny, maxx, maxy = bounds
    
    # Create grid cells
    x_points = np.arange(minx, maxx + size, size)
    y_points = np.arange(miny, maxy + size, size)
    
    cells = []
    for i in range(len(x_points) - 1):
        for j in range(len(y_points) - 1):
            cells.append(box(x_points[i], y_points[j], x_points[i+1], y_points[j+1]))
    
    return gpd.GeoDataFrame(geometry=cells, crs=working_crs)

# Grid cell size (adjust based on your study area size and data density)
# Use a more appropriate grid size based on the study area extent
study_area_width = study_area.total_bounds[2] - study_area.total_bounds[0]
study_area_height = study_area.total_bounds[3] - study_area.total_bounds[1]
study_area_extent = max(study_area_width, study_area_height)

# Use a grid size that gives reasonable number of cells (aim for 10-20 cells)
grid_size = max(50, int(study_area_extent / 15))  # meters or CRS units
print(f"Study area extent: {study_area_extent:.1f}m, using grid size: {grid_size}m")
grid = create_grid(study_area.total_bounds, grid_size)

# Count points in each grid cell
print(f"Grid has {len(grid)} cells")
print(f"Grid bounds: {grid.total_bounds}")

# Check if grid cells are valid
if grid.total_bounds[2] <= grid.total_bounds[0] or grid.total_bounds[3] <= grid.total_bounds[1]:
    print("Warning: Invalid grid bounds detected")
    print(f"Grid bounds: {grid.total_bounds}")

original_count = gpd.sjoin(grid, original_tires, how='left', predicate='contains').groupby(level=0).size()
refined_count = gpd.sjoin(grid, refined_tires, how='left', predicate='contains').groupby(level=0).size()
new_count = gpd.sjoin(grid, new_tires, how='left', predicate='contains').groupby(level=0).size()

print(f"Original count range: {original_count.min()} to {original_count.max()}")
print(f"Refined count range: {refined_count.min()} to {refined_count.max()}")
print(f"New count range: {new_count.min()} to {new_count.max()}")

# Create DataFrame with counts
grid_counts = pd.DataFrame(index=grid.index)
grid_counts['original'] = original_count.reindex(grid.index, fill_value=0)
grid_counts['refined'] = refined_count.reindex(grid.index, fill_value=0)
grid_counts['new'] = new_count.reindex(grid.index, fill_value=0)

# Calculate ratio of new to original (adding small epsilon to avoid division by zero)
epsilon = 0.1  # Small value to avoid division by zero
grid_counts['ratio'] = grid_counts['new'] / (grid_counts['original'] + epsilon)

# Remove cells with no tires in either dataset for the analysis
analysis_cells = grid_counts[(grid_counts['original'] > 0) | (grid_counts['refined'] > 0)]

# Plot the enrichment analysis
ax5 = fig.add_subplot(325)
scatter = ax5.scatter(analysis_cells['original'], analysis_cells['ratio'], 
                     alpha=0.6, c=analysis_cells['original'], cmap='viridis')
ax5.set_xlabel('Original Tire Count per Grid Cell')
ax5.set_ylabel('Ratio of New:Original Tires')
ax5.set_title(f'E) Tire Detection Enrichment by Original Density\nGrid size: {grid_size}m\nHigh ratios in low-density areas suggest model found many missed tires')

# Add trend line and statistics
if len(analysis_cells) > 1:
    # Log transform for better fit
    mask = (analysis_cells['original'] > 0)
    if mask.sum() > 1:
        x = analysis_cells.loc[mask, 'original']
        y = analysis_cells.loc[mask, 'ratio']
        slope, intercept, r_value, p_value, _ = stats.linregress(np.log1p(x), y)
        x_pred = np.linspace(0, analysis_cells['original'].max(), 100)
        y_pred = slope * np.log1p(x_pred) + intercept
        ax5.plot(x_pred, y_pred, 'r--', linewidth=2)
        
        # Add comprehensive statistics
        stats_text = f'R² = {r_value**2:.3f}, p = {p_value:.4f}\n'
        stats_text += f'Slope = {slope:.3f}\n'
        stats_text += f'Grid cells: {len(analysis_cells)}'
        
        ax5.annotate(stats_text, xy=(0.05, 0.08), xycoords='axes fraction', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    else:
        ax5.annotate('Insufficient data for trend analysis', 
                    xy=(0.5, 0.45), xycoords='axes fraction', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
else:
    ax5.annotate('Insufficient data for analysis', 
                xy=(0.5, 0.45), xycoords='axes fraction', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.colorbar(scatter, ax=ax5, label='Original Tire Count')

# Print grid analysis summary
print(f"Grid analysis: {len(grid)} cells, {len(analysis_cells)} cells with data")
print(f"Original tires in grid: {analysis_cells['original'].sum()}")
print(f"Refined tires in grid: {analysis_cells['refined'].sum()}")
print(f"New tires in grid: {analysis_cells['new'].sum()}")

# 4. HOTSPOT ANALYSIS
# ----------------------------------------

# Create a spatial weights matrix
# We'll use original and refined datasets separately

# Function to create weight matrix and run Local Getis-Ord
def get_local_G(points, k=8):
    """
    Create a spatial weights matrix and calculate Local Getis-Ord G* statistic.
    Improved to handle connectivity issues and edge cases.
    """
    if len(points) < 2:
        return np.zeros(len(points))
    
    # Create a distance-based weights matrix
    coords = np.vstack((points.geometry.x, points.geometry.y)).T
    
    # Use a larger k to ensure better connectivity and capture appropriate spatial scale
    # For a 4.2km study area, we want neighborhoods that capture cluster patterns
    k_neighbors = min(k + 1, len(points) - 1)
    if k_neighbors < 1:
        return np.zeros(len(points))
    
    # Calculate typical distance between points to understand spatial scale
    if len(points) > 1:
        coords = np.vstack((points.geometry.x, points.geometry.y)).T
        distances = []
        for i in range(min(100, len(points))):  # Sample first 100 points
            for j in range(i+1, min(i+11, len(points))):  # Check 10 nearest neighbors
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)
        if distances:
            avg_distance = np.mean(distances)
            print(f"Average distance between points: {avg_distance:.1f}m")
            # Adjust k if needed based on spatial scale
            if avg_distance > 100:  # If points are far apart, use larger neighborhoods
                k_neighbors = min(k_neighbors * 2, len(points) - 1)
                print(f"Adjusted k to {k_neighbors} due to large spatial separation")
    
    try:
        knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(coords)
        distances, indices = knn.kneighbors(coords)
        
        # Create weights dictionary with minimum connectivity
        weights_dict = {}
        for i in range(len(points)):
            neighbors = {}
            for j in range(1, len(indices[i])):  # Start from 1 to skip self
                if indices[i, j] < len(points):  # Check bounds
                    # Use inverse distance weighting for better spatial relationships
                    if distances[i, j] > 0:
                        weight = 1.0 / distances[i, j]
                    else:
                        weight = 1.0  # Fallback for zero distance
                    neighbors[indices[i, j]] = weight
            weights_dict[i] = neighbors
        
        # Create weights object
        w = libpysal.weights.W(weights_dict)
        
        # Check if weights matrix is fully connected
        if len(w.islands) > 0:
            print(f"Warning: Weights matrix has {len(w.islands)} isolated points, attempting to connect...")
            # Try to connect isolated points to their nearest neighbor
            for island in w.islands:
                if island < len(points):
                    # Find nearest neighbor for isolated point
                    point_coords = coords[island].reshape(1, -1)
                    distances_nn, indices_nn = knn.kneighbors(point_coords, n_neighbors=min(2, len(points)))
                    if len(indices_nn[0]) > 1:
                        nearest = indices_nn[0, 1]
                        if nearest < len(points) and nearest != island:
                            # Add bidirectional connection
                            if island not in weights_dict:
                                weights_dict[island] = {}
                            if nearest not in weights_dict:
                                weights_dict[nearest] = {}
                            
                            # Add weights (inverse distance)
                            if distances_nn[0, 1] > 0:
                                weight = 1.0 / distances_nn[0, 1]
                            else:
                                weight = 1.0
                            
                            weights_dict[island][nearest] = weight
                            weights_dict[nearest][island] = weight
            
            # Recreate weights object
            w = libpysal.weights.W(weights_dict)
        
        # Ensure minimum connectivity by adding nearest neighbor if needed
        if len(w.islands) > 0:
            print(f"Still have {len(w.islands)} isolated points, forcing minimum connectivity...")
            for island in w.islands:
                if island < len(points):
                    # Find the absolute nearest neighbor
                    point_coords = coords[island].reshape(1, -1)
                    all_distances = np.linalg.norm(coords - point_coords, axis=1)
                    all_distances[island] = np.inf  # Exclude self
                    nearest_idx = np.argmin(all_distances)
                    
                    if nearest_idx < len(points):
                        # Force connection
                        if island not in weights_dict:
                            weights_dict[island] = {}
                        if nearest_idx not in weights_dict:
                            weights_dict[nearest_idx] = {}
                        
                        # Add bidirectional connection with unit weight
                        weights_dict[island][nearest_idx] = 1.0
                        weights_dict[nearest_idx][island] = 1.0
            
            # Final weights object
            w = libpysal.weights.W(weights_dict)
        
        w.transform = 'r'  # Row-standardized
        
        # Calculate Local G with error handling
        # Use spatial coordinates to create meaningful variation for Local G*
        coords_array = np.vstack((points.geometry.x, points.geometry.y)).T
        
        # Normalize coordinates to 0-1 range to avoid scale issues
        x_norm = (coords_array[:, 0] - coords_array[:, 0].min()) / (coords_array[:, 0].max() - coords_array[:, 0].min())
        y_norm = (coords_array[:, 1] - coords_array[:, 1].min()) / (coords_array[:, 1].max() - coords_array[:, 1].min())
        
        # Use a combination of normalized coordinates as the variable
        y = x_norm + y_norm  # This creates spatial variation
        
        # Check if we have sufficient variation in the data
        if len(np.unique(y)) < 2:
            print("Warning: Insufficient variation in data for Local G calculation")
            return np.zeros(len(points))
        
        try:
            lg = G_Local(y, w)
            z_scores = lg.Zs
            
            # Handle invalid values more robustly
            z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Additional validation: replace extreme values
            z_threshold = 10.0  # Reasonable threshold for z-scores
            z_scores = np.clip(z_scores, -z_threshold, z_threshold)
            
            # Check if we got meaningful results
            if np.all(z_scores == 0):
                print("Warning: All z-scores are zero, Local G may not be meaningful")
            
            return z_scores
        except Exception as e:
            print(f"Warning: Local G calculation failed: {e}")
            return np.zeros(len(points))
            
    except Exception as e:
        print(f"Warning: KNN calculation failed: {e}")
        return np.zeros(len(points))

# We need enough points for a meaningful analysis
if len(original_tires) >= 10 and len(refined_tires) >= 10:  # Lowered threshold
    print("Running hotspot analysis...")
    
    # Get Local G z-scores with more robust parameters
    # Use larger neighborhoods to better capture the spatial scale of clusters
    # The previous k=12 was too small for the 4.2km study area
    k_original = min(20, len(original_tires)-1)  # Larger neighborhood for original data
    k_refined = min(25, len(refined_tires)-1)    # Even larger for refined data
    
    print(f"Using k={k_original} neighbors for original data, k={k_refined} for refined data")
    
    original_tires['g_zscore'] = get_local_G(original_tires, k=k_original)
    refined_tires['g_zscore'] = get_local_G(refined_tires, k=k_refined)
    
    # Classify hotspots (significant positive z-scores)
    # Using 1.0 for 84% confidence (z-score) - more sensitive to detect actual clusters
    # The previous threshold of 1.65 was too restrictive and missed obvious clusters
    confidence_threshold = 1.0
    original_tires['hotspot'] = original_tires['g_zscore'] > confidence_threshold
    refined_tires['hotspot'] = refined_tires['g_zscore'] > confidence_threshold
    
    # Also check for negative hotspots (cold spots) for completeness
    original_tires['coldspot'] = original_tires['g_zscore'] < -confidence_threshold
    refined_tires['coldspot'] = refined_tires['g_zscore'] < -confidence_threshold
    
    # Print hotspot statistics
    orig_hotspots = original_tires[original_tires['hotspot']]
    ref_hotspots = refined_tires[refined_tires['hotspot']]
    print(f"Original hotspots: {len(orig_hotspots)} ({len(orig_hotspots)/len(original_tires)*100:.1f}%)")
    print(f"Refined hotspots: {len(ref_hotspots)} ({len(ref_hotspots)/len(refined_tires)*100:.1f}%)")
    
    # Debug: Show z-score distribution
    print(f"Original z-scores: min={original_tires['g_zscore'].min():.3f}, max={original_tires['g_zscore'].max():.3f}")
    print(f"Refined z-scores: min={refined_tires['g_zscore'].min():.3f}, max={refined_tires['g_zscore'].max():.3f}")
    print(f"Original z-scores > 1.0: {(original_tires['g_zscore'] > 1.0).sum()}")
    print(f"Refined z-scores > 1.0: {(refined_tires['g_zscore'] > 1.0).sum()}")
    
    if len(orig_hotspots) > 0 and len(ref_hotspots) > 0:
        # Create a new field in original points to track if they remain hotspots
        original_buffered = original_tires.copy()
        original_buffered.geometry = original_tires.geometry.buffer(buffer_distance).values
        
        # Find which original hotspots intersect with refined hotspots
        hotspots_original = original_tires[original_tires['hotspot']]
        hotspots_refined = refined_tires[refined_tires['hotspot']]
        
        # Buffer the original hotspots
        hotspots_original_buffered = hotspots_original.copy()
        hotspots_original_buffered.geometry = hotspots_original.geometry.buffer(buffer_distance * 5).values  # Larger buffer for hotspot regions
        
        # Find which refined hotspots intersect with original hotspot areas
        joined_hotspots = gpd.sjoin(hotspots_refined, hotspots_original_buffered, how='left', predicate='intersects')
        
        # Refined hotspots that don't intersect with any original hotspot areas are new
        joined_hotspots_valid = joined_hotspots.dropna(subset=['index_right'])
        new_hotspots_idx = ~hotspots_refined.index.isin(joined_hotspots_valid.index)
        new_hotspots = hotspots_refined.loc[new_hotspots_idx]
        
        # Original hotspots that don't have a refined hotspot in their area are diminished
        joined_original = gpd.sjoin(hotspots_original_buffered, hotspots_refined, how='left', predicate='intersects')
        joined_original_valid = joined_original.dropna(subset=['index_right'])
        diminished_hotspots_idx = ~hotspots_original.index.isin(joined_original_valid.index)
        diminished_hotspots = hotspots_original.loc[diminished_hotspots_idx]
        
        # Persistent hotspots are the rest
        persistent_hotspots_idx = list(set(hotspots_original.index) - set(diminished_hotspots.index))
        persistent_hotspots = hotspots_original.loc[persistent_hotspots_idx]
        
        # Plot the hotspot analysis
        ax6 = fig.add_subplot(326)
        
        # Plot background (all points in gray)
        ax6.scatter(all_tires.geometry.x, all_tires.geometry.y, c='lightgray', s=10, alpha=0.3, label='All Tires')
        
        # Plot the hotspot categories
        ax6.scatter(persistent_hotspots.geometry.x, persistent_hotspots.geometry.y, 
                   c='red', s=30, alpha=0.7, label='Persistent Hotspots')
        ax6.scatter(new_hotspots.geometry.x, new_hotspots.geometry.y, 
                   c='orange', s=30, alpha=0.7, label='New Hotspots')
        ax6.scatter(diminished_hotspots.geometry.x, diminished_hotspots.geometry.y, 
                   c='blue', s=30, alpha=0.7, label='Diminished Hotspots')
        
        ax6.set_title(f'F) Hotspot Analysis of Tire Distribution\n(84% confidence, threshold={confidence_threshold})\nPersistent: areas significant in both datasets, New: only in refined data, Diminished: no longer significant')
        ax6.set_aspect('equal')
        ax6.add_artist(ScaleBar(1000, dimension='si-length', units='m'))
        
        ax6.set_axis_off()
        
        # Position legend to avoid obscuring the plot - use upper left with more space
        ax6.legend(loc='upper left', bbox_to_anchor=(0.02, 0.95))
        
        # Print counts
        print(f"Persistent hotspots: {len(persistent_hotspots)}")
        print(f"New hotspots: {len(new_hotspots)}")
        print(f"Diminished hotspots: {len(diminished_hotspots)}")
        
        # Add summary statistics to plot - position to avoid legend overlap
        stats_text = f'Persistent: {len(persistent_hotspots)}\nNew: {len(new_hotspots)}\nDiminished: {len(diminished_hotspots)}'
        ax6.text(0.98, 0.85, stats_text, transform=ax6.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    else:
        ax6 = fig.add_subplot(326)
        ax6.text(0.5, 0.45, "No hotspots detected\n(threshold may be too high)", 
                 ha='center', va='center', fontsize=12)
        ax6.set_title(f'F) Hotspot Analysis of Tire Distribution\n(84% confidence, threshold={confidence_threshold})\nPersistent: areas significant in both datasets, New: only in refined data, Diminished: no longer significant')
        ax6.set_axis_off()
else:
    ax6 = fig.add_subplot(326)
    ax6.text(0.5, 0.45, "Insufficient data for hotspot analysis\n(requires at least 10 points in each dataset)", 
             ha='center', va='center', fontsize=12)
    ax6.set_title('F) Hotspot Analysis of Tire Distribution\nPersistent: areas significant in both datasets, New: only in refined data, Diminished: no longer significant')
    ax6.set_axis_off()

# Adjust layout with more generous spacing
plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=4.0)
fig.subplots_adjust(top=0.94, bottom=0.02, left=0.02, right=0.98)

# Save the figure
fig.savefig(os.path.join(output_dir, 'tire_spatial_analysis.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(output_dir, 'tire_spatial_analysis.pdf'), bbox_inches='tight')

# Add a standalone map showing all points with background imagery for context
fig2, ax = plt.subplots(figsize=(12, 12))
try:
    # Try to add basemap with appropriate zoom level
    ctx.add_basemap(ax, crs=working_crs, source=ctx.providers.CartoDB.Positron, zoom=18)
except Exception as e:
    print(f"Warning: Could not add basemap: {e}")
    # Fallback: just plot the points without basemap
    pass

ax.scatter(original_tires.geometry.x, original_tires.geometry.y, c='blue', s=10, alpha=0.7, label='Original Annotations')
ax.scatter(new_tires.geometry.x, new_tires.geometry.y, c='red', s=10, alpha=0.7, label='New Detections')
ax.set_title('All Tire Annotations with Basemap')
ax.legend()

# Add scalebar with proper aspect ratio
ax.set_aspect('equal')
try:
    ax.add_artist(ScaleBar(1, dimension='si-length', units='m', rotation='horizontal-only'))
except Exception as e:
    print(f"Warning: Could not add scalebar: {e}")

ax.set_axis_off()
fig2.savefig(os.path.join(output_dir, 'tire_locations_map.png'), dpi=300, bbox_inches='tight')

# Save the analysis data for future reference
grid_counts.to_csv(os.path.join(output_dir, 'grid_analysis_data.csv'))

# Final cleanup
optimize_memory()

# Print final summary
print("\n" + "="*60)
print("SPATIAL ANALYSIS SUMMARY")
print("="*60)
print(f"Original tires: {len(original_tires)}")
print(f"Refined tires: {len(refined_tires)}")
print(f"New tires (found by model): {len(new_tires)}")
print(f"Study area extent: {study_area_extent:.1f}m x {study_area_extent:.1f}m")
print(f"Grid size used: {grid_size}m")
print(f"Working CRS: {working_crs}")
print(f"Results saved to: {output_dir}")
print(f"Cache saved to: {cache_dir}")
print("="*60)
print("Analysis complete!")