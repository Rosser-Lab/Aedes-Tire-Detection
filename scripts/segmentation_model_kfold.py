#!/usr/bin/env python3
"""
Segmentation Model K-Fold Cross Validation Script
Converted from segmentation_model_kfold.ipynb for Stanford Sherlock cluster execution
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import pickle
from PIL import Image
import random
from pathlib import Path
import cv2
import argparse
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable for calculated class weight
calculated_class_weight = 20.0  # Default fallback value

class Config:
    def __init__(self, config_file=None):
        # Set basic parameters first (these can be overridden by JSON)
        self.base_path = '/scratch/groups/jrosser/drone_dengue'  # Use jrosser scratch for performance
        self.image_suffix = '_image.tif'
        self.mask_suffix = '_mask_tires.tif'
        self.tile_size = 256
        self.overlap = 64
        self.batch_size = 16
        self.model_iteration = None  # Will be auto-generated if not set
        self.num_epochs = 50
        self.k_folds = 5
        self.random_seed = 42
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        self.train_threshold = 0.65
        self.val_threshold = 0.99
        self.model_def = 'UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1, encoder_weights="imagenet")'
        self.use_curriculum = True
        self.use_negative_samples = True
        self.use_staged_curriculum = True
        self.curriculum_start_ratio = 0.0
        self.curriculum_end_ratio = 1.0
        self.curriculum_rate = 1.25
        self.max_negative_ratio = self.curriculum_end_ratio
        self.curriculum_stages = 8
        self.epochs_per_stage = 5
        self.holdout_percentage = 0.05
        
        # Additional parameters for class weight calculation and loss function
        self.use_calculated_class_weight = False
        self.calculate_from_training_data = False
        self.bce_weight = 1.0
        self.dice_weight = 1.0
        
        # Load configuration from file if provided (this can override any of the above)
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # NOW calculate all derived paths using the final values (after JSON overrides)
        self._setup_paths()
    
    def _generate_model_iteration(self):
        """Generate unique model iteration name with architecture and auto-increment"""
        # Extract model name from model_def (e.g., "UnetPlusPlus" -> "unetplusplus")
        model_name = self.model_def.split('(')[0].lower()
        
        # Base directory for this model
        base_dir = os.path.join(self.base_path, 'model_data')
        
        # Find existing runs and increment
        counter = 1
        while True:
            iteration_name = f"{model_name}_run_{counter:03d}"
            if not os.path.exists(os.path.join(base_dir, iteration_name)):
                break
            counter += 1
        
        return iteration_name
    
    def _setup_paths(self):
        """Set up all file and directory paths after config is loaded"""
        # Auto-generate model_iteration ONLY if it wasn't set in JSON config
        if self.model_iteration is None:
            self.model_iteration = self._generate_model_iteration()
            logger.info(f"Auto-generated model iteration: {self.model_iteration}")
        else:
            logger.info(f"Using config-specified model iteration: {self.model_iteration}")
        
        # Calculate derived paths
        self.base_path_model = os.path.join(self.base_path, 'model_data')
        self.data_folder = os.path.join(self.base_path, 'data')
        self.image_folder = os.path.join(self.data_folder, 'images')
        self.mask_folder = os.path.join(self.data_folder, 'masks')
        
        # Extract model name from model_def (e.g., "UnetPlusPlus" from "UnetPlusPlus(encoder_name=...)")
        model_name = self.model_def.split('(')[0] if '(' in self.model_def else 'cnn'
        
        # Set up model weight paths
        self.initial_model_weights = os.path.join(self.base_path_model, self.model_iteration, f'{model_name.lower()}_trash_detection_initial.pth')
        self.best_model_weights = os.path.join(self.base_path_model, self.model_iteration, f'{model_name.lower()}_trash_detection_best.pth')
        self.model_definition_file = os.path.join(self.base_path_model, self.model_iteration, 'model_definition.txt')
        self.save_path = os.path.join(self.base_path_model, self.model_iteration, f'{model_name.lower()}_trash_detection.pth')
        self.checkpoint_path = os.path.join(self.base_path_model, self.model_iteration, 'checkpoint.pth')
        self.optimizer_path = os.path.join(self.base_path_model, self.model_iteration, 'optimizer.pth')
        self.scheduler_path = os.path.join(self.base_path_model, self.model_iteration, 'scheduler.pth')
    
    def load_from_file(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.info(f"Updated config.{key} = {value}")
        except Exception as e:
            logger.warning(f"Could not load config file {config_file}: {e}")
    
    def save_config_to_output(self):
        """Save current config parameters to a text file in the output directory"""
        # Use the directory path (without the .pth filename) for saving config
        config_dir = os.path.dirname(self.save_path)
        config_file_path = os.path.join(config_dir, 'experiment_config.txt')
        
        with open(config_file_path, 'w') as f:
            f.write("=== EXPERIMENT CONFIGURATION ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Architecture: {self.model_def}\n")
            f.write(f"Model Iteration: {self.model_iteration}\n")
            f.write(f"Base Path: {self.base_path}\n")
            f.write(f"Data Folder: {self.data_folder}\n")
            f.write(f"Image Folder: {self.image_folder}\n")
            f.write(f"Mask Folder: {self.mask_folder}\n")
            f.write(f"Tile Size: {self.tile_size}\n")
            f.write(f"Overlap: {self.overlap}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Number of Epochs: {self.num_epochs}\n")
            f.write(f"K-Folds: {self.k_folds}\n")
            f.write(f"Random Seed: {self.random_seed}\n")
            f.write(f"Initial Learning Rate: {self.initial_lr}\n")
            f.write(f"Weight Decay: {self.weight_decay}\n")
            f.write(f"Train Threshold: {self.train_threshold}\n")
            f.write(f"Validation Threshold: {self.val_threshold}\n")
            f.write(f"Use Curriculum: {self.use_curriculum}\n")
            f.write(f"Use Negative Samples: {self.use_negative_samples}\n")
            f.write(f"Use Staged Curriculum: {self.use_staged_curriculum}\n")
            f.write(f"Curriculum Start Ratio: {self.curriculum_start_ratio}\n")
            f.write(f"Curriculum End Ratio: {self.curriculum_end_ratio}\n")
            f.write(f"Curriculum Rate: {self.curriculum_rate}\n")
            f.write(f"Curriculum Stages: {self.curriculum_stages}\n")
            f.write(f"Epochs Per Stage: {self.epochs_per_stage}\n")
            f.write(f"Holdout Percentage: {self.holdout_percentage}\n")
            f.write(f"Use Calculated Class Weight: {self.use_calculated_class_weight}\n")
            f.write(f"Calculate From Training Data: {self.calculate_from_training_data}\n")
            f.write(f"BCE Weight: {self.bce_weight}\n")
            f.write(f"Dice Weight: {self.dice_weight}\n")
            f.write(f"Image Suffix: {self.image_suffix}\n")
            f.write(f"Mask Suffix: {self.mask_suffix}\n")
            f.write(f"Initial Model Weights: {self.initial_model_weights}\n")
            f.write(f"Best Model Weights: {self.best_model_weights}\n")
            f.write(f"Save Path: {self.save_path}\n")
            f.write(f"Checkpoint Path: {self.checkpoint_path}\n")
            f.write(f"Optimizer Path: {self.optimizer_path}\n")
            f.write(f"Scheduler Path: {self.scheduler_path}\n")
            
            # Add environment info
            f.write(f"\n=== ENVIRONMENT INFO ===\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"PyTorch Version: {torch.__version__}\n")
            f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA Version: {torch.version.cuda}\n")
                f.write(f"GPU Device: {torch.cuda.get_device_name(0)}\n")
        
        logger.info(f"Configuration saved to: {config_file_path}")

class DroneTrashDataset(Dataset):
    def __init__(self, config, transforms, mode='train', indices=None):
        self.config = config
        self.transforms = transforms
        self.mode = mode
        
        self.image_mask_pairs = get_image_mask_pairs(config)
        self.image_srcs = [rasterio.open(img_path) for img_path, _ in self.image_mask_pairs]
        self.mask_srcs = [rasterio.open(mask_path) for _, mask_path in self.image_mask_pairs]
        
        self.total_tiles = 0
        self.tile_offsets = [0]
        self.n_tiles_width = []
        self.n_tiles_height = []
        
        for img_src in self.image_srcs:
            image_width = img_src.width
            image_height = img_src.height
            n_tiles_w = (image_width + config.tile_size - config.overlap - 1) // (config.tile_size - config.overlap)
            n_tiles_h = (image_height + config.tile_size - config.overlap - 1) // (config.tile_size - config.overlap)
            n_tiles = n_tiles_w * n_tiles_h
            self.total_tiles += n_tiles
            self.tile_offsets.append(self.total_tiles)
            self.n_tiles_width.append(n_tiles_w)
            self.n_tiles_height.append(n_tiles_h)

        if indices is None:
            self.indices = list(range(self.total_tiles))
        else:
            self.indices = indices

        logger.info(f"{self.mode.capitalize()} dataset - Total tiles: {self.total_tiles}, Valid tiles: {len(self.indices)}")

    def filter_tiles(self):
        valid_indices = []
        with tqdm(total=self.total_tiles, desc="Filtering tiles") as pbar:
            for idx in range(self.total_tiles):
                mask_tile = self.get_mask_tile(idx)
                if not np.all(mask_tile == 255):  # Exclude if all pixels are NoData
                    valid_indices.append(idx)
                pbar.update(1)
                if idx % 1000 == 0 or idx == self.total_tiles - 1:
                    pbar.set_postfix(valid_tiles=len(valid_indices))
        return valid_indices

    def get_mask_tile(self, idx):
        img_idx = next(i for i, offset in enumerate(self.tile_offsets) if offset > idx) - 1
        local_idx = idx - self.tile_offsets[img_idx]
        mask_src = self.mask_srcs[img_idx]
        n_tiles_width = self.n_tiles_width[img_idx]
        row = local_idx // n_tiles_width
        col = local_idx % n_tiles_width
        x_offset = col * (self.config.tile_size - self.config.overlap)
        y_offset = row * (self.config.tile_size - self.config.overlap)
        mask_tile = mask_src.read(
            1,
            window=Window(x_offset, y_offset, self.config.tile_size, self.config.tile_size)
        )
        return mask_tile

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Index must be an integer, not {type(idx)}")

        try:
            real_idx = self.indices[idx]
            
            # Find which image/mask pair this index belongs to
            img_idx = next(i for i, offset in enumerate(self.tile_offsets) if offset > real_idx) - 1
            
            local_idx = real_idx - self.tile_offsets[img_idx]

            image_src = self.image_srcs[img_idx]
            mask_src = self.mask_srcs[img_idx]
            
            n_tiles_width = self.n_tiles_width[img_idx]
            
            row = local_idx // n_tiles_width
            col = local_idx % n_tiles_width
            
            x_offset = col * (self.config.tile_size - self.config.overlap)
            y_offset = row * (self.config.tile_size - self.config.overlap)
           
            # Read image tile
            image_tile = image_src.read(
                [1, 2, 3],
                window=Window(x_offset, y_offset, self.config.tile_size, self.config.tile_size)
            )
            image_tile = np.transpose(image_tile, (1, 2, 0))  # CHW to HWC
            image_tile = image_tile.astype(np.float32) / 255.0  # Scale to [0, 1]
            
            # Read mask tile
            mask_tile = mask_src.read(
                1,
                window=Window(x_offset, y_offset, self.config.tile_size, self.config.tile_size)
            )
            mask_tile = mask_tile.squeeze()  # Remove the channel dimension
            
            # Handle NoData and binarize mask
            mask_tile = np.where(mask_tile == 255, 0, mask_tile)
            mask_tile = np.where(mask_tile > 0, 1, 0).astype(np.float32)

            # Apply transformations
            if self.transforms:
                if self.mode == 'train':
                    # Concatenate image and mask
                    combined = np.concatenate([image_tile, mask_tile[..., None]], axis=-1)
                    # Apply spatial transformations
                    augmented = self.transforms['spatial'](image=combined)
                    combined_transformed = augmented['image']

                    # Split the combined array back into image and mask
                    image_tile = combined_transformed[..., :3]
                    mask_tile = combined_transformed[..., 3]
                    mask_tile = np.where(mask_tile > 0, 1, 0)  # ensure the masks remain binary

                    # Convert to uint8 for photometric transformations
                    image_tile = (image_tile * 255).astype(np.uint8)
        
                    # Apply photometric transformations to the image only
                    image_tile = self.transforms['photometric'](image=image_tile)['image']
        
                    # Convert back to float32
                    image_tile = image_tile.astype(np.float32) / 255.0

                elif self.mode == 'subset':
                    # Apply only resize for subsetting
                    augmented = self.transforms(image=image_tile, mask=mask_tile)
                    image_tile = augmented['image']
                    mask_tile = augmented['mask']
                else:  # val, test, base, or any other mode
                    # Apply resize and normalization to image, only resize to mask
                    image_tile = self.transforms['val_image'](image=image_tile)['image']
                    mask_tile = self.transforms['val_mask'](image=mask_tile)['image']

            # Ensure mask is binary
            mask_tile = np.where(mask_tile > 0, 1, 0).astype(np.float32)

            # Convert to tensor if not already
            if not isinstance(image_tile, torch.Tensor):
                image_tile = torch.from_numpy(image_tile).float()
            if not isinstance(mask_tile, torch.Tensor):
                mask_tile = torch.from_numpy(mask_tile).float()

            # Ensure correct shapes
            image_tile = image_tile.permute(2, 0, 1) if image_tile.ndim == 3 else image_tile
            mask_tile = mask_tile.unsqueeze(0) if mask_tile.ndim == 2 else mask_tile

            return image_tile, mask_tile

        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            return None

    def __del__(self):
        for src in self.image_srcs + self.mask_srcs:
            src.close()

class CurriculumDataset(Dataset):
    def __init__(self, dataset, positive_indices, negative_indices, num_epochs, mode='train', config=None):
        self.dataset = dataset
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.mode = mode
        self.config = config
        self.update_epoch(0)

    def update_epoch(self, epoch):
        self.current_epoch = epoch
        num_positives = len(self.positive_indices)
        num_negatives = 0  # Initialize num_negatives to 0
    
        if self.config.use_negative_samples:
            if self.config.use_curriculum:
                if self.config.use_staged_curriculum:
                    stage = min(epoch // self.config.epochs_per_stage, self.config.curriculum_stages - 1)
                    progress = stage / (self.config.curriculum_stages - 1)
                else:
                    progress = epoch / (self.num_epochs - 1) if self.num_epochs > 1 else 1
    
                if self.mode == 'train':
                    ratio = self.config.curriculum_start_ratio + (self.config.curriculum_end_ratio - self.config.curriculum_start_ratio) * progress
                else:  # validation
                    ratio = self.config.curriculum_start_ratio + (self.config.curriculum_end_ratio - self.config.curriculum_start_ratio) * progress * self.config.curriculum_rate
            else:
                ratio = self.config.max_negative_ratio
    
            num_negatives = int(ratio * num_positives)
            num_negatives = min(num_negatives, len(self.negative_indices))
            self.indices = self.positive_indices + random.sample(self.negative_indices, num_negatives)
        else:
            self.indices = self.positive_indices
    
        random.shuffle(self.indices)
        
        if num_positives > 0:
            ratio = num_negatives / num_positives
        else:
            ratio = 0
        
        logger.info(f"Epoch {epoch+1} ({self.mode.capitalize()}): {num_positives} positive, {num_negatives} negative samples (ratio: {ratio:.2f})")

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=None):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # BCE Loss
        bce = self.bce_loss(inputs, targets)

        # Dice Loss
        inputs = torch.sigmoid(inputs)  # Apply sigmoid since BCEWithLogits expects raw logits
        smooth = 1.0  # Smoothing constant to prevent division by zero
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = 1 - ((2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth))

        # Combined loss
        combined_loss = (self.bce_weight * bce) + (self.dice_weight * dice)
        return combined_loss

def get_image_mask_pairs(config):
    """
    Find image-mask pairs based on naming convention.
    
    Image files: "Makassar_Tallo_June2024_image.tif"
    Mask files: "Makassar_Tallo_June2024_682_mask_tires.tif" (with tire count)
    
    The function matches images to masks by finding files that:
    1. Start with the same base name
    2. Contain the mask pattern (e.g., contain "_mask_tires.tif")
    """
    image_files = [f for f in os.listdir(config.image_folder) if f.endswith(config.image_suffix)]
    pairs = []
    
    for image_file in image_files:
        # Clean up corrupted file names by extracting just the filename part
        # Handle cases where full Windows paths got copied
        clean_image_file = os.path.basename(image_file)
        if '\\' in clean_image_file:
            # Extract just the filename from a corrupted path
            clean_image_file = clean_image_file.split('\\')[-1]
            logger.info(f"Cleaned corrupted image filename: {image_file} -> {clean_image_file}")
        
        base_name = clean_image_file[:-len(config.image_suffix)]
        # Look for mask files that start with base_name and match the exact mask suffix
        # This handles patterns like: "Makassar_Tallo_June2024_682_mask_tires.tif" or "Makassar_Tallo_June2024_998_mask_tires.tif"
        expected_mask_name = base_name + config.mask_suffix
        mask_files = [f for f in os.listdir(config.mask_folder) 
                     if f == expected_mask_name]
        
        if mask_files:
            # Use the exact matching mask file
            mask_file = mask_files[0]
            mask_path = os.path.join(config.mask_folder, mask_file)
            pairs.append((
                os.path.join(config.image_folder, image_file),
                mask_path
            ))
            logger.info(f"Matched: {clean_image_file} -> {mask_file}")
        else:
            logger.warning(f"No matching mask found for {clean_image_file}")
            # Debug: show what files exist in mask folder
            all_mask_files = [f for f in os.listdir(config.mask_folder) if f.endswith('.tif')]
            logger.warning(f"Available mask files in {config.mask_folder}: {all_mask_files}")
    
    return pairs

def get_transforms(config):
    spatial_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Resize(config.tile_size, config.tile_size, always_apply=True),
    ])

    photometric_transforms = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])

    val_image_transform = A.Compose([
        A.Resize(config.tile_size, config.tile_size, always_apply=True),
    ])

    val_mask_transform = A.Resize(config.tile_size, config.tile_size, always_apply=True)

    return {
        'spatial': spatial_transforms,
        'photometric': photometric_transforms,
        'val_image': val_image_transform,
        'val_mask': val_mask_transform
    }

def calculate_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Handle cases where only one class is present
    if len(np.unique(y_true)) == 1:
        accuracy = accuracy_score(y_true, y_pred)
        if y_true[0] == 1:  # All positive
            precision = accuracy
            recall = 1.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mcc = 0  # MCC is undefined when there's only one class
        else:  # All negative
            precision = 1.0
            recall = accuracy
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mcc = 0
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
    
    return accuracy, precision, recall, f1, mcc

def evaluate_model(model, val_loader, criterion, config, pbar=None):
    model.eval()
    val_loss = 0.0
    total_samples = 0
    confusion_mat = np.zeros((2, 2), dtype=np.int64)
    
    try:
        with torch.no_grad():
            for images, masks in val_loader:
                if images is None:
                    continue
                
                # Skip batches with insufficient samples for BatchNorm (DeepLabV3+ requirement)
                if images.size(0) < 2:
                    logger.warning(f"Skipping validation batch with size {images.size(0)} - insufficient for BatchNorm")
                    continue

                images = images.to(device).float()
                masks = masks.to(device).float()

                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                preds = (outputs > config.val_threshold).float()
                
                # Update confusion matrix
                batch_confusion = confusion_matrix(masks.cpu().numpy().flatten(), 
                                                   preds.cpu().numpy().flatten(),
                                                   labels=[0, 1])
                confusion_mat += batch_confusion
                
                total_samples += images.size(0)

                if pbar is not None:
                    pbar.update(1)

        val_loss /= total_samples
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = confusion_mat.ravel()
        
        # Calculate metrics
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # MCC calculation
        mcc_num = tp * tn - fp * fn
        mcc_den = np.sqrt(tp + fp) * np.sqrt(tp + fn) * np.sqrt(tn + fp) * np.sqrt(tn + fn)
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0

        return val_loss, accuracy, precision, recall, f1, mcc

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning("WARNING: ran out of memory during validation")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.error(f"RuntimeError during validation: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        raise e

def train_model(model, config, train_loader, val_loader, criterion, optimizer, scheduler):
    epoch_train_losses = []
    epoch_train_f1_scores = []
    epoch_train_mcc_scores = []
    epoch_train_accuracies = []
    epoch_train_precisions = []
    epoch_train_recalls = []
    epoch_val_losses = []
    epoch_val_f1_scores = []
    epoch_val_mcc_scores = []
    epoch_val_accuracies = []
    epoch_val_precisions = []
    epoch_val_recalls = []
    
    best_mcc = -1.0
    best_model_state = None
    best_loss = float('inf')
    best_epoch = -1
    
    # Calculate class weight from training data only if requested
    if hasattr(config, 'calculate_from_training_data') and config.calculate_from_training_data:
        logger.info("Calculating class weight from training data only...")
        train_positive_pixels = 0
        train_negative_pixels = 0
        
        for images, masks in train_loader:
            if images is None or masks is None:
                continue
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            
            # Count pixels in this batch
            valid_mask = masks != 255  # Exclude nodata pixels
            batch_positive = torch.sum((masks == 1) & valid_mask).item()
            batch_negative = torch.sum((masks == 0) & valid_mask).item()
            
            train_positive_pixels += batch_positive
            train_negative_pixels += batch_negative
        
        if train_positive_pixels > 0:
            train_pos_weight = train_negative_pixels / train_positive_pixels
            logger.info(f"Training data pixel analysis:")
            logger.info(f"  Training positive pixels: {train_positive_pixels:,}")
            logger.info(f"  Training negative pixels: {train_negative_pixels:,}")
            logger.info(f"  Training positive ratio: {train_positive_pixels/(train_positive_pixels + train_negative_pixels):.4f}")
            logger.info(f"  Training pos_weight: {train_pos_weight:.2f}")
            
            # Update the criterion with the new weight
            if hasattr(criterion, 'bce_loss') and hasattr(criterion.bce_loss, 'pos_weight'):
                criterion.bce_loss.pos_weight = torch.tensor([train_pos_weight]).to(device)
                logger.info(f"Updated criterion with training data pos_weight: {train_pos_weight:.2f}")
        else:
            logger.warning("No positive pixels found in training data!")

    try:
        for epoch in range(config.num_epochs):
            if isinstance(train_loader.dataset, CurriculumDataset):
                train_loader.dataset.update_epoch(epoch)
            if isinstance(val_loader.dataset, CurriculumDataset):
                val_loader.dataset.update_epoch(epoch)
            
            model.train()
            running_loss = 0.0
            running_f1 = 0.0
            running_mcc = 0.0
            running_accuracy = 0.0
            running_precision = 0.0
            running_recall = 0.0
            total_batches = len(train_loader)

            for images, masks in train_loader:
                if images is None or masks is None:
                    continue
                
                # Skip batches with insufficient samples for BatchNorm (DeepLabV3+ requirement)
                if images.size(0) < 2:
                    logger.warning(f"Skipping batch with size {images.size(0)} - insufficient for BatchNorm")
                    continue
    
                images = images.float().to(device)
                masks = masks.float().to(device)
    
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
    
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item() * images.size(0)
                preds = (outputs > config.train_threshold).float()
                accuracy, precision, recall, f1, mcc = calculate_metrics(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
                running_f1 += f1
                running_mcc += mcc
                running_accuracy += accuracy
                running_precision += precision
                running_recall += recall
    
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_f1 = running_f1 / total_batches
            epoch_mcc = running_mcc / total_batches
            epoch_accuracy = running_accuracy / total_batches
            epoch_precision = running_precision / total_batches
            epoch_recall = running_recall / total_batches
    
            epoch_train_losses.append(epoch_loss)
            epoch_train_f1_scores.append(epoch_f1)
            epoch_train_mcc_scores.append(epoch_mcc)
            epoch_train_accuracies.append(epoch_accuracy)
            epoch_train_precisions.append(epoch_precision)
            epoch_train_recalls.append(epoch_recall)
    
            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_mcc = evaluate_model(model, val_loader, criterion, config)
    
            epoch_val_losses.append(val_loss)
            epoch_val_f1_scores.append(val_f1)
            epoch_val_mcc_scores.append(val_mcc)
            epoch_val_accuracies.append(val_accuracy)
            epoch_val_precisions.append(val_precision)
            epoch_val_recalls.append(val_recall)
    
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
    
            if val_mcc > best_mcc or (val_mcc == best_mcc and val_loss < best_loss):
                best_mcc = val_mcc
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1  # +1 because epoch is 0-indexed
                logger.info(f"New best model at epoch {best_epoch} with MCC: {best_mcc:.6f} and loss: {best_loss:.6f}")
    
            logger.info(f"Epoch {epoch+1}/{config.num_epochs} - "
                       f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Train F1: {epoch_f1:.4f}, Val F1: {val_f1:.4f}, "
                       f"Train MCC: {epoch_mcc:.4f}, Val MCC: {val_mcc:.4f}, "
                       f"Train Precision: {epoch_precision:.4f}, Val Precision: {val_precision:.4f}, "
                       f"Train Recall: {epoch_recall:.4f}, Val Recall: {val_recall:.4f}, "
                       f"LR: {current_lr:.6f}")
        
        return (epoch_train_losses, epoch_train_f1_scores, epoch_train_mcc_scores, epoch_train_accuracies,
                epoch_train_precisions, epoch_train_recalls,
                epoch_val_losses, epoch_val_f1_scores, epoch_val_mcc_scores, epoch_val_accuracies,
                epoch_val_precisions, epoch_val_recalls,
                best_mcc, best_model_state, best_epoch)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current state...")
        # Extract model name for consistent naming
        model_name = config.model_def.split('(')[0] if '(' in config.model_def else 'cnn'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, f'{config.base_path_model}/{config.model_iteration}/{model_name.lower()}_interrupted_checkpoint.pth')
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise e

def plot_cv_results(train_losses, train_f1_scores, train_mcc_scores, train_accuracies,
                    train_precisions, train_recalls,
                    val_losses, val_f1_scores, val_mcc_scores, val_accuracies,
                    val_precisions, val_recalls,
                    num_folds, num_epochs, config):
    
    # Reshape the data
    train_losses = np.array(train_losses).reshape(num_folds, num_epochs)
    val_losses = np.array(val_losses).reshape(num_folds, num_epochs)
    train_f1_scores = np.array(train_f1_scores).reshape(num_folds, num_epochs)
    val_f1_scores = np.array(val_f1_scores).reshape(num_folds, num_epochs)
    train_mcc_scores = np.array(train_mcc_scores).reshape(num_folds, num_epochs)
    val_mcc_scores = np.array(val_mcc_scores).reshape(num_folds, num_epochs)
    train_accuracies = np.array(train_accuracies).reshape(num_folds, num_epochs)
    val_accuracies = np.array(val_accuracies).reshape(num_folds, num_epochs)
    train_precisions = np.array(train_precisions).reshape(num_folds, num_epochs)
    val_precisions = np.array(val_precisions).reshape(num_folds, num_epochs)
    train_recalls = np.array(train_recalls).reshape(num_folds, num_epochs)
    val_recalls = np.array(val_recalls).reshape(num_folds, num_epochs)

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 30))

    # Loss plot
    plt.subplot(6, 1, 1)
    plot_metric(epochs, train_losses, val_losses, "Loss")

    # F1 Score plot
    plt.subplot(6, 1, 2)
    plot_metric(epochs, train_f1_scores, val_f1_scores, "F1 Score")

    # MCC plot
    plt.subplot(6, 1, 3)
    plot_metric(epochs, train_mcc_scores, val_mcc_scores, "MCC")

    # Accuracy plot
    plt.subplot(6, 1, 4)
    plot_metric(epochs, train_accuracies, val_accuracies, "Accuracy")

    # Precision plot
    plt.subplot(6, 1, 5)
    plot_metric(epochs, train_precisions, val_precisions, "Precision")

    # Recall plot
    plt.subplot(6, 1, 6)
    plot_metric(epochs, train_recalls, val_recalls, "Recall")

    plt.tight_layout()
    plt.savefig(f'{config.base_path_model}/{config.model_iteration}/cv_results.png')
    plt.close()

def plot_metric(epochs, train_metric, val_metric, metric_name):
    train_mean = np.mean(train_metric, axis=0)
    train_std = np.std(train_metric, axis=0)
    val_mean = np.mean(val_metric, axis=0)
    val_std = np.std(val_metric, axis=0)

    plt.plot(epochs, train_mean, 'b-', label=f'Training {metric_name}')
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='b')
    plt.plot(epochs, val_mean, 'r-', label=f'Validation {metric_name}')
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='r')

    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        # Return a minimal valid batch to avoid BatchNorm issues
        # This should rarely happen with proper data filtering
        logger.warning("Empty batch detected in custom_collate - this may cause issues")
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def stratified_k_fold_cross_validation(config, full_dataset, transforms):
    """Performs stratified k-fold cross-validation."""
    # Create labels for stratification and track pixel counts for class weights
    all_indices = np.arange(len(full_dataset))
    
    labels = []
    total_pixels = 0
    positive_pixels = 0
    
    for i in tqdm(all_indices, desc="Generating Stratification Labels"):
        try:
            _, mask = full_dataset[i]
            
            # Convert to numpy if it's a tensor
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = mask
            
            # Count pixels (excluding nodata pixels, which are 255 in this dataset)
            valid_mask = mask_np != 255  # Exclude nodata pixels
            valid_pixels = np.sum(valid_mask)
            tire_pixels = np.sum((mask_np == 1) & valid_mask)
            
            total_pixels += valid_pixels
            positive_pixels += tire_pixels
            
            if torch.sum(mask == 1) > 0:
                labels.append(1)  # Trash present
            else:
                labels.append(0)  # No trash
        except Exception as e:
            logger.error(f"Error processing item {i}: {str(e)}")
            labels.append(0)  # Treat as no trash

    labels = np.array(labels)
    
    # Calculate and log class weights
    if positive_pixels > 0:
        calculated_pos_weight = (total_pixels - positive_pixels) / positive_pixels
        logger.info(f"Dataset pixel analysis:")
        logger.info(f"  Total valid pixels: {total_pixels:,}")
        logger.info(f"  Positive (tire) pixels: {positive_pixels:,}")
        logger.info(f"  Negative (background) pixels: {total_pixels - positive_pixels:,}")
        logger.info(f"  Positive pixel ratio: {positive_pixels/total_pixels:.4f}")
        logger.info(f"  Calculated pos_weight: {calculated_pos_weight:.2f}")
        
        # Store the calculated weight globally for use in loss function
        global calculated_class_weight
        calculated_class_weight = calculated_pos_weight
    else:
        logger.warning("No positive pixels found in dataset!")
        calculated_class_weight = 20.0
    
    # Create loss function now that class weights are calculated
    if hasattr(config, 'use_calculated_class_weight') and config.use_calculated_class_weight:
        # Check if we should calculate from training data only
        if hasattr(config, 'calculate_from_training_data') and config.calculate_from_training_data:
            logger.info("Will calculate class weight from training data only (not full dataset)")
            # We'll calculate this later in the training loop
            pos_weight = torch.tensor([20.0]).to(device)  # Temporary fallback
        else:
            # Use the class weight calculated during stratification
            logger.info("Using calculated class weight from dataset analysis...")
            pos_weight = torch.tensor([calculated_class_weight]).to(device)
            logger.info(f"Using calculated pos_weight: {pos_weight.item():.2f}")
    else:
        # Use hardcoded class weight
        pos_weight = torch.tensor([20.0]).to(device)
        logger.info("Using hardcoded pos_weight: 20.0")
    
    # Get loss function weights from config
    bce_weight = getattr(config, 'bce_weight', 1.0)
    dice_weight = getattr(config, 'dice_weight', 1.0)
    
    global criterion
    criterion = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight, pos_weight=pos_weight).to(device)
    
    # Split into train+val and hold-out sets
    train_val_indices, holdout_indices, train_val_labels, holdout_labels = train_test_split(
        all_indices, labels, test_size=config.holdout_percentage, stratify=labels, random_state=config.random_seed
    )

    # Use StratifiedKFold
    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_seed)
                          
    best_mcc_overall = -1.0
    best_model_state_overall = None
    best_fold = -1
    best_epoch = -1

    # Initialize lists to store metrics across all folds
    all_train_losses, all_train_f1_scores, all_train_mcc_scores, all_train_accuracies = [], [], [], []
    all_train_precisions, all_train_recalls = [], []
    all_val_losses, all_val_f1_scores, all_val_mcc_scores, all_val_accuracies = [], [], [], []
    all_val_precisions, all_val_recalls = [], []

    # Create a single progress bar for all folds
    for fold, (train_index, val_index) in enumerate(skf.split(train_val_indices, train_val_labels)):
        logger.info(f"\n--- Fold {fold + 1}/{config.k_folds} ---")

        train_indices = train_val_indices[train_index]
        val_indices = train_val_indices[val_index]

        train_dataset = DroneTrashDataset(config, transforms, mode='train', indices=train_indices)
        val_dataset = DroneTrashDataset(config, transforms, mode='val', indices=val_indices)

        # Identify positive and negative samples
        train_pos = [i for i, idx in enumerate(train_indices) if labels[idx] == 1]
        train_neg = [i for i, idx in enumerate(train_indices) if labels[idx] == 0]
        val_pos = [i for i, idx in enumerate(val_indices) if labels[idx] == 1]
        val_neg = [i for i, idx in enumerate(val_indices) if labels[idx] == 0]

        train_curriculum = CurriculumDataset(train_dataset, train_pos, train_neg, config.num_epochs, mode='train', config=config)
        val_curriculum = CurriculumDataset(val_dataset, val_pos, val_neg, config.num_epochs, mode='val', config=config)

        train_loader = DataLoader(train_curriculum, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_curriculum, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate)

        # Reset model
        model.load_state_dict(torch.load(config.initial_model_weights))

        # Reset optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay)

        # Reset scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Log data distribution
        logger.info(f"Train set size: {len(train_indices)}, Validation set size: {len(val_indices)}")
        logger.info(f"Positive samples in train: {len(train_pos)}, in validation: {len(val_pos)}")

        (fold_train_losses, fold_train_f1_scores, fold_train_mcc_scores, fold_train_accuracies,
         fold_train_precisions, fold_train_recalls,
         fold_val_losses, fold_val_f1_scores, fold_val_mcc_scores, fold_val_accuracies,
         fold_val_precisions, fold_val_recalls,
         fold_best_mcc, fold_best_model_state, fold_best_epoch) = train_model(model, config, train_loader, val_loader, criterion, optimizer, scheduler)

        all_train_losses.extend(fold_train_losses)
        all_train_f1_scores.extend(fold_train_f1_scores)
        all_train_mcc_scores.extend(fold_train_mcc_scores)
        all_train_accuracies.extend(fold_train_accuracies)
        all_train_precisions.extend(fold_train_precisions)
        all_train_recalls.extend(fold_train_recalls)
        all_val_losses.extend(fold_val_losses)
        all_val_f1_scores.extend(fold_val_f1_scores)
        all_val_mcc_scores.extend(fold_val_mcc_scores)
        all_val_accuracies.extend(fold_val_accuracies)
        all_val_precisions.extend(fold_val_precisions)
        all_val_recalls.extend(fold_val_recalls)

        if fold_best_mcc > best_mcc_overall:
            best_mcc_overall = fold_best_mcc
            best_model_state_overall = fold_best_model_state
            best_fold = fold + 1  # +1 because fold is 0-indexed
            best_epoch = fold_best_epoch
            logger.info(f"New best overall MCC: {best_mcc_overall:.6f} achieved in fold {best_fold} at epoch {best_epoch}. Saving model...")
            # Extract model name for consistent naming
            model_name = config.model_def.split('(')[0] if '(' in config.model_def else 'cnn'
            torch.save(best_model_state_overall, f'{config.base_path_model}/{config.model_iteration}/{model_name.lower()}_trash_detection_best.pth')

    # Plot the performance metrics across all epochs and folds
    plot_cv_results(all_train_losses, all_train_f1_scores, all_train_mcc_scores, all_train_accuracies,
                    all_train_precisions, all_train_recalls,
                    all_val_losses, all_val_f1_scores, all_val_mcc_scores, all_val_accuracies,
                    all_val_precisions, all_val_recalls,
                    config.k_folds, config.num_epochs, config)
    
    # Print final summary
    logger.info(f"\nTraining completed. Best model performance:")
    logger.info(f"Best MCC: {best_mcc_overall:.6f}")
    logger.info(f"Achieved in fold {best_fold} at epoch {best_epoch}")

    # After k-fold cross-validation, evaluate on hold-out set
    holdout_dataset = DroneTrashDataset(config, transforms, mode='test', indices=holdout_indices)
    holdout_loader = DataLoader(holdout_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate)

    # Load the best model
    best_model = eval(config.model_def).to(device)
    # Extract model name for consistent naming
    model_name = config.model_def.split('(')[0] if '(' in config.model_def else 'cnn'
    best_model.load_state_dict(torch.load(f'{config.base_path_model}/{config.model_iteration}/{model_name.lower()}_trash_detection_best.pth'))

    # Evaluate on hold-out set
    holdout_loss, holdout_accuracy, holdout_precision, holdout_recall, holdout_f1, holdout_mcc = evaluate_model(best_model, holdout_loader, criterion, config)

    logger.info(f"\nHold-out Set Performance:")
    logger.info(f"Loss: {holdout_loss:.4f}")
    logger.info(f"Accuracy: {holdout_accuracy:.4f}")
    logger.info(f"Precision: {holdout_precision:.4f}")
    logger.info(f"Recall: {holdout_recall:.4f}")
    logger.info(f"F1 Score: {holdout_f1:.4f}")
    logger.info(f"MCC: {holdout_mcc:.4f}")

    return best_model_state_overall, best_mcc_overall, best_fold, best_epoch, (holdout_loss, holdout_f1, holdout_mcc, holdout_accuracy)

def main():
    parser = argparse.ArgumentParser(description='Run segmentation model k-fold cross validation')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data_path', type=str, help='Override base data path')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--k_folds', type=int, help='Override number of folds')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(args.config)
    
    # Override config with command line arguments if provided
    if args.data_path:
        config.base_path = args.data_path
        config.base_path_model = os.path.join(config.base_path, 'model_data')
        config.data_folder = os.path.join(config.base_path, 'data')
        config.image_folder = os.path.join(config.data_folder, 'images')
        config.mask_folder = os.path.join(config.data_folder, 'masks')
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.k_folds:
        config.k_folds = args.k_folds
    
    # Set random seed for reproducibility
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    global model
    model = eval(config.model_def).to(device)
    
    # Loss function will be created after class weights are calculated during stratification
    
    # Create directories
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.optimizer_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.scheduler_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    
    try:
        # Get transforms
        transforms = get_transforms(config)
        
        # Create the full dataset
        full_dataset = DroneTrashDataset(config, transforms, mode='base')
        logger.info(f"Found {len(full_dataset.image_mask_pairs)} image-mask pairs:")
        for i, (img, mask) in enumerate(full_dataset.image_mask_pairs):
            logger.info(f"Pair {i+1}: Image: {os.path.basename(img)}, Mask: {os.path.basename(mask)}")
        logger.info(f"Total tiles: {full_dataset.total_tiles}, Valid tiles: {len(full_dataset.indices)}")

        # Create directory for this model iteration
        Path(os.path.join(config.base_path_model, config.model_iteration)).mkdir(parents=True, exist_ok=True)

        # Save initial model weights
        torch.save(model.state_dict(), config.initial_model_weights)

        # Write model definition to file
        with open(config.model_definition_file, 'w') as f:
            f.write(config.model_def)

        # Save config to output directory
        config.save_config_to_output()

        # Start k-fold cross-validation and get hold-out set performance
        best_model_state, best_mcc, best_fold, best_epoch, holdout_metrics = stratified_k_fold_cross_validation(config, full_dataset, transforms)
        
        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        raise e

if __name__ == "__main__":
    main()
