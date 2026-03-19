import rasterio
from rasterio.windows import Window
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import morphology
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import cv2
import geopandas as gpd
from shapely.geometry import Polygon
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import prediction_analysis as pa
import model_evaluation as me

class PredictionConfig:
    """Configuration class for prediction parameters."""
    def __init__(
        self,
        model_weights_path: str,
        model_definition: str = 'UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1, encoder_weights="imagenet")',
        tile_size: int = 256,
        batch_size: int = 16,
        val_threshold: float = 0.99,
        min_object_size: int = 100,
        process_percentage: float = 100.0,
        image_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        ground_truth_path: Optional[str] = None
    ):
        self.model_weights_path = model_weights_path
        self.model_definition = model_definition
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.val_threshold = val_threshold
        self.min_object_size = min_object_size
        self.process_percentage = process_percentage
        self.image_path = image_path
        self.output_dir = output_dir
        self.ground_truth_path = ground_truth_path

    @classmethod
    def from_json(cls, json_path: str) -> 'PredictionConfig':
        """Load configuration from a JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, json_path: str) -> None:
        """Save configuration to a JSON file."""
        config_dict = self.__dict__
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

class DroneImagePredictor:
    """Class for making predictions on drone imagery using a trained model."""
    
    def __init__(self, config: PredictionConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.transforms = self.get_transforms()

    def setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = self.output_dir / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Initializing predictor with config: {self.config.__dict__}")

    def load_model(self) -> torch.nn.Module:
        """Load the pre-trained model."""
        try:
            model = eval(self.config.model_definition)
            model.load_state_dict(torch.load(self.config.model_weights_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            logging.info(f"Model loaded successfully from {self.config.model_weights_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def get_transforms(self) -> A.Compose:
        """Get the transformation pipeline."""
        return A.Compose([
            A.Resize(self.config.tile_size, self.config.tile_size, always_apply=True),
            ToTensorV2()
        ])

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        image = image.astype(np.float32) / 255.0
        augmented = self.transforms(image=image)
        image_tensor = augmented["image"]

        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 3 and image_tensor.shape[0] != 3:
            image_tensor = image_tensor.permute(2, 0, 1)

        return image_tensor

    def pad_to_multiple(self, tensor: torch.Tensor, multiple: int = 32) -> Tuple[torch.Tensor, int, int]:
        """Pad tensor dimensions to be multiples of the given number."""
        height, width = tensor.shape[-2:]
        pad_h = (multiple - height % multiple) if height % multiple != 0 else 0
        pad_w = (multiple - width % multiple) if width % multiple != 0 else 0
        padding = [0, pad_w, 0, pad_h]
        return F.pad(tensor, padding, mode='constant', value=0), pad_h, pad_w

    def remove_small_objects(self, mask: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than min_size."""
        labeled, num_labels = morphology.label(mask, return_num=True)
        if num_labels == 1:
            if np.sum(mask) >= self.config.min_object_size:
                return mask
            return np.zeros_like(mask)
        return morphology.remove_small_objects(labeled, min_size=self.config.min_object_size).astype(bool)

    def predict_on_patch(self, patch: np.ndarray) -> np.ndarray:
        """Make prediction on a single patch."""
        patch_tensor = self.preprocess_image(patch).to(self.device)
        
        if patch_tensor.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, but got {patch_tensor.shape[0]}")

        padded_patch_tensor, pad_h, pad_w = self.pad_to_multiple(patch_tensor)
        padded_patch_tensor = padded_patch_tensor.unsqueeze(0)

        with torch.no_grad():
            try:
                output = self.model(padded_patch_tensor)
                output = output[:, :, :patch_tensor.shape[1], :patch_tensor.shape[2]]
                output = torch.sigmoid(output)
                pred_mask = (output > self.config.val_threshold).float().cpu().numpy()
                pred_mask = self.remove_small_objects(pred_mask[0, 0] > 0.5)
            except RuntimeError as e:
                logging.error(f"Error processing patch: {str(e)}")
                return np.zeros((self.config.tile_size, self.config.tile_size), dtype=np.float32)

        return pred_mask.astype(np.float32)

    def process_tile(self, image_tile: np.ndarray) -> np.ndarray:
        """Process a single tile of the image."""
        if image_tile.shape[0] == 3:
            image_tile = np.transpose(image_tile, (1, 2, 0))
        
        tile_height, tile_width, _ = image_tile.shape
        prediction_tile = np.zeros((tile_height, tile_width), dtype=np.float32)
        
        for i in range(0, tile_height, self.config.tile_size):
            for j in range(0, tile_width, self.config.tile_size):
                patch = image_tile[i:i+self.config.tile_size, j:j+self.config.tile_size, :]
                
                if patch.shape[:2] != (self.config.tile_size, self.config.tile_size):
                    padded_patch = np.zeros((self.config.tile_size, self.config.tile_size, 3), dtype=patch.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded_patch
                
                if np.all(patch == 0):
                    continue
                
                patch_prediction = self.predict_on_patch(patch)
                prediction_tile[i:i+min(self.config.tile_size, tile_height-i), 
                              j:j+min(self.config.tile_size, tile_width-j)] = patch_prediction[:min(self.config.tile_size, tile_height-i), 
                                                                                              :min(self.config.tile_size, tile_width-j)]
                del patch_prediction

        return prediction_tile

    def predict_image(self, image_path: str, output_name: Optional[str] = None) -> str:
        """Process the entire image and save predictions."""
        if output_name is None:
            output_name = Path(image_path).stem + "_prediction.tif"
        
        output_path = self.output_dir / output_name
        logging.info(f"Starting prediction on {image_path}")
        logging.info(f"Output will be saved to {output_path}")

        with rasterio.open(image_path) as src:
            profile = src.profile
            all_windows = list(src.block_windows(1))
            num_windows_to_process = int(len(all_windows) * (self.config.process_percentage / 100))
            
            valid_windows = []
            for idx, window in all_windows:
                alpha_band = src.read(4, window=window)
                if np.any(alpha_band == 255):
                    valid_windows.append((idx, window))
                
                if len(valid_windows) >= num_windows_to_process:
                    break
            
            windows_to_process = valid_windows[:num_windows_to_process]
            
            if self.config.process_percentage == 100:
                out_height, out_width = src.height, src.width
            else:
                last_window = windows_to_process[-1][1]
                out_height = last_window.row_off + last_window.height
                out_width = src.width
            
            profile.update(height=out_height, width=out_width, dtype=rasterio.float32, count=1, compress='lzw')
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                pbar = tqdm(total=len(windows_to_process), desc="Processing tiles", unit="tile")
                
                for _, window in windows_to_process:
                    image_tile = src.read([1, 2, 3, 4], window=window)
                    alpha_mask = image_tile[3] == 255
                    masked_tile = image_tile[:3] * alpha_mask
                    prediction_tile = self.process_tile(masked_tile)
                    prediction_tile *= alpha_mask
                    dst.write(prediction_tile, window=window, indexes=1)
                    pbar.update(1)
                
                pbar.close()

        logging.info(f"Prediction completed and saved to {output_path}")
        return str(output_path)

    def save_prediction_shp(self, prediction_path: str, output_name: Optional[str] = None) -> str:
        """Convert prediction raster to shapefile."""
        if output_name is None:
            output_name = Path(prediction_path).stem + ".shp"
        
        output_path = self.output_dir / output_name
        logging.info(f"Converting prediction to shapefile: {output_path}")

        with rasterio.open(prediction_path) as src:
            image = src.read(1)
            transform = src.transform
            profile = src.profile

        binary_mask = (image > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                coords = [transform * (x, y) for x, y in contour.squeeze()]
                poly = Polygon(coords)
                polygons.append(poly)
        
        gdf = gpd.GeoDataFrame({'geometry': polygons})
        gdf.crs = profile['crs']
        gdf.to_file(output_path)
        
        logging.info(f"Shapefile saved to {output_path}")
        return str(output_path)

    def analyze_prediction(self, prediction_path: str, ground_truth_path: str, original_image_path: str) -> Dict[str, Any]:
        """Analyze prediction against ground truth using prediction_analysis module."""
        logging.info("Starting prediction analysis")
        
        # Create analysis output directory
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Get base name for analysis outputs
        base_name = Path(prediction_path).stem
        
        try:
            # Generate evaluation report
            logging.info("Generating model evaluation report...")
            evaluation_dir = analysis_dir / "evaluation"
            evaluation_results = me.evaluate_model(
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
                output_dir=str(evaluation_dir)
            )
            
            # Run the visual analysis
            logging.info("Running visual analysis...")
            visual_analysis_results = pa.extreme_memory_process(
                image_path=original_image_path,  # Use original RGB image for visualization
                mask_paths={
                    'ground_truth': ground_truth_path,
                    'prediction': prediction_path
                },
                output_dir=str(analysis_dir / "visual_analysis"),
                base_name=base_name,
                chunk_size=1024,
                sigma=3.0,
                downscale_factor=8,
                quick_test=False,
                use_cache=True,
                save_cache=True
            )
            
            # Combine results
            results = {
                'metrics': evaluation_results['metrics'],
                'report_path': evaluation_results['report_path'],
                'visual_analysis': visual_analysis_results
            }
            
            logging.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            logging.error(f"Error during analysis: {str(e)}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Drone imagery prediction with trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--image_path", type=str, help="Path to input drone image (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Directory for output files (overrides config)")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth mask (optional)")
    parser.add_argument("--model-weights", type=str, help="Path to model weights (optional)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = PredictionConfig.from_json(args.config)

    # Override config values with command line arguments if provided
    if args.model_weights:
        config.model_weights_path = args.model_weights
    if args.image_path:
        config.image_path = args.image_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.ground_truth:
        config.ground_truth_path = args.ground_truth
    
    # Check required parameters
    if not hasattr(config, 'image_path') or not config.image_path:
        raise ValueError("image_path must be specified either in config.json or via --image_path")
    if not hasattr(config, 'output_dir') or not config.output_dir:
        raise ValueError("output_dir must be specified either in config.json or via --output_dir")
   
    # Initialize predictor
    predictor = DroneImagePredictor(config, config.output_dir)
    
    # Make prediction
    prediction_path = predictor.predict_image(config.image_path)
    
    # Save as shapefile
    shapefile_path = predictor.save_prediction_shp(prediction_path)
    logging.info(f"Shapefile saved to {shapefile_path}")
    
    # If ground truth is provided, analyze prediction
    if hasattr(config, 'ground_truth_path') and config.ground_truth_path:
        analysis_results = predictor.analyze_prediction(
            prediction_path=prediction_path,
            ground_truth_path=config.ground_truth_path,
            original_image_path=config.image_path  # Pass the original image path
        )
        
        # Save analysis results
        results_path = Path(config.output_dir) / "analysis_results.json"
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=4)
        
        logging.info(f"Analysis results saved to {results_path}")

if __name__ == "__main__":
    main() 