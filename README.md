# Aedes-Tire-Detection

Code for the manuscript:

> Hassan MM\*, Chamberlin AJ\*, Tarpenning MS, Weber WC, Dave Coombe K, De Leo GA, Junaid M, Soma AS, Ansariadi, Rosser JI. **Enhancing *Aedes aegypti* breeding site detection: A deep learning approach to tire identification in unmanned aerial vehicle imagery from urban Indonesia.** *Remote Sensing Applications: Society and Environment*, in press. (\*Co-first authors)

## Overview

This repository contains code to train and evaluate U-Net++ and DeepLabV3+ segmentation models for detecting discarded tires — a key *Aedes aegypti* breeding site — in UAV imagery, along with scripts for post-prediction spatial analysis.

**Study area:** Tallo sub-district, Makassar, Indonesia

## Repository Structure

```
scripts/
  segmentation_model_kfold.py   # Model training with stratified k-fold cross-validation
  predict.py                    # Run inference on a new orthomosaic
  prediction_analysis.py        # Object-based accuracy assessment (TP/FP/FN matching)
  model_evaluation.py           # Pixel-level evaluation metrics
  spatial_analysis_tires.py     # Spatial statistics and hotspot mapping of detections
notebooks/
  segmentation_model_kfold.ipynb  # Interactive notebook version of training pipeline
configs/
  config_example.json           # Template configuration for prediction
```

## Requirements

```bash
conda env create -f environment.yml
conda activate drone_dengue
```

The conda environment includes dependencies for both model training/prediction and spatial analysis. `requirements.txt` and `requirements_spatial_analysis.txt` are provided as pip-installable references for each component.

## Usage

### 1. Model training (k-fold cross-validation)

```bash
python scripts/segmentation_model_kfold.py --config path/to/your/config.json
```

The training script expects a JSON config file specifying data paths, model architecture, and hyperparameters. Key configurable parameters are documented at the top of the script.

### 2. Prediction on a new orthomosaic

Copy and edit `configs/config_example.json` with your paths and parameters, then:

```bash
python scripts/predict.py --config configs/your_config.json
```

Outputs a binary prediction GeoTIFF and vectorized detections (GeoPackage).

### 3. Accuracy assessment

```bash
python scripts/prediction_analysis.py \
  --image path/to/orthomosaic.tif \
  --ground-truth path/to/ground_truth_mask.tif \
  --prediction path/to/prediction_mask.tif \
  --output-dir path/to/output/
```

### 4. Spatial analysis

```bash
python scripts/spatial_analysis_tires.py \
  --prediction path/to/prediction_mask.tif \
  --image path/to/orthomosaic.tif \
  --output-dir path/to/output/
```

Produces KDE density maps, nearest-neighbor distance statistics, and Getis-Ord G* hotspot maps.

## Data Availability

UAV imagery and annotation masks are available upon reasonable request to the corresponding authors. Data sharing is subject to agreements with local collaborators in Makassar, Indonesia.

## License

See [LICENSE](LICENSE) for details.
