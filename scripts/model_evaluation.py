import numpy as np
import rasterio
from sklearn.metrics import (
    f1_score, 
    matthews_corrcoef, 
    precision_score, 
    recall_score, 
    accuracy_score,
    confusion_matrix
)
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, Any, Tuple

def load_and_prepare_masks(prediction_path: str, ground_truth_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare prediction and ground truth masks for evaluation."""
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
        nodata_pred = src.nodata if src.nodata is not None else 255
    
    with rasterio.open(ground_truth_path) as src:
        ground_truth = src.read(1)
        nodata_gt = src.nodata if src.nodata is not None else 255
    
    # Create valid data mask (where neither prediction nor ground truth is nodata)
    valid_mask = (prediction != nodata_pred) & (ground_truth != nodata_gt)
    
    # Extract only valid pixels
    prediction = prediction[valid_mask]
    ground_truth = ground_truth[valid_mask]
    
    # Convert to binary masks
    prediction = (prediction > 0.5).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)
    
    # Ensure masks have same shape
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Mask shapes do not match: prediction {prediction.shape} vs ground truth {ground_truth.shape}")
    
    logging.info(f"Total valid pixels: {len(prediction)}")
    logging.info(f"Ground truth positive pixels: {np.sum(ground_truth)}")
    logging.info(f"Predicted positive pixels: {np.sum(prediction)}")
    
    return prediction.flatten(), ground_truth.flatten()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive set of performance metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix - calculate in chunks to avoid memory issues
    chunk_size = 1000000  # Process 1M pixels at a time
    tn = fp = fn = tp = 0
    
    for i in range(0, len(y_true), chunk_size):
        chunk_true = y_true[i:i + chunk_size]
        chunk_pred = y_pred[i:i + chunk_size]
        chunk_cm = confusion_matrix(chunk_true, chunk_pred, labels=[0, 1]).ravel()
        tn_chunk, fp_chunk, fn_chunk, tp_chunk = chunk_cm
        tn += tn_chunk
        fp += fp_chunk
        fn += fn_chunk
        tp += tp_chunk
    
    # Additional metrics
    metrics['sensitivity'] = metrics['recall']  # Same as recall
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    # Raw counts
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    metrics['total_pixels'] = int(len(y_true))
    metrics['positive_pixels'] = int(np.sum(y_true))
    metrics['predicted_positive_pixels'] = int(np.sum(y_pred))
    
    # Calculate percentage of positive pixels
    metrics['ground_truth_positive_percentage'] = (metrics['positive_pixels'] / metrics['total_pixels']) * 100
    metrics['predicted_positive_percentage'] = (metrics['predicted_positive_pixels'] / metrics['total_pixels']) * 100
    
    return metrics

def create_confusion_matrix_plot(metrics: Dict[str, float], output_path: str):
    """Create and save confusion matrix visualization."""
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_metrics_plot(metrics: Dict[str, float], output_path: str):
    """Create and save performance metrics visualization."""
    # Select metrics to plot
    plot_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1'],
        'Specificity': metrics['specificity']
    }
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(plot_metrics.keys(), plot_metrics.values())
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_report(metrics: Dict[str, float], output_dir: Path) -> str:
    """Generate a comprehensive HTML report."""
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            .metric-table th, .metric-table td {{ 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left; 
            }}
            .metric-table th {{ background-color: #f5f5f5; }}
            .section {{ margin: 20px 0; }}
            .images {{ display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; }}
            .image-container {{ text-align: center; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        <p>Generated on: {report_time}</p>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            <table class="metric-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td>{metrics['accuracy']:.4f}</td>
                    <td>Overall prediction accuracy</td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>Positive predictive value</td>
                </tr>
                <tr>
                    <td>Recall (Sensitivity)</td>
                    <td>{metrics['recall']:.4f}</td>
                    <td>True positive rate</td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td>{metrics['f1']:.4f}</td>
                    <td>Harmonic mean of precision and recall</td>
                </tr>
                <tr>
                    <td>Specificity</td>
                    <td>{metrics['specificity']:.4f}</td>
                    <td>True negative rate</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Detailed Statistics</h2>
            <table class="metric-table">
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>True Positives</td>
                    <td>{metrics['true_positives']:,}</td>
                </tr>
                <tr>
                    <td>False Positives</td>
                    <td>{metrics['false_positives']:,}</td>
                </tr>
                <tr>
                    <td>True Negatives</td>
                    <td>{metrics['true_negatives']:,}</td>
                </tr>
                <tr>
                    <td>False Negatives</td>
                    <td>{metrics['false_negatives']:,}</td>
                </tr>
                <tr>
                    <td>Total Valid Pixels</td>
                    <td>{metrics['total_pixels']:,}</td>
                </tr>
                <tr>
                    <td>Ground Truth Positive Pixels</td>
                    <td>{metrics['positive_pixels']:,}</td>
                </tr>
                <tr>
                    <td>Ground Truth Positive Percentage</td>
                    <td>{metrics['ground_truth_positive_percentage']:.2f}%</td>
                </tr>
                <tr>
                    <td>Predicted Positive Pixels</td>
                    <td>{metrics['predicted_positive_pixels']:,}</td>
                </tr>
                <tr>
                    <td>Predicted Positive Percentage</td>
                    <td>{metrics['predicted_positive_percentage']:.2f}%</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Visualizations</h2>
            <div class="images">
                <div class="image-container">
                    <img src="confusion_matrix.png" alt="Confusion Matrix">
                    <p>Confusion Matrix</p>
                </div>
                <div class="image-container">
                    <img src="metrics_plot.png" alt="Performance Metrics">
                    <p>Performance Metrics</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    report_path = output_dir / "evaluation_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return str(report_path)

def evaluate_model(prediction_path: str, ground_truth_path: str, output_dir: str) -> Dict[str, Any]:
    """Main function to evaluate model performance and generate report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Loading and preparing masks...")
    y_pred, y_true = load_and_prepare_masks(prediction_path, ground_truth_path)
    
    logging.info("Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    
    logging.info("Creating visualizations...")
    create_confusion_matrix_plot(metrics, str(output_dir / "confusion_matrix.png"))
    create_metrics_plot(metrics, str(output_dir / "metrics_plot.png"))
    
    logging.info("Generating HTML report...")
    report_path = generate_report(metrics, output_dir)
    
    # Save metrics to JSON for programmatic access
    metrics_path = output_dir / "metrics.json"
    pd.Series(metrics).to_json(metrics_path)
    
    return {
        'metrics': metrics,
        'report_path': report_path,
        'metrics_json_path': str(metrics_path)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth")
    parser.add_argument("--prediction", required=True, help="Path to prediction mask")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth mask")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    
    args = parser.parse_args()
    
    results = evaluate_model(args.prediction, args.ground_truth, args.output_dir)
    print(f"Evaluation complete. Report saved to: {results['report_path']}") 