# PointWSSIS Evaluation Framework

## Overview

This comprehensive evaluation framework tests the PointWSSIS instance segmentation pipeline across different weak supervision scenarios and generates detailed performance analysis.

## Features

✅ **Comprehensive Evaluation**: Test models with varying proportions of weak labels (1%, 2%, 5%, 10%, 20%, 30%, 50%)

✅ **Multiple Metrics**: Compute AP, AP50, AP75, APs, APm, APl for thorough performance assessment

✅ **Weak Label Types Support**:

- **U**: Unlabeled data
- **I**: Image-level labels
- **P**: Point labels (default)
- **B**: Box labels
- **F**: Full mask labels

✅ **Rich Visualizations**: Generate 6 types of performance comparison graphs

✅ **Export Formats**: JSON, CSV, LaTeX tables, and publication-ready plots

## Files Structure

```
PointWSSIS/
├── evaluate_pipeline.py           # Main evaluation script
├── visualize_results.py           # Visualization generation
├── evaluation_utils.py            # Utility functions
├── evaluation_config.yaml         # Configuration file
├── run_evaluation.sh              # Quick-start bash script
└── EVALUATION_GUIDE.md            # This file
```

## Installation

### Prerequisites

```bash
# Install required packages
pip install torch torchvision
pip install matplotlib seaborn
pip install pyyaml numpy tqdm
pip install pycocotools

# Install detectron2 and AdelaiDet (already in repo)
cd detectron2
python setup.py build develop --user

cd ../AdelaiDet
python setup.py build develop --user
```

## Quick Start

### 1. Update Configuration

Edit `evaluation_config.yaml`:

```yaml
dataset:
  data_root: "/path/to/your/coco"  # Update this!

experiments:
  coco_5p:
    teacher_weights: "training_dir/SOLOv2_R101_coco5p_teacher/model_final.pth"
    student_weights: "training_dir/SOLOv2_R101_coco5p_student/model_final.pth"
  # ... update paths for other proportions
```

### 2. Run Evaluation

#### Option A: Using the script (recommended)

```bash
bash run_evaluation.sh
```

#### Option B: Manual execution

```bash
# Run evaluation
python evaluate_pipeline.py \
    --data_root /path/to/coco \
    --training_dir training_dir \
    --output_dir evaluation_results

# Generate visualizations
python visualize_results.py \
    --results evaluation_results/evaluation_results.json \
    --output_dir evaluation_results/plots
```

### 3. View Results

Results are saved in `evaluation_results/`:

- `evaluation_results.json` - All metrics in JSON format
- `summary_table.txt` - Summary table
- `plots/` - All visualization plots
- `evaluation_log.txt` - Detailed logs

## Usage Guide

### Evaluate Specific Proportions

```bash
python evaluate_pipeline.py \
    --data_root /path/to/coco \
    --training_dir training_dir \
    --output_dir evaluation_results \
    --proportions 5p 10p 20p
```

### Generate Specific Plots

```bash
# Generate only AP trends and dashboard
python visualize_results.py \
    --results evaluation_results/evaluation_results.json \
    --plots trends dashboard
```

Available plot types:

- `trends` - AP trends across proportions
- `metrics` - Multi-metric comparison
- `improvement` - Performance improvement analysis
- `efficiency` - Label efficiency analysis
- `heatmap` - Performance heatmap
- `dashboard` - Comprehensive dashboard
- `all` - All plots (default)

### Using Custom Configuration

```bash
python evaluate_pipeline.py \
    --data_root /path/to/coco \
    --config my_custom_config.yaml \
    --output_dir my_results
```

## Output Description

### 1. Evaluation Results (`evaluation_results.json`)

```json
{
  "coco_5p": {
    "full_label_percent": 5,
    "weak_label_percent": 95,
    "weak_type": "P",
    "teacher_metrics": {
      "AP": 28.5,
      "AP50": 48.2,
      "AP75": 29.1,
      "APs": 12.3,
      "APm": 31.2,
      "APl": 42.1
    },
    "student_metrics": {
      "AP": 33.7,
      "AP50": 54.1,
      "AP75": 35.6,
      "APs": 16.8,
      "APm": 36.9,
      "APl": 48.3
    }
  },
  ...
}
```

### 2. Summary Table (`summary_table.txt`)

```
========================================================================================================================
Proportion      Full%    Weak%    Type   Teacher AP   Student AP   Improvement 
========================================================================================================================
coco_5p         5        95       P      28.50        33.70        +5.20       
coco_10p        10       90       P      31.20        35.80        +4.60       
coco_20p        20       80       P      34.50        37.10        +2.60       
...
========================================================================================================================
```

### 3. Visualization Plots

#### a) AP Trends (`ap_trends.png`)

Main performance comparison showing teacher vs student AP across different label proportions.

#### b) Multi-Metric Comparison (`multi_metric_comparison.png`)

6-panel plot showing AP, AP50, AP75, APs, APm, APl comparisons.

#### c) Improvement Analysis (`improvement_analysis.png`)

Bar charts showing absolute and relative performance gains.

#### d) Label Efficiency (`label_efficiency.png`)

Performance vs annotation cost analysis.

#### e) Performance Heatmap (`performance_heatmap.png`)

Heatmap visualization of all metrics across proportions.

#### f) Comprehensive Dashboard (`comprehensive_dashboard.png`)

All-in-one dashboard with 9 subplots for complete overview.

## Advanced Usage

### Using the Utility Module

```python
from evaluation_utils import (
    load_json_results,
    compute_metrics_statistics,
    format_metric_table,
    export_to_csv,
    generate_latex_table
)

# Load results
results = load_json_results('evaluation_results/evaluation_results.json')

# Compute statistics
stats = compute_metrics_statistics(results)
print(f"Mean AP: {stats['mean']['AP']:.2f}")

# Export to CSV
export_to_csv(results, 'results.csv')

# Generate LaTeX table
latex_table = generate_latex_table(results)
print(latex_table)
```

### Comparing with Baselines

```python
from evaluation_utils import compare_with_baseline

# Compare with fully supervised baseline
comparisons = compare_with_baseline(
    results, 
    baseline_ap=39.7,  # Fully supervised COCO
    baseline_name="Fully Supervised"
)

for prop, comp in comparisons.items():
    print(f"{prop}: {comp['pct_of_baseline']:.1f}% of baseline")
```

### Finding Best Configuration

```python
from evaluation_utils import find_best_proportion

# Find most efficient configuration
best_prop, best_data = find_best_proportion(
    results,
    metric='AP',
    criterion='efficiency'
)

print(f"Best proportion: {best_prop}")
print(f"AP: {best_data['student_metrics']['AP']:.2f}")
```

## Metrics Explanation

| Metric | Description | Range |
|--------|-------------|-------|
| **AP** | Average Precision across IoU 0.5:0.05:0.95 | 0-100 |
| **AP50** | AP at IoU threshold 0.50 | 0-100 |
| **AP75** | AP at IoU threshold 0.75 | 0-100 |
| **APs** | AP for small objects (area < 32²) | 0-100 |
| **APm** | AP for medium objects (32² < area < 96²) | 0-100 |
| **APl** | AP for large objects (area > 96²) | 0-100 |

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size in config or use CPU evaluation

```yaml
evaluation:
  batch_size: 1
  use_cuda: false
```

### Issue: Missing model weights

**Solution**: Verify paths in `evaluation_config.yaml` and ensure models are trained

```bash
ls -l training_dir/SOLOv2_R101_coco5p_teacher/model_final.pth
```

### Issue: Dataset not found

**Solution**: Set DETECTRON2_DATASETS environment variable

```bash
export DETECTRON2_DATASETS=/path/to/coco
```

### Issue: Import errors

**Solution**: Reinstall detectron2 and AdelaiDet

```bash
cd detectron2 && python setup.py build develop --user
cd ../AdelaiDet && python setup.py build develop --user
```

## Expected Results

Based on the paper, expected Student AP values:

| Proportion | Teacher AP | Student AP | Improvement |
|------------|-----------|------------|-------------|
| COCO 1%    | ~18-20    | ~24.0      | +4-6        |
| COCO 2%    | ~20-22    | ~25.3      | +3-5        |
| COCO 5%    | ~28-30    | ~33.7      | +4-6        |
| COCO 10%   | ~31-33    | ~35.8      | +3-5        |
| COCO 20%   | ~34-36    | ~37.1      | +2-3        |
| COCO 30%   | ~35-37    | ~38.0      | +2-3        |
| COCO 50%   | ~37-39    | ~38.8      | +1-2        |

## Citation

If you use this evaluation framework, please cite:

```bibtex
@inproceedings{kim2023pointwssis,
  title={The Devil is in the Points: Weakly Semi-Supervised Instance Segmentation via Point-Guided Mask Representation},
  author={Kim, Beomyoung and Jeong, Joonhyun and Han, Dongyoon and Hwang, Sung Ju},
  booktitle={CVPR},
  year={2023}
}
```

## Support

For issues and questions:

1. Check the [original repo](https://github.com/clovaai/PointWSSIS)
2. Review the evaluation logs in `evaluation_results/evaluation_log.txt`
3. Verify all paths and configurations

## License

This evaluation framework follows the same license as PointWSSIS (Apache-2.0).
