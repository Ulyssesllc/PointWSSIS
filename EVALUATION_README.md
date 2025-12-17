# PointWSSIS Evaluation Framework - README

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://www.python.org/)

## 📋 Overview

This comprehensive evaluation framework provides tools to test and analyze the PointWSSIS instance segmentation pipeline across different weak supervision scenarios.

### Key Features

- ✅ **Automated Evaluation**: Test models with 1%, 2%, 5%, 10%, 20%, 30%, 50% labeled data
- ✅ **Multiple Metrics**: AP, AP50, AP75, APs, APm, APl
- ✅ **Weak Label Types**: Unlabeled (U), Image-level (I), Point (P), Box (B), Full (F)
- ✅ **Rich Visualizations**: 6 types of publication-ready plots
- ✅ **Export Formats**: JSON, CSV, LaTeX, PNG, PDF

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch torchvision matplotlib seaborn pyyaml numpy tqdm pycocotools

# Build detectron2 and AdelaiDet
cd detectron2 && python setup.py build develop --user
cd ../AdelaiDet && python setup.py build develop --user
```

### 2. Update Configuration

Edit `evaluation_config.yaml` and set your data path:

```yaml
dataset:
  data_root: "/path/to/your/coco"
```

### 3. Run Evaluation

#### Option A: Quick Script

```bash
bash run_evaluation.sh --data_root /path/to/coco
```

#### Option B: Manual

```bash
# Run evaluation
python evaluate_pipeline.py --data_root /path/to/coco

# Generate plots
python visualize_results.py --results evaluation_results/evaluation_results.json
```

#### Option C: Try Example

```bash
# Run with sample data (for testing)
python example_evaluation.py
```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `evaluate_pipeline.py` | Main evaluation script - computes all metrics |
| `visualize_results.py` | Generate 6 types of visualization plots |
| `evaluation_utils.py` | Utility functions for metrics and analysis |
| `evaluation_config.yaml` | Configuration for experiments |
| `run_evaluation.sh` | Automated bash script for full pipeline |
| `example_evaluation.py` | Quick example with sample data |
| `EVALUATION_GUIDE.md` | Comprehensive documentation |

## 📊 Output Files

After running evaluation, you'll get:

```
evaluation_results/
├── evaluation_results.json      # All metrics in JSON
├── summary_table.txt            # Quick summary table
├── evaluation_log.txt           # Detailed logs
├── results_export.csv           # CSV format results
├── results_table.tex            # LaTeX table
└── plots/
    ├── ap_trends.png            # Main AP comparison
    ├── multi_metric_comparison.png
    ├── improvement_analysis.png
    ├── label_efficiency.png
    ├── performance_heatmap.png
    └── comprehensive_dashboard.png
```

## 📈 Visualization Examples

The framework generates 6 types of plots:

1. **AP Trends**: Compare teacher vs student performance
2. **Multi-Metric Comparison**: 6-panel plot with all metrics
3. **Improvement Analysis**: Absolute and relative gains
4. **Label Efficiency**: Performance vs annotation cost
5. **Performance Heatmap**: All metrics across proportions
6. **Comprehensive Dashboard**: All-in-one overview

## 💻 Usage Examples

### Evaluate Specific Proportions

```bash
python evaluate_pipeline.py \
    --data_root /path/to/coco \
    --proportions 5p 10p 20p
```

### Generate Specific Plots

```bash
python visualize_results.py \
    --results evaluation_results.json \
    --plots trends dashboard
```

### Use Utility Functions

```python
from evaluation_utils import (
    load_json_results,
    format_metric_table,
    export_to_csv,
    compare_with_baseline
)

# Load and analyze
results = load_json_results('evaluation_results.json')
table = format_metric_table(results)
print(table)

# Export
export_to_csv(results, 'results.csv')
```

## 📖 Metrics Explained

| Metric | Description |
|--------|-------------|
| **AP** | Average Precision @ IoU 0.5:0.05:0.95 (primary metric) |
| **AP50** | AP at IoU threshold 0.50 |
| **AP75** | AP at IoU threshold 0.75 |
| **APs** | AP for small objects (area < 32²) |
| **APm** | AP for medium objects (32² < area < 96²) |
| **APl** | AP for large objects (area > 96²) |

## 🎯 Expected Results

Based on the PointWSSIS paper (CVPR 2023):

| Setup | Teacher AP | Student AP | Gain |
|-------|-----------|------------|------|
| 1% + Point | ~18-20 | 24.0 | +4-6 |
| 5% + Point | ~28-30 | 33.7 | +4-6 |
| 10% + Point | ~31-33 | 35.8 | +3-5 |
| 20% + Point | ~34-36 | 37.1 | +2-3 |
| 50% + Point | ~37-39 | 38.8 | +1-2 |

**Baseline**: Fully supervised (100% labels) = 39.7 AP

## 🔧 Configuration

The `evaluation_config.yaml` file controls:

- Dataset paths and splits
- Model configurations
- Experiment proportions
- Evaluation settings
- Visualization options
- Output formats

Edit this file to customize your evaluation setup.

## 🐛 Troubleshooting

### CUDA Out of Memory

Reduce batch size or use CPU:

```yaml
evaluation:
  batch_size: 1
  use_cuda: false
```

### Missing Weights

Verify paths in config:

```bash
ls training_dir/SOLOv2_R101_coco5p_teacher/model_final.pth
```

### Dataset Not Found

Set environment variable:

```bash
export DETECTRON2_DATASETS=/path/to/coco
```

### Import Errors

Reinstall packages:

```bash
cd detectron2 && python setup.py build develop --user
cd ../AdelaiDet && python setup.py build develop --user
```

## 📚 Documentation

- **Quick Start**: This README
- **Comprehensive Guide**: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- **Original Paper**: [arXiv:2303.15062](https://arxiv.org/abs/2303.15062)
- **Original Repo**: [github.com/clovaai/PointWSSIS](https://github.com/clovaai/PointWSSIS)

## 📝 Citation

If you use this evaluation framework, please cite:

```bibtex
@inproceedings{kim2023pointwssis,
  title={The Devil is in the Points: Weakly Semi-Supervised Instance Segmentation via Point-Guided Mask Representation},
  author={Kim, Beomyoung and Jeong, Joonhyun and Han, Dongyoon and Hwang, Sung Ju},
  booktitle={CVPR},
  year={2023}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Test your changes
2. Update documentation
3. Follow existing code style
4. Add examples for new features

## 📄 License

Apache License 2.0 (same as PointWSSIS)

---

## 🎓 Academic Use

This framework is designed for research and academic evaluation. It provides:

- **Reproducible Results**: Consistent evaluation across runs
- **Publication-Ready Figures**: High-quality plots for papers
- **LaTeX Export**: Easy integration with academic papers
- **Comprehensive Metrics**: All COCO evaluation metrics

## 🔍 Advanced Usage

### Custom Weak Label Types

Evaluate with different supervision types:

```python
experiments = {
    'box_10p': {
        'full_label_percent': 10,
        'weak_label_percent': 90,
        'weak_type': 'B',  # Box labels
        ...
    }
}
```

### Multi-Dataset Evaluation

Extend to other datasets:

```yaml
dataset:
  name: "BDD100K"
  num_classes: 8
```

### Batch Evaluation

Evaluate multiple checkpoints:

```bash
for ckpt in teacher student; do
    python evaluate_pipeline.py --model_type $ckpt
done
```

## 📞 Support

For questions and issues:

1. Check [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
2. Review logs in `evaluation_results/evaluation_log.txt`
3. Open an issue on GitHub
4. Contact the original authors

## ⭐ Acknowledgments

This evaluation framework builds upon:

- [PointWSSIS](https://github.com/clovaai/PointWSSIS) - Original implementation
- [Detectron2](https://github.com/facebookresearch/detectron2) - Detection framework
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) - Instance segmentation tools
- [COCO API](https://github.com/cocodataset/cocoapi) - Evaluation metrics

---

**Made with ❤️ for the Computer Vision research community**
