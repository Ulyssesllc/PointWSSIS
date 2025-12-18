# Quick Start Guide

Get PointWSSIS running in 5 minutes!

## Prerequisites

- Python 3.7+ with pip
- CUDA 10.2+ with nvcc
- COCO 2017 dataset downloaded

## Step-by-Step Setup

### 1. Install Dependencies (2 min)

```bash
cd PointWSSIS
pip install -r requirements.txt
pip install portalocker
```

### 2. Build C++ Extensions (5-10 min)

```bash
# Build detectron2
cd detectron2
python -m pip install -e .

# Build AdelaiDet (includes custom CUDA ops)
cd ../AdelaiDet
python setup.py build develop --user

# Verify builds
cd ..
python -c "from detectron2 import _C; from adet import _C; print('✓ Extensions built')"
```

### 3. Set Environment Variable

```bash
# For Kaggle:
export DETECTRON2_DATASETS=/kaggle/input/coco-2017-dataset

# For Colab:
export DETECTRON2_DATASETS=/content/data

# For local machine:
export DETECTRON2_DATASETS=/path/to/your/coco

# Make it permanent (optional):
echo 'export DETECTRON2_DATASETS=/path/to/your/coco' >> ~/.bashrc
```

### 4. Download Point Annotations (5 min)

```bash
# Quick download for 1%, 2%, 5% scenarios
python scripts/download_annotations.py 1 2 5

# Or download all scenarios (takes longer)
python scripts/download_annotations.py all

# Include pre-generated pseudo labels (optional, to skip training steps)
python scripts/download_annotations.py 1 5 --pseudo --refined
```

### 5. Verify Setup (30 sec)

```bash
python scripts/verify_setup.py --scenario 1p
```

Expected output:

```
============================================================
PointWSSIS Setup Verification
============================================================
Checking Python version...
  ✓ Python 3.x.x
Checking dependencies...
  ✓ torch
  ✓ detectron2
  ✓ adet
  ...
Checking C++ extensions...
  ✓ detectron2._C
  ✓ adet._C
============================================================
Result: 6/6 checks passed
============================================================
✓ All checks passed! Your setup is ready.
```

### 6. Run Training

```bash
# Train on COCO 5% scenario (recommended for testing)
bash scripts/coco_5p.sh

# Or other scenarios:
bash scripts/coco_1p.sh   # 1% strong + 99% weak
bash scripts/coco_10p.sh  # 10% strong + 90% weak
bash scripts/coco_20p.sh  # 20% strong + 80% weak
```

## What Each Script Does

The training scripts run a 4-step pipeline:

1. **Teacher Network** - Train on strong (fully-labeled) subset
2. **MaskRefineNet** - Train mask refinement network
3. **Pseudo Labels** - Generate labels for weak (point-labeled) subset
4. **Student Network** - Train final model on strong + pseudo labels

Full training takes several hours on GPU.

## Common Issues

### "ModuleNotFoundError: No module named 'adet._C'"

**Solution:** C++ extensions not built. Run:

```bash
cd AdelaiDet
python setup.py build develop --user
```

### "FileNotFoundError: instances_train2017_1p_s.json"

**Solution:** Point annotations not downloaded. Run:

```bash
python scripts/download_annotations.py 1
```

### "DETECTRON2_DATASETS not set"

**Solution:** Export the environment variable:

```bash
export DETECTRON2_DATASETS=/path/to/coco
```

## Directory Structure

After setup, your workspace should look like:

```
PointWSSIS/
├── AdelaiDet/              # Detection framework (modified)
├── detectron2/             # Facebook's detection library (modified)
├── MaskRefineNet/          # Mask refinement network
├── scripts/                # Training scripts
│   ├── coco_1p.sh
│   ├── coco_5p.sh
│   ├── download_annotations.py
│   └── verify_setup.py
├── SETUP_GUIDE.md         # Detailed setup instructions
├── FIXES_SUMMARY.md       # All fixes made
├── requirements.txt       # Python dependencies
└── README.md             # Project overview

$DETECTRON2_DATASETS/
└── coco/
    ├── train2017/         # Training images (118K images)
    ├── val2017/           # Validation images (5K images)
    └── annotations/
        ├── instances_train2017.json          # Standard COCO
        ├── instances_val2017.json            # Standard COCO
        ├── instances_train2017_1p_s.json     # 1% strong subset
        ├── instances_train2017_1p_w.json     # 1% weak subset
        ├── instances_train2017_5p_s.json     # 5% strong subset
        ├── instances_train2017_5p_w.json     # 5% weak subset
        └── ... (other percentages)
```

## Next Steps

1. **Monitor training:** Check `training_dir/` for logs and checkpoints
2. **Evaluate results:** Model outputs mask predictions on COCO val2017
3. **Adjust hyperparameters:** Edit shell scripts to modify learning rate, iterations, etc.
4. **Try different scenarios:** Compare 1% vs 5% vs 10% strong supervision

## Resources

- **Detailed Setup:** See [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Troubleshooting:** See [SETUP_GUIDE.md](SETUP_GUIDE.md) Common Issues section
- **What Was Fixed:** See [FIXES_SUMMARY.md](FIXES_SUMMARY.md)
- **Original Paper:** <https://arxiv.org/abs/2303.15062>
- **COCO Dataset:** <https://cocodataset.org/#download>

## Getting Help

If verification fails:

1. Check the error message carefully
2. Review [SETUP_GUIDE.md](SETUP_GUIDE.md) Common Issues section
3. Ensure CUDA and GCC versions are compatible
4. On Kaggle/Colab, you may need to restart runtime after installing packages

## Training Time Estimates (on V100 GPU)

- COCO 1%: ~3-4 hours total
- COCO 5%: ~8-10 hours total
- COCO 10%: ~15-20 hours total
- COCO 50%: ~48+ hours total

Results will be saved in `training_dir/SOLOv2_R101_coco*_*/`.
