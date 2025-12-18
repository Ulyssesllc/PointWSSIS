# Kaggle Setup Guide for PointWSSIS

This guide provides specific instructions for running PointWSSIS on Kaggle.

## Prerequisites

1. **Kaggle Notebook with GPU**
   - Go to Kaggle and create a new notebook
   - Enable GPU accelerator (Settings → Accelerator → GPU T4 x2 or P100)
   - Enable Internet access if downloading annotations

2. **COCO 2017 Dataset**
   - Add the [COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) to your notebook
   - It will be available at `/kaggle/input/coco-2017-dataset/coco/`

## Setup Instructions

### 1. Clone Repository

```bash
cd /kaggle/working
git clone https://github.com/clovaai/PointWSSIS.git
cd PointWSSIS
```

### 2. Fix NumPy Compatibility

Kaggle's pre-installed packages may have NumPy 2.x which causes binary incompatibility. Downgrade:

```bash
pip install "numpy<2.0.0" --force-reinstall --no-deps
```

### 3. Install Dependencies

```bash
# Install PointWSSIS requirements
pip install -r requirements.txt

# Install detectron2
python -m pip install -e detectron2

# Install AdelaiDet
python -m pip install -e AdelaiDet
```

### 4. Set Environment Variables

In Kaggle, set the data directory to point to the COCO dataset:

```bash
export DETECTRON2_DATASETS="/kaggle/input/coco-2017-dataset/coco"
```

**Important**: This points to the read-only input directory for reading images, but annotations will be downloaded to `/kaggle/working/`.

### 5. Download PointWSSIS Annotations

The annotation files need to be in a writable location. Download them to `/kaggle/working/`:

```bash
# Download annotations (they will automatically go to /kaggle/working/coco/annotations/)
python scripts/download_annotations.py all --output-dir /kaggle/working/coco/annotations
```

Or download specific percentages:

```bash
python scripts/download_annotations.py 1 2 5 --output-dir /kaggle/working/coco/annotations
```

### 6. Update Configuration Files

Since annotations are in `/kaggle/working/` but images are in `/kaggle/input/`, you need to update the configs:

**For training scripts**, create symbolic links or modify the annotation paths:

```bash
# Option 1: Create a symbolic link structure
mkdir -p /kaggle/working/coco
ln -s /kaggle/input/coco-2017-dataset/coco/train2017 /kaggle/working/coco/train2017
ln -s /kaggle/input/coco-2017-dataset/coco/val2017 /kaggle/working/coco/val2017
# Annotations are already at /kaggle/working/coco/annotations/

# Then use this as DETECTRON2_DATASETS
export DETECTRON2_DATASETS="/kaggle/working/coco"
```

**Or Option 2**: Modify the training scripts to specify separate paths for images and annotations.

### 7. Verify Setup

```bash
python scripts/verify_setup.py
```

## Training Examples

### Stage 1: Train Point-supervised Model

```bash
cd AdelaiDet
export DETECTRON2_DATASETS="/kaggle/working/coco"

# Train with 1% annotations
bash ../scripts/coco_1p.sh
```

### Stage 2: Generate Pseudo Labels

```bash
cd MaskRefineNet

# Make sure to update paths in the script if needed
python main.py \
    --data_root /kaggle/working/coco \
    --percent 1
```

### Stage 3: Merge and Train Final Model

```bash
cd MaskRefineNet
python merge_strong_and_refined_weak_labels.py \
    --data_root /kaggle/working/coco \
    --percent 1

cd ../AdelaiDet
# Train final model with merged labels
# (use appropriate training script)
```

## Common Issues and Solutions

### Issue 1: Read-only file system error

**Error**: `OSError: [Errno 30] Read-only file system: '/kaggle/input/...'`

**Solution**: This happens when trying to write to `/kaggle/input/`. Always write outputs to `/kaggle/working/`:

- Annotations: `/kaggle/working/coco/annotations/`
- Model outputs: `/kaggle/working/output/`
- Logs: `/kaggle/working/logs/`

### Issue 2: NumPy binary incompatibility

**Error**: `ValueError: numpy.dtype size changed, may indicate binary incompatibility`

**Solution**:

```bash
pip install "numpy<2.0.0" --force-reinstall --no-deps
# Then restart the kernel
```

### Issue 3: rapidfuzz import error

**Error**: `ImportError: cannot import name 'string_metric' from 'rapidfuzz'`

**Solution**: This is fixed in the latest code. Make sure you have the latest version:

```bash
cd /kaggle/working/PointWSSIS
git pull
```

### Issue 4: Missing annotation files

**Error**: `FileNotFoundError: .../instances_train2017_1p_s.json`

**Solution**: Download annotations first:

```bash
python scripts/download_annotations.py all --output-dir /kaggle/working/coco/annotations
```

### Issue 5: Out of Memory (OOM)

**Solution**:

- Use smaller batch size in config files
- Use gradient checkpointing
- Reduce image resolution
- Use a smaller percentage (1% or 2% instead of 5%)

## Directory Structure on Kaggle

After setup, your directory structure should look like:

```
/kaggle/
├── input/                          # Read-only
│   └── coco-2017-dataset/
│       └── coco/
│           ├── train2017/         # Images
│           ├── val2017/           # Images
│           └── annotations/       # Original COCO annotations (not used)
│
└── working/                        # Writable
    ├── PointWSSIS/                # Cloned repository
    │   ├── AdelaiDet/
    │   ├── detectron2/
    │   ├── MaskRefineNet/
    │   └── scripts/
    │
    └── coco/                       # Working COCO directory
        ├── annotations/            # PointWSSIS annotations (downloaded)
        │   ├── instances_train2017_1p_s.json
        │   ├── instances_train2017_1p_w.json
        │   └── ...
        ├── train2017/             # Symlink to /kaggle/input/.../train2017
        └── val2017/               # Symlink to /kaggle/input/.../val2017
```

## Performance Tips

1. **Use Kaggle's GPUs efficiently**:
   - Enable "GPU T4 x2" for parallel training
   - Monitor GPU usage with `nvidia-smi`

2. **Save checkpoints to Kaggle Datasets**:
   - Kaggle notebooks timeout after 12 hours
   - Save important checkpoints as Kaggle Datasets for persistence

3. **Use Kaggle Secrets** for GitHub tokens if needed

4. **Enable Session Persistence** to avoid re-running setup

## Additional Resources

- [Kaggle Docs - GPU Quotas](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle Docs - Datasets](https://www.kaggle.com/docs/datasets)
- [PointWSSIS Main README](README.md)
- [Quick Start Guide](QUICKSTART.md)
