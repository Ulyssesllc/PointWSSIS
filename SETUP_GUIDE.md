# PointWSSIS Installation and Setup Guide

## Quick Start

### 1. Install Missing Dependencies

If you encounter the `ModuleNotFoundError: No module named 'portalocker'` error, install it:

```bash
pip install portalocker
```

Or install all requirements at once:

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Before running any scripts, set the data root path:

```bash
export DETECTRON2_DATASETS=/path/to/your/coco/data
```

For example:

```bash
export DETECTRON2_DATASETS=/kaggle/input/coco-2017-dataset
# or
export DETECTRON2_DATASETS=/content/data
```

### 3. Verify Setup

Check that your data directory has the correct structure:

```
$DETECTRON2_DATASETS/
├── coco/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   └── instances_train2017_*p_s.json  # Point annotation files
│   ├── train2017/
│   └── val2017/
```

### 4. Run Training Scripts

Now you can run the training scripts:

```bash
cd /path/to/PointWSSIS
bash scripts/coco_5p.sh
```

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError: No module named 'portalocker'

**Solution:**

```bash
pip install portalocker
```

### Issue 2: FileNotFoundError for annotation files

**Solution:**
Make sure `DETECTRON2_DATASETS` is set correctly:

```bash
export DETECTRON2_DATASETS=/path/to/your/data
echo $DETECTRON2_DATASETS  # Verify it's set
```

### Issue 3: "YOUR_DATA_ROOT" in error messages

**Solution:**
The environment variable is not set. Set it before running:

```bash
export DETECTRON2_DATASETS=/your/actual/path
```

### Issue 4: Scripts can't find Python files

**Solution:**
All scripts now use absolute paths. Make sure you run them from any directory:

```bash
bash /full/path/to/PointWSSIS/scripts/coco_5p.sh
```

## Building from Source

If you need to build detectron2 and AdelaiDet from source:

```bash
# Build detectron2
cd detectron2
python -m pip install -e .

# Build AdelaiDet
cd ../AdelaiDet
python -m pip install -e .
```

## Environment Setup for Different Platforms

### Google Colab

```python
import os
os.environ['DETECTRON2_DATASETS'] = '/content/drive/MyDrive/coco'
```

### Kaggle

```bash
export DETECTRON2_DATASETS=/kaggle/input/coco-2017-dataset
```

### Local Machine

```bash
# Add to ~/.bashrc or ~/.zshrc for persistence
export DETECTRON2_DATASETS=/home/user/datasets
```

## Verification Commands

```bash
# Check if portalocker is installed
python -c "import portalocker; print('portalocker installed')"

# Check if environment variable is set
echo $DETECTRON2_DATASETS

# Check if data directory exists
ls -la $DETECTRON2_DATASETS/coco/annotations/

# Test imports
python -c "from detectron2.data import MetadataCatalog; print('detectron2 OK')"
```

## Additional Notes

- All shell scripts now work from any directory
- Data root can be set via environment variable or command-line argument
- Scripts automatically detect project root and use absolute paths
- PIL.Image.LINEAR compatibility issue has been fixed (now uses BILINEAR)
