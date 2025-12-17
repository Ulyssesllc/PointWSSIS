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

**IMPORTANT:** Before running any scripts, you **MUST** set the data root path:

```bash
export DETECTRON2_DATASETS=/path/to/your/coco/data
```

For example:

```bash
export DETECTRON2_DATASETS=/kaggle/input/coco-2017-dataset
# or
export DETECTRON2_DATASETS=/content/data
```

**Note:** The scripts will validate this environment variable and exit with a clear error message if not set.

**What the scripts do automatically:**

- Set `PYTHONPATH` to include AdelaiDet, detectron2, and MaskRefineNet modules
- Validate that `DETECTRON2_DATASETS` is set before running
- Export the environment variable for all child processes

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

### Issue 1: ModuleNotFoundError: No module named 'adet'

**Cause:** The `adet` module is not in Python's search path.

**Solution:**
The scripts now automatically add the required directories to `PYTHONPATH`. If you're running Python files directly (not through the scripts), set it manually:

```bash
export PYTHONPATH="${PWD}/AdelaiDet:${PWD}/detectron2:${PWD}/MaskRefineNet:${PYTHONPATH}"
```

### Issue 2: ModuleNotFoundError: No module named 'portalocker'

**Solution:**

```bash
pip install portalocker
```

### Issue 3: FileNotFoundError for annotation files

**Solution:**
Make sure `DETECTRON2_DATASETS` is set correctly:

```bash
export DETECTRON2_DATASETS=/path/to/your/data
echo $DETECTRON2_DATASETS  # Verify it's set
```

### Issue 4: "YOUR_DATA_ROOT" or "Please set DETECTRON2_DATASETS"

**Solution:**
The environment variable is not set. The scripts will now exit with a clear error message. Set it before running:

```bash
export DETECTRON2_DATASETS=/your/actual/path
bash scripts/coco_5p.sh
```

### Issue 5: Scripts can't find Python files

**Solution:**
All scripts now use absolute paths. Make sure you run them from any directory:

```bash
bash /full/path/to/PointWSSIS/scripts/coco_5p.sh
```

## What the Scripts Do Automatically

All training scripts (`coco_*p.sh`) now handle the following automatically:

1. **Detect Project Root**: Automatically find the project root directory
2. **Set PYTHONPATH**: Add AdelaiDet, detectron2, and MaskRefineNet to Python's module search path
3. **Validate Environment**: Check that `DETECTRON2_DATASETS` is set and exit with a helpful error if not
4. **Use Absolute Paths**: All Python script paths are absolute, so scripts work from any directory

**What You Still Need to Do:**

- Set `DETECTRON2_DATASETS` environment variable before running scripts
- Ensure all dependencies are installed (`pip install -r requirements.txt`)
- Have your COCO dataset in the correct location

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
