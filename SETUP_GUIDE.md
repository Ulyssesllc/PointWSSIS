# PointWSSIS Installation and Setup Guide

## Quick Start

For a streamlined setup process, use the provided helper scripts:

```bash
# 1. Verify your setup
python scripts/verify_setup.py

# 2. Download annotation files
python scripts/download_annotations.py 1 2 5

# 3. Run training
bash scripts/coco_5p.sh
```

See below for detailed step-by-step instructions.

## Prerequisites

Before you begin, ensure you have:

- Python 3.7+
- CUDA 10.2+ (for GPU support)
- GCC 5+ (for building C++ extensions)
- COCO 2017 dataset

## Helper Scripts

We provide utility scripts to simplify setup:

- **verify_setup.py**: Check installation completeness

  ```bash
  python scripts/verify_setup.py [--scenario 1p]
  ```

- **download_annotations.py**: Download point annotation files

  ```bash
  python scripts/download_annotations.py 1 2 5 [--pseudo] [--refined]
  ```

## Complete Setup Process

### Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install additional required packages
pip install portalocker
```

### Step 2: Build C++ Extensions (REQUIRED)

**CRITICAL:** You must build the C++ extensions before running any training scripts.

```bash
# Build detectron2
cd detectron2
python -m pip install -e .

# Build AdelaiDet (includes _C extension)
cd ../AdelaiDet
python setup.py build develop --user

# Verify the build
python -c "from adet import _C; print('AdelaiDet _C module built successfully')"
```

**Common build issues:**

- If you get CUDA errors, ensure `nvcc --version` matches your PyTorch CUDA version
- On Kaggle/Colab, you may need to run: `pip install ninja` first

### Step 3: Prepare Point Annotation Files (REQUIRED)

The point annotation files (`instances_train2017_*p_s.json`) are **NOT** part of the standard COCO dataset.

#### Quick Method: Use Download Script

We provide a convenient script to download all annotations:

```bash
# Download specific scenarios
python scripts/download_annotations.py 1 2 5

# Download all scenarios
python scripts/download_annotations.py all

# Include pseudo labels (optional, to skip training steps)
python scripts/download_annotations.py 1 5 10 --pseudo --refined
```

#### Option A: Download Pre-Generated Annotations (Recommended)

Download from the [official GitHub releases](https://github.com/clovaai/PointWSSIS/releases/tag/annotation_coco):

```bash
cd $DETECTRON2_DATASETS/coco/annotations/

# Download for specific scenarios (examples):
# For COCO 1%:
wget https://github.com/clovaai/PointWSSIS/releases/download/annotation_coco/instances_train2017_1p_s.json
wget https://github.com/clovaai/PointWSSIS/releases/download/annotation_coco/instances_train2017_1p_w.json

# For COCO 2%:
wget https://github.com/clovaai/PointWSSIS/releases/download/annotation_coco/instances_train2017_2p_s.json
wget https://github.com/clovaai/PointWSSIS/releases/download/annotation_coco/instances_train2017_2p_w.json

# For COCO 5%:
wget https://github.com/clovaai/PointWSSIS/releases/download/annotation_coco/instances_train2017_5p_s.json
wget https://github.com/clovaai/PointWSSIS/releases/download/annotation_coco/instances_train2017_5p_w.json

# Similarly for 10p, 20p, 30p, 50p scenarios
```

**File suffixes explained:**

- `*_s.json`: Strong (fully-labeled) subset - small portion with full masks
- `*_w.json`: Weak (point-labeled) subset - larger portion with only point annotations
- `*_sw.json`: Combined pseudo labels (optional, generated during training)
- `*_sw_refined.json`: Refined pseudo labels with MaskRefineNet (optional)

#### Option B: Generate Annotations Yourself

Use the detectron2 PointSup tool (generates different format):

```bash
cd detectron2/projects/PointSup/tools

# Generate with N points per instance
python prepare_coco_point_annotations_without_masks.py 10

# Output: instances_train2017_n10_v1_without_masks.json
```

**Note**: Option B generates a different format than PointWSSIS expects. For reproducibility with the paper, **use Option A**.

#### On Kaggle/Colab without wget

```python
import requests
import os

annotations_dir = os.path.join(os.getenv('DETECTRON2_DATASETS'), 'coco/annotations')
base_url = 'https://github.com/clovaai/PointWSSIS/releases/download/annotation_coco/'

files = [
    'instances_train2017_1p_s.json',
    'instances_train2017_1p_w.json',
    # Add more files as needed
]

for file in files:
    url = base_url + file
    response = requests.get(url)
    filepath = os.path.join(annotations_dir, file)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print(f'Downloaded {file}')
```

### Step 4: Set Environment Variables

**IMPORTANT:** Set the data root path:

```bash
export DETECTRON2_DATASETS=/path/to/your/coco/data
```

For example:

```bash
export DETECTRON2_DATASETS=/kaggle/input/coco-2017-dataset
# or
export DETECTRON2_DATASETS=/content/data
```

### Step 5: Verify Setup

**Use the verification script to check everything:**

```bash
# Basic verification
python scripts/verify_setup.py

# Verify specific scenario
python scripts/verify_setup.py --scenario 1p
```

The script checks:

- ✓ Python version (3.7+)
- ✓ All dependencies installed
- ✓ C++ extensions built (detectron2._C and adet._C)
- ✓ CUDA availability
- ✓ DETECTRON2_DATASETS environment variable
- ✓ Data directory structure
- ✓ Annotation files present

**Manual verification:**

Check that your data directory has the correct structure:

```
$DETECTRON2_DATASETS/
├── coco/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   └── instances_train2017_*p_s.json  # Point annotation files (MUST EXIST)
│   ├── train2017/
│   └── val2017/
```

Verify all required files exist:

```bash
# Check annotation files
ls -la $DETECTRON2_DATASETS/coco/annotations/instances_train2017_1p_s.json
ls -la $DETECTRON2_DATASETS/coco/annotations/instances_train2017_2p_s.json
ls -la $DETECTRON2_DATASETS/coco/annotations/instances_train2017_5p_s.json
# ... and other percentages

# Check C++ extension is built
python -c "from adet import _C; print('✓ AdelaiDet C++ extension ready')"

# Check environment variable
echo "DETECTRON2_DATASETS=$DETECTRON2_DATASETS"
```

### Step 6: Run Training Scripts

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
