#!/bin/bash

################################################################################
# PointWSSIS Evaluation Pipeline Runner
# This script runs the complete evaluation pipeline and generates visualizations
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DATA_ROOT="${DATA_ROOT:-/path/to/coco}"
TRAINING_DIR="${TRAINING_DIR:-training_dir}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results}"
CONFIG_FILE="${CONFIG_FILE:-evaluation_config.yaml}"
PROPORTIONS="${PROPORTIONS:-}"  # Empty means all proportions

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --training_dir)
            TRAINING_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --proportions)
            shift
            PROPORTIONS=""
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                PROPORTIONS="$PROPORTIONS $1"
                shift
            done
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_root DIR       Path to COCO dataset (default: /path/to/coco)"
            echo "  --training_dir DIR    Path to trained models (default: training_dir)"
            echo "  --output_dir DIR      Output directory (default: evaluation_results)"
            echo "  --config FILE         Config file (default: evaluation_config.yaml)"
            echo "  --proportions LIST    Specific proportions (e.g., 5p 10p 20p)"
            echo "  --help               Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  DATA_ROOT, TRAINING_DIR, OUTPUT_DIR, CONFIG_FILE, PROPORTIONS"
            echo ""
            echo "Examples:"
            echo "  $0 --data_root /data/coco --training_dir models"
            echo "  $0 --proportions 5p 10p 20p"
            echo "  export DATA_ROOT=/data/coco && $0"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        return 1
    fi
    return 0
}

check_dir() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        return 1
    fi
    return 0
}

################################################################################
# Pre-flight Checks
################################################################################

print_header "PointWSSIS Evaluation Pipeline"

print_info "Configuration:"
echo "  Data root:     $DATA_ROOT"
echo "  Training dir:  $TRAINING_DIR"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Config file:   $CONFIG_FILE"
if [ -n "$PROPORTIONS" ]; then
    echo "  Proportions:   $PROPORTIONS"
else
    echo "  Proportions:   All (1p, 2p, 5p, 10p, 20p, 30p, 50p)"
fi
echo ""

print_header "Pre-flight Checks"

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found"
    exit 1
fi
print_success "Python found: $(python --version)"

# Check required files
if ! check_file "evaluate_pipeline.py"; then exit 1; fi
if ! check_file "visualize_results.py"; then exit 1; fi
if ! check_file "evaluation_utils.py"; then exit 1; fi
print_success "Evaluation scripts found"

# Check config file
if [ -f "$CONFIG_FILE" ]; then
    print_success "Config file found: $CONFIG_FILE"
else
    print_warning "Config file not found: $CONFIG_FILE"
    print_info "Using default configuration"
fi

# Check data directory
if ! check_dir "$DATA_ROOT"; then
    print_error "Data directory not found: $DATA_ROOT"
    print_warning "Please set DATA_ROOT to your COCO dataset path"
    exit 1
fi
print_success "Data directory found"

# Check training directory
if ! check_dir "$TRAINING_DIR"; then
    print_warning "Training directory not found: $TRAINING_DIR"
    print_info "Make sure you have trained models available"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/plots"
print_success "Output directory created: $OUTPUT_DIR"

################################################################################
# Run Evaluation
################################################################################

print_header "Running Evaluation"

EVAL_CMD="python evaluate_pipeline.py \
    --data_root $DATA_ROOT \
    --training_dir $TRAINING_DIR \
    --output_dir $OUTPUT_DIR"

if [ -f "$CONFIG_FILE" ]; then
    EVAL_CMD="$EVAL_CMD --config $CONFIG_FILE"
fi

if [ -n "$PROPORTIONS" ]; then
    EVAL_CMD="$EVAL_CMD --proportions $PROPORTIONS"
fi

print_info "Command: $EVAL_CMD"
echo ""

if $EVAL_CMD; then
    print_success "Evaluation completed successfully"
else
    print_error "Evaluation failed"
    exit 1
fi

################################################################################
# Generate Visualizations
################################################################################

RESULTS_FILE="$OUTPUT_DIR/evaluation_results.json"

if [ ! -f "$RESULTS_FILE" ]; then
    print_error "Results file not found: $RESULTS_FILE"
    exit 1
fi

print_header "Generating Visualizations"

VIZ_CMD="python visualize_results.py \
    --results $RESULTS_FILE \
    --output_dir $OUTPUT_DIR/plots \
    --plots all"

print_info "Command: $VIZ_CMD"
echo ""

if $VIZ_CMD; then
    print_success "Visualizations generated successfully"
else
    print_error "Visualization generation failed"
    exit 1
fi

################################################################################
# Generate Summary Report
################################################################################

print_header "Summary Report"

if [ -f "$OUTPUT_DIR/summary_table.txt" ]; then
    echo ""
    cat "$OUTPUT_DIR/summary_table.txt"
    echo ""
fi

################################################################################
# Final Output
################################################################################

print_header "Evaluation Complete"

print_info "Results saved to:"
echo "  📊 Results JSON:   $OUTPUT_DIR/evaluation_results.json"
echo "  📋 Summary table:  $OUTPUT_DIR/summary_table.txt"
echo "  📈 Plots:          $OUTPUT_DIR/plots/"
echo "  📝 Logs:           $OUTPUT_DIR/evaluation_log.txt"
echo ""

print_success "All tasks completed successfully!"

# List generated plots
if [ -d "$OUTPUT_DIR/plots" ]; then
    print_info "Generated plots:"
    for plot in "$OUTPUT_DIR/plots"/*.png; do
        if [ -f "$plot" ]; then
            echo "  • $(basename "$plot")"
        fi
    done
    echo ""
fi

print_info "To view results:"
echo "  • Open plots in: $OUTPUT_DIR/plots/"
echo "  • Read summary:  cat $OUTPUT_DIR/summary_table.txt"
echo "  • Check logs:    cat $OUTPUT_DIR/evaluation_log.txt"
echo ""

print_success "Done! 🎉"
