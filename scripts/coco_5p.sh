#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Add AdelaiDet and detectron2 to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/AdelaiDet:${PROJECT_ROOT}/detectron2:${PROJECT_ROOT}/MaskRefineNet:${PYTHONPATH}"

# Set data root from environment or use placeholder
root=${DETECTRON2_DATASETS:-"YOUR_DATA_ROOT"}

# Validate data root is set
if [ "$root" = "YOUR_DATA_ROOT" ]; then
    echo "Error: Please set DETECTRON2_DATASETS environment variable"
    echo "Example: export DETECTRON2_DATASETS=/path/to/coco"
    exit 1
fi

export DETECTRON2_DATASETS=${root}

# Change to AdelaiDet directory so config paths work correctly
cd "${PROJECT_ROOT}/AdelaiDet" || exit 1

config_file="configs/PointWSSIS/R101_teacher.yaml"
exp_name="SOLOv2_R101_coco5p_teacher"
trainsets="('coco_2017_train_5p_s',)"
testsets="('coco_2017_val',)"
prompt="point"
eval_pseudo_label="True"

learning_rate="0.05"
decay_steps="(20000,25000)"
train_iter="30001"

if command -v nvidia-smi >/dev/null 2>&1; then
    ngpus=$(nvidia-smi --list-gpus | wc -l)
else
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        ngpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    else
        ngpus=1
    fi
fi

if command -v torchrun >/dev/null 2>&1; then
    RUNNER="torchrun --standalone --nnodes=1 --nproc_per_node=${ngpus}"
else
    RUNNER="python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=${ngpus}"
fi

# step 1
OMP_NUM_THREADS=1 python3 -W ignore tools/train_net.py \
    --config-file ${config_file} \
    --num-gpus ${ngpus} \
    SEED 1 \
    OUTPUT_DIR training_dir/${exp_name} \
    DATASETS.TRAIN ${trainsets} \
    DATASETS.TEST ${testsets} \
    SOLVER.STEPS ${decay_steps} \
    SOLVER.MAX_ITER ${train_iter} \
    SOLVER.BASE_LR ${learning_rate} \
    MODEL.SOLOV2.PROMPT ${prompt} \
    MODEL.SOLOV2.EVAL_PSEUDO_LABEL ${eval_pseudo_label} \
    TEST.EVAL_PERIOD 5000
    
# step 2
testsets="('coco_2017_train_5p_w',)"
prompt="point"
inference_dir="inference_dir"
OMP_NUM_THREADS=1 python3 -W ignore tools/train_net.py \
    --config-file ${config_file} \
    --num-gpus ${ngpus} \
    --eval-only \
    MODEL.WEIGHTS training_dir/${exp_name}/model_final.pth \
    OUTPUT_DIR inference_dir/${exp_name} \
    MODEL.SOLOV2.FPN_SCALE_RANGES "((1,100000),(1,100000),(1,100000),(1,100000),(1,100000))" \
    MODEL.SOLOV2.NMS_TYPE mask \
    MODEL.SOLOV2.PROMPT ${prompt} \
    DATASETS.TEST ${testsets} \
    
# step 3 (optional - without MaskRefineNet)
# python3 tools/merge_strong_and_weak_labels.py \
#     --workers 8 \
#     --save_path ${root}/coco/annotations/instances_train2017_5p_sw.json \
#     --strong_json ${root}/coco/annotations/instances_train2017_5p_s.json \
#     --full_json ${root}/coco/annotations/instances_train2017.json \
#     --weak_pth inference_dir/${exp_name}/inference/instances_predictions.pth

# config_file_sw="configs/SOLOv2/R101_3x.yaml"
# trainsets="('coco_2017_train_5p_sw',)"
# testsets="('coco_2017_val',)"
# exp_name_sw="SOLOv2_R101_coco5p_sw"

# OMP_NUM_THREADS=1 python3 -W ignore tools/train_net.py \
#     --config-file ${config_file_sw} \
#     --num-gpus ${ngpus} \
#     SEED 1 \
#     OUTPUT_DIR training_dir/${exp_name_sw} \
#     DATASETS.TRAIN ${trainsets} \
#     DATASETS.TEST ${testsets} \
#     TEST.EVAL_PERIOD 5000

# step 4
testsets="('coco_2017_train_5p_s',)"
prompt="point_with_size"

OMP_NUM_THREADS=1 python3 -W ignore tools/train_net.py \
    --config-file ${config_file} \
    --num-gpus ${ngpus} \
    --eval-only \
    MODEL.WEIGHTS training_dir/${exp_name}/model_0014999.pth \
    OUTPUT_DIR inference_dir/${exp_name}_strong_1 \
    MODEL.SOLOV2.NMS_TYPE mask \
    MODEL.SOLOV2.PROMPT ${prompt} \
    DATASETS.TEST ${testsets} \

OMP_NUM_THREADS=1 python3 -W ignore tools/train_net.py \
    --config-file ${config_file} \
    --num-gpus ${ngpus} \
    --eval-only \
    MODEL.WEIGHTS training_dir/${exp_name}/model_0024999.pth \
    OUTPUT_DIR inference_dir/${exp_name}_strong_2 \
    MODEL.SOLOV2.NMS_TYPE mask \
    MODEL.SOLOV2.PROMPT ${prompt} \
    DATASETS.TEST ${testsets} \


# step 5
train_iters=200000
warm_iters=2000
val_interval=5000

MRN_exp_name="MRN_5p"
gt_json="instances_train2017_5p_s.json"

weak_pth="${PROJECT_ROOT}/AdelaiDet/inference_dir/${exp_name}_strong_1/inference/instances_predictions.pth ${PROJECT_ROOT}/AdelaiDet/inference_dir/${exp_name}_strong_2/inference/instances_predictions.pth"
eval_pth="${PROJECT_ROOT}/AdelaiDet/inference_dir/${exp_name}/inference/instances_predictions.pth"

${RUNNER} "${PROJECT_ROOT}/MaskRefineNet/main.py" \
    --data_root ${root} \
    --workspace results \
    --exp_name ${MRN_exp_name} \
    --train_iters ${train_iters} \
    --warm_iters ${warm_iters} \
    --val_interval ${val_interval} \
    --weak_pth ${weak_pth} \
    --gt_json ${gt_json} \
    --eval_pth ${eval_pth} \
    --amp

${RUNNER} "${PROJECT_ROOT}/MaskRefineNet/merge_strong_and_refined_weak_labels.py" \
    --data_root ${root} \
    --ckpt results/${MRN_exp_name}/ckpt/best_AP.pt \
    --dataset coco \
    --size 256 \
    --weak_pth ${eval_pth} \
    --weak_json ${root}/coco/annotations/instances_train2017_5p_w.json \
    --strong_json ${root}/coco/annotations/instances_train2017_5p_s.json \
    --save_path ${root}/coco/annotations/instances_train2017_5p_sw_refined.json


# step 6
config_file="configs/SOLOv2/R101_3x.yaml"
trainsets="('coco_2017_train_5p_sw_refined',)"
testsets="('coco_2017_val',)"
exp_name="SOLOv2_R101_coco5p_sw_refined"

OMP_NUM_THREADS=1 python3 -W ignore tools/train_net.py \
    --config-file ${config_file} \
    --num-gpus ${ngpus} \
    SEED 1 \
    OUTPUT_DIR training_dir/${exp_name} \
    DATASETS.TRAIN ${trainsets} \
    DATASETS.TEST ${testsets} \
    TEST.EVAL_PERIOD 5000

