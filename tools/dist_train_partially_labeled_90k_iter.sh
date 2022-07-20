#!/usr/bin/env bash
set -x

CONFIG=configs/PseCo/PseCo_faster_rcnn_r50_caffe_fpn_coco_180k.py   
work_dir=           # define your experiment path here

FOLD=1
PERCENT=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

export unsup_start_iter=5000
export unsup_warmup_iter=2000

python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-29500} \
    $(dirname "$0")/train.py $CONFIG --work-dir $work_dir --launcher=pytorch \
    --cfg-options fold=${FOLD} \
                  percent=${PERCENT} \
                  runner.max_iters=90000 \
                  lr_config.step=\[60000\] \