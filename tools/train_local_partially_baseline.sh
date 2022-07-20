#!/usr/bin/env bash

CONFIG=configs/baseline/retinanet_r50_fpn_coco_partial_90k_align_train.py
work_dir=../Experiments/debug

FOLD=1
PERCENT=10

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python $(dirname "$0")/train.py $CONFIG --work-dir $work_dir \
    --cfg-options fold=${FOLD} \
                    percent=${PERCENT} \
                    data.train.ann_file=../data/annotations/semi_supervised/instances_train2017.${FOLD}@${PERCENT}.json \
                    data.train.img_prefix=../data/train2017 \
                    data.val.ann_file=../data/annotations/instances_val2017.json \
                    data.val.img_prefix=../data/val2017 \
                    data.samples_per_gpu=1 \
                    data.workers_per_gpu=1 
                    # model.backbone.init_cfg.checkpoint="open-mmlab://detectron2/resnet50_caffe" \

