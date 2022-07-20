#!/usr/bin/env bash

CONFIG=configs/baseline/faster_rcnn_r50_caffe_fpn_coco_full_180k_multiscale_train.py
work_dir=../Experiments/debug

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python $(dirname "$0")/train.py $CONFIG --work-dir $work_dir \
    --cfg-options data.train.ann_file=../data/annotations/instances_train2017.json \
                    data.train.img_prefix=../data/train2017 \
                    data.val.ann_file=../data/annotations/instances_val2017.json \
                    data.val.img_prefix=../data/val2017 \
                    model.backbone.init_cfg.checkpoint=/home/SENSETIME/ligang2/Resource/model_zoo/resnet50_msra-5891d200.pth \
                    data.samples_per_gpu=1 \
                    data.workers_per_gpu=1 \

