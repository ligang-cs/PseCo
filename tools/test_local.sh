#!/usr/bin/env bash

CONFIG=configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py
CHECKPOINT=../checkpoints/10_partial_baseline_iter_180000.pth 

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval bbox --out ../checkpoints/results_new.pkl \
    --cfg-options data.test.ann_file=../data/annotations/instances_val2017.json \
                  data.test.img_prefix=../data/val2017 \
                  data.workers_per_gpu=1 \
                  data.samples_per_gpu=1 \
