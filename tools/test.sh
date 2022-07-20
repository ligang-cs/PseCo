#!/usr/bin/env bash

CONFIG=configs/PseCo/PseCo_faster_rcnn_r50_caffe_fpn_coco_180k.py
CHECKPOINT=         # path to checkpoint

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval bbox --out xxx.pkl \
    --cfg-options data.test.ann_file=../data/instances_val2017.json \
                  data.test.img_prefix=../data/val2017 \
                  data.workers_per_gpu=1 \
                  data.samples_per_gpu=1 \
