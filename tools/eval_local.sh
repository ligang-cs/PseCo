#!/usr/bin/env bash

CONFIG=configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py
RESULTS=../checkpoints/results.pkl 

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python thirdparty/mmdetection/tools/eval.py $RESULTS $CONFIG \
            --cfg-options data.test.ann_file=../data/annotations/instances_val2017.json \
                      data.test.img_prefix=../data/val2017  