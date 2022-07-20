#!/usr/bin/env bash
set -x

CONFIG=configs/soft_teacher/unlabel_average_voting_faster_rcnn_r50_caffe_fpn_coco_full_720k.py
work_dir=../Experiments/debug

FOLD=1
PERCENT=10

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT:-29500} \
    $(dirname "$0")/train.py $CONFIG --work-dir $work_dir --launcher=pytorch \
    --cfg-options fold=${FOLD} \
                    percent=${PERCENT} \
                    data.train.sup.ann_file=../data/annotations/instances_train2017.json \
                    data.train.sup.img_prefix=../data/train2017 \
                    data.train.unsup.ann_file=../data/annotations/instances_unlabeled2017.json \
                    data.train.unsup.img_prefix=../data/unlabeled2017 \
                    data.val.ann_file=../data/annotations/instances_val2017.json \
                    data.val.img_prefix=../data/val2017 \
                    data.workers_per_gpu=1 \
                    data.samples_per_gpu=2 \
                    data.sampler.train.sample_ratio=[1,1] \
                    model.backbone.init_cfg.checkpoint=/home/SENSETIME/ligang2/Resource/model_zoo/resnet50_msra-5891d200.pth \
                    load_from=../checkpoints/iter_80000.pth 