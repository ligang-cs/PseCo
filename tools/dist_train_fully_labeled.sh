#!/usr/bin/env bash
set -x

CONFIG=configs/PseCo/PseCo_faster_rcnn_r50_caffe_fpn_coco_180k.py   
work_dir=           # define your experiment path here

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

export unsup_start_iter=5000
export unsup_warmup_iter=2000

python -m torch.distributed.launch --nproc_per_node=8 --master_port=${PORT:-29500} \
    $(dirname "$0")/train.py $CONFIG --work-dir $work_dir --launcher=pytorch \
    --cfg-options data.train.sup.ann_file=../data/instances_train2017.json \
                    data.train.sup.img_prefix=../data/train2017 \
                    data.train.unsup.ann_file=../data/instances_unlabeled2017.json \
                    data.train.unsup.img_prefix=../data/unlabeled2017 \
                    semi_wrapper.train_cfg.unsup_weight=1.0 \
                    data.samples_per_gpu=8 \
                    data.sampler.train.sample_ratio=[1,1] \
                    runner.max_iters=720000 \
                    lr_config.step=\[480000\] \
                   

