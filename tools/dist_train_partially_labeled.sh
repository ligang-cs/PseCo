#!/usr/bin/env bash
set -x

CONFIG=configs/soft_teacher/PseCo_faster_rcnn_r50_caffe_fpn_coco_180k.py   
work_dir=../Experiments/debug

FOLD=1
PERCENT=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

export unsup_start_iter=0
export unsup_warmup_iter=0

python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT:-29500} \
    $(dirname "$0")/train.py $CONFIG --work-dir $work_dir --launcher=pytorch \
    --cfg-options fold=${FOLD} \
                    percent=${PERCENT} \
                    data.workers_per_gpu=1 \
                    data.samples_per_gpu=2 \
                    data.sampler.train.sample_ratio=[1,1] \
                    model.backbone.init_cfg.checkpoint=/home/SENSETIME/ligang2/Resource/model_zoo/resnet50_msra-5891d200.pth \
                    optimizer.lr=0.0 \
                    auto_resume=False \
                   

# load_from=../checkpoints/PseCo_CELoss/iter_180000.pth \
# load_from=../checkpoints/retinanet/labeled_only_iter_8000.pth \              