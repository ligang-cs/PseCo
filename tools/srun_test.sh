#!/usr/bin/env bash

CONFIG=configs/soft_teacher/semi_align_train_faster_rcnn_r50_caffe_fpn_coco_180k.py
CHECKPOINT=../Experiments/semi/align_train/high_mimic/iter_90000.pth 

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

cluster=17

g=$(($1<8?$1:8))

case $cluster in
    1986)
    srun --mpi=pmi2 -p Model-16gv100 -x SH-IDC1-10-198-6-242 -n$1 --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=2 \
                --quotatype=auto --job-name=$job_name \
    python -u $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval bbox \
        --launcher="slurm" \
        --cfg-options data.test.ann_file=/mnt/lustre/share_data/coco/annotations/instances_val2017.json \
                    data.test.img_prefix=/mnt/lustre/share_data/coco/val2017  \
    ;;
    *)
    srun --mpi=pmi2 -p Model -x BJ-IDC1-10-10-17-23 -n$1 --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=2 \
                --quotatype=auto --job-name=$job_name \
    python -u $(dirname "$0")/test.py $CONFIG $CHECKPOINT --eval bbox \
        --launcher="slurm" \
        --cfg-options data.test.ann_file=/mnt/lustre/share/DSK/datasets/mscoco2017/annotations/instances_val2017.json \
                      data.test.img_prefix=/mnt/lustre/share/DSK/datasets/mscoco2017/val2017  \
    ;;
esac 