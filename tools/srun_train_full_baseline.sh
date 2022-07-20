#!/usr/bin/env bash

CONFIG=configs/baseline/faster_rcnn_r50_caffe_fpn_coco_full_180k_multiscale_train.py
job_name=multiscale_train
work_dir=../Experiments/Baseline/$job_name

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
cluster=17

if [ ! -d "$work_dir" ]; then
    mkdir $work_dir
    echo "Create work_dir folder."
else
    echo "Folder exists."
fi

g=$(($1<8?$1:8))

case $cluster in
    1986)
    srun --mpi=pmi2 -p Model-16gv100 -x SH-IDC1-10-198-6-242 -n$1 --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=2 \
            --quotatype=auto --job-name=$job_name \
    python -u $(dirname "$0")/train.py $CONFIG \
            --work-dir ${work_dir} \
            --launcher="slurm" \
            --cfg-options data.train.sup.img_prefix=/mnt/lustre/share_data/coco/train2017 \
                    data.train.unsup.img_prefix=/mnt/lustre/share_data/coco/train2017 \
                    data_root=/mnt/lustre/share_data/coco \
                    data.val.ann_file=/mnt/lustre/share_data/coco/annotations/instances_val2017.json \
                    data.val.img_prefix=/mnt/lustre/share_data/coco/val2017  
    ;;
    
    *)
    srun --mpi=pmi2 -p Model -x BJ-IDC1-10-10-17-23 -n$1 --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=2 \
            --quotatype=auto --async --job-name=$job_name \
    python -u $(dirname "$0")/train.py $CONFIG \
            --work-dir ${work_dir} \
            --launcher="slurm" 
    ;;
esac 