#!/usr/bin/env bash

CONFIG=configs/soft_teacher/unlabel_average_voting_faster_rcnn_r50_caffe_fpn_coco_full_720k.py
job_name=unlabeled2017
work_dir=../Experiments/semi/$job_name

cluster=1986

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [ ! -d "$work_dir" ]; then
    mkdir $work_dir
    echo "Create work_dir folder."
else
    echo "Folder exists."
fi

g=$(($1<8?$1:8))

case $cluster in
    1986)
    srun --mpi=pmi2 -p Model -x SH-IDC1-10-198-6-242 -n$1 --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=4 \
            --quotatype=reserved --async --job-name=$job_name \
    python -u $(dirname "$0")/train.py $CONFIG \
            --work-dir ${work_dir} \
            --launcher="slurm" \
            --cfg-options data.train.sup.img_prefix=/mnt/lustre/share_data/coco/train2017 \
                    data.train.unsup.img_prefix=/mnt/lustre/ligang2/Works/SSOD/data/unlabeled2017 \
                    data_root=/mnt/lustre/share_data/coco \
                    data.val.ann_file=/mnt/lustre/share_data/coco/annotations/instances_val2017.json \
                    data.val.img_prefix=/mnt/lustre/share_data/coco/val2017 \
                    auto_resume=True 
    ;;
    
    *)
    srun --mpi=pmi2 -p Model -x BJ-IDC1-10-10-17-[23,68] -n$1 --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=2 \
            --quotatype=auto --async --job-name=$job_name \
    python -u $(dirname "$0")/train.py $CONFIG \
            --work-dir ${work_dir} \
            --launcher="slurm" \
            --cfg-options fold=${FOLD} \
                    percent=${PERCENT} \
                    auto_resume=True \
                    semi_wrapper.train_cfg.adap_assign=False \
                    semi_wrapper.train_cfg.disable_bbox_loss=True \
                    semi_wrapper.train_cfg.pred_var=False \
                    semi_wrapper.train_cfg.cls_pseudo_threshold=0.5 \
                    semi_wrapper.train_cfg.unlabel_view="both_views" \
                    data.samples_per_gpu=5 \
                    data.sampler.train.sample_ratio=\[1,4\]
    ;;
esac 
                         







