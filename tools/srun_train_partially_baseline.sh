#!/usr/bin/env bash

CONFIG=configs/baseline/retinanet_r50_fpn_coco_partial_90k.py
job_name=retina_partial
work_dir=../Experiments/Baseline/$job_name

FOLD=1
PERCENT=10

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [ ! -d "$work_dir" ]; then
    mkdir $work_dir
    echo "Create work_dir folder."
else
    echo "Folder exists."
fi

g=$(($1<8?$1:8))
srun --mpi=pmi2 -p Model -x BJ-IDC1-10-10-17-23 -n$1 --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=2 \
            --quotatype=auto --async --job-name=$job_name \
python -u $(dirname "$0")/train.py $CONFIG \
        --work-dir $work_dir \
        --launch='slurm' \
        --cfg-options fold=${FOLD} \
                  percent=${PERCENT}

