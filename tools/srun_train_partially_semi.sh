#!/usr/bin/env bash
T=`date +%m%d-%H%M`
CONFIG=configs/soft_teacher/PseCo_retinanet_r50_coco_180k.py
job_name=CELoss
work_dir=../Experiments/camera_ready/$job_name
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

FOLD=1
PERCENT=10

export unsup_start_iter=4000

cluster=SH
case $cluster in
    SH)
    train_img_prefix="1984:s3://openmmlab/datasets/detection/coco/train2017"
    val_img_prefix="1984:s3://openmmlab/datasets/detection/coco/val2017"
    ;;
    *)
    train_img_prefix="s3://academic_datasets/mscoco2017/train2017"
    val_img_prefix="s3://academic_datasets/mscoco2017/val2017"
    ;;
esac

if [ ! -d "$work_dir" ]; then
    mkdir $work_dir
    echo "Create work_dir folder."
else
    echo "Folder exists."
fi

g=$(($1<8?$1:8))
srun --mpi=pmi2 -p Model-1080ti -n$1 --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=4 \
        --quotatype=auto --async --job-name=$job_name -o ./srun_logs/$T.log \
python -u $(dirname "$0")/train.py $CONFIG \
        --work-dir ${work_dir} \
        --launcher="slurm" \
        --cfg-options data.train.sup.img_prefix=${train_img_prefix} \
                data.train.unsup.img_prefix=${train_img_prefix} \
                data.val.img_prefix=${val_img_prefix} \
                fold=${FOLD} \
                percent=${PERCENT} \
                data.samples_per_gpu=5 \
                data.sampler.train.sample_ratio=\[1,4\] \
               
  
# -x BJ-IDC1-10-10-17-[23,68] 






