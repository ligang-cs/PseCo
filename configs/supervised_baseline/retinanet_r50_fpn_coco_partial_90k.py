_base_ = "base_one_stage.py"

fold = 1
percent = 10

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file="../data/semi_supervised/instances_train2017.${fold}@${percent}.json",
        img_prefix="/mnt/lustre/share/DSK/datasets/mscoco2017/train2017/",
    ),
)

