_base_ = "base.py"
fold = 1
percent = 1
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        ann_file="../data/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
        img_prefix="../data/train2017/",
    ),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
