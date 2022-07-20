_base_ = "base.py"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file="data/coco/annotations/instances_train2017.json",
        img_prefix="data/coco/train2017/",
    ),
)

model = dict(
    neck=dict(
        start_level=1,
        add_extra_convs='on_input'
    ),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64]),
    ),
)

optimizer = dict(lr=0.02)
lr_config = dict(step=[120000, 160000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000)

