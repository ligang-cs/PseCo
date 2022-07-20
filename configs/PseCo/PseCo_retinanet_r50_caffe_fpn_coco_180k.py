mmdet_base = "../../thirdparty/mmdetection/configs"
_base_ = [
    f"{mmdet_base}/_base_/models/retinanet_r50_fpn.py",
    f"{mmdet_base}/_base_/datasets/coco_detection.py",
    f"{mmdet_base}/_base_/schedules/schedule_1x.py",
    f"{mmdet_base}/_base_/default_runtime.py",
]

model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="/mnt/lustre/ligang2/Resource/resnet50_msra-5891d200.pth" 
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=6),
    bbox_head=dict(
        anchor_generator=dict(
            _delete_=True,
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
    ),
    train_cfg=dict(
        assigner=dict(_delete_=True, type='ATSSAssigner', topk=9)
    )
)

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],      
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],   
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type=k)
                            for k in [
                                "Identity",
                                "AutoContrast",
                                "RandEqualize",
                                "RandSolarize",
                                "RandColor",
                                "RandContrast",
                                "RandBrightness",
                                "RandSharpness",
                                "RandPosterize",
                            ]
                        ],
                    ),
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
weak_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility
    # dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type="SemiDataset",
        sup=dict(
            type="CocoDataset",
            ann_file="../data/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="/mnt/lustre/share/DSK/datasets/mscoco2017/train2017/",
            pipeline=train_pipeline,
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="../data/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="/mnt/lustre/share/DSK/datasets/mscoco2017/train2017/",
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
        ),
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio=[1, 4],
            by_prob=True,
            # at_least_one=True,
            epoch_length=7330,
        )
    ),
)

semi_wrapper = dict(
    type="PseCo_RetinaNet",
    model="${model}",
    train_cfg=dict(
        pseudo_label_initial_score_thr=0.3,
        cls_pseudo_threshold=0.4,
        min_pseduo_box_size=0,
        unsup_weight=2.0,
        use_MSL=True,
        use_PCV=True,
    ),
    test_cfg=dict(
        inference_on="student"
        ),
)

fold = 1
percent = 1
interval=1

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval="${interval}", warm_up=0),
]

evaluation = dict(type="SubModulesDistEvalHook", interval=5000)
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)   
lr_config = dict(step=[60000, 80000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=15, create_symlink=False)

fp16 = dict(loss_scale="dynamic")

log_config = dict(
    interval=50,   
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)

find_unused_parameters=True 