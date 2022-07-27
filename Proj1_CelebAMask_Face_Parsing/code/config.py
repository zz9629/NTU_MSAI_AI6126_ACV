norm_cfg = dict(type='BN', requires_grad=True)

# log2:  [1.805, 1.976, 5.6004, 8.5621, 8.8014, 8.8059, 7.8778, 7.9143, 7.7323, 8.0199, 8.3939, 7.9185, 7.2026, 1.6725, 6.695, 8.7228, 12.7002, 4.6004, 4.8802]

# log10: [0.5434, 0.5948, 1.6859, 2.5774, 2.6495, 2.6508, 2.3715, 2.3824, 2.3276, 2.4142, 2.5268, 2.3837, 2.1682, 0.5035, 2.0154, 2.6258, 3.8231, 1.3849, 1.4691]
# bisenetv2

model = dict(
    type='EncoderDecoder',
    # init_cf=dict(type='Pretrained', checkpoint='open-mmlab://resnet50_v1c',),
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        pretrained='open-mmlab://resnet50_v1c'),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
        class_weight=[1.805, 1.976, 5.6004, 8.5621, 8.8014, 8.8059, 7.8778, 7.9143,
                      7.7323, 8.0199, 8.3939, 7.9185, 7.2026, 1.6725, 6.695, 8.7228, 12.7002, 4.6004, 4.8802])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
        class_weight=[1.805, 1.976, 5.6004, 8.5621, 8.8014, 8.8059, 7.8778, 7.9143,
                      7.7323, 8.0199, 8.3939, 7.9185, 7.2026, 1.6725, 6.695, 8.7228, 12.7002, 4.6004, 4.8802])),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'FaceParsingDataset'
data_root = 'AI6126_dataset_public'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='FaceParsingDataset',
        data_root='AI6126_dataset_public',
        img_dir='train/train_image',
        ann_dir='train/train_mask',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomFlip', prob=0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split='splits/train.txt'),
    val=dict(
        type='FaceParsingDataset',
        data_root='AI6126_dataset_public',
        img_dir='val/val_image',
        ann_dir='val/val_mask',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/val.txt'),
    test=dict(
        type='FaceParsingDataset',
        data_root='AI6126_dataset_public',
        img_dir='test/test_image',
        ann_dir='val/val_mask',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/val.txt'))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'work_dirs/CosineAnnealing/iter_40000.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005,)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=4000,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-5)
runner = dict(type='IterBasedRunner', max_iters=60000)
# checkpoint_config = dict(by_epoch=False, interval=200)
checkpoint_config = dict(by_epoch=False, interval=5000, meta=dict(
    CLASSES=('background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
             'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth'),
    PALETTE=[[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
             [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
             [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]])
                         )
evaluation = dict(interval=2000, metric=['mIoU', 'mDice'], pre_eval=True)
# project_name = "ai6126"
expt_name = "name"
work_dir = f"./work_dirs/{expt_name}"
# work_dir = './work_dirs/'
seed = 0
gpu_ids = range(0, 1)

