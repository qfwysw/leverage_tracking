_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
model = dict(
    type='Detr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        pretrained = 'ckpts/resnet101_caffe-3ad79236.pth',
        ),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='PETRHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            # type='NMSFreeClsCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            # cls_cost=dict(type='ClassificationCost', weight=2.0),
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = '/data/Dataset/nuScenes/'
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))
ida_aug_conf = {
        "resize_lim": (0.8, 1.0),
        "final_dim": (512, 1408),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        # "rand_flip": False,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[1.0, 1.0],
            reverse_angle=True,
            training=True
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality),
    test=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality))

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=4,
#     train=dict(
#         type='CBGSDataset',
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             ann_file=data_root + 'nuscenes_infos_train.pkl',
#             pipeline=train_pipeline,
#             classes=class_names,
#             modality=input_modality,
#             test_mode=False,
#             use_valid_flag=True,
#             # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#             # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#             box_type_3d='LiDAR'),
#     ),
#     val=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality),
#     test=dict(pipeline=test_pipeline, classes=class_names, modality=input_modality))

optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    # by_epoch=False
    )
total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)
find_unused_parameters = False

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='ckpts/fcos3d.pth'
resume_from=None


###c5 imagenet pretrain
# mAP: 0.3290
# mATE: 0.8005
# mASE: 0.2777
# mAOE: 0.6102
# mAVE: 0.9941
# mAAE: 0.2292
# NDS: 0.3733
# Eval time: 179.3s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.518   0.594   0.152   0.107   1.051   0.238
# truck   0.264   0.850   0.229   0.171   0.994   0.265
# bus     0.359   0.781   0.208   0.144   2.277   0.392
# trailer 0.131   1.092   0.251   0.650   0.604   0.110
# construction_vehicle    0.063   1.033   0.500   1.228   0.151   0.358
# pedestrian      0.418   0.728   0.297   1.024   0.783   0.304
# motorcycle      0.327   0.762   0.263   0.859   1.576   0.134
# bicycle 0.291   0.727   0.267   1.143   0.518   0.032
# traffic_cone    0.515   0.617   0.321   nan     nan     nan
# barrier 0.405   0.820   0.288   0.165   nan     nan

###c5 + cbgs imagenet pretrain
# mAP: 0.3447
# mATE: 0.7436
# mASE: 0.2779
# mAOE: 0.5260
# mAVE: 0.8410
# mAAE: 0.2085
# NDS: 0.4126
# Eval time: 181.4s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.552   0.534   0.154   0.085   0.948   0.206
# truck   0.300   0.790   0.221   0.150   0.745   0.215
# bus     0.365   0.842   0.224   0.115   2.108   0.410
# trailer 0.150   1.116   0.254   0.579   0.523   0.079
# construction_vehicle    0.089   1.002   0.447   1.159   0.129   0.401
# pedestrian      0.378   0.702   0.309   1.050   0.805   0.235
# motorcycle      0.339   0.685   0.267   0.587   1.174   0.112
# bicycle 0.267   0.637   0.276   0.897   0.295   0.010
# traffic_cone    0.529   0.531   0.344   nan     nan     nan
# barrier 0.476   0.598   0.282   0.112   nan     nan

###p4 + fcos pretrain
# mAP: 0.3587
# mATE: 0.7727
# mASE: 0.2716
# mAOE: 0.4346
# mAVE: 0.8520
# mAAE: 0.2125
# NDS: 0.4250
# Eval time: 190.8s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.549   0.554   0.150   0.083   0.939   0.206
# truck   0.317   0.826   0.218   0.117   0.848   0.220
# bus     0.377   0.833   0.204   0.135   1.982   0.390
# trailer 0.168   1.034   0.244   0.467   0.566   0.164
# construction_vehicle    0.092   1.098   0.485   1.034   0.114   0.347
# pedestrian      0.435   0.707   0.292   0.575   0.481   0.211
# motorcycle      0.348   0.728   0.251   0.552   1.431   0.151
# bicycle 0.317   0.677   0.262   0.799   0.455   0.012
# traffic_cone    0.527   0.579   0.327   nan     nan     nan
# barrier 0.456   0.692   0.282   0.148   nan     nan

###900 query
# mAP: 0.3597
# mATE: 0.7647
# mASE: 0.2693
# mAOE: 0.4370
# mAVE: 0.8626
# mAAE: 0.2079
# NDS: 0.4257
# Eval time: 240.9s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.551   0.553   0.150   0.084   0.928   0.209
# truck   0.320   0.808   0.216   0.113   0.841   0.232
# bus     0.392   0.838   0.204   0.120   2.065   0.376
# trailer 0.154   1.057   0.233   0.506   0.609   0.145
# construction_vehicle    0.093   1.041   0.484   1.054   0.134   0.367
# pedestrian      0.434   0.707   0.291   0.579   0.480   0.202
# motorcycle      0.354   0.707   0.251   0.597   1.378   0.118
# bicycle 0.308   0.697   0.263   0.727   0.466   0.013
# traffic_cone    0.533   0.551   0.318   nan     nan     nan
# barrier 0.459   0.688   0.282   0.153   nan     nan

###no dropout
# mAP: 0.3584
# mATE: 0.7739
# mASE: 0.2681
# mAOE: 0.4556
# mAVE: 0.8858
# mAAE: 0.2036
# NDS: 0.4205
# Eval time: 255.7s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.552   0.544   0.150   0.084   0.951   0.209
# truck   0.318   0.809   0.220   0.116   0.885   0.206
# bus     0.384   0.861   0.205   0.122   2.113   0.382
# trailer 0.152   1.111   0.231   0.500   0.514   0.104
# construction_vehicle    0.076   1.073   0.470   1.148   0.128   0.342
# pedestrian      0.429   0.705   0.290   0.579   0.492   0.215
# motorcycle      0.357   0.717   0.249   0.602   1.466   0.159
# bicycle 0.314   0.677   0.262   0.799   0.536   0.013
# traffic_cone    0.535   0.571   0.325   nan     nan     nan
# barrier 0.468   0.670   0.279   0.150   nan     nan


