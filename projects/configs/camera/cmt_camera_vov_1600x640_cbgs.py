plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]  # [x_min, y_min, z_min, x_max, y_max, z_max]
class_names = [ 
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2] # 体素大小
out_size_factor = 8 #控制feature map的大小，缩放因子，默认为8
evaluation = dict(interval=20) #每20个epoch评估一次
dataset_type = 'CustomNuScenesDataset' 
# 自定义数据集类：CustomNuScenesDataset 是在标准 NuScenesDataset 的基础上进行扩展的自定义数据集类。它通常用于处理特定任务或特定数据格式的需求。#
# 功能：除了继承 NuScenesDataset 的基本功能外，CustomNuScenesDataset 可以根据需要进行自定义扩展，例如添加新的数据预处理步骤、支持新的数据格式或调整数据增强方法。

data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False, #TODO：新增radar链路
    use_map=False,
    use_external=False)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
# 预训练模型的均值和标准差，用于图像归一化处理
    
ida_aug_conf = {# 图像增强配置
        "resize_lim": (0.94, 1.25), #图像缩放范围
        "final_dim": (640, 1600), #最终图像尺寸
        "bot_pct_lim": (0.0, 0.0),#图像下边缘裁剪百分比范围
        "rot_lim": (0.0, 0.0),#图像旋转角度范围
        "H": 900,#目标高度
        "W": 1600,#目标宽度
        "rand_flip": True,#是否随机翻转图像
    }

train_pipeline = [ #数据处理
    dict(type='LoadMultiViewImageFromFiles'), #加载多视图图像
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),#加载3D标注
    dict(type='GlobalRotScaleTransImage', #用于数据增强的全局旋转、缩放和平移
            rot_range=[-0.3925, 0.3925], # ±22.5°
            translation_std=[0, 0, 0],#[x,y,z]平移标准差
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True #仅在训练时使用
            ),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range), #过滤超出范围的点云
    dict(type='ObjectNameFilter', classes=class_names), #保留class_names中的类别
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),#数据增强，只在训练时使用
    dict(type='NormalizeMultiviewImage', **img_norm_cfg), #对每个像素进行处理 pixel = (pixel - mean) / std
    dict(type='PadMultiViewImage', size_divisor=32), #将图像填充到32的倍数
    dict(type='DefaultFormatBundle3D', class_names=class_names), 
    #图像数据格式化：将图像数据转换为张量（Tensor）
    #3D 边界框格式化：将 3D 边界框数据转换为张量
    #类别标签格式化：将类别标签转换为张量
    
   
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'], #[处理后的图像，3d bboxes坐标, 类型标签]
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d'))
]
test_pipeline = [ 
    # 测试阶段的数据处理流程
    #一般不包含复杂数据增强，只对图像进行缩放和平移操作
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
            #非训练阶段，使用相同的数据增强配置，没有区别,TODO: 这里删除或者改变测试阶段的数据增强配置
            dict(type='NormalizeMultiviewImage', **img_norm_cfg), #**用于解包字典
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False), #预测标签
            dict(type='Collect3D', keys=['img']) #只收集原始图像数据，不包含3D标注信息
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type='CBGSDataset', #主要功能是通过类别平衡的采样方法，确保每个类别在训练过程中被均匀地采样
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/nuscenes_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')), #表示使用 LiDAR 坐标系来表示 3D 边界框
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='CmtDetector',
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='CmtImageHead',
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
         tasks=[
            dict(num_class=10, class_names=[
                'car', 'truck', 'construction_vehicle',
                'bus', 'trailer', 'barrier',
                'motorcycle', 'bicycle',
                'pedestrian', 'traffic_cone'
            ]),
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='CmtImageTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),

                    feedforward_channels=1024, #unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                # cls_cost=dict(type='ClassificationCost', weight=2.0),
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200
        )))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(
    type='CustomFp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
)

lr_config = dict(
    policy='cyclic',
    target_ratio=(8, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from='ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
