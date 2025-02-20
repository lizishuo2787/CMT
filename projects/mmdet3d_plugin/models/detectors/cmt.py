# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import mmcv
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply 
#用于对一个函数进行多次应用。这在处理多任务或多尺度数据时非常有用。例如，在目标检测中，可能需要对不同尺度的特征图应用相同的处理逻辑
from mmdet.models import DETECTORS
#注册器，用于注册检测器类。通过使用 @DETECTORS.register_module() 装饰器，可以将自定义的检测器类注册到这个注册器中，从而在配置文件中使用这些类。
from mmdet.models.builder import build_backbone
#根据配置构建骨干网络。这在定义模型时非常有用，因为它允许灵活地选择和配置不同的骨干网络
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
# Box3DMode：定义了 3D 边界框的表示方式，例如 'LiDAR'、'Depth' 或 'Camera'。
# Coord3DMode：定义了 3D 坐标系的表示方式，例如 'LiDAR' 或 'Camera'。
# bbox3d2result，用于将 3D 边界框的预测结果转换为标准格式。这在后处理阶段非常有用，因为它可以将模型的输出格式化为易于理解和使用的格式。
# merge_aug_bboxes_3d，用于合并多尺度或多视角的 3D 边界框预测结果。这在多尺度测试或多视角测试中非常有用，因为它可以提高检测的准确性和鲁棒性。
# show_result，用于可视化 3D 检测结果。这在调试和展示模型性能时非常有用，因为它可以直观地显示检测到的 3D 边界框。
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
# 一个基础类，用于实现多模态（Multi-modality）的两阶段（Two-Stage）3D 目标检测器。这个类提供了多模态数据融合和两阶段检测的框架，支持点云和图像数据的融合

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
# 实现网格掩码（Grid Mask）数据增强。网格掩码是一种数据增强方法，通过在图像上随机生成网格状的掩码，可以提高模型对局部特征的鲁棒性
from projects.mmdet3d_plugin import SPConvVoxelization
# 将点云数据转换为体素网格。这在处理点云数据时非常有用，因为它可以将不规则的点云数据转换为规则的体素网格，从而便于后续的处理和分析

@DETECTORS.register_module()
class CmtDetector(MVXTwoStageDetector): #继承自两阶段检测器MVXTwoStageDetector

    def __init__(self,
                 use_grid_mask=False,
                 **kwargs): #父类的额外参数  
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None) #参数中如果没有pts_voxel_layer，则默认为None
        kwargs['pts_voxel_layer'] = None
        super(CmtDetector, self).__init__(**kwargs) #调用父类的初始化方法
        
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        # [x_flag, y_flag,rotate,offset,ratio网格掩码的比率,mode,prob应用网格掩码的概率]  
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg) #pts_voxel_cfg 中的配置初始化 SPConvVoxelization 对象

    def init_weights(self):
        """Initialize model weights."""
        super(CmtDetector, self).init_weights()

    @auto_fp16(apply_to=('img'), out_fp32=True) 
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:] #[B,N,C,H,W] 取H，W
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape) 
            # 更新每个图像的实际输入形状。这有助于后续处理时了解每个图像的真实尺寸

            if img.dim() == 5 and img.size(0) == 1: # batch_size= 1
                img.squeeze_(0) #[B,N,C,H,W]->[B*N,C,H,W] = [1*N,C,H,W]
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W) #[B,N,C,H,W]->[B*N,C,H,W]
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float()) # backbone network
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            # 检查提取的特征 img_feats 是否是一个字典。
            # 如果是，则将其转换为特征值的列表。这通常是因为不同的骨干网络可能返回不同格式的特征
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats) # neck
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        
        # 点云网络 
        
        #input：
        #       pts: 输入的点云数据形状为[B,N,C] C = [x,y,z,intensity,timestamp_diff] 5
        voxels, num_points, coors = self.voxelize(pts) # 将点云数据转换为体素网格
        #output & next input:
        #       voxels: 体素网格数据，形状为[M,max_points,C] M为体素数量，max_points为每个体素中的允许最大点数，为超参数，C = [x,y,z,intensity,timestamp_diff] 5
        #       num_points: 每个体素中的点的数量，形状为[M,] M为体素数量
        #       coors: 体素中心坐标，形状为[M,3] M为体素数量，3代表体素中心坐标的xyz三个维度        
        
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,)
        #体素特征：体素特征，形状为[M,D] M为体素数量，D为特征维度，与C不同
        
        batch_size = coors[-1, 0] + 1
        #x: (B, C, H, W, D)，其中 B 是批量大小，C 是特征通道数，H、W 和 D 是体素网格的尺寸
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        # 只有B为batch size保持不变，其他均和网络设计相关
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        # 遍历
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        # 合并在第0维度    
        # example：
        # a = torch.tensor([[1, 2], [3, 4]])
        # b = torch.tensor([[5, 6], [7, 8]])
        # result = torch.cat((a, b), dim=0)
        
        # a:
        # [[1, 2],
        #  [3, 4]]

        # b:
        # [[5, 6],
        #  [7, 8]]
        
        # tensor([[1, 2],
        #         [3, 4],
        #         [5, 6],
        #         [7, 8]])
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            # coor: 输入的张量，形状为 (M, 3)，表示 M 个体素的坐标 (x, y, z)。
            # (1, 0): 一个元组，表示在每个维度上填充的宽度。F.pad 的填充宽度是按照 (padding_left, padding_right, padding_top, padding_bottom) 的顺序指定的。在这里，(1, 0) 表示在第 0 维上左边填充 1 个元素，右边填充 0 个元素。
            # example：coor_pad = F.pad(coor, (1, 2, 3, 4), mode='constant', value=i) 左1右2上3下4
            # mode='constant': 填充的模式，'constant' 表示用常数值填充。
            # value=i: 填充的值，这里用的是样本的索引 i。
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        # example：
        # coors_sample1 = torch.tensor([[0, 1, 2], [3, 4, 5]])
        # coors_sample2 = torch.tensor([[6, 7, 8], [9, 10, 11]])
        # coors = [coors_sample1, coors_sample2]
        # coor_pad1 = torch.tensor([[0, 0, 1, 2], [0, 3, 4, 5]])
        # coor_pad2 = torch.tensor([[1, 6, 7, 8], [1, 9, 10, 11]])
        # coors_batch = torch.tensor([
        # [0, 0, 1, 2],
        # [0, 3, 4, 5],
        # [1, 6, 7, 8],
        # [1, 9, 10, 11]
        # ])
        
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats or img_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        return losses

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        for var, name in [(points, 'points'), (img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
    
    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        
        bbox_list = [dict() for i in range(len(img_metas))]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
