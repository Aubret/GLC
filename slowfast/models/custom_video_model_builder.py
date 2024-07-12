#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from torch.nn.init import trunc_normal_
from functools import partial
from fairscale.nn.checkpoint import checkpoint_wrapper

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.attention import MultiScaleBlock, MultiScaleDecoderBlock
from slowfast.models.global_attention import GlobalLocalBlock
from slowfast.models.utils import round_width, validate_checkpoint_wrapper_import
from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

"""A More Flexible Video models."""

# ResNet_Baseline with maxpool: ['2D', '3D_full', '3D_a', '3D_b', '3D_c', '3D_d', '3D_e', '3D_cd']
# ResNet_3D with conv: ['2D_conv', '3D_a_conv', '3D_b_conv', '3D_c_conv', '3D_d_conv', '3D_e_conv', '3D_dev1', '3D_dev2', '3D_de_conv', '3D_full_conv']

model = '3D_full_56'

##############################################################
# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


# @MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x

@MODEL_REGISTRY.register()
class ResNet_Baseline(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet_Baseline, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        # [[[5]], [[3]], [[3, 1]], [[3, 1]], [[1, 3]]] 

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],    # [5] + [7, 7]
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],    # [2, 3, 3]
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )
        if model == '2D':
            # Adding fcn module here:
            # without temporal conv: I3D+2DFCNs
            self.relu    = nn.ReLU(inplace=True) 
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(2, 1, 1))
            # fcn module ends here
        elif model == '3D_full':
            # Adding fcn module here:
            # with all 5 temporal conv: I3D+Full3DFCNs
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            # previous 3d_full: better. 
            # self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            # self.bn5     = nn.BatchNorm3d(64)
            # self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            # self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(1, 1, 1))
            # maxpool-cla sequence
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            # self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(3, 1, 1))  # make dense prediction in temporal dim
            # self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.classifier = nn.Conv3d(64, 1, kernel_size=1)
            # try this
            # self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            # self.bn5     = nn.BatchNorm3d(64)
            # self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            # self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(2, 1, 1))
            # fcn module ends here

        elif model == '3D_full_56':  # set the output size as 56x56
            # Adding fcn module here:
            # with all 5 temporal conv: I3D+Full3DFCNs
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1), bias=True)
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1), bias=True)
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1), bias=True)
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0), bias=True)
            self.bn4     = nn.BatchNorm3d(64)
            # self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            # self.bn5     = nn.BatchNorm3d(64)
            # self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(3, 1, 1))  # make dense prediction in temporal dim
            # self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.classifier = nn.Conv3d(64, 1, kernel_size=1)

        elif model == '3D_a':
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(2, 1, 1))
            # fcn module ends here
        elif model == '3D_b':
            # Adding fcn module here:
            # with 1 temporal conv: I3D+3DFCNs (b)
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(1, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(2, 1, 1))
            # fcn module ends here
        elif model == '3D_c':
            # Adding fcn module here:
            # with 1 temporal conv: I3D+3DFCNs (c)
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(1, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(2, 1, 1))
            # fcn module ends here
        elif model == '3D_d':
            # Adding fcn module here:
            # with 1 temporal conv: I3D+3DFCNs (d)
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(2, 1, 1))
            # fcn module ends here
        elif model == '3D_e':
            # Joint fi-m-la with 3D (e):
            # with 1 temporal conv bottom: I3D+3DFCNs (e)
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3 = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4 = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(64)
            self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(2, 1, 1))
            # fcn module ends here
        elif model == '3D_cd':
            # Adding fcn module here:
            # with 1 temporal conv: I3D+3DFCNs (cd)
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn3 = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4 = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(64)
            self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(2, 1, 1))
            # fcn module ends here
        elif model == '3D_de':
            # Adding fcn module here:
            # with 1 temporal conv: I3D+3DFCNs (cd)
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3 = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4 = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(64)
            self.classifier = nn.Conv3d(64, 2, kernel_size=1)
            self.maxpool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(2, 1, 1))
            # fcn module ends here

        # torch.Size([1, 2048, 4, 7, 7])        x5
        # torch.Size([1, 1024, 4, 14, 14])      x4
        # torch.Size([1, 512, 4, 28, 28])       x3
        # torch.Size([1, 256, 4, 56, 56])       x2
        # torch.Size([1, 64, 8, 56, 56])        x1
        # torch.Size([1, 2, 1, 224, 224])       return

    def forward(self, x, bboxes=None):
        # torch.Size([1, 3, 8, 224, 224])
        x1 = self.s1(x)  # torch.Size([1, 64, 8, 56, 56])
        x2 = self.s2(x1)   # torch.Size([1, 256, 8, 56, 56])
        pool = getattr(self, "pathway{}_pool".format(0))
        x2[0] = pool(x2[0])  # torch.Size([1, 256, 4, 56, 56])
        x3 = self.s3(x2)  # torch.Size([1, 512, 4, 28, 28])
        x4 = self.s4(x3)  # torch.Size([1, 1024, 4, 14, 14])
        x5 = self.s5(x4)  # torch.Size([1, 2048, 4, 7, 7])

        # print(x5[0].shape)                                # torch.Size([1, 2048, 4, 7, 7])        x5
        score = self.bn1(self.relu(self.deconv1(x5[0])))    # torch.Size([1, 1024, 4, 14, 14])      x4
        score = score + x4[0]

        score = self.bn2(self.relu(self.deconv2(score)))    # torch.Size([1, 512, 4, 28, 28])       x3
        score = score + x3[0]

        score = self.bn3(self.relu(self.deconv3(score)))    # torch.Size([1, 256, 4, 56, 56])       x2
        score = score + x2[0]

        score = self.bn4(self.relu(self.deconv4(score)))    # torch.Size([1, 64, 8, 56, 56])        x1
        score = score + x1[0]

        # score = self.bn5(self.relu(self.deconv5(score)))    # torch.Size([1, 64, 12, 224, 224])

        # score = self.maxpool(score)
        # score = functional.adaptive_max_pool3d(score, (x[0].size(2), score.size(3), score.size(4)))

        score = self.classifier(score)                      # torch.Size([1, 1, 8, 56, 56])

        return score

# torch.Size([1, 3, 8, 224, 224])
# torch.Size([1, 64, 8, 56, 56])        x1
# torch.Size([1, 256, 4, 56, 56])       x2
# torch.Size([1, 512, 4, 28, 28])       x3
# torch.Size([1, 1024, 4, 14, 14])      x4
# torch.Size([1, 2048, 4, 7, 7])        x5
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# torch.Size([1, 2048, 4, 7, 7])        x5
# torch.Size([1, 1024, 4, 14, 14])      x4
# torch.Size([1, 512, 4, 28, 28])       x3
# torch.Size([1, 256, 4, 56, 56])       x2
# torch.Size([1, 64, 8, 56, 56])        x1
######################################
# torch.Size([1, 64, 8, 56, 56])
# torch.Size([1, 64, 8, 224, 224])
# torch.Size([1, 2, 8, 224, 224])
# torch.Size([1, 2, 3, 224, 224])       return

@MODEL_REGISTRY.register()
class ResNet_3D(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet_3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        # [[[5]], [[3]], [[3, 1]], [[3, 1]], [[1, 3]]] 

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],    # [5] + [7, 7]
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],    # [2, 3, 3]
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if model == '2D_conv':
            # Adding fcn module here:
            # without temporal conv: I3D+2DFCNs
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            
            self.deconv6 = nn.Conv3d(64, 32, kernel_size=(4, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn7 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            
            # fcn module ends here
        elif model == '3D_a_conv':
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            self.deconv6 = nn.Conv3d(64, 32, kernel_size=(4, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn7 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            # fcn module ends here
        elif model == '3D_b_conv':
            # Adding fcn module here:
            # with 1 temporal conv: I3D+3DFCNs (b)
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(1, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            self.deconv6 = nn.Conv3d(64, 32, kernel_size=(4, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn7 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            # fcn module ends here
        elif model == '3D_c_conv':
            # full 3d + 2 linear
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(1, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(64)
            self.deconv6 = nn.Conv3d(64, 32, kernel_size=(4, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn7 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            # fcn module ends here
        elif model == '3D_d_conv':
            # Adding fcn module here:
            # with 1 temporal conv: I3D+3DFCNs (d)
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)
            # 3D_d_conv-original
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5     = nn.BatchNorm3d(64)
            self.deconv6 = nn.Conv3d(64, 32, kernel_size=(4, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn7 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)


        elif model == '3D_e_conv':
            # 2d + temporal conv at e + 2 linear
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                              dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                              dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3 = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1),
                                              dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4 = nn.BatchNorm3d(64)
            # adding more layers here
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(64)
            self.deconv6 = nn.Conv3d(64, 32, kernel_size=(4, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn7 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            # fcn module ends here
        elif model == '3D_cd_conv':
            # Adding fcn module here:
            # with 1 temporal conv: I3D+3DFCNs (cd)
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn3 = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4 = nn.BatchNorm3d(64)
            self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(64)
            self.deconv6 = nn.Conv3d(64, 32, kernel_size=(4, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn7 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            # fcn module ends here
        elif model == '3D_dev1':
            # 2d + temporal conv 112-224
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                              dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                              dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3 = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1),
                                              dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4 = nn.BatchNorm3d(64)
            # adding more layers here
            self.deconv5 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(32)
            self.deconv6 = nn.ConvTranspose3d(32, 32, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 0, 0))
            self.bn7 = nn.BatchNorm3d(16)
            self.deconv8 = nn.ConvTranspose3d(16, 16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 0, 0))
            self.bn8 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            # fcn module ends here
        elif model == '3D_dev2':
            # 2d + temporal conv 112-224 + 2 linear
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                              dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                              dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3 = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(1, 3, 3), stride=(3, 1, 1), padding=(1, 1, 1),
                                              dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4 = nn.BatchNorm3d(64)
            # adding more layers here
            self.deconv5 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(32)
            self.deconv6 = nn.ConvTranspose3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                              dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7 = nn.Conv3d(32, 32, kernel_size=(4, 1, 1))
            self.bn7 = nn.BatchNorm3d(32)
            self.deconv8 = nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn8 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            # fcn module ends here
        elif model == '3D_de_conv':
            # 2d + temporal conv at d + 2 linear
            self.relu = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2 = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
            self.bn3 = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4 = nn.BatchNorm3d(64)
            # adding more layers here
            self.deconv5 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(32)
            self.deconv6 = nn.Conv3d(32, 32, kernel_size=(4, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn7 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            # fcn module ends here
        elif model == '3D_full_conv':
            # full 3d + 2 linear
            self.relu    = nn.ReLU(inplace=True)
            self.deconv1 = nn.ConvTranspose3d(2048, 1024, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn1     = nn.BatchNorm3d(1024)
            self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn2     = nn.BatchNorm3d(512)
            self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn3     = nn.BatchNorm3d(256)
            self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
            self.bn4     = nn.BatchNorm3d(64)

            # adding more layers here
            self.deconv5 = nn.ConvTranspose3d(64, 32, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
            self.bn5 = nn.BatchNorm3d(32)
            self.deconv6 = nn.Conv3d(32, 32, kernel_size=(4, 1, 1))
            self.bn6 = nn.BatchNorm3d(32)
            self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
            self.bn7 = nn.BatchNorm3d(16)
            self.classifier = nn.Conv3d(16, 2, kernel_size=1)
            # fcn module ends here

        # torch.Size([1, 2048, 4, 7, 7])        x5
        # torch.Size([1, 1024, 4, 14, 14])      x4
        # torch.Size([1, 512, 4, 28, 28])       x3
        # torch.Size([1, 256, 4, 56, 56])       x2
        # torch.Size([1, 64, 8, 56, 56])        x1
        # torch.Size([1, 2, 1, 224, 224])       return


    def forward(self, x, bboxes=None):
        # torch.Size([1, 3, 8, 224, 224])
        x1 = self.s1(x)  # torch.Size([1, 64, 8, 56, 56])
        x2 = self.s2(x1)   # torch.Size([1, 256, 8, 56, 56])
        # for pathway in range(self.num_pathways):
        #     pool = getattr(self, "pathway{}_pool".format(pathway))
        #     x[pathway] = pool(x[pathway])
        pool = getattr(self, "pathway{}_pool".format(0))
        x2[0] = pool(x2[0]) # torch.Size([1, 256, 4, 56, 56])
        x3 = self.s3(x2)  # torch.Size([1, 512, 4, 28, 28])
        x4 = self.s4(x3)  # torch.Size([1, 1024, 4, 14, 14])
        x5 = self.s5(x4)  # torch.Size([1, 2048, 4, 7, 7])

        #############################################
        #    FCN decoder part starting from here    #
        #############################################
        # FCN decoder here
        # print(x5[0].shape)                                # torch.Size([1, 2048, 4, 7, 7])        x5
        score = self.bn1(self.relu(self.deconv1(x5[0])))    # torch.Size([1, 1024, 4, 14, 14])      x4
        score = score + x4[0]
        # print(score.shape)
        score = self.bn2(self.relu(self.deconv2(score)))    # torch.Size([1, 512, 4, 28, 28])       x3
        score = score + x3[0]
        # print(score.shape)
        score = self.bn3(self.relu(self.deconv3(score)))    # torch.Size([1, 256, 4, 56, 56])       x2
        score = score + x2[0]
        # print(score.shape)
        score = self.bn4(self.relu(self.deconv4(score)))    # torch.Size([1, 64, 8, 56, 56])        x1
        score = score + x1[0]
        # print(score.shape)
        # print('######################################')
        # fi-m-la 3D_full_conv/3D_a_conv/3D_b_conv/3D_c_conv/3D_e_conv/3D_de_conv here:
        score = self.bn5(self.relu(self.deconv5(score)))  # torch.Size([1, 64, 8, 224, 224])
        # print(score.shape)
        score = self.bn6(self.relu(self.deconv6(score)))  # torch.Size([1, 32, 5, 224, 224])
        # print(score.shape)
        score = self.bn7(self.relu(self.deconv7(score)))  # torch.Size([1, 16, 3, 224, 224])
        # print(score.shape)
        score = self.classifier(score)  # torch.Size([1, 2, 3, 224, 224])
        # print(score.shape)

        return score  # size=(N, n_class, x.H/1, x.W/1)

# torch.Size([1, 3, 8, 224, 224])
# torch.Size([1, 64, 8, 56, 56])        x1
# torch.Size([1, 256, 4, 56, 56])       x2
# torch.Size([1, 512, 4, 28, 28])       x3
# torch.Size([1, 1024, 4, 14, 14])      x4
# torch.Size([1, 2048, 4, 7, 7])        x5
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# torch.Size([1, 2048, 4, 7, 7])        x5
# torch.Size([1, 1024, 4, 14, 14])      x4
# torch.Size([1, 512, 4, 28, 28])       x3
# torch.Size([1, 256, 4, 56, 56])       x2
# torch.Size([1, 64, 8, 56, 56])        x1
# torch.Size([1, 2, 1, 224, 224])       return


# @MODEL_REGISTRY.register()
class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            self._round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = self._round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = self._round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = self._round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _round_width(self, width, multiplier, min_depth=8, divisor=8):
        """Round width of filters based on width multiplier."""
        if not multiplier:
            return width

        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(
            min_depth, int(width + divisor / 2) // divisor * divisor
        )
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = self._round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = self._round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if cfg.X3D.CHANNELWISE_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            NotImplementedError
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x


@MODEL_REGISTRY.register()
class GLC_Gaze(nn.Module):
    """
    Multiscale Vision Transformers with Global-Local Correlation for Egocentric Gaze Estimation
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST  # False

        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        # temporal_size = cfg.DATA.NUM_FRAMES  # 8
        temporal_size = 8
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D  # default false
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride

        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM  # 96

        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS  # True
        self.drop_rate = cfg.MVIT.DROPOUT_RATE  # 0
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE  # 0.2
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.global_embed_on = cfg.MVIT.GLOBAL_EMBED_ON
        self.global_embed_num = 1
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        # Input embedding
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [self.input_dims[i] // self.patch_stride[i] for i in range(len(self.input_dims))]
        num_patches = math.prod(self.patch_dims)
        # Global embedding
        if self.global_embed_on:
            self.global_embed = stem_helper.CasGlobalEmbed(
                dim_in=embed_dim,
                dim_embed=embed_dim,
                conv_2d=use_2d_patch,
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # size (1, 1, 96)
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_dims[0], embed_dim))
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        self.stage_end = []  # save the number of blocks before downsampling

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
            self.stage_end.append(cfg.MVIT.POOL_Q_STRIDE[i][0]-1)
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:  # if there's a stride in q
                    _stride_kv = [max(_stride_kv[d] // stride_q[i][d], 1) for d in range(len(_stride_kv))]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [s + 1 if s > 1 else s for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]]
        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None  # None

        self.blocks = nn.ModuleList()

        if cfg.MODEL.ACT_CHECKPOINT:  # False
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(embed_dim, dim_mul[i + 1], divisor=round_width(num_heads, head_mul[i + 1]),)
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                has_global_embed=self.global_embed_on,
                global_embed_num=self.global_embed_num,
                pool_first=pool_first,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

        self.global_fuse = GlobalLocalBlock(
            dim=self.blocks[15].dim_out,
            dim_out=self.blocks[15].dim_out,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=self.drop_rate,
            drop_path=0.4,
            norm_layer=norm_layer,
            kernel_q=[3, 3, 3],
            kernel_kv=[3, 3, 3],
            stride_q=[1, 1, 1],
            stride_kv=[1, 1, 1],
            mode=mode,
            has_cls_embed=self.cls_embed_on,
            has_global_embed=self.global_embed_on,
            global_embed_num=self.global_embed_num,
            pool_first=pool_first
        )

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim * 2)

        # TransDecoder
        decode_dim_in = [768*2, 768, 384, 192]
        decode_dim_out = [768, 384, 192, 96]
        decode_num_heads = [8, 4, 4, 2]
        decode_kernel_q = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        decode_kernel_kv = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        decode_stride_q = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 1, 1]]  # upsample stride
        decode_stride_kv = [[1, 2, 2], [1, 4, 4], [1, 8, 8], [1, 16, 16]]
        for i in range(len(decode_dim_in)):
            decoder_block = MultiScaleDecoderBlock(
                dim=decode_dim_in[i],
                dim_out=decode_dim_out[i],
                num_heads=decode_num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=0,
                norm_layer=norm_layer,
                kernel_q=decode_kernel_q[i] if len(decode_kernel_q) > i else [],
                kernel_kv=decode_kernel_kv[i] if len(decode_kernel_kv) > i else [],
                stride_q=decode_stride_q[i] if len(decode_stride_q) > i else [],
                stride_kv=decode_stride_kv[i] if len(decode_stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                has_global_embed=self.global_embed_on,
                global_embed_num=self.global_embed_num,
                pool_first=pool_first,
            )

            setattr(self, f'decode_block{i+1}', decoder_block)

        self.classifier = nn.Conv3d(96, 1, kernel_size=1)

        # Initialization
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {"pos_embed_spatial", "pos_embed_temporal", "pos_embed_class", "cls_token"}
                else:
                    return {"pos_embed_spatial", "pos_embed_temporal", "pos_embed_class"}
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x, return_glc=False):
        inpt = x[0]  # size (B, 3, 8, 256, 256)
        x = self.patch_embed(inpt)  # size (B, 16384, 96)  16384 = 4*64*64

        # T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]  # 4
        T = 8 // self.patch_stride[0]  # 4
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]  # 64
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]  # 64
        B, N, C = x.shape  # B, 16384, 96

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.global_embed_on:
            # share the first conv with patch embedding, followed by multi-conv (best now)
            global_tokens = x.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
            global_tokens = self.global_embed(global_tokens)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(1, self.patch_dims[0], 1) \
                        + torch.repeat_interleave(self.pos_embed_temporal, self.patch_dims[1] * self.patch_dims[2], dim=1)
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.global_embed_on:
            x = torch.cat((global_tokens, x), dim=1)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        inter_feat = [[x, thw]]  # record features to be integrated in decoder
        for i, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)

            if i in self.stage_end:
                inter_feat.append([x, thw])

            if i == 15:
                if not return_glc:
                    x_fuse, thw = self.global_fuse(x, thw)
                else:
                    x_fuse, thw, glc = self.global_fuse(x, thw, return_glc=True)
                x = torch.cat([x, x_fuse], dim=2)

        x = self.norm(x)  # x size [B, 256, 768]

        # Decoder (Transformer)
        feat, thw = self.decode_block1(x, thw)  # (B, 1024, 768)  1024 = 4*16*16
        feat = feat + inter_feat[-1][0]

        feat, thw = self.decode_block2(feat, thw)  # (B, 4096, 384)  4096 = 4*32*32
        feat = feat + inter_feat[-2][0]

        feat, thw = self.decode_block3(feat, thw)  # (B, 16384, 192)  16384 = 4*64*64
        feat = feat + inter_feat[-3][0]

        feat, thw = self.decode_block4(feat, thw)  # (B, 32768, 96)  16384 = 8*64*64
        if self.global_embed_on:
            feat = feat[:, self.global_embed_num:, :]
        feat = feat.reshape(feat.size(0), *thw, feat.size(2)).permute(0, 4, 1, 2, 3)
        en_feat, thw = inter_feat[-4]
        if self.global_embed_on:
            en_feat = en_feat[:, self.global_embed_num:, :]
        en_feat = en_feat.reshape(en_feat.size(0), *thw, en_feat.size(2)).permute(0, 4, 1, 2, 3)
        feat = feat + F.interpolate(en_feat, size=(thw[0]*2, thw[1], thw[2]), mode='trilinear')

        feat = self.classifier(feat)

        if not return_glc:
            return feat
        else:
            return [feat, glc]


@MODEL_REGISTRY.register()
class GLC_Action(nn.Module):
    """
    Action recognition with Global-Local Correlation
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST  # False

        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES  # 8
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D  # default false
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride

        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM  # 96

        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS  # True
        self.drop_rate = cfg.MVIT.DROPOUT_RATE  # 0
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE  # 0.2
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.global_embed_on = cfg.MVIT.GLOBAL_EMBED_ON
        self.global_embed_num = 1
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        # Input embedding
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [self.input_dims[i] // self.patch_stride[i] for i in range(len(self.input_dims))]
        num_patches = math.prod(self.patch_dims)
        # Global embedding
        if self.global_embed_on:
            # Use multiple conv after patch embedding
            self.global_embed = stem_helper.CasGlobalEmbed(
                dim_in=embed_dim,
                dim_embed=embed_dim,
                conv_2d=use_2d_patch,
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # size (1, 1, 96)
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.patch_dims[1] * self.patch_dims[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.patch_dims[0], embed_dim))
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        self.stage_end = []  # save the number of blocks before downsampling

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
            self.stage_end.append(cfg.MVIT.POOL_Q_STRIDE[i][0]-1)
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:  # if there's a stride in q
                    _stride_kv = [max(_stride_kv[d] // stride_q[i][d], 1) for d in range(len(_stride_kv))]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [s + 1 if s > 1 else s for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]]
        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None  # None

        self.blocks = nn.ModuleList()

        if cfg.MODEL.ACT_CHECKPOINT:  # False
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(embed_dim, dim_mul[i + 1], divisor=round_width(num_heads, head_mul[i + 1]),)
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                has_global_embed=self.global_embed_on,
                global_embed_num=self.global_embed_num,
                pool_first=pool_first,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

        self.global_fuse = GlobalLocalBlock(
            dim=self.blocks[15].dim_out,
            dim_out=self.blocks[15].dim_out,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=self.drop_rate,
            drop_path=0.25,
            norm_layer=norm_layer,
            kernel_q=[3, 3, 3],
            kernel_kv=[3, 3, 3],
            stride_q=[1, 1, 1],
            stride_kv=[1, 1, 1],
            mode=mode,
            has_cls_embed=self.cls_embed_on,
            has_global_embed=self.global_embed_on,
            global_embed_num=self.global_embed_num,
            pool_first=pool_first
        )

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim * 2)

        # Classification Head
        self.head = head_helper.TransformerBasicHead(
            embed_dim * 2,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

        # Initialization
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {"pos_embed_spatial", "pos_embed_temporal", "pos_embed_class", "cls_token"}
                else:
                    return {"pos_embed_spatial", "pos_embed_temporal", "pos_embed_class"}
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x, return_glc=False):
        inpt = x[0]  # size (B, 3, 8, 256, 256)
        x = self.patch_embed(inpt)  # size (B, 16384, 96)  16384 = 4*64*64

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]  # 4
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]  # 64
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]  # 64
        B, N, C = x.shape  # B, 16384, 96

        if self.global_embed_on:
            # share the first conv with patch embedding, followed by multi-conv (best now)
            global_tokens = x.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
            global_tokens = self.global_embed(global_tokens)

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(1, self.patch_dims[0], 1) \
                        + torch.repeat_interleave(self.pos_embed_temporal, self.patch_dims[1] * self.patch_dims[2], dim=1)
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.global_embed_on:
            x = torch.cat((global_tokens, x), dim=1)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for i, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)

            if i == 15:
                if not return_glc:
                    x_fuse, thw = self.global_fuse(x, thw)
                else:
                    x_fuse, thw, glc = self.global_fuse(x, thw, return_glc=True)
                x = torch.cat([x, x_fuse], dim=2)

        x = self.norm(x)  # x size [B, 256, 768]

        # Classifier
        if self.global_embed_on:
            x = x[:, self.global_embed_num:, :]
        if self.cls_embed_on:
            x = x[:, 0]  # size [B, 768]
        else:
            x = x.mean(1)
        x = self.head(x)

        if not return_glc:
            return x
        else:
            return [x, glc]
