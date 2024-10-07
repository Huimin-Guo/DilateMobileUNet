# -*- coding: utf-8 -*-
# @Author  : Guo Huimin

from typing import Sequence, Tuple, Union

import torch
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from torch import nn

from models.DilateMobileFormer import DilateMobileFormer


class DilateMobileVitUnet(nn.Module):
    def __init__(self,
                 sets,
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 norm_name: Union[Tuple, str] = "instance",
                 res_block: bool = True,
                 ):
        super(DilateMobileVitUnet, self).__init__()

        if sets.model_type == "s":
            self.expansion = 4
            self.dims = [144, 192, 240]
            self.channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        elif sets.model_type == "xs":
            self.expansion = 4
            self.dims = [96, 120, 144]
            self.channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
        elif sets.model_type == "xxs":
            self.expansion = 2
            self.dims = [66, 72, 96]
            self.channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
        else:
            raise NotImplementedError

        self.mobile_vit = DilateMobileFormer(in_channels=in_channels, dims=self.dims,
                                             channels=self.channels, kernel_size=3, expansion=self.expansion)

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.channels[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.channels[10],
            out_channels=self.channels[6],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.channels[6],
            out_channels=self.channels[4],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.channels[4],
            out_channels=self.channels[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.channels[2],
            out_channels=self.channels[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.channels[0],
            out_channels=self.channels[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=self.channels[0], out_channels=out_channels)

    def forward(self, x_in):
        base, hidden_features = self.mobile_vit(x_in)
        x2 = hidden_features[0]
        x3 = hidden_features[1]
        x4 = hidden_features[2]
        x5 = hidden_features[3]

        enc1 = self.encoder1(x_in)
        dec4 = self.decoder5(base, x5)
        dec3 = self.decoder4(dec4, x4)
        dec2 = self.decoder3(dec3, x3)
        dec1 = self.decoder2(dec2, x2)
        out = self.decoder1(dec1, enc1)
        return self.out(out)


if __name__ == '__main__':
    image_size = (64, 64)
    net = DilateMobileVitUnet(
        spatial_dims=2, in_channels=3, out_channels=2).cuda()

    image = torch.randn(2, 3, 64, 64).cuda()
    pred = net(image)
    print(pred.shape)
    pass
