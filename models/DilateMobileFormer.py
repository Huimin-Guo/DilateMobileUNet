import torch
import torch.nn as nn
from einops import rearrange


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation,
                                dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1, H*W]
                      ).permute(0, 1, 4, 3, 2)
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size *
                                    self.kernel_size, H*W]).permute(0, 1, 4, 2, 3)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size *
                                    self.kernel_size, H*W]).permute(0, 1, 4, 3, 2)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C //
                                  self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        x = x.reshape(B, self.num_dilation, C//self.num_dilation,
                      H, W).permute(1, 0, 3, 4, 2).clone()
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](
                qkv[i][0], qkv[i][1], qkv[i][2])
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiDilatelocalAttention(dim=dim, num_heads=num_heads, qkv_bias=True,
                        qk_scale=None, attn_drop=dropout, kernel_size=3, dilation=[1, 2, 3])),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DilateMobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, mlp_dim, num_heads=6, dropout=0.):
        super().__init__()

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, num_heads, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        x = x.permute(0, 2, 3, 1)
        x = self.transformer(x)
        x = x.permute(0, 3, 1, 2)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class DilateMobileFormer(nn.Module):
    def __init__(self, in_channels, dims, channels, expansion=4, kernel_size=3):
        super().__init__()
        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(in_channels, channels[0], stride=2)

        self.mv2_0 = MV2Block(channels[0], channels[1], 1, expansion)
        self.mv2_1 = MV2Block(channels[1], channels[2], 2, expansion)
        self.mv2_2 = MV2Block(channels[2], channels[3], 1, expansion)
        self.mv2_3 = MV2Block(channels[2], channels[3], 1, expansion)  # Repeat
        self.mv2_4 = MV2Block(channels[3], channels[4], 2, expansion)
        self.mv2_5 = MV2Block(channels[5], channels[6], 2, expansion)
        self.mv2_6 = MV2Block(channels[7], channels[8], 2, expansion)

        self.mvit_0 = DilateMobileViTBlock(
            dims[0], L[0], channels[5], kernel_size, int(dims[0]*2))
        self.mvit_1 = DilateMobileViTBlock(
            dims[1], L[1], channels[7], kernel_size, int(dims[0]*2))
        self.mvit_2 = DilateMobileViTBlock(
            dims[2], L[2], channels[9], kernel_size, int(dims[0]*2))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

    def forward(self, x):
        features = []
        x = self.conv1(x)
        features.append(x)  # down-sample //2
        x = self.mv2_0(x)

        x = self.mv2_1(x)
        features.append(x)  # down-sample //4
        x = self.mv2_2(x)
        x = self.mv2_3(x)  # Repeat

        x = self.mv2_4(x)
        features.append(x)  # down-sample //8
        x = self.mvit_0(x)

        x = self.mv2_5(x)
        features.append(x)  # down-sample //16
        x = self.mvit_1(x)

        x = self.mv2_6(x)
        x = self.mvit_2(x)
        x = self.conv2(x)  # //32

        return x, features
