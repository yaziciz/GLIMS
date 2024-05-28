import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Sequence, Tuple, Type, Union

from Modules.conv_generator import get_conv_layer
from monai.utils import optional_import
rearrange, _ = optional_import("einops", name="rearrange")

class GroupConv(nn.Module):
    def __init__(self, in_c, out_c, kernel = 3, stride = 1, padding = 1, dilation = 1, spatial_dims = 3, act = True):
        super(GroupConv, self).__init__()

        self.spatial_dims = spatial_dims
        self.in_c = in_c
        self.out_c = out_c
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.act = act

        self.conv = nn.Conv3d(in_c, in_c, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, groups=in_c)
        self.norm = nn.GroupNorm(in_c, in_c)

        if(act):
            self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if(self.act):
            out = self.prelu(out)
        return out

class DACB(nn.Module):
    def __init__(self, in_c, emb_dim, kernel = 3, spatial_dims = 3, norm_name: Union[Tuple, str] = "instance"):
        super(DACB, self).__init__()

        self.spatial_dims = spatial_dims
        self.in_c = in_c
        self.emb_dim = emb_dim
        self.norm_name = norm_name

        #Depthwise Atrious Convolution
        self.conv_3x3 = GroupConv(in_c, in_c, kernel, stride=1, padding=1, dilation=1)
        self.conv_3x3_dil = GroupConv(in_c, in_c, kernel, stride = 1, padding = 2, dilation = 2)
        self.conv_3x3_dil2 = GroupConv(in_c, in_c, kernel, stride = 1, padding = 3, dilation = 3)

        #Pointwise Convolution
        self.conv_1x1 = get_conv_layer(spatial_dims, in_c * 3, emb_dim, kernel_size=1, stride=1, padding=0, norm=norm_name, conv_only=False)
        self.conv_1x1_2 = get_conv_layer(spatial_dims, emb_dim, in_c, kernel_size=1, stride=1, padding=0, norm=norm_name, conv_only=False)

        self.prelu = nn.PReLU()

        self.norm2 = nn.InstanceNorm3d(in_c)
        self.prelu2 = nn.PReLU()
        
    def forward(self, x):
        out = self.conv_3x3(x) #dilated? larger kernel?
        out2 = self.conv_3x3_dil(x)
        out3 = self.conv_3x3_dil2(x)

        out = torch.cat((out, out2, out3), dim=1)
        out = self.conv_1x1(out)
        out = self.conv_1x1_2(out)

        out = out + x
        out = self.norm2(out)
        out = self.prelu2(out)
        return out
    
class DADB(nn.Module):
    def __init__(self, in_c, emb_dim, kernel, out_c, spatial_dims = 3, norm_name: Union[Tuple, str] = "instance"):
        super(DADB, self).__init__()

        self.spatial_dims = spatial_dims
        self.in_c = in_c
        self.out_c = out_c
        self.norm_name = norm_name

        self.conv_3x3 = get_conv_layer(spatial_dims, in_c, out_c, kernel_size=2, stride=2, padding=0, conv_only=False) #2 as conv? or 3?
        
    def forward(self, x):
        out = self.conv_3x3(x)

        return out
    
class DMSF(nn.Module):
    def __init__(self, in_channels, embed_dim, out_c, kernel, spatial_dims = 3, norm_name: Union[Tuple, str] = "instance"):
        super(DMSF, self).__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_c = out_c
        self.kernel = kernel
        self.spatial_dims = spatial_dims
        self.norm_name = norm_name

        
        self.l1 = DACB(in_channels, embed_dim, kernel, spatial_dims, norm_name)

        self.l2 = DACB(in_channels, embed_dim, kernel, spatial_dims, norm_name)
        self.ds = DADB(in_channels, embed_dim, kernel, out_c, spatial_dims, norm_name)

    def forward(self, x):
        skip = self.l1(x)
        skip = self.l2(skip) #ommit if attn or not?

        out = self.ds(skip)
        return out, skip
    
class DAUB(nn.Module):
    def __init__(self, in_c, emb_dim, out_c, kernel = 3, spatial_dims = 3, norm_name: Union[Tuple, str] = "instance"):
        super(DAUB, self).__init__()

        self.spatial_dims = spatial_dims
        self.in_c = in_c
        self.emb_dim = emb_dim
        self.out_c = out_c
        self.norm_name = norm_name

        #Depthwise Atrious Convolution, stride 2, transpose
        self.convT = get_conv_layer(spatial_dims, in_c, out_c, kernel_size=2, stride=2, output_padding=0, padding=0, norm=norm_name, is_transposed=True, conv_only=False)
        
    def forward(self, x):
        out = self.convT(x)

        return out
    
class DMSU(nn.Module):
    def __init__(self, in_c, emb_dim, out_c, kernel = 5, spatial_dims = 3, depths: Sequence[int] = (2, 2), norm_name: Union[Tuple, str] = "instance"):
        super(DMSU, self).__init__()

        self.in_c = in_c
        self.emb_dim = emb_dim
        self.out_c = out_c
        self.kernel = kernel
        self.spatial_dims = spatial_dims
        self.depths = depths
        self.norm_name = norm_name

        self.up = DAUB(in_c, emb_dim, out_c, kernel, spatial_dims, norm_name)
        self.l1 = DACB(out_c, in_c, kernel, spatial_dims, norm_name)

        self.l2 = DACB(out_c, in_c, kernel, spatial_dims, norm_name)

        self.conv1x1 = get_conv_layer(spatial_dims, out_c * 2, out_c, kernel_size=1, conv_only=False, padding=0, norm=norm_name)

    def forward(self, x, skip = None):
        out_up = self.up(x)
        
        if(skip is not None):
            cat = torch.cat((out_up, skip), dim=1)
            out_up = self.conv1x1(cat)

        skip = self.l1(out_up)
        out = self.l2(skip)

        return out