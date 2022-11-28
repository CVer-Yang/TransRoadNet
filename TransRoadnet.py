import torch
from torch import nn, einsum
import numpy as np
from torchvision import models
from einops import rearrange, repeat
from torch.nn import init
from torch.nn import functional as F
from functools import partial


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


nonlinearity = partial(F.relu, inplace=True)


class FPN5(nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super(FPN5, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv1x1 = nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.relu =nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.conv1x1(x1)
        x1 = self.up2(x1)
        x = self.relu(x1 + x2)
        return x


class FPN4(nn.Module):
    def __init__(self,  in_channels=256, out_channels=128):
        super(FPN4, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.conv1x1(x1)
        x1 = self.up2(x1)
        x = self.relu(x1 + x2)
        return x


class FPN3(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super(FPN3, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.conv1x1(x1)
        x1 = self.up2(x1)
        x = self.relu(x1 + x2)
        return x

class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

#文章提到的位置注意力机制
class SCA_Blocak(nn.Module):
    def __init__(self, inchannel=256, h=64, w=64):
        super(SCA_Blocak, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0,stride=1)
        self.conv3x3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1,padding=0)
        self.sigmoid = [nn.Sigmoid(), nn.Sigmoid()]
        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))

        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.relu = [nn.ReLU(inplace=True), nn.ReLU(inplace=True)]

    def forward(self, x):
        meanh1 = self.avg_pool_x(x).permute(0, 1, 3, 2)
        meanh1 = torch.mean(meanh1, dim=1, keepdim=True)
        meanh2,_ = torch.max(meanh1, dim=1, keepdim=True)
        meanh = self.conv1x1(torch.cat([meanh1, meanh2], dim=1))

        meanw1 = self.avg_pool_y(x)
        meanw1 = torch.mean(meanw1, dim=1, keepdim=True)
        meanw2,_ = torch.max(meanw1, dim=1, keepdim=True)
        meanw = self.conv1x1(torch.cat([meanw1, meanw2], dim=1))

        s_h = self.sigmoid[0]((meanh.permute(0, 1, 3, 2)))
        s_w = self.sigmoid[1]((meanw))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class TransRoadNet(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1, head_dim=32, window_size=8,
                 downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(512)
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        """可选择在线加载权重,修改预训练权重为Ture.权重置于networks目录下"""
        #resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.up2 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=4)
        self.up8 = nn.Upsample(scale_factor=8)
        self.up16 = nn.Upsample(scale_factor=16)
        self.up32 = nn.Upsample(scale_factor=32)

        self.stage1 = StageModule(in_channels=64, hidden_dimension=128, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=64, hidden_dimension=128, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage5 = StageModule(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 16, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)


        self.finaldeconv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.finalconv1 = nn.Conv2d(64,32,3,padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32,1, 3, padding=1)
        self.conv2 =nn.Conv2d(32,1,3,padding=1)
        self.finalconv4 =nn.Conv2d(2,1,1,padding=0)
        self.finalrelu3 = nonlinearity
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv256 = nn.Conv2d(768, 256, 3, padding=1)
        self.conv128 = nn.Conv2d(640, 128, 3, padding=1)
        self.conv64 = nn.Conv2d(576, 64, 3, padding=1)
        self.conv32 = nn.Conv2d(576, 64, 3, padding=1)

        self.decoder5 = DecoderBlock(hidden_dim * 8, hidden_dim * 4, relu=False, last=True)  # 256
        self.decoder4 = DecoderBlock(hidden_dim * 4, hidden_dim * 2, relu=False)  # 128
        self.decoder3 = DecoderBlock(hidden_dim * 2, hidden_dim * 1, relu=False)  # 64
        self.decoder2 = DecoderBlock(hidden_dim, hidden_dim)  # 64
        self.decoder11 = DecoderBlock(hidden_dim, hidden_dim)  # 32
        self.conv1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,padding=0,stride=1)
        self.conv_last = ConvBnRelu(in_planes=512, out_planes=512, ksize=1, stride=1, pad=0)
        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.gamma = nn.Parameter(torch.zeros(1))

        """由于文章采用的1024图像，在编码器第三层时特征图为64，因此SCA模块中h,w参数设置为64.如果特征图尺寸不一致，需要对应修改"""
        self.SCA = SCA_Blocak()

        self.FPN5 = FPN5()
        self.FPN4 = FPN4()
        self.FPB3 = FPN3()

    def forward(self, img):
        x = self.firstconv(img)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x1 = self.firstmaxpool(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x3_1 = self.SCA(x3)
        x3_1 = self.stage4(x3_1)

        #文章提到的自适应融合
        x4 = self.relu(self.bn(self.conv1(self.gamma * x4 + (1 - self.gamma) * x3_1)))


        """文章中提到的上下文结构消息补充"""
        x3_2 = self.up2(x3_1)#64
        x3_2 = self.relu1(self.conv256(torch.cat((x3_2,x3),dim=1)))
        x3_3 = self.up4(x3_1)#128
        x3_3 = self.relu2(self.conv128(torch.cat((x3_3,x2),dim=1)))
        x3_4 = self.up8(x3_1)#256
        x3_4 = self.relu3(self.conv64(torch.cat((x3_4,x1),dim=1)))
        x3_5 = self.up16(x3_1)#512
        x3_5 = self.relu4(self.conv32(torch.cat((x3_5,x),dim=1)))

        d4 = self.decoder5(x4)+x3_2
        d3 = self.decoder4(d4)+x3_3
        d2 = self.decoder3(d3)+x3_4
        d1 = self.decoder2(d2)+x3_5
        out = self.finaldeconv1(d1)
        out = self.finalconv1(out)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes=768, out_planes=768, ksize=3, stride=1, pad=1, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(DecoderBlock, self).__init__()

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last == False:
            x = self.conv_3x3(x)
            # x=self.sap(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x

"""模型调用接口 ，在train.py引入TransRoad模型"""
def TransRoad(hidden_dim=64, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    net = TransRoadNet(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
    return net





