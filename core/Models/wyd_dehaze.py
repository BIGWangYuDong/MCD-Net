import torch
import torch.nn as nn
import torch.nn.functional as F
from Dehaze.core.Models.builder import NETWORK, build_backbone
from Dehaze.core.Models.base_model import BaseNet
from Dehaze.core.Models.weight_init import normal_init, xavier_init
from Dehaze.core.Models.DCN import DeformConv2dPack as DCN


@NETWORK.register_module()
class MCDNet(BaseNet):
    def __init__(self,
                 backbone,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True,):
        super(MCDNet, self).__init__(backbone, pretrained, init_weight_type, get_parameter)
        self.backbone = build_backbone(backbone)
        if init_weight_type is not None:
            self.init_weight_type = init_weight_type
        self.get_parameter = get_parameter
        self._init_layers()
        self.init_weight(pretrained=pretrained)
        self.get_parameters()

    def _init_layers(self):
        self.UNet = UNet(input_nc=3, output_nc=3, nf=8)
        self.SA_UNet = SpatialAttention()
        self.CA_UNet = ChannelAttention(3)

        self.DUB1 = DUB(1024, 512, 256)
        self.DUB2 = DUB(512+256, 256, 128)
        self.DUB3 = DUB(256+128, 128, 256)
        self.DUB4 = DUB(256, 64, 128)
        self.DUB5 = DUB(128, 32, 16)
        self.instance1 = nn.InstanceNorm2d(512, affine=False)
        self.instance2 = nn.InstanceNorm2d(256, affine=False)
        self.SA_EnDeCoder = SpatialAttention()
        self.CA_EnDeCoder = ChannelAttention(16)

        self.EnhanceHead = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32)
        )
        self.Enhance = EMBlock(32, 32, 32)
        self.WAB1 = WAB(32)
        self.WAB2 = WAB(32)
        self.WAB3 = WAB(32)
        self.EnhanceTail = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 13, 3, 1, 1, bias=True)
        )
        self.SA_Enhance = SpatialAttention()
        self.CA_Enhance = ChannelAttention(13)

        # self.CASA_Conv1 = nn.Conv2d(32, 64, 3, 1, 1)

        self.SA_tail = SpatialAttention()
        self.CA_tail = ChannelAttention(32)

        self.tail = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DCN(32, 3, 3, 1, 1)
        )
        self.upsample = F.upsample_nearest


    def init_weight(self, pretrained=None):
        # super(DehazeNet, self).init_weight(pretrained)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        self.backbone.init_weights(pretrained)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        UNet_out = self.UNet(x)

        UNet_CASA_out = self.CA_UNet(UNet_out) * UNet_out
        UNet_CASA_out = self.SA_UNet(UNet_CASA_out) * UNet_CASA_out

        enhance_out = self.EnhanceHead(x)
        enhance_out = self.Enhance(enhance_out)
        enhance_out = self.WAB1(self.WAB2(self.WAB3(enhance_out)))
        enhance_out = self.EnhanceTail(enhance_out)

        enhance_CASA_out = self.CA_Enhance(enhance_out) * enhance_out
        enhance_CASA_out = self.SA_Enhance(enhance_CASA_out) * enhance_CASA_out

        dense_out1, dense_out2, dense_out3, dense_out4, dense_out5 = self.backbone(x)
        shape_out4 = dense_out4.data.size()
        shape_out4 = shape_out4[2:4]

        shape_out3 = dense_out3.data.size()
        shape_out3 = shape_out3[2:4]

        shape_out2 = dense_out2.data.size()
        shape_out2 = shape_out2[2:4]

        shape_out1 = dense_out1.data.size()
        shape_out1 = shape_out1[2:4]

        shape_out = x.data.size()
        shape_out = shape_out[2:4]


        up_4 = self.DUB1(dense_out5)
        up_4 = self.upsample(up_4, size=shape_out4)
        dense_out4 = self.instance1(dense_out4)

        up_3 = self.DUB2(torch.cat([up_4, dense_out4], 1))
        up_3 = self.upsample(up_3, size=shape_out3)
        dense_out3 = self.instance2(dense_out3)

        up_2 = self.DUB3(torch.cat([up_3, dense_out3], 1))
        up_2 = self.upsample(up_2, size=shape_out2)

        up_1 = self.DUB4(up_2)
        up_1 = self.upsample(up_1, size=shape_out1)

        EnDeCoder_out = self.DUB5(up_1)
        EnDeCoder_out = self.upsample(EnDeCoder_out, size=shape_out)

        EnDeCoder_CASA_out = self.CA_EnDeCoder(EnDeCoder_out) * EnDeCoder_out
        EnDeCoder_CASA_out = self.SA_EnDeCoder(EnDeCoder_CASA_out) * EnDeCoder_CASA_out

        CASA = torch.cat([UNet_CASA_out, enhance_CASA_out, EnDeCoder_CASA_out], 1)
        out = torch.cat([UNet_out, enhance_out, EnDeCoder_out], 1)
        CASA = self.CA_tail(CASA) * CASA
        CASA = self.SA_tail(CASA) * CASA

        out = CASA + out

        out = self.tail(out)

        # out = self.relu(out)
        return F.tanh(out)
        # return out.clamp(0,1)
        # return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class EMBlock(nn.Module):
    def __init__(self, in_fea, mid_fea, out_fea):
        super(EMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_fea, mid_fea, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(mid_fea)
        self.conv2 = nn.Conv2d(mid_fea, mid_fea, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(mid_fea)
        self.avgpool32 = nn.AvgPool2d(32)
        self.avgpool16 = nn.AvgPool2d(16)
        self.avgpool8 = nn.AvgPool2d(8)
        self.avgpool4 = nn.AvgPool2d(4)
        self.avgpool2 = nn.AvgPool2d(2)

        self.upsample = F.upsample_nearest

        self.conv3_4 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.conv3_8 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.conv3_16 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)
        self.conv3_32 = nn.Conv2d(mid_fea, int(mid_fea / 4), kernel_size=1, stride=1, padding=0)

        self.conv4 = nn.Conv2d(mid_fea + mid_fea, out_fea, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_4 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_8 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_16 = nn.LeakyReLU(0.2, inplace=True)
        self.relu3_32 = nn.LeakyReLU(0.2, inplace=True)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # b, c, h, w = x.shape
        # mod1 = h % 32
        # mod2 = w % 32
        # if (mod1):
        #     down1 = 64 - mod1
        #     x = F.pad(x, (0, 0, 0, down1), "reflect")
        # if (mod2):
        #     down2 = 64 - mod2
        #     x = F.pad(x, (0, down2, 0, 0), "reflect")
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out_2 = self.avgpool2(out)
        shape_out = out_2.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        out_32 = self.avgpool32(out)
        out_16 = self.avgpool16(out)
        out_8 = self.avgpool8(out)
        out_4 = self.avgpool4(out)

        out_32 = self.upsample(self.relu3_32(self.conv3_32(out_32)), size=shape_out)
        out_16 = self.upsample(self.relu3_16(self.conv3_16(out_16)), size=shape_out)
        out_8 = self.upsample(self.relu3_8(self.conv3_8(out_8)), size=shape_out)
        out_4 = self.upsample(self.relu3_4(self.conv3_4(out_4)), size=shape_out)
        out = torch.cat((out_32, out_16, out_8, out_4, out_2), dim=1)
        out = self.relu4(self.conv4(out))
        # if(mod1):out=out[:,:,:-down1,:]
        # if(mod2):out=out[:,:,:,:-down2]
        return out


class WAB(nn.Module):
    def __init__(self,n_feats,expand=4):
        super(WAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * expand,3,1,1, bias=True),
            nn.BatchNorm2d(n_feats * expand),
            nn.ReLU(True),
            nn.Conv2d(n_feats* expand, n_feats , 3, 1, 1, bias=True),
            nn.BatchNorm2d(n_feats)
        )

    def forward(self, x):
        res = self.body(x).mul(0.2)+x
        return res


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block

class UNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=16):
        super(UNet, self).__init__()
        # input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=False, relu=False, dropout=False)

        ## NOTE: decoder
        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        # dlayer6 = blockUNet(nf*16, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
        dlayer6 = blockUNet(nf*8, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer5 = blockUNet(nf*16, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer4 = blockUNet(nf*16, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer3 = blockUNet(nf*8, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer2 = blockUNet(nf*4, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.tail_conv = nn.Conv2d(nf*2, output_nc, 3,padding=1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        mod1 = h % 64
        mod2 = w % 64
        if (mod1):
            down1 = 64 - mod1
            x = F.pad(x, (0, 0, 0, down1), "reflect")
        if (mod2):
            down2 = 64 - mod2
            x = F.pad(x, (0, down2, 0, 0), "reflect")
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        dout1 = self.tail_conv(dout1)
        if (mod1): dout1 = dout1[:, :, :-down1, :]
        if (mod2): dout1 = dout1[:, :, :, :-down2]
        # dout1=torch.sigmoid(dout1)
        return dout1

class DUB(nn.Module):
    def __init__(self, a,b,c):
        super(DUB, self).__init__()
        self.conv3=nn.Sequential(
            nn.BatchNorm2d(a),
            nn.ReLU(inplace=True),
            DCN(a, b, kernel_size=3, stride=1,padding=1)
        )
        self.conv1=nn.Sequential(
            nn.BatchNorm2d(a+b),
            nn.ReLU(inplace=True),
            DCN(a+b, c, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        y = self.conv3(x)
        x = self.conv1(torch.cat([x,y],1))
        return x