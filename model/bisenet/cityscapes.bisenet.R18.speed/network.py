# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# from torchvision.models import resnet50, resnet101, resnet152

import numpy as np
from config import config
from base_model import resnet18
from seg_opr.seg_oprs import ConvBnRelu, DeConvBnRelu, AttentionRefinement, FeatureFusion


def get():
    return BiSeNet(config.num_classes, None, None)


class BiSeNet(nn.Module):
    def __init__(self, out_planes, is_training,
                 criterion, ohem_criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(BiSeNet, self).__init__()
        self.context_path = resnet18(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=config.bn_eps,
                                     bn_momentum=config.bn_momentum,
                                     deep_stem=False, stem_width=64)

        self.business_layer = []
        self.is_training = is_training

        self.spatial_path = SpatialPath(3, 128, norm_layer)

        conv_channel = 128
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(512, conv_channel, 1, 1, 0,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        )

        # stage = [512, 256, 128, 64]
        arms = [AttentionRefinement(512, conv_channel, norm_layer),
                AttentionRefinement(256, conv_channel, norm_layer)]
        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False)]

        if is_training:
            #heads = [BiSeNetHead(conv_channel, out_planes, 16,
            #                     True, norm_layer),
            #         BiSeNetHead(conv_channel, out_planes, 8,
            #                     True, norm_layer),
            #         BiSeNetHead(conv_channel * 2, out_planes, 8,
            #                     False, norm_layer)]
            heads = [XmHead(conv_channel, out_planes, 16,
                                 True, norm_layer),
                     XmHead(conv_channel, out_planes, 8,
                                 True, norm_layer),
                     XmHead(conv_channel * 2, out_planes, 8,
                                 False, norm_layer)]
        else:
            #heads = [None, None,
            #         BiSeNetHead(conv_channel * 2, out_planes, 8,
            #                     False, norm_layer)]
            heads = [None, None,
                     XmHead(conv_channel * 2, out_planes, 8,
                                 False, norm_layer)]

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2,
                                 1, norm_layer)

        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)

        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

        if is_training:
            self.criterion = criterion
            self.ohem_criterion = ohem_criterion

    def forward(self, data, label=None):
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        #fix_sizes = [(24, 48), (48, 96), (96, 192)]
        fix_sizes = [(11, 20), (22, 40), (44, 80)]
        global_context_avg = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context_avg,
                                       #size=context_blocks[0].size()[2:],
                                       size=fix_sizes[0],
                                       mode='bilinear', align_corners=True)
                                       # mode='nearest')

        last_fm = global_context
        pred_out = []

        #print('cotexxt block size:')
        #print(context_blocks[0].size()[2:])
        #print(context_blocks[1].size()[2:])
        #print(context_blocks[2].size()[2:])

        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm, 
                                    #size=(context_blocks[i + 1].size()[2:]),
                                    size=fix_sizes[i + 1],
                                    mode='bilinear', align_corners=True)
                                    #mode='nearest')
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        pred_out.append(concate_fm)

        if self.is_training:
            #aux_loss0 = self.ohem_criterion(self.heads[0](pred_out[0]), label)
            #aux_loss1 = self.ohem_criterion(self.heads[1](pred_out[1]), label)
            main_loss = self.ohem_criterion(self.heads[-1](pred_out[2]), label)

            loss = main_loss
            #loss = main_loss + aux_loss0 + aux_loss1
            return loss

        head_out = self.heads[-1](pred_out[-1])
        #out_data = head_out.cpu().numpy()
        #print('head_out shape:', out_data.shape)
        #np.save('head_out.npy', out_data);
        #for i in range(5):
        #    np.savetxt('head_out_' + str(i) + 'txt', out_data[0][i])
        #lsf_result = F.log_softmax(head_out, dim=1)
        #lsf_result = torch.exp(lsf_result)
        #lsf_result = F.interpolate(lsf_result, size=(768, 768*2), mode= 'bilinear', align_corners=True)
        #lsf_result = lsf_result.permute(0, 2, 3, 1)
        lsf_result = head_out.permute(0, 2, 3, 1)
        lsf_result = lsf_result.argmax(3)
        #np.save('torch_lsf_result.npy', lsf_result.cpu().numpy());
        #np.save('torch_spatial_out.npy', spatial_out.cpu().numpy());
        #np.save('torch_context_block_0.npy', context_blocks[0].cpu().numpy());
        #np.save('torch_concate_fm.npy', concate_fm.cpu().numpy());
        #np.save('torch_head_out.npy', head_out.cpu().numpy());
        #np.save('torch_data.npy', data.cpu().numpy());

        #return lsf_result, spatial_outs[1], spatial_outs[2] 
        #return lsf_result, spatial_out, context_blocks[0], global_context_avg, global_context, context_out 
        return lsf_result 


class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)
        return output


class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 128, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(128, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            if self.scale == 8:
                #hacked for onnx export, scale-factor not supported now
                output = F.interpolate(output, 
                                   #scale_factor=(8),
                                   size=(352, 640),
                                   mode='bilinear',
                                   align_corners=True)
            else:
                output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output

class XmHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(XmHead, self).__init__()
        if is_aux:
            #assert(scale == 16 and in_planes ==256)
            dconv0 =  DeConvBnRelu(in_planes, 128, 3, 2, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)

            dconv1 =  DeConvBnRelu(128, 64, 3, 2, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
            dconv2 =  DeConvBnRelu(64, 32, 3, 2, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)

            deconv_arr = [dconv0, dconv1, dconv2] 
        else:
            assert(scale == 8 and in_planes == 256)
            dconv0 =  DeConvBnRelu(in_planes, 128, 3, 2, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)

            dconv1 =  DeConvBnRelu(128, 64, 3, 2, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
            dconv2 =  DeConvBnRelu(64, 32, 3, 2, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)

            deconv_arr = [dconv0, dconv1, dconv2] 

        self.deconv_3x3_arr = nn.ModuleList(deconv_arr)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(32, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(32, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale
        self.in_planes = in_planes

    def forward(self, x):
        #aux not supported now
        #if self.scale == 8:
        fm = self.deconv_3x3_arr[0](x)
        fm = self.deconv_3x3_arr[1](fm)
        fm = self.deconv_3x3_arr[2](fm)
        # fm = self.dropout(fm)
        output = self.conv_1x1(fm)
        return output


if __name__ == "__main__":
    model = BiSeNet(22, None)
    # print(model)
