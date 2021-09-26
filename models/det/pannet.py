import torch
from torch import nn

from models.det.resnet import *
from models.det.segmentation_head import FPEM_FFM, FPN
from models.det.shufflenetv2 import *

backbone_dict = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
                 'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
                 'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
                 'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
                 'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
                 'resnet50_32x4d': {'models': resnext50_32x4d, 'out': [256, 512, 1024, 2048]},
                 'resnet101_32x8d': {'models': resnext101_32x8d, 'out': [256, 512, 1024, 2048]},
                 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}
                 }

segmentation_head_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}


# 'MobileNetV3_Large': {'models': MobileNetV3_Large, 'out': [24, 40, 160, 160]},
# 'MobileNetV3_Small': {'models': MobileNetV3_Small, 'out': [16, 24, 48, 96]},
# 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}}


class PANNet(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANNet
        :param model_config: 模型配置
        """
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_head = segmentation_head_dict[segmentation_head](backbone_out, **model_config)
        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        # for b in backbone_out:
        #     print(b.shape)
        segmentation_head_out, feature = self.segmentation_head(backbone_out)

        segmentation_head_out = torch.squeeze(segmentation_head_out, 1)

        # y = F.interpolate(segmentation_head_out, size=(H, W), mode='bilinear', align_corners=True)
        return segmentation_head_out, feature


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 800, 800).to(device)

    model_config = {
        'backbone': 'resnet34',
        'fpem_repeat': 4,  # fpem模块重复的次数
        'pretrained': False,  # backbone 是否使用imagenet的预训练模型
        'result_num': 7,
        'segmentation_head': 'FPEM_FFM'  # 分割头，FPN or FPEM_FFM
    }
    model = PANNet(model_config=model_config).to(device)
    print("{} parameters in total".format(sum(x.numel() for x in model.parameters())))
    y, feature = model(x)
    print(y.shape)
    print(feature.shape)
    # print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
