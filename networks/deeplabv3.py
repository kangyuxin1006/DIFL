import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.aspp import build_aspp
from networks.decoder import build_decoder
from networks.backbone import build_backbone
from networks.encoder import build_encoderDC
from networks.randomizations import StyleRandomization

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=False, num_domain=3, freeze_bn=False, lam =0.9,style_mode=True):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.style_mode=style_mode
        self.lam = lam
        self.centroids = nn.Parameter(torch.randn(num_domain, 304, 64, 64), requires_grad=False)
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.style_randomization=StyleRandomization()
        self.decoder = build_decoder(num_classes, num_domain, backbone, BatchNorm)
        self.last_conv_mask = nn.Sequential(BatchNorm(3),
                                                nn.ReLU(),
                                                nn.Dropout(0.5),
                                                nn.Conv2d(3, num_domain, kernel_size=1, stride=1))
        self.encoder_d=build_encoderDC(num_domain, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def update_memory(self, feature):
        _feature = torch.mean(torch.mean(feature, 3, True), 2, True)
        lam = self.lam
        self.centroids[0].data = lam * self.centroids[0].data + (1 - lam) * torch.mean(_feature[0:8], 0, True)
        self.centroids[1].data = lam * self.centroids[1].data + (1 - lam) * torch.mean(_feature[8:16], 0, True)
        self.centroids[2].data = lam * self.centroids[2].data + (1 - lam) * torch.mean(_feature[16:24], 0, True)

    def forward(self, input, extract_feature=False):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x) #remove feature
        domain_code = self.encoder_d(x)
        if self.style_mode:
            x_style, low_level_feat_style, idx_swap = self.style_randomization(x,low_level_feat)

            domain_code_style = self.encoder_d(x_style)

            x_style = self.decoder(x_style,low_level_feat_style)
            x_style = F.interpolate(x_style, size=input.size()[2:], mode='bilinear', align_corners=True)

        else:
            x_style=None
            domain_code_style=None
            idx_swap=None

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x,domain_code,x_style,domain_code_style,idx_swap

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_para(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        # for param in self.aspp.parameters():
        #     param.requires_grad = False


    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab_DomainCode(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


