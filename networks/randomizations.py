import torch
import torch.nn as nn


class StyleRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x,low_level_feat):
        N, C, H, W = x.size()
        N2, C2, H2, W2 = low_level_feat.size()

        if self.training:
            x = x.view(N, C, -1)

            low_level_feat = low_level_feat.view(N, C2, -1)

            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)



            low_level_feat_mean=low_level_feat.mean(-1, keepdim=True)
            low_level_feat_var = low_level_feat.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()



            low_level_feat=(low_level_feat - low_level_feat_mean) / (low_level_feat_var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)

            if x.is_cuda:
                alpha = alpha.cuda()
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]



            low_level_feat_mean = alpha * low_level_feat_mean + (1 - alpha) * low_level_feat_mean[idx_swap]
            low_level_feat_var = alpha * low_level_feat_var + (1 - alpha) * low_level_feat_var[idx_swap]


            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)



            low_level_feat = low_level_feat * (low_level_feat_var + self.eps).sqrt() + low_level_feat_mean
            low_level_feat = low_level_feat.view(N2, C2, H2, W2)

        else:
            idx_swap = torch.randperm(N)


        return x,low_level_feat,idx_swap


class ContentRandomization_single(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            x = x[idx_swap].detach()

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x


class ContentRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x_list):
        feature_len = len(x_list)
        N = x_list[0].size()[0]
        idx_swap = torch.randperm(N)
        x_list_new = []
        if N != 1:
            for i in range(feature_len):
                x = x_list[i]
                N, C, H, W = x.size()
                if self.training:
                    x = x.view(N, C, -1)
                    mean = x.mean(-1, keepdim=True)
                    var = x.var(-1, keepdim=True)

                    x = (x - mean) / (var + self.eps).sqrt()
                    x = x[idx_swap].detach()

                    x = x * (var + self.eps).sqrt() + mean
                    x = x.view(N, C, H, W)

                    x_list_new.append(x)

            return x_list_new
        else:
            return x_list
