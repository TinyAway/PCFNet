import resnet as models
import torch
from torch import nn
import torch.nn.functional as F
import vgg as vgg_models


def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4


class MatchingNet(nn.Module):
    def __init__(self, shot=5):
        super(MatchingNet, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.shot = shot
        vgg_models.BatchNorm = nn.BatchNorm2d
        vgg16 = vgg_models.vgg16_bn(pretrained=True)
        self.layer0, self.layer1, self.layer2, \
        self.layer3, _ = get_vgg16_layer(vgg16)
        # self.backbone = models.resnet50_MLC(pretrained=True)

    def forward(self, img_q, img_s_list, mask_s_list):
        h, w = img_q.shape[-2:]

        # feature maps of support images
        feature_s_list = []
        for k in range(self.shot):
            support_feature = self.layer0(img_s_list[:, k, :, :, :])
            support_feature = self.layer1(support_feature)
            support_feature = self.layer2(support_feature)
            support_feature = self.layer3(support_feature)
            feature_s_list.append(support_feature)
        # feature map of query image
        feature_q = self.layer0(img_q)
        feature_q = self.layer1(feature_q)
        feature_q = self.layer2(feature_q)
        feature_q = self.layer3(feature_q)

        # for k in range(shot):
        #     feature_s_list.append(self.backbone(img_s_list[:, k, :, :, :]))
        #     # feature map of query image
        # feature_q = self.backbone(img_q)

        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []
        for k in range(self.shot):
            feature_fg_list.append(self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[:, k, :, :] == 1).float())[None, :])
            feature_bg_list.append(self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[:, k, :, :] == 0).float())[None, :])
        # average K foreground prototypes and K background prototypes
        feature_fg = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0)
        feature_bg = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0)

        # measure the similarity of query features to fg/bg prototypes
        similarity_fg = F.cosine_similarity(feature_q, feature_fg[..., None, None], dim=1)
        similarity_bg = F.cosine_similarity(feature_q, feature_bg[..., None, None], dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out

    def masked_average_pooling(self, feature, mask):
        feature = F.interpolate(feature, size=mask.shape[-2:], mode="bilinear", align_corners=True)
        masked_feature = torch.sum(feature * mask[:, None, ...], dim=(2, 3)) \
                         / (mask[:, None, ...].sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        loss = self.cross_entropy_loss(logit_mask, gt_mask)
        return loss


if __name__ == "__main__":
    batch = {
        'query_img': torch.rand(4, 3, 256, 256),
        'support_imgs': torch.rand(4, 5, 3, 256, 256),
        'support_masks': torch.randint(0, 2, (4, 5, 256, 256)),
        'org_query_imsize': (256, 256)
    }
    model = MatchingNet('resnet50', shot=5)
    out = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
    print(out.size())


