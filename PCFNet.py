import torch
from torch import nn
import torch.nn.functional as F
from HSNet import extract_feat_res, extract_feat_vgg
from correlation import Correlation
from torchvision.models import resnet
from torchvision.models import vgg
from functools import reduce
from operator import add


class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, group=4):

            building_block_layers = []
            for idx, outch in enumerate(out_channels):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                building_block_layers.append(nn.Conv2d(inch, outch, 3, 1, 1))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0] * 2, [outch1, outch2, outch3])
        self.encoder_layer3 = make_building_block(inch[1] * 2, [outch1, outch2, outch3])
        self.encoder_layer2 = make_building_block(inch[2] * 2, [outch1, outch2, outch3])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3])

        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))



    def forward(self, hypercorr_pyramid):

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = F.interpolate(hypercorr_sqz4, hypercorr_sqz3.shape[-2:], mode='bilinear', align_corners=True)
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = F.interpolate(hypercorr_mix43, hypercorr_sqz2.shape[-2:], mode='bilinear', align_corners=True)
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)


        # Decode the encoded 4D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_mix432)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.decoder2(hypercorr_decoded)

        return logit_mask


class PCFNet(nn.Module):
    def __init__(self, backbone, shot, use_original_imgsize):
        super(PCFNet, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, query_img, support_img, support_mask):
        with torch.no_grad():
            support_feats_list = []
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            for i in range(self.shot):
                support_feats = self.extract_feats(support_img[:, i, :, :, :], self.backbone, self.feat_ids,
                                               self.bottleneck_ids, self.lids)
                support_feats_list.append(support_feats)

        corr_list = []
        for i in range(self.shot):
            corr = Correlation.correlation(query_feats, support_feats_list[i], self.stack_ids,
                                           support_mask[:, i, :, :].clone())
            corr_list.append(corr)
        corr = corr_list[0]
        if self.shot > 1:
            for i in range(1, self.shot):
                corr[0] += corr_list[i][0]
                corr[1] += corr_list[i][1]
                corr[2] += corr_list[i][2]
            corr[0] /= self.shot
            corr[1] /= self.shot
            corr[2] /= self.shot
        logit_mask = self.hpn_learner(corr)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[-2:], mode='bilinear', align_corners=True)

        return logit_mask

    def compute_objective(self, logit_mask, gt_mask, scale_score):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        scale_score = scale_score.view(bsz, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        loss_pixel = self.cross_entropy_loss(logit_mask, gt_mask)
        loss_pixel = (loss_pixel * scale_score).mean()
        return loss_pixel

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging


if __name__ == "__main__":
    batch = {
        'query_img': torch.rand(4, 3, 256, 256),
        'support_imgs': torch.rand(4, 5, 3, 256, 256),
        'support_masks': torch.randint(0, 2, (4, 5, 256, 256)),
        'org_query_imsize': (256, 256)
    }
    model = PCFNet('resnet50', 5, False)
    out = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
    print(out.size())