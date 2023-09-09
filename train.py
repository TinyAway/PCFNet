import utils
from dataset import FSSiSAIDDataset
from logger import Logger, AverageMeter
import argparse
import torch.optim as optim
import torch
from HSNet import HypercorrSqueezeNetwork
from evaluation import Evaluator
from MLC import MatchingNet
from PCFNet import PCFNet
from DCAMA import DCAMA


def train(epoch, model, dataloader, optimizer, training):
    r""" Train PCFNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.train_mode() if training else model.eval()

    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        # 1. forward pass
        batch = utils.to_cuda(batch)
        logit_mask = model(batch['query_img'], batch['support_imgs'],batch['support_masks'])
        pred_mask = logit_mask.argmax(1)

        # 2. Compute loss & update model parameters
        loss = model.compute_objective(logit_mask, batch['query_mask'], batch['scale_score'])
        # loss = model.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        # average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=100)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='MaskNet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default="F:/datasets/iSAID_5i/iSAID_patches")
    # parser.add_argument('--datapath', type=str, default="F:/datasets/DLRSD")
    parser.add_argument('--benchmark', type=str, default='iSAID', choices=['pascal', 'coco', 'fss', 'iSAID', 'DLRSD'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--nworker', type=int, default=2)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'vgg16'])
    args = parser.parse_args(args=[])
    Logger.initialize(args, training=True)

    model = PCFNet(args.backbone, shot=args.shot, use_original_imgsize=False)
    # model = HypercorrSqueezeNetwork(args.backbone, False)
    # model = DCAMA(args.backbone, False)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()

    # Dataset initialization
    FSSiSAIDDataset.initialize(datapath=args.datapath)
    dataloader_trn = FSSiSAIDDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', args.shot)
    dataloader_val = FSSiSAIDDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.shot)

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.niter):
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')