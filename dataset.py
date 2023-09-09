from torch.utils.data import Dataset
import torch
import numpy as np
import PIL.Image as Image
import os
import albumentations as albu
from torch.utils.data import DataLoader
import random
import cv2


def mask_small_object(mask):
    """
    :param mask: input mask
    :return: weight map
    """

    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, retval):
        mask[labels == i] = (1.0 - np.log(stats[i][4] / 65536))
    mask[labels == 0] = 1.0
    return mask


class DatasetISAID(Dataset):
    def __init__(self, datapath, fold, transform, split, shot):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'iSAID'
        self.shot = shot

        self.img_path = os.path.join(os.path.join(datapath, split), 'images')
        self.ann_path = os.path.join(os.path.join(datapath, split), 'semantic_png')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata)
        # return 10000 if self.split == 'trn' else 5000

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = ('F:/datasets/iSAID_5i/iSAID_patches/%s/list/split%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data[:-3], int(data.split('_')[-1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        if self.split == 'trn':
            class_sample = random.choice(self.class_ids)
            query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        else:
            query_name, class_sample = self.img_metadata[idx]
        support_names = self.sample_episode(query_name, class_sample)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names, class_sample)
        support_imgs = torch.stack([support_img for support_img in support_imgs])
        support_masks = torch.stack(support_masks)
        scale_score = mask_small_object(np.array(query_mask.clone(), dtype=np.uint8))
        scale_score = torch.from_numpy(scale_score)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'scale_score': scale_score,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def sample_episode(self, query_name, class_sample):
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name and support_name not in support_names:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break
        return support_names

    def load_frame(self, query_name, support_names, class_sample):
        query_img, query_mask = self.read_imgpair(query_name, class_sample)
        support_imgs = []
        support_masks = []
        for name in support_names:
            support_img, support_mask = self.read_imgpair(name, class_sample)
            support_imgs.append(support_img)
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks

    def read_imgpair(self, img_name, class_sample):
        """load images and masks after augmentation"""
        image = np.array(Image.open(os.path.join(self.img_path, img_name.replace('_instance_color_RGB', ''))))
        mask = np.array(Image.open(os.path.join(self.ann_path, img_name)))
        mask[mask != (class_sample + 1)] = 0
        mask[mask == (class_sample + 1)] = 1
        augmented = self.transform(image=image, mask=mask)
        image = torch.tensor(np.transpose(augmented["image"], [2, 0, 1]))
        mask = torch.tensor(augmented["mask"]).float()
        return image, mask


class DatasetDLRSD(Dataset):
    def __init__(self, datapath, fold, transform, split, shot):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'DLRSD'
        self.shot = shot

        self.img_path = os.path.join(datapath, 'UCMerced_LandUse/Images')
        self.ann_path = os.path.join(datapath, 'DLRSD/semantic_masks')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        # return len(self.img_metadata) if self.split == 'trn' else 1000
        return 5000 if self.split == 'trn' else 1000

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = ('F:/datasets/DLRSD_5i/%s/split%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('_')[0], int(data.split('_')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        if self.split == 'trn':
            class_sample = random.choice(self.class_ids)
            query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        else:
            query_name, class_sample = self.img_metadata[idx]
        support_names= self.sample_episode(query_name, class_sample)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names, class_sample)
        support_imgs = torch.stack([support_img for support_img in support_imgs])
        support_masks = torch.stack(support_masks)
        scale_score = mask_small_object(np.array(query_mask.clone(), dtype=np.uint8))
        scale_score = torch.from_numpy(scale_score)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'scale_score': scale_score,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def sample_episode(self, query_name, class_sample):
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name and support_name not in support_names:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break
        return support_names

    def load_frame(self, query_name, support_names, class_sample):
        query_img, query_mask = self.read_imgpair(query_name, class_sample)
        support_imgs = []
        support_masks = []
        for name in support_names:
            support_img, support_mask = self.read_imgpair(name, class_sample)
            support_imgs.append(support_img)
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks

    def read_imgpair(self, img_name, class_sample):
        """load images and masks after augmentation"""
        image = np.array(Image.open(os.path.join(self.img_path, img_name) + '.tif'))
        mask = np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png'))
        mask[mask != (class_sample + 1)] = 0
        mask[mask == (class_sample + 1)] = 1
        augmented = self.transform(image=image, mask=mask)
        image = torch.tensor(np.transpose(augmented["image"], [2, 0, 1]))
        mask = torch.tensor(augmented["mask"]).float()
        return image, mask


class FSSiSAIDDataset:

    @classmethod
    def initialize(cls, datapath):

        cls.datasets = {
            'iSAID': DatasetISAID,
            'DLRSD': DatasetDLRSD
        }
        cls.datapath = datapath
        cls.transform_trn = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            # albu.Resize(256, 256),
            albu.Normalize(mean=(0.378, 0.380, 0.356), std=(0.162, 0.156, 0.152)),  # iSAID
            # albu.Normalize(mean=(0.432, 0.626, 0.270), std=(0.277, 0.255, 0.187)),  # DLRSD
        ])

        cls.transform_val = albu.Compose([
            # albu.Resize(256, 256),
            albu.Normalize(mean=(0.378, 0.380, 0.356), std=(0.162, 0.156, 0.152)),    # iSAID
            # albu.Normalize(mean=(0.432, 0.626, 0.270), std=(0.277, 0.255, 0.187)),  # DLRSD
        ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1, way=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility

        if split == 'trn':
            shuffle = True
            nworker = nworker
            transform = cls.transform_trn
        else:
            shuffle = False
            nworker = 0
            transform = cls.transform_val

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split, shot=shot)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader