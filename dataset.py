import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class KonIQ10KDataset(Dataset):
    """
    load KonIQ-10K dataset
    """

    def __init__(self, mos_df, images_folder, training=True, dist=True):
        """
        Args:
            mos_df (DataFrame): mos detail about KonIQ-10k
            images_folder (str): path to real image
            training (bool, optional): whether in training process. Defaults to True.
            dist (bool, optional): to predict distribution of mos. Defaults to True.
        """
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.distribution = dist

        if training:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(3, expand=True),
                transforms.CenterCrop((768, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.images_folder = images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, mos_detail.image_name)
        image = self.transforms(Image.open(image_path))

        if self.distribution:
            mos_distribution = (mos_detail.c1, mos_detail.c2,
                                mos_detail.c3, mos_detail.c4, mos_detail.c5)
            label = tuple([m/mos_detail.c_total for m in mos_distribution])
        else:
            label = [mos_detail.MOS / 5]
        return image, torch.Tensor(label)


class LiveDataSet(Dataset):
    def __init__(self, mos_df, images_folder):
        self.mos_df = mos_df
        self.len = len(self.mos_df)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.images_folder = images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, mos_detail[0])
        image = self.transforms(Image.open(image_path))
        label = [mos_detail[1] / 20]
        return image, torch.Tensor(label)
