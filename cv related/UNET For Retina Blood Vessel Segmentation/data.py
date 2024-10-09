import cv2
import imageio
import numpy as np
from albumentations import HorizontalFlip, \
    VerticalFlip, Rotate

from tqdm import tqdm

import os

import torch
from torch.utils.data import Dataset


class RetinaDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples


def augment_data(images, masks, save_path, size_image, augment=True):
    """ Applies augmentations to images """

    size = size_image

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ File name extraction """
        name = x.split('\\')[-1].split('.')[0]

        """ Reading image and mask"""
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment is True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f'{name}_{index}.png'
            tmp_mask_name = f'{name}_{index}.png'

            image_path = os.path.join(save_path, 'image', tmp_image_name)
            mask_path = os.path.join(save_path, 'mask', tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1
