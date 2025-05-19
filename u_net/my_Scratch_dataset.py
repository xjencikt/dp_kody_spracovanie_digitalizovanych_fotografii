import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class ScratchDataset(Dataset):
    def __init__(self, image_directory, mask_directory, transform=None, target_size=(256, 256)):
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.transform = transform
        self.target_size = target_size

        self.image_filenames = [
            image for image in os.listdir(image_directory)
            if os.path.exists(os.path.join(mask_directory, image))
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        filename = self.image_filenames[index]

        image_path = os.path.join(self.image_directory, filename)
        mask_path = os.path.join(self.mask_directory, filename)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size)
        mask = (mask > 0).astype(np.uint8)
        mask = torch.tensor(mask, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, mask