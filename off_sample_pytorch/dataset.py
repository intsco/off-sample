from collections import Counter
import numpy as np

import torch
import torchvision
import torchvision as torchvision
from torchvision.datasets.folder import find_classes, make_dataset, IMG_EXTENSIONS, default_loader


class OffSampleDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform):
        self.classes, self.class_to_idx = find_classes(root)
        self.samples = make_dataset(root, self.class_to_idx, extensions=IMG_EXTENSIONS)  # path, target
        self.loader = default_loader
        self.data = []
        self.targets = []
        self.transform = transform

        for path, target in self.samples:
            img = self.loader(path)
            self.data.append(img)
            self.targets.append(target)

    def __getitem__(self, index):
        img = self.transform(self.data[index])
        return img, self.targets[index]

    def __len__(self):
        return len(self.data)


class TrainValidListDataset(torch.utils.data.Dataset):

    def __init__(self, root_path, train_dir, valid_dir):
        self.classes, self.class_to_idx = find_classes(root_path / 'train')
        train_samples = make_dataset(root_path / 'train', self.class_to_idx, extensions=IMG_EXTENSIONS)
        valid_samples = make_dataset(root_path / 'valid', self.class_to_idx, extensions=IMG_EXTENSIONS)
        self.samples = train_samples + valid_samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class SubsetDataset(object):

    def __init__(self, dataset, inds, transform):
        self.dataset = dataset
        self.inds = inds
        self.transform = transform
        self.loader = default_loader
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __getitem__(self, index):
        path, target = self.dataset[self.inds[index]]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.inds)
