import os
import copy
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class KuzushijiMNIST(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        if train:
            self.data = np.load(os.path.join(root, 'kmnist-train-imgs.npz'))['arr_0']
            self.targets = np.load(os.path.join(root, 'kmnist-train-labels.npz'))['arr_0']
        else:
            self.data = np.load(os.path.join(root, 'kmnist-test-imgs.npz'))['arr_0']
            self.targets = np.load(os.path.join(root, 'kmnist-test-labels.npz'))['arr_0']
        # min max normalization
        self.data = self.data / 255
        # to torch tensor
        self.data = torch.from_numpy(self.data).float()
        self.targets = torch.from_numpy(self.targets).long()
        self.classes = np.arange(10)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # add channel dim
        img = img.unsqueeze(0)
        
        return img, target
    
    def __len__(self):
        return len(self.data)
    
def statstic_info(dataset: Dataset):
    sample_num = len(dataset)
    num_classes = len(dataset.classes)
    class_sample_num = [0 for _ in range(num_classes)]
    for _, label in dataset:
        class_sample_num[label] += 1
    return sample_num, num_classes, class_sample_num

def get_subset(dataset, idxs) -> Dataset:
    subset = copy.deepcopy(dataset)
    subset.targets = subset.targets[idxs]
    subset.data = subset.data[idxs]
    return subset

def get_dataset(name, data_root='~/storage/public_datasets'):
    if name == 'cifar10':
        transforms_cifar_train = transforms.Compose([
            # transforms.RandomResizedCrop((224, 224)), # For ViT
            transforms.RandomCrop(32, padding=4), # For ResNet
            transforms.RandomHorizontalFlip(), # For ResNet
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # normalization
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
        ])

        transforms_cifar_test = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # normalization
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
        ])

        train_set = datasets.CIFAR10(
            root = data_root,
            train = True,                         
            transform = transforms_cifar_train, 
            download = True,            
        )
        train_set.targets = np.array(train_set.targets)

        test_set = datasets.CIFAR10(
            root = data_root, 
            train = False, 
            transform = transforms_cifar_test,
            download = True,  
        )
        test_set.targets = np.array(test_set.targets)

    elif name == 'cifar100':
        transforms_cifar_train = transforms.Compose([
            # transforms.RandomResizedCrop((224, 224)), # For ViT
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'), # For ResNet
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), # For ResNet
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                (0.26733428587941854, 0.25643846292120615, 0.2761504713263903)
            ) # normalization
        ])
        transforms_cifar_test = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                (0.26733428587941854, 0.25643846292120615, 0.2761504713263903)
            ) # normalization
        ])

        train_set = datasets.CIFAR100(
            root = data_root,
            train = True,
            transform = transforms_cifar_train,
            download = True,
        )
        train_set.targets = np.array(train_set.targets)

        test_set = datasets.CIFAR100(
            root = data_root, 
            train = False,
            transform = transforms_cifar_test,
            download = True,
        )
        test_set.targets = np.array(test_set.targets)

    elif name == 'mnist':
        train_set = datasets.MNIST(
            root = data_root,
            train = True,                         
            transform = transforms.ToTensor(),
            download = True,            
        )
        test_set = datasets.MNIST(
            root = data_root, 
            train = False, 
            transform = transforms.ToTensor(),
        )
    elif name == 'mnistFashion':
        train_set = datasets.FashionMNIST(
            root = data_root,
            train = True,                         
            transform = transforms.ToTensor(),
            download = True,            
        )
        test_set = datasets.FashionMNIST(
            root = data_root, 
            train = False, 
            transform = transforms.ToTensor(),
        )
    elif name == 'mnistKuzushiji':
        train_set = KuzushijiMNIST(
            root = '../../Dataset/KMNIST',
            train = True,                         
        )
        test_set = KuzushijiMNIST(
            root = '../../Dataset/KMNIST', 
            train = False, 
        )

    return train_set, test_set


def get_dataloader(dataset, batch_size, shuffle):
    loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        pin_memory = True
    )

    return loader