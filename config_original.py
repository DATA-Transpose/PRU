from models.AllCNN import AllCNN
from models.ResNet import ResNet18

from utils.metric import test_all_in_one
from utils.train_tools import fit_one_cycle
from data.dataset import get_dataset

import torch

def get_configs(dataset_name, model_name):
    config_name = f'{dataset_name}-{model_name}'

    CONFIGS = {
        'mnist-AllCNN': {
            'model': AllCNN(n_channels=1, num_classes=10),
            'batch_size': 64,
            'epochs': 10,
            'optimizer_type': torch.optim.SGD,
            'optimizer_params': {'lr': 0.01, 'weight_decay': 1e-4, 'momentum': 0.9},
            # 'scheduler_type': torch.optim.lr_scheduler.OneCycleLR,
            'scheduler_type': None,
            'scheduler_params': {'max_lr': 0.01, 'epochs': 30},
            'grad_clip': None,
            'training_func': fit_one_cycle,
            'test_func': test_all_in_one,
            'unlearn_classes_set': range(10),
        },
        'cifar10-AllCNN': {
            'model': AllCNN(n_channels=3, num_classes=10),
            'batch_size': 64,
            'epochs': 30,
            'optimizer_type': torch.optim.SGD,
            'optimizer_params': {'lr': 0.1, 'weight_decay': 1e-4, 'momentum': 0.9},
            # 'scheduler_type': torch.optim.lr_scheduler.OneCycleLR,
            'scheduler_type': None,
            'scheduler_params': {'max_lr': 0.1, 'epochs': 30},
            'grad_clip': None,
            'training_func': fit_one_cycle,
            'test_func': test_all_in_one,
            'unlearn_classes_set': range(10),
        },
        'cifar10-ResNet18': {
            'model': ResNet18(num_classes=10),
            'batch_size': 64,
            'epochs': 20,
            'optimizer_type': torch.optim.SGD,
            'optimizer_params': {'lr': 0.01, 'weight_decay': 5e-4, 'momentum': 0.9},
            # 'scheduler_type': torch.optim.lr_scheduler.OneCycleLR,
            'scheduler_type': None,
            'scheduler_params': {'max_lr': 0.01, 'epochs': 30},
            'grad_clip': None,
            'training_func': fit_one_cycle,
            'test_func': test_all_in_one,
            'unlearn_classes_set': range(10),
        },
    }

    if config_name not in CONFIGS:
        raise ValueError(f'Config {config_name} not found! Please config it in config.py')
    return CONFIGS[config_name]

