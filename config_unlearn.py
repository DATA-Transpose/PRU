from models.AllCNN import AllCNN
from models.ResNet import ResNet18

from utils.metric import test_all_in_one
from utils.train_tools import fit_one_cycle
from data.dataset import get_dataset

import torch

def get_unlearn_configs(dataset_name, model_name):
    config_name = f'{dataset_name}-{model_name}'

    CONFIGS = {
        'mnist-AllCNN': {
            'shift_epoch': 10,
            'shift_lr': 5 * 1e-3,
            'shift_lamb': 1,
            'refine_epoch': 1,
            'refine_lr': 1 * 1e-1,
        },

        'cifar10-AllCNN': {
            'shift_epoch': 10,
            'shift_lr': 1 * 1e-2,
            'shift_lamb': 1,
            'refine_epoch': 1,
            'refine_lr': 1 * 1e-5,
        },
        
        'cifar10-ResNet18': {
            'shift_epoch': 10,
            'shift_lr': 4 * 1e-4,
            'shift_lamb': 1,
            'refine_epoch': 1,
            'refine_lr': 1 * 1e-4,
        },
    }

    if config_name not in CONFIGS:
        raise ValueError(f'Config {config_name} not found! Please config it in config.py')
    return CONFIGS[config_name]
