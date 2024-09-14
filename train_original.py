import os
import copy
import pickle
import argparse
import numpy as np

import torch

from config_original import get_configs
from utils.seed import set_seed
from utils.metric import test_all_in_one
from utils.unlearn_tools import get_idx_by_unlearn_class
from data.dataset import get_dataloader, get_dataset, get_subset, statstic_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='mnist', choices=['mnist', 'mnistFashion', 'mnistKuzushiji', 'cifar10'])
    parser.add_argument('--model', type=str, default='AllCNN', choices=['AllCNN', 'ResNet18', 'ViT'])
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)

    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    seed = args.seed
    trials = args.trials
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prepared_data_path_template = './runs/prepared_data/%s/trial_%s/'
    save_path_template = prepared_data_path_template                    # original model is a part of prepared stuff
    training_history_path_template = './runs/training_history/%s/original_model/trial_%s/'

    raw_train_set, raw_test_set = get_dataset(dataset)
    num_classes = len(raw_train_set.classes)

    # get model and training config
    CONFIGS = get_configs(dataset, args.model)
    unlearn_classes_set = CONFIGS['unlearn_classes_set']

    res = []
    for trial in range(trials):
        print(f'{"-"*10} Trial {trial}, set seed {seed+trial} {"-"*10}')
        set_seed(seed + trial)
        in_trial_model_performance = []
        # for cls in range(num_classes):

        prepared_data_path = prepared_data_path_template % (dataset, trial)
        if not os.path.exists(prepared_data_path):
            raise ValueError(f'{prepared_data_path} does not exist!')
        train_idx = np.load(prepared_data_path + 'train_idx.npy')
        val_idx = np.load(prepared_data_path + 'val_idx.npy')

        train_set = get_subset(raw_train_set, train_idx)
        val_set = get_subset(raw_train_set, val_idx)
        print(f'Train set: {statstic_info(train_set)}')
        print(f'Val set: {statstic_info(val_set)}')

        # prepare dataloader
        train_loader = get_dataloader(train_set, CONFIGS['batch_size'], shuffle=True)
        val_loader = get_dataloader(val_set, CONFIGS['batch_size'], shuffle=True)
        test_loader = get_dataloader(raw_test_set, CONFIGS['batch_size'], shuffle=False)

        model = copy.deepcopy(CONFIGS['model']).to(device)
        train_history, model = CONFIGS['training_func'](
            CONFIGS['epochs'], model, train_loader, val_loader,
            CONFIGS['optimizer_type'], CONFIGS['optimizer_params'],
            CONFIGS['scheduler_type'], CONFIGS['scheduler_params'],
            grad_clip=CONFIGS['grad_clip'], device=device, output_activation=False
        )

        # save model and training history
        model_save_title = f'original_model_{model.__class__.__name__}.pt'
        save_path = save_path_template % (dataset, trial)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + model_save_title)
        print(f'Model saved at {save_path + model_save_title}')

        training_history_path = training_history_path_template % (dataset, trial)
        if not os.path.exists(training_history_path):
            os.makedirs(training_history_path)
        training_history_save_title = f'original_model_{model.__class__.__name__}_training_history.pkl'
        with open(training_history_path + training_history_save_title, 'wb') as f:
            pickle.dump(train_history, f)
        print(f'History saved at {training_history_path + model_save_title}')

        # evaluate model
        for unlearn_class in unlearn_classes_set:
            unlearn_test_idx = get_idx_by_unlearn_class(raw_test_set.targets, [unlearn_class])
            retain_test_idx = np.setdiff1d(np.arange(len(raw_test_set)), unlearn_test_idx)

            test_res = test_all_in_one(model, copy.deepcopy(test_loader), unlearn_test_idx, retain_test_idx, unlearn_classes_set, device, output_activation=False)
            unlearn_acc = test_res['unlearn_acc'][0]
            remain_acc = test_res['remain_acc'][0]
            overall_acc = test_res['overall_acc'][0]
            print(f'Origin model performance: {unlearn_acc}, {remain_acc}, {overall_acc}')
