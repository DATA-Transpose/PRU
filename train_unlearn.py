import os
import time
import copy
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
    parser.add_argument('--unlearn-method', type=str, default='embedding_shift', choices=['retrain', 'embedding_shift',])
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)

    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    seed = args.seed
    unlearn_method = args.unlearn_method
    trials = args.trials
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prepared_data_path_template = './runs/prepared_data/%s/trial_%s/'
    save_path_template = './runs/unlearned_models/%s/trial_%s/uncls_%s/'
    # save_path_template = './runs/DEV_unlearned_models/%s/trial_%s/uncls_%s/'

    raw_train_set, raw_test_set = get_dataset(dataset)
    num_classes = len(raw_train_set.classes)

    # get model and training config
    CONFIGS = get_configs(dataset, args.model)
    unlearn_classes_set = CONFIGS['unlearn_classes_set']

    res = []
    for trial in range(trials):
        print(f'{"="*10} Trial {trial}, set seed {seed+trial} {"-"*10}')
        set_seed(seed + trial)
        in_trial_model_performance = []

        for unlearn_class in unlearn_classes_set:
            prepared_data_path = prepared_data_path_template % (dataset, trial)
            if not os.path.exists(prepared_data_path):
                raise ValueError(f'{prepared_data_path} does not exist!')
            save_path = save_path_template % (dataset, trial, unlearn_class)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            train_idx = np.load(prepared_data_path + 'train_idx.npy')
            val_idx = np.load(prepared_data_path + 'val_idx.npy')
            request_idx = np.load(prepared_data_path + 'request_idx.npy')

            # for simulating the unseen unlearning requests
            request_set = get_subset(raw_train_set, request_idx)
            # for training the original model
            train_set = get_subset(raw_train_set, train_idx)
            val_set = get_subset(raw_train_set, val_idx)
            # print statstic info
            print(f'Request set: {statstic_info(request_set)}')
            print(f'Train set: {statstic_info(train_set)}')
            print(f'Val set: {statstic_info(val_set)}')

            # prepare unlearning idx
            unlearn_train_idx = get_idx_by_unlearn_class(train_set.targets, [unlearn_class])
            unlearn_val_idx = get_idx_by_unlearn_class(val_set.targets, [unlearn_class])
            unlearn_request_idx = get_idx_by_unlearn_class(request_set.targets, [unlearn_class])
            unlearn_test_idx = get_idx_by_unlearn_class(raw_test_set.targets, [unlearn_class])

            retain_train_idx = np.setdiff1d(np.arange(len(train_set)), unlearn_train_idx)
            retain_val_idx = np.setdiff1d(np.arange(len(val_set)), unlearn_val_idx)
            retain_request_idx = np.setdiff1d(np.arange(len(request_set)), unlearn_request_idx)
            retain_test_idx = np.setdiff1d(np.arange(len(raw_test_set)), unlearn_test_idx)

            # prepare unlearning set
            unlearn_train_set = get_subset(train_set, unlearn_train_idx)
            unlearn_val_set = get_subset(val_set, unlearn_val_idx)
            unlearn_request_set = get_subset(request_set, unlearn_request_idx)
            # prepare retain set
            retain_train_set = get_subset(train_set, retain_train_idx)
            retain_val_set = get_subset(val_set, retain_val_idx)
            retain_request_set = get_subset(request_set, retain_request_idx)
            print(f'Unlearn train set: {statstic_info(unlearn_train_set)}')
            print(f'Retain train set: {statstic_info(retain_train_set)}')
            print(f'Unlearn val set: {statstic_info(unlearn_val_set)}')
            print(f'Retain val set: {statstic_info(retain_val_set)}')
            print(f'Unlearn request set: {statstic_info(unlearn_request_set)}')
            print(f'Retain request set: {statstic_info(retain_request_set)}')
            assert len(unlearn_train_set) + len(retain_train_set) == len(train_set)
            assert len(unlearn_val_set) + len(retain_val_set) == len(val_set)
            assert len(unlearn_request_set) + len(retain_request_set) == len(request_set)

            # load origin model
            ori_model = copy.deepcopy(CONFIGS['model']).to(device)
            model_save_title = f'original_model_{ori_model.__class__.__name__}.pt'
            ori_model.load_state_dict(torch.load(prepared_data_path + model_save_title))

            # prepare dataloader
            test_loader = get_dataloader(raw_test_set, CONFIGS['batch_size'], shuffle=False)

            # unlearning
            start_at = time.time()
            if unlearn_method == 'retrain':
                from utils.train_tools import fit_one_cycle
                retain_train_loader = get_dataloader(retain_train_set, CONFIGS['batch_size'], shuffle=True)
                retain_val_loader = get_dataloader(retain_val_set, CONFIGS['batch_size'], shuffle=True)
                unlearned_model = copy.deepcopy(CONFIGS['model']).to(device)
                hist, unlearn_model = fit_one_cycle(
                    CONFIGS['epochs'], unlearned_model, retain_train_loader, retain_val_loader,
                    CONFIGS['optimizer_type'], CONFIGS['optimizer_params'], CONFIGS['scheduler_type'], CONFIGS['scheduler_params'],
                    grad_clip=CONFIGS['grad_clip'], device=device, output_activation=False
                )
            elif unlearn_method == 'embedding_shift':
                from unlearn_methods.embedding_shift import embedding_shift_unlearning_correspond
                from config_unlearn import get_unlearn_configs
                UNLEARN_CONFIG = get_unlearn_configs(dataset, args.model)
                u_set_ratio = 1
                u_set = get_subset(unlearn_train_set, np.random.choice(len(unlearn_train_set), int(len(unlearn_train_set) * u_set_ratio), replace=False))
                print(f'len(u_set): {len(u_set)}')
                unlearn_train_loader = get_dataloader(
                    u_set,
                    CONFIGS['batch_size'] * 2, shuffle=True
                )
                # unlearn_train_loader = get_dataloader(unlearn_request_set, CONFIGS['batch_size'] * 2, shuffle=True)
                unlearn_model = embedding_shift_unlearning_correspond(ori_model, unlearn_train_loader, unlearn_class, device='cuda', **UNLEARN_CONFIG)


            end_at = time.time()
            print(f'Method: {unlearn_method} Unlearning time: {end_at - start_at:.4f}s')

            model_save_title = f'unlearn_model_{unlearn_method}_{unlearn_model.__class__.__name__}.pt'
            torch.save(unlearn_model.state_dict(), save_path + model_save_title)
            print(f'Model saved at {save_path + model_save_title}')

            # test unlearn model
            test_res = test_all_in_one(unlearn_model, copy.deepcopy(test_loader), unlearn_test_idx, retain_test_idx, unlearn_classes_set, device, output_activation=False)
            unlearn_acc = test_res['unlearn_acc'][0]
            remain_acc = test_res['remain_acc'][0]
            overall_acc = test_res['overall_acc'][0]
            print(f'{unlearn_method} unlearned model performance: {unlearn_acc}, {remain_acc}, {overall_acc}\n')
