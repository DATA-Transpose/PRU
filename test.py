import os
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
    parser.add_argument('--dataset',type=str, default='cifar10', choices=['mnist', 'mnistFashion', 'mnistKuzushiji', 'cifar10'])
    parser.add_argument('--model', type=str, default='AllCNN', choices=['AllCNN', 'ResNet18', 'ViT'])
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)

    args = parser.parse_args()
    print(args)

    methods_list = [
        'original_model',
        'unlearn_model_retrain',
        'unlearn_model_embedding_shift',
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = args.seed
    dataset_name = args.dataset
    model_name = args.model
    trials = args.trials

    CONFIG = get_configs(dataset_name, model_name)
    unlearn_classes_set = CONFIG['unlearn_classes_set']

    prepared_data_path_template = './runs/prepared_data/%s/trial_%s/'
    save_path_template = './runs/unlearned_models/%s/trial_%s/uncls_%s/'

    raw_train_set, raw_test_set = get_dataset(dataset_name)
    num_classes = len(raw_train_set.classes)

    res = {}
    for m in methods_list:
        res[m] = []

    for trial in range(trials):
        print('=' * 20 + f'Trial {trial}, set seed {seed + trial}' + '-' * 20)
        set_seed(seed + trial)

        in_trial_performances = {}
        for m in methods_list:
            in_trial_performances[m] = []

        for unlearn_class in unlearn_classes_set:
            print(f'{"-" * 10} Unlearn class {unlearn_class} {"-" * 10}')
            save_path = save_path_template % (dataset_name, trial, unlearn_class)

            unlearn_test_idx = get_idx_by_unlearn_class(raw_test_set.targets, [unlearn_class])
            retain_test_idx = np.setdiff1d(np.arange(len(raw_test_set)), unlearn_test_idx)
            test_loader = get_dataloader(raw_test_set, batch_size=CONFIG['batch_size'], shuffle=False)

            # load models
            for m in methods_list:
                model = CONFIG['model'].to(device)
                try:
                    if m == 'original_model':
                        prepared_data_path = prepared_data_path_template % (dataset_name, trial)
                        print(prepared_data_path + f'{m}_{model.__class__.__name__}.pt')
                        model.load_state_dict(torch.load(prepared_data_path + f'{m}_{model.__class__.__name__}.pt'))
                    else:
                        model.load_state_dict(torch.load(save_path + f'{m}_{model.__class__.__name__}.pt'))

                    test_res = test_all_in_one(model, copy.deepcopy(test_loader), unlearn_test_idx, retain_test_idx, unlearn_classes_set, device)
                    unlearn_acc = test_res['unlearn_acc'][0]
                    remain_acc = test_res['remain_acc'][0]
                    overall_acc = test_res['overall_acc'][0]
                    in_trial_performances[m].append([unlearn_acc, remain_acc, overall_acc])
                    print(f'{m} unlearn_acc: {unlearn_acc:.4f}, remain_acc: {remain_acc:.4f}, overall_acc: {overall_acc:.4f}')
                except:
                    print(f'{m} does not exist!')
                    in_trial_performances[m].append([-1, -1, -1])
                    continue

        for m in methods_list:
            res[m].append(in_trial_performances[m])

    for m in methods_list:
        res[m] = np.array(res[m])
        # print(res[m].shape)
        unlearn_acc = res[m][:, :, 0].copy()
        remain_acc = res[m][:, :, 1].copy()
        overall_acc = res[m][:, :, 2].copy()
        punlearn_acc = np.mean(unlearn_acc, axis=0).tolist()
        premain_acc = np.mean(remain_acc, axis=0).tolist()
        poverall_acc = np.mean(overall_acc, axis=0).tolist()

        print(f'{m} unlearn_acc: {unlearn_acc.mean():.4f} +- {unlearn_acc.std():.4f}')
        print(f'{m} remain_acc: {remain_acc.mean():.4f} +- {remain_acc.std():.4f}')
        print(f'{m} overall_acc: {remain_acc.mean():.4f} +- {remain_acc.std():.4f}')
        print('-' * 50)
    
    save_path = f'./runs/test_res/{dataset_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for m in methods_list:
        np.save(save_path + f'{m}_{model.__class__.__name__}.npy', res[m])