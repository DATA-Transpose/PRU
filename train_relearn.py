
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

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='mnist', choices=['mnist', 'mnistFashion', 'mnistKuzushiji', 'cifar10'])
    parser.add_argument('--model', type=str, default='AllCNN', choices=['AllCNN', 'ResNet18', 'ViT'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)

    args = parser.parse_args()
    print(args)

    methods_list = [
        'original_model',
        'unlearn_model_retrain',
        'unlearn_model_embedding_shift',
    ]

    res = {}
    for m in methods_list:
        res[m] = []

    dataset = args.dataset
    seed = args.seed
    trials = args.trials
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prepared_data_path_template = './runs/prepared_data/%s/trial_%s/'
    save_path_template = './runs/unlearned_models/%s/trial_%s/uncls_%s/'
    relearn_lr = args.lr
    relearn_epochs = args.epochs

    raw_train_set, raw_test_set = get_dataset(dataset)
    num_classes = len(raw_train_set.classes)

    # get model and training config
    CONFIGS = get_configs(dataset, args.model)
    unlearn_classes_set = CONFIGS['unlearn_classes_set']
    # unlearn_classes_set = [0]

    for trial in range(trials):
        print(f'{"="*10} Trial {trial}, set seed {seed+trial} {"-"*10}')
        set_seed(seed + trial)
        in_trial_model_performance = []
        # for cls in range(num_classes):

        in_trial_performances = {}
        for m in methods_list:
            in_trial_performances[m] = []

        for unlearn_class in unlearn_classes_set:
            print(f'{"-"*10} Unlearn class {unlearn_class} {"-"*10}')
            save_path = prepared_data_path_template % (dataset, trial)
            if not os.path.exists(save_path):
                raise ValueError(f'{save_path} does not exist!')
            train_idx = np.load(save_path + 'train_idx.npy')
            request_idx = np.load(save_path + 'request_idx.npy')
            val_idx = np.load(save_path + 'val_idx.npy')

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


            for m in methods_list:
                print(f'{"-"*10} Method {m} {"-"*10}')
                method_res = [] # epoch * 3
                # load unlearned model
                unlearn_model = CONFIGS['model']
                save_path = save_path_template % (dataset, trial, unlearn_class)
                model_save_title = f'{m}_{unlearn_model.__class__.__name__}.pt'
                if m == 'original_model':
                    unlearn_model.load_state_dict(torch.load(prepared_data_path_template % (dataset, trial) + model_save_title))
                else:
                    unlearn_model.load_state_dict(torch.load(save_path + model_save_title))
                unlearn_model = unlearn_model.to(device)


                # relearn_lr = 0.0001
                # CONFIGS['batch_size'] = 64

                # split retain_request_set into train and val
                rand_idx = np.random.permutation(len(retain_request_set))
                retain_request_set_train = get_subset(retain_request_set, rand_idx[:int(len(rand_idx) * 0.8)])
                retain_request_set_val = get_subset(retain_request_set, rand_idx[int(len(rand_idx) * 0.8):])

                retain_request_loader = get_dataloader(retain_request_set_train, CONFIGS['batch_size'], shuffle=True)
                val_loader = get_dataloader(retain_request_set_val, CONFIGS['batch_size'], shuffle=True)
                test_loader = get_dataloader(raw_test_set, CONFIGS['batch_size'], shuffle=False)

                # relearned_model = relearn(unlearn_model, retain_request_loader, relearn_epochs, relearn_lr, device=device)
                relearned_model = copy.deepcopy(unlearn_model)
                CONFIGS['optimizer_params']['lr'] = relearn_lr
                for e in tqdm(range(relearn_epochs)):
                    # relearned_model = relearn(relearned_model, copy.deepcopy(retain_request_loader), 1, relearn_lr, device=device)
                    train_history, relearned_model = CONFIGS['training_func'](
                        1, relearned_model, copy.deepcopy(retain_request_loader), val_loader,
                        CONFIGS['optimizer_type'], CONFIGS['optimizer_params'],
                        CONFIGS['scheduler_type'], CONFIGS['scheduler_params'],
                        grad_clip=CONFIGS['grad_clip'], device=device, output_activation=False
                    )

                    if e in [0, 4, 9]:
                        test_res = test_all_in_one(relearned_model, copy.deepcopy(test_loader), unlearn_test_idx, retain_test_idx, unlearn_classes_set, device)
                        unlearn_acc = test_res['unlearn_acc'][0]
                        remain_acc = test_res['remain_acc'][0]
                        overall_acc = test_res['overall_acc'][0]

                        method_res.append([unlearn_acc, remain_acc, overall_acc])

                        print(f'Relearn epoch {e}: \t{np.round(unlearn_acc, 4)} \t {np.round(remain_acc, 4)} \t {np.round(overall_acc, 4)}')

                in_trial_performances[m].append(method_res)
        for m in methods_list:
            res[m].append(in_trial_performances[m])
    for m in methods_list:
        res[m] = np.array(res[m])
        print(res[m].shape)
        unlearn_acc = res[m][:, :, :, 0].copy()
        remain_acc = res[m][:, :, :, 1].copy()
        overall_acc = res[m][:, :, :, 2].copy()

        def get_mean(data):
            data = data.copy()
            data = np.mean(data, axis=1)
            data = np.mean(data, axis=0)
            return data
        unlearn_acc = get_mean(unlearn_acc)
        remain_acc = get_mean(remain_acc)
        overall_acc = get_mean(overall_acc)
        print(f'{m} unlearn_acc: {unlearn_acc}')
        print(f'{m} remain_acc: {remain_acc}')
        print(f'{m} overall_acc: {overall_acc}')
        print('-' * 50)

    relearn_path_template = f'./runs/relearn_records/{dataset}/'
    if not os.path.exists(relearn_path_template):
        os.makedirs(relearn_path_template)
    for m in methods_list:
        np.save(relearn_path_template + f'{m}_{unlearn_model.__class__.__name__}.npy', res[m])
