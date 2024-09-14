import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from models.AllCNN import AllCNN
from models.ResNet import ResNet
from models.ViT import ViT

################
# Section 4.2.1
################
def embedding_shift(ori_model, model, loader, forget_class, epoch=5, lr=1e-4, lamb=10, device='cuda', CELoss=False, no_cluster=False, no_sample=False, random_label=False):
    reference_model = copy.deepcopy(ori_model)
    reference_model.eval()
    # fix the reference model
    for name, param in reference_model.named_parameters():
        param.requires_grad = False

    unlearn_model = copy.deepcopy(model)
    unlearn_model.train()
    # fix the classifier
    for name, param in unlearn_model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    for epoch in range(epoch):
        in_epoch_loss = []
        in_epoch_losses = []

        optimizer = optim.SGD(unlearn_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        for batch_idx, iter_item in enumerate(loader):
            if isinstance(ori_model, AllCNN) or isinstance(ori_model, ResNet) or isinstance(ori_model, ViT):
                data, target = iter_item
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # get embeddings
            if isinstance(ori_model, AllCNN) or isinstance(ori_model, ResNet) or isinstance(ori_model, ViT):
                ref_emb = reference_model.get_representation(data)
                unl_emb = unlearn_model.get_representation(data)

            # get supervision information from reference model
            if isinstance(ori_model, AllCNN) or isinstance(ori_model, ResNet) or isinstance(ori_model, ViT):
                output = reference_model(data)
            prob = F.softmax(output, dim=1)
            unlearn_prob = prob[:, forget_class]
            unlearn_prob = torch.stack([unlearn_prob] * prob.shape[1], dim=1)
            prob[:, forget_class] = 0
            reasign_ref = prob / prob.sum(dim=1, keepdim=True)
            adding_reasign = reasign_ref * unlearn_prob
            reasigned_prob = prob + adding_reasign
            reasigned_prob = reasigned_prob / reasigned_prob.sum(dim=1, keepdim=True)
            # reasigned_label
            if random_label:
                reasigned_label = target
                CELoss = True
            else:
                reasigned_label = torch.argmax(reasigned_prob, dim=1)

            ################
            # 1. enlarge clustering distance. Eq. 3
            ################
            unique_reasigned_labels = torch.unique(reasigned_label)
            grouped_embs = []
            for unique_reasigned_label in unique_reasigned_labels:
                idx = reasigned_label.eq(unique_reasigned_label)
                emd_for_single_cls = unl_emb[idx]
                grouped_embs.append(torch.mean(emd_for_single_cls, dim=0))
            grouped_embs = torch.stack(grouped_embs)
            cluster_dist = torch.cdist(grouped_embs, grouped_embs, p=2)
            cluster_dist = torch.sum(cluster_dist) / (cluster_dist.shape[0] * (cluster_dist.shape[1] - 1) + 1e-6)
            assert lamb > 0
            cluster_dist /= lamb
            cluster_dist_loss = (1 / (cluster_dist + 1e-6))

            ################
            # 2. min sample distance. Eq. 4
            ################
            unique_reasigned_labels = torch.unique(reasigned_label)
            sample_dist = 0
            single_sample_cluster_count = 0
            for unique_reasigned_label in unique_reasigned_labels:
                idx = reasigned_label.eq(unique_reasigned_label)
                emd_for_single_cls = unl_emb[idx]
                dist = torch.cdist(emd_for_single_cls, emd_for_single_cls, p=2)
                if dist.shape[0] == 1:
                    single_sample_cluster_count += 1
                    continue
                dist = dist.sum() / (dist.shape[0] * (dist.shape[1] - 1))
                sample_dist += dist
            if len(unique_reasigned_labels) - single_sample_cluster_count == 0:
                sample_dist_loss = torch.tensor(0.0).to(device)
            else:
                sample_dist_loss = sample_dist / (len(unique_reasigned_labels) - single_sample_cluster_count)

            # 3. Task loss
            if isinstance(ori_model, AllCNN) or isinstance(ori_model, ResNet) or isinstance(ori_model, ViT):
                output = unlearn_model(data)
            if CELoss:
                loss = F.cross_entropy(output, reasigned_label)
            else:
                ################
                # Eq. 5
                ################
                loss = F.kl_div(torch.log(F.softmax(output, dim=1) + 1e-6), reasigned_prob, reduction='batchmean')

            # Final loss
            in_epoch_losses.append([loss.item(), cluster_dist_loss.item(), sample_dist_loss.item()])

            c_idx = 1 if not no_cluster else 0
            s_idx = 1 if not no_sample else 0

            loss = loss + c_idx * cluster_dist_loss + s_idx * sample_dist_loss
            in_epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        in_epoch_losses = np.array(in_epoch_losses)

    return unlearn_model

################
# Section 4.2.2
################
def boundary_refine(ori_model, model, loader, forget_class, epoch=3, lr=1e-2, device='cuda', CELoss=False, random_label=False):
    reference_model = copy.deepcopy(ori_model)
    reference_model.eval()
    # fix the reference model
    for name, param in reference_model.named_parameters():
        param.requires_grad = False

    unlearn_model = copy.deepcopy(model)
    unlearn_model.train()
    # fix the feature extractor
    for name, param in unlearn_model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for epoch in range(epoch):
        in_epoch_loss = []
        in_epoch_losses = []

        optimizer = optim.SGD(unlearn_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        for batch_idx, iter_item in enumerate(loader):
            if isinstance(ori_model, AllCNN) or isinstance(ori_model, ResNet) or isinstance(ori_model, ViT):
                data, target = iter_item
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # get supervision information from reference model
            if isinstance(ori_model, AllCNN) or isinstance(ori_model, ResNet) or isinstance(ori_model, ViT):
                output = reference_model(data)
            prob = F.softmax(output, dim=1)
            unlearn_prob = prob[:, forget_class]
            unlearn_prob = torch.stack([unlearn_prob] * prob.shape[1], dim=1)
            prob[:, forget_class] = 0
            reasign_ref = prob / prob.sum(dim=1, keepdim=True)
            adding_reasign = reasign_ref * unlearn_prob
            reasigned_prob = prob + adding_reasign
            reasigned_prob = reasigned_prob / reasigned_prob.sum(dim=1, keepdim=True)
            # reasigned_label
            if random_label:
                reasigned_label = target
                CELoss = True
            else:
                reasigned_label = torch.argmax(reasigned_prob, dim=1)

            # classification loss
            if isinstance(ori_model, AllCNN) or isinstance(ori_model, ResNet) or isinstance(ori_model, ViT):
                output = unlearn_model(data)

            if CELoss:
                loss = F.cross_entropy(output, reasigned_label)
            else:
                loss = F.kl_div(torch.log(F.softmax(output, dim=1) + 1e-6), reasigned_prob, reduction='batchmean')

            in_epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()


    return unlearn_model



def embedding_shift_unlearning_correspond(
        ori_model, loader, forget_class,
        shift_epoch=5, shift_lr=1e-4, shift_lamb=10,
        refine_epoch=3, refine_lr=1e-2,
        device='cuda',
        CELoss=False, no_cluster=False, no_sample=False, random_label=False
):
    reference_model = copy.deepcopy(ori_model)
    unlearn_model = copy.deepcopy(ori_model)
    if random_label:
        retaining_class = [i for i in range(10) if i != forget_class]
        random_y = np.random.choice(retaining_class, len(loader.dataset.targets))
        loader.dataset.targets = random_y
        CELoss = True
    for e in tqdm(range(shift_epoch)):
        # Section 4.2.1
        unlearn_model = embedding_shift(reference_model, unlearn_model, copy.deepcopy(loader), forget_class, 1, shift_lr, shift_lamb, device, CELoss=CELoss, no_cluster=no_cluster, no_sample=no_sample, random_label=random_label)
        # Section 4.2.2
        unlearn_model = boundary_refine(reference_model, unlearn_model, copy.deepcopy(loader), forget_class, refine_epoch, refine_lr, device, CELoss=CELoss, random_label=random_label)

    return unlearn_model
