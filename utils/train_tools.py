import time
import copy

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def training_step(model, batch, device, output_activation=True):
    images, labels = batch 
    images, labels = images.to(device), labels.to(device)
    if output_activation:
        out, *_ = model(images)                  # Generate predictions
    else:
        out = model(images)
    loss = F.cross_entropy(out, labels) # Calculate loss
    return loss

def validation_step(model, batch, device, output_activation=True):
    images, labels = batch 
    images, labels = images.to(device), labels.to(device)
    if output_activation:
        out, *_ = model(images)                    # Generate predictions
    else:
        out = model(images)
    loss = F.cross_entropy(out, labels)   # Calculate loss
    acc = accuracy(out, labels)           # Calculate accuracy
    return {'Loss': loss.detach(), 'Acc': acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}

def epoch_end(model, epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['Loss'], result['Acc']))
    
@torch.no_grad()
def evaluate(model, val_loader, device='cuda', output_activation=True):
    model.eval()
    outputs = [validation_step(model, batch, device, output_activation=output_activation) for batch in val_loader]
    return validation_epoch_end(model, outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(
        epochs, model, train_loader, val_loader,
        optimizer_type, optimizer_params,
        scheduler_type, scheduler_params,
        grad_clip=None, device='cuda', output_activation=False
):
    torch.cuda.empty_cache()
    history = {
        'training_loss': [],
        'val_loss': [],
        'val_acc': [],
        'lrs': [],
    }

    optimizer = optimizer_type(model.parameters(), **optimizer_params)
    if scheduler_type == torch.optim.lr_scheduler.OneCycleLR:
        scheduler = scheduler_type(optimizer, **scheduler_params, steps_per_epoch=len(train_loader))
    
    for epoch in tqdm(range(epochs)): 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device, output_activation=output_activation)
            train_losses.append(loss.item())
            loss.backward()
            
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            if scheduler_type == torch.optim.lr_scheduler.OneCycleLR:
                scheduler.step()
            
        
        # Validation phase
        result = evaluate(model, val_loader, device, output_activation=output_activation)
        history['training_loss'].append(np.mean(train_losses))
        history['lrs'].append(np.mean(lrs))
        history['val_loss'].append(result['Loss'])
        history['val_acc'].append(result['Acc'])
    return history, model
