import torch
import torch.nn.functional as F

import numpy as np

def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def precision(y_pred, y_true, unlearn_classes_set):
    res = np.zeros(len(unlearn_classes_set))
    for idx, cls in enumerate(unlearn_classes_set):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        if tp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        res[idx] = precision
    return res

def recall(y_pred, y_true, unlearn_classes_set):
    res = np.zeros(len(unlearn_classes_set))
    for idx, cls in enumerate(unlearn_classes_set):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        if tp == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        res[idx] = recall
    return res

def f1_score(y_pred, y_true, unlearn_classes_set):
    res = np.zeros(len(unlearn_classes_set))
    for idx, cls in enumerate(unlearn_classes_set):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        if tp == 0:
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        res[idx] = f1
    return res

@torch.no_grad()
def test_all_in_one(net, loader, unlearn_idxs, retain_idxs, unlearn_classes_set, device, output_activation=False):
    net.eval()
    pred = np.array([])
    label = np.array([])
    record_loss = np.array([])
    logits = []
    with torch.no_grad():
        if 'Text' not in net.__class__.__name__:
            for i, (x, y) in enumerate(loader):
                x = torch.tensor(x.clone().detach().numpy(), dtype=torch.float32).to(device)
                y = y.to(device)
                # y = y.cpu().numpy()
                label = np.concatenate((label, y.cpu().numpy()))

                if output_activation:
                    y_pred, *_ = net(x)
                else:
                    y_pred = net(x)
                softmax_logits = F.softmax(y_pred, dim=1)
                logits.append(y_pred.cpu().detach().numpy())

                loss = F.cross_entropy(y_pred, y, reduction='none')
                record_loss = np.concatenate((record_loss, loss.cpu().numpy()))
                _, predicted = torch.max(y_pred.data, 1)
                predicted = predicted.cpu().numpy()
                pred = np.concatenate((pred, predicted))
        else:
            for i, (y, x, offsets) in enumerate(loader):
                y, x, offsets = y.to(device), x.to(device), offsets.to(device)
                label = np.concatenate((label, y.cpu().numpy()))

                y_pred = net(x, offsets)
                
                softmax_logits = F.softmax(y_pred, dim=1)
                logits.append(y_pred.cpu().detach().numpy())

                loss = F.cross_entropy(y_pred, y, reduction='none')
                record_loss =  np.concatenate((record_loss, loss.cpu().numpy()))
                _, predicted = torch.max(y_pred.data, 1)
                predicted = predicted.cpu().numpy()
                pred = np.concatenate((pred, predicted))

    # record logits
    logits = np.concatenate(logits)
    # print(logits.shape)
    logits_by_class = {}
    for cls in unlearn_classes_set:
        logits_by_class[cls] = logits[label == cls]
    # sort logits_by_class by key
    logits_by_class = dict(sorted(logits_by_class.items(), key=lambda x: x[0]))

    res = {}
    # unlearn acc
    unlearn_pred = pred[unlearn_idxs]
    unlearn_label = label[unlearn_idxs]
    unlearn_loss = record_loss[unlearn_idxs].mean()
    acc = accuracy(unlearn_pred, unlearn_label)
    p = precision(unlearn_pred, unlearn_label, unlearn_classes_set)
    r = recall(unlearn_pred, unlearn_label, unlearn_classes_set)
    f1 = f1_score(unlearn_pred, unlearn_label, unlearn_classes_set)
    res['unlearn_acc'] = [acc, p, r, f1, unlearn_loss]

    # remain acc
    remain_pred = pred[retain_idxs]
    remain_label = label[retain_idxs]
    remain_loss = record_loss[retain_idxs].mean()
    acc = accuracy(remain_pred, remain_label)
    p = precision(remain_pred, remain_label, unlearn_classes_set)
    r = recall(remain_pred, remain_label, unlearn_classes_set)
    f1 = f1_score(remain_pred, remain_label, unlearn_classes_set)
    res['remain_acc'] = [acc, p, r, f1, remain_loss]

    # overall acc
    overall_pred = pred[np.concatenate((unlearn_idxs, retain_idxs))]
    overall_label = label[np.concatenate((unlearn_idxs, retain_idxs))]
    overall_loss = record_loss[np.concatenate((unlearn_idxs, retain_idxs))].mean()
    acc = accuracy(overall_pred, overall_label)
    p = precision(overall_pred, overall_label, unlearn_classes_set)
    r = recall(overall_pred, overall_label, unlearn_classes_set)
    f1 = f1_score(overall_pred, overall_label, unlearn_classes_set)
    res['overall_acc'] = [acc, p, r, f1, overall_loss]

    res['loss'] = np.mean(record_loss)
    res['logits_by_class'] = logits_by_class
    res['logits'] = logits
    # print(f'my loss: {np.mean(record_loss)}')

    return res


@torch.no_grad()
def test(net, loader, idxs, class_count, device):
    net.eval()
    pred = np.array([])
    label = np.array([])
    record_loss = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = torch.tensor(x.clone().detach().numpy(), dtype=torch.float32).to(device)
            y = y.to(device)
            # y = y.cpu().numpy()
            label = np.concatenate((label, y.cpu().numpy()))

            y_pred, *_ = net(x)

            loss = F.cross_entropy(y_pred, y)
            record_loss.append(loss.item())
            _, predicted = torch.max(y_pred.data, 1)
            predicted = predicted.cpu().numpy()
            pred = np.concatenate((pred, predicted))

    pred = pred[idxs]
    label = label[idxs]
    
    # accuracy
    acc = accuracy(pred, label)

    # precision
    p = precision(pred, label, class_count)

    # recall
    r = recall(pred, label, class_count)

    # f1 score
    f1 = f1_score(pred, label, class_count)
    # save pred
    # np.save('pred.npy', pred)
    # np.save('label.npy', label)
    
    return acc, p, r, f1, np.mean(record_loss)
        