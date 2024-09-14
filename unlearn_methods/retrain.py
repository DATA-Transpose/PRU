import copy
from utils.train_tools import fit_one_cycle


def retrain(
        epochs, max_lr, model, train_loader, val_loader, weight_decay, grad_clip, opt_func, device, output_activation=True
):
    retrained_model = copy.deepcopy(model).to(device)
    retrained_model, model_history = fit_one_cycle(epochs, max_lr, retrained_model, train_loader, val_loader, weight_decay, grad_clip, opt_func, device, output_activation=True)

    return retrained_model
