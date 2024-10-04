import torch
import torch.nn as nn
from torch.cuda.amp import autocast


def reset_bn_stats(model, loader, device, epochs=1, load_type='loader'):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None  # use simple average
            m.reset_running_stats()
    # run a single train epoch with augmentations to recalc stats
    model.to(device)
    model.train()
    for _ in range(epochs):
        with torch.no_grad(), autocast():
            for images, _ in loader:
                output = model(images.to(device))