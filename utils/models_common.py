from glob import glob
import os
from pathlib import Path
import torch
import cv2
import torchvision
from tqdm import tqdm
import logging
import psutil
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def np_to_tensor(x, device):
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contigious().pin_memory().to(device=device, non_blocking=True)


class Nop(object):
    def nop(*args, **kw): pass

    def __getattr__(self, _): return self.nop


def train_(train_dataloader, eval_dataloader, caller):
    if caller.config["tensorboard"]:
        writer = SummaryWriter(caller.config["experiment_dir"])
    else:
        writer = Nop()

    max_score = 0
    old_path = None
    has_validated = False
    for epoch in range(caller.config["epochs"]):

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{caller.config["epochs"]}')
        caller.model.train()

        for (x, y, z) in pbar:
            if has_validated:
                caller.optimizer.zero_grad()

