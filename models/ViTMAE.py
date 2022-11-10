import os
from glob import glob
import logging
import os
from pathlib import Path
from utils.models_common import FolderDataset, np_to_tensor, train_

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, ViTMAEForPreTraining
import torchvision
import torchvision.transforms as T

PROJECT_PATH = Path(os.path.dirname(os.path.dirname(__file__)))


class ViTMAE():
    def __init__(self, config):
        self.config = config
        self.step = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # automatically select device
        self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def save_model(self, appendix=''):
        path = self.config['experiment_dir'] / f'checkpoint_{self.step}_{appendix}.pt'
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_sate_dict': self.optimizer.state_dict(),
        }, path)

    def train(self):
        transforms = None
        train_dataset = FolderDataset(self.config["train_dataset"], transforms, feat = True)
        val_dataset = FolderDataset(self.config["val_dataset"], transforms, feat = True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
        loss_fn = nn.BCELoss()
        metric_fns = {}

        def loss_fn_bce(y_hat, y):
            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(y_hat, y)
            return loss

        def loss_fn_mse(y_hat, y):
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(y_hat, y)
            return loss

        loss_fn = loss_fn_bce if self.config['loss'] == 'bce' else loss_fn_mse
        train_(train_dataloader, val_dataloader, loss_fn, metric_fns, self)
