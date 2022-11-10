import os
from glob import glob
import logging
import os
from pathlib import Path
from utils.models_common import FolderDataset, np_to_tensor, train_

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

PROJECT_PATH = Path(os.path.dirname(os.path.dirname(__file__)))


# noinspection SpellCheckingInspection
class ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"][0]

        layers = []
        prev_channels = self.img_size

        for i, n_channels in enumerate(config["conv_layers"]):
            layers += [nn.Conv2d(prev_channels, n_channels, (3, 3), padding=(1, 1))]
            layers += [nn.ReLU(inplace=True)]
            if i%1==0:
                layers += [nn.MaxPool2d(2,2)]
            prev_channels = n_channels
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        x = self.layers(x)
        #print(x.shape)
        return x


# noinspection SpellCheckingInspection
class ConvDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"][0]
        layers = []
        prev_channels = 0

        for i in range(len(config["conv_layers"])-1, 0, -1):
            if i >= 2:
                layers += [nn.ConvTranspose2d(config["conv_layers"][i], config["conv_layers"][i-1], (2,2), stride=(2, 2))]
            else:
                layers += [nn.ConvTranspose2d(config["conv_layers"][i], config["conv_layers"][i-1], (2,2), stride=(2, 2), output_padding=1)]
            layers += [nn.ReLU(inplace=True)]
            prev_channels = config["conv_layers"][i-1]
        layers += [nn.ConvTranspose2d(prev_channels, self.img_size, (2,2), stride=(2, 2), output_padding=1, dilation=2)]
        layers += [nn.ReLU(inplace=True)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #print("Decoder Start", x.shape)
        x = self.layers(x)
        #print("Decoder End", x.shape)
        return x



class AutoEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = ConvEncoder(config)
        self.decoder = ConvDecoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AE():
    def __init__(self, config):
        self.config = config
        self.step = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # automatically select device
        self.model = AutoEncoder(config).to(self.device)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def save_model(self, appendix=''):
        path = self.config['experiment_dir'] / f'checkpoint_{self.step}_{appendix}.pt'
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_sate_dict': self.optimizer.state_dict(),
        }, path)

    def train(self):
        #transforms = T.Compose([T.ToTensor()])
        transforms = None
        train_dataset = FolderDataset(self.config["train_dataset"], transforms)
        val_dataset = FolderDataset(self.config["val_dataset"], transforms)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
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
