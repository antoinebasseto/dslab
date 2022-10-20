import os
from glob import glob
import logging
import os
from pathlib import Path
from utils.models_common import FolderDataset, np_to_tensor, train_

import torch
import torch.nn as nn
import torchvision.transforms as T

PROJECT_PATH = Path(os.path.dirname(os.path.dirname(__file__)))


# noinspection SpellCheckingInspection
class ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"][0]
        self.conv1 = nn.Conv2d(self.img_size, 16, (2, 2))

        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (2, 2))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (2, 2))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (2, 2))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(128, 256, (2, 2))
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        # Downscale the image with conv maxpool etc.
        #print("1", x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        #x = self.maxpool1(x)

        #print("2", x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        #x = self.maxpool2(x)

        #print("3: ",x.shape)

        x = self.conv3(x)
        x = self.relu3(x)
        #x = self.maxpool3(x)

        #print("4", x.shape)

        #x = self.conv4(x)
        #x = self.relu4(x)
        #x = self.maxpool4(x)

        #print("5", x.shape)

        #x = self.conv5(x)
        #x = self.relu5(x)
        #x = self.maxpool5(x)

        #print("6: ",x.shape)
        return x


# noinspection SpellCheckingInspection
class ConvDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"][0]
        self.deconv1 = nn.ConvTranspose2d(256, 128, (2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(64, 32, (2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(32, 16, (2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose2d(16, self.img_size, (2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        #print("1: ",x.shape)
        #x = self.deconv1(x)
        #x = self.relu1(x)
        #print("1: ",x.shape)

        #x = self.deconv2(x)
        #x = self.relu2(x)
        #print("1: ", x.shape)

        x = self.deconv3(x)
        x = self.relu3(x)
        #print("1: ",x.shape)

        x = self.deconv4(x)
        x = self.relu4(x)
        #print("1: ",x.shape)

        x = self.deconv5(x)
        x = self.relu5(x)
        #print("1: ",x.shape)
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


class DeepRanking():
    def __init__(self, config):
        self.config = config
        self.step = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # automatically select device
        self.model = AutoEncoder(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def save_model(self, appendix=''):
        path = self.config['experiment_dir'] / f'checkpoint_{self.step}_{appendix}.pt'
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_sate_dict': self.optimizer.state_dict(),
        }, path)

    def train(self):
        transforms = T.Compose([T.ToTensor()])
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
