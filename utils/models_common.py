from glob import glob
import os
from pathlib import Path
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import logging
import psutil
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def np_to_tensor(x, device):
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contigious().pin_memory().to(device=device, non_blocking=True)


def create_embeddings(model, dataloader, caller):
    model.encoder.eval()
    embedding = torch.randn(caller.config["img_shape"])

    with torch.no_grad():
        print("Creating embeddings ...")
        pbar = tqdm(dataloader)
        for train_img, target_img in pbar:
            train_img = train_img.to(caller.device)
            enc_output = model.encoder(train_img).cpu()
            embedding = torch.cat((embedding, enc_output), 0)

    return embedding


class Nop(object):
    def nop(*args, **kw): pass

    def __getattr__(self, _): return self.nop


class FolderDataset(Dataset):

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        print(self.main_dir)
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image.to(device), tensor_image.to(device)


def train_(train_dataloader, eval_dataloader, loss_fn, metric_fns, caller):
    if caller.config["tensorboard"] == True:
        writer = SummaryWriter(caller.config["experiment_dir"])
    else:
        writer = Nop()

    max_loss = 700000
    old_path = None
    has_validated = False
    for epoch in range(caller.config["epochs"]):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': [], 'memory': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_' + k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{caller.config["epochs"]}')
        # training
        caller.model.train()
        for (x, y) in pbar:
            if has_validated:
                caller.optimizer.zero_grad()  # zero out gradients
                y_hat = caller.model(x)  # forward pass
                loss = loss_fn(y_hat, y)
                loss.backward()  # backward pass
                caller.optimizer.step()  # optimize weights

                metrics['loss'].append(loss.item())
                writer.add_scalar('loss', loss.item(), caller.step)
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                metrics['memory'].append(mem)
                writer.add_scalar('memory', mem, caller.step)
                for k, fn in metric_fns.items():
                    v = fn(y_hat, y).item()
                    metrics[k].append(v)
                    writer.add_scalar(k, v, caller.step)

            pbar.set_postfix({k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0 and k in ['loss', 'acc']})
            if caller.step % caller.config["steps_per_checkpoint"] == 0:
                caller.save_model()
            if (not has_validated) or caller.step % caller.config["steps_per_validation"] == 0:
                has_validated = True
                # validation
                caller.model.eval()
                with torch.no_grad():  # do not keep track of gradients
                    metrics_val = {'val_loss': []}
                    for k, _ in metric_fns.items():
                        metrics_val['val_' + k] = []
                    for (x, y) in tqdm(eval_dataloader, desc="validating model"):
                        y_hat = caller.model(x).cpu()  # forward pass
                        y = y.cpu()
                        loss = loss_fn(y_hat, y)
                        # print(y_hat.shape)
                        # print(y.shape)
                        # print(x.shape)
                        metrics_val['val_loss'].append(loss.item())
                        for k, fn in metric_fns.items():
                            v = fn(y_hat, y).item()
                            metrics_val['val_' + k].append(v)

                    for k, v in metrics_val.items():
                        writer.add_scalar(k, sum(v) / len(v), caller.step)

                    logging.info(' '.join(
                        ['\t- ' + str(k) + ' = ' + str(sum(v) / len(v)) + '\n ' for (k, v) in metrics_val.items()]))

                    val_loss = max([metrics_val[f'val_loss'][0]])
                    logging.info(f"val_f1 = {val_loss}, max_val_f1 = {max_loss}")

                    if val_loss < max_loss:
                        print("Saving model.")
                        new_path = caller.save_model(str(val_loss))
                        if old_path is not None:
                            os.remove(old_path)
                        old_path = new_path
                        max_loss = val_loss

            caller.step += 1
    print("Saving model.")
    caller.save_model()

    logging.info('Finished Training')
