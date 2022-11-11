from glob import glob
import os
from pathlib import Path
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize
import logging
import psutil
from torch.utils.tensorboard import SummaryWriter
from utils.droplet_retreiver import create_dataset, resize_patch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, ViTMAEModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def np_to_tensor(x, device):
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contigious().pin_memory().to(device=device, non_blocking=True)


def load_image_tensor(img_id, caller):
    image_path = f"{caller.config['train_dataset']}/img_{img_id}"
    image_tensor = T.ToTensor()(Image.open(image_path))
    image_tensor = image_tensor.unsqueeze(0)
    # print(image_tensor.shape)
    # input_images = image_tensor.to(device)
    return image_tensor


def compute_similar_images(img_id, embedding, caller, model):
    image_tensor = load_image_tensor(img_id)
    # image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = model.encoder(image_tensor).cpu().detach().numpy()

    # print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    # print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=caller.config["num_images"], metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list


def create_dataset_images(video, project_path, image_path, droplet_table_path, allFrames=False, allChannels=False,
                          buffer=3, suppress_rest=True, suppression_slack=1, discard_boundaries=False):
    dataset = create_dataset([0, 1, 2, 3, 4, 5, 6, 7], ['BF', 'DAPI', 'Cy5'], image_path, droplet_table_path, allFrames, allChannels,
                             buffer, suppress_rest, suppression_slack, discard_boundaries)
    DATA_PATH = os.path.join(project_path, "data")
    EXPERIMENT_DATA_DIR = os.path.join(DATA_PATH, str(video))
    try:
        os.mkdir(EXPERIMENT_DATA_DIR)
    except FileExistsError as _:
        pass
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            patch = resize_patch(dataset[i][j]['patch'], 224)
            np.save(os.path.join(EXPERIMENT_DATA_DIR, str(i + 1) + str(j).zfill(4)), patch)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def create_embeddings(dataloader, caller, model):

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

    def __init__(self, main_dir, transform=None, feat=False):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        self.feat = feat

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = np.load(img_loc).astype(float)
        norm_img = normalized(image)
        # image = image.astype(np.float32)

        if self.transform is not None:
            tensor_image = self.transform(image)
        elif not self.feat:
            tensor_image = torch.tensor(image)

        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
            tensor_image = feature_extractor(images = Image.fromarray(image.astype(float)*255, 'RGB'), return_tensors = 'pt')
            #print(tensor_image.shape)

        # tensor_image = torch.nn.functional.normalize(tensor_image, 1)

        return tensor_image.to(device, dtype=torch.float), tensor_image.to(device, torch.float)


def train_(train_dataloader, eval_dataloader, loss_fn, metric_fns, caller):
    if caller.config["tensorboard"] == True:
        writer = SummaryWriter(caller.config["experiment_dir"])
    else:
        writer = Nop()

    max_loss = 7000009000000000000
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
                loss, y_hat, mask = caller.model(x)  # forward pass
                y_hat = y_hat.cpu()
                #loss = loss_fn(y_hat, y)
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
                        #print(caller.model(x).shape)
                        loss, y_hat, mask = caller.model(x)
                        y_hat = y_hat.cpu()# forward pass
                        y = y.cpu()
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

#experiment = 2
#project_path = ""
#image_path = "data/Smallmvt1.nd2"
#droplet_table_path = "utils/droplets_and_cells/finished_outputs/smallMovement1_droplets.csv"

#create_dataset_images(experiment, project_path, image_path, droplet_table_path)

