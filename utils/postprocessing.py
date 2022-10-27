import torch
import numpy as np
from models.AE import AutoEncoder
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T


def load_image_tensor(img_id, caller):
    image_path = f"{caller.config['train_dataset']}/img_{img_id}"
    image_tensor = T.ToTensor()(Image.open(image_path))
    image_tensor = image_tensor.unsqueeze(0)
    # print(image_tensor.shape)
    # input_images = image_tensor.to(device)
    return image_tensor


def compute_similar_images(img_id, embedding, caller):
    image_tensor = load_image_tensor(img_id)
    # image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = caller.model.encoder(image_tensor).cpu().detach().numpy()

    # print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    # print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=caller.config["num_images"], metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list


def plot_similar_images(indices_list, caller):
    indices = indices_list[0]
    for index in indices:
        if index == 0:
            # index 0 is a dummy embedding.
            pass
        else:
            img_name = str(index - 1) + ".jpg"
            img_path = os.path.join(caller.config["train_dataset"] + img_name)
            # print(img_path)
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.show()
            img.save(f"../outputs/query_image_3/recommended_{index - 1}.jpg")
