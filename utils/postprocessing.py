import torch
import numpy as np
from models.AE import AutoEncoder
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T

def load_image_tensor(image_path, device):
    """
    Loads a given image to device.
    Args:
    image_path: path to image to be loaded.
    device: "cuda" or "cpu"
    """
    image_tensor = T.ToTensor()(Image.open(image_path))
    image_tensor = image_tensor.unsqueeze(0)
    # print(image_tensor.shape)
    # input_images = image_tensor.to(device)
    return image_tensor


def compute_similar_images(image_path, num_images, model, embedding, device):
    """
    Given an image and number of similar images to generate.
    Returns the num_images closest neares images.
    Args:
    image_path: Path to image whose similar images are to be found.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """

    image_tensor = load_image_tensor(image_path, device)
    # image_tensor = image_tensor.to(device)

    with torch.no_grad():
        image_embedding = model.encoder(image_tensor).cpu().detach().numpy()

    # print(image_embedding.shape)

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    # print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list


def plot_similar_images(indices_list, config):
    """
    Plots images that are similar to indices obtained from computing simliar images.
    Args:
    indices_list : List of List of indexes. E.g. [[1, 2, 3]]
    """

    indices = indices_list[0]
    for index in indices:
        if index == 0:
            # index 0 is a dummy embedding.
            pass
        else:
            img_name = str(index - 1) + ".jpg"
            img_path = os.path.join(config["train_dataset"]+ img_name)
            # print(img_path)
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.show()
            img.save(f"../outputs/query_image_3/recommended_{index - 1}.jpg")