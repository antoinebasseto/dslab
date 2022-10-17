import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from torch.autograd import Variable

"""
# Code heavily based on https://github.com/Zhenye-Na/image-similarity-using-deep-ranking/blob/master/src/net.py
model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def image_loader(path):
    Image Loader helper function.
    return Image.open(path.rstrip("\n")).convert('RGB')


class TripletImageLoader(Dataset):
    Image Loader for Tiny ImageNet.

    def __init__(self, base_path, triplets_filename, transform=None,
                 train=True, loader=image_loader):
        
        Image Loader Builder.
        Args:
            base_path: path to triplets.txt
            filenames_filename: text file with each line containing the path to an image e.g., `images/class1/sample.JPEG`
            triplets_filename: A text file with each line containing three images
            transform: torchvision.transforms
            loader: loader for each image
        
        self.base_path = base_path
        self.transform = transform
        self.loader = loader

        self.train_flag = train

        # load training data
        if self.train_flag:
            triplets = []
            for line in open(triplets_filename):
                line_array = line.split(",")
                triplets.append((line_array[0], line_array[1], line_array[2]))
            self.triplets = triplets

        # load test data
        else:
            singletons = []
            test_images = os.listdir(os.path.join(
                "../tiny-imagenet-200", "val", "images"))
            for test_image in test_images:
                loaded_image = self.loader(os.path.join(
                    "../tiny-imagenet-200", "val", "images", test_image))
                singletons.append(loaded_image)
            self.singletons = singletons

    def __getitem__(self, index):
        Get triplets in dataset.
        # get trainig triplets
        if self.train_flag:
            path1, path2, path3 = self.triplets[index]
            a = self.loader(os.path.join(self.base_path, path1))
            p = self.loader(os.path.join(self.base_path, path2))
            n = self.loader(os.path.join(self.base_path, path3))
            if self.transform is not None:
                a = self.transform(a)
                p = self.transform(p)
                n = self.transform(n)
            return a, p, n

        # get test image
        else:
            img = self.singletons[index]
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        Get the length of dataset.
        if self.train_flag:
            return len(self.triplets)
        else:
            return len(self.singletons)


def resnet101(pretrained=False, **kwargs):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 23, 3])
    return EmbeddingNet(model)


class TripletNet(nn.Module):
    Triplet Network.

    def __init__(self, embeddingnet):
        Triplet Network Builder.
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, a, p, n):
        Forward pass.
        # anchor
        embedded_a = self.embeddingnet(a)

        # positive examples
        embedded_p = self.embeddingnet(p)

        # negative examples
        embedded_n = self.embeddingnet(n)

        return embedded_a, embedded_p, embedded_n


class EmbeddingNet(nn.Module):
    EmbeddingNet using ResNet-101.

    def __init__(self, resnet):
        Initialize EmbeddingNet model.
        super(EmbeddingNet, self).__init__()

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 4096)

    def forward(self, x):
        Forward pass of EmbeddingNet.
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out
"""


class DeepRanking():
    def __init__(self, config):
        self.config = config
        self.step = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # automatically select device
        print("Stuff is working so far")

    def train(self):
        print("Things work part 2")
