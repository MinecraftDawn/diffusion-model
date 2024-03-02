import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.transforms.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

import einops
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def showImage(imgs):
    plt.figure(figsize=(10, 10))
    imgs = einops.rearrange(imgs, 'b c h w -> b h w c')
    imgs = imgs.numpy().astype(np.float32)

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i])

    plt.show()

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size'
    )

    parser.add_argument(
        '--epoches', type=int, default=10, help='Epoches'
    )

    parser.add_argument(
        '--noise_step', type=int, default=8, help='Noise step'
    )

    parser.add_argument(
        '--learning_rate', type=float, default=0.001, help='Learning rate'
    )

    args = vars(parser.parse_args())

    BATCH_SIZE = args['batch_size']
    EPOCHES = args['epoches']
    NOISE_STEP = args['noise_step']
    LEARNING_RATE = args['learning_rate']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    dataset = MNIST(root='./datasets', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    originImg, _ = next(iter(loader))
    showImage(originImg)

if __name__ == '__main__':
    main()