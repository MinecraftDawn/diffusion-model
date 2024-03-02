import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.transforms.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.distributions import normal

import random
import einops
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser


def showImage(imgs, size=(5, 5)):
    plt.figure(figsize=size)
    imgs = einops.rearrange(imgs, 'b c h w -> b h w c')
    imgs = imgs.numpy().astype(np.float32)

    for i in range(25):
        plt.subplot(size[0], size[1], i+1)
        plt.imshow(imgs[i], cmap='gray')

    plt.show()


def addNoise(imgs, noiseScale, NOISE_STEP, count=25):
    imgs = imgs[:count]

    noise = torch.normal(mean=0.5, std=0.5, size=imgs.shape)
    steps = torch.randint(low=0, high=NOISE_STEP, size=(count, ))

    cur = noiseScale[steps]
    nxt = noiseScale[steps+1]

    cur = cur.reshape((-1,1,1,1))
    nxt = nxt.reshape((-1,1,1,1))

    curImg = imgs * (1-cur) + noise * cur
    nxtImg = imgs * (1-nxt) + noise * nxt

    return curImg, nxtImg


def main():
    SEED = 0
    random.seed(0)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
    print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')

    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    dataset = MNIST(root='./datasets', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    originImg, _ = next(iter(loader))
    # showImage(originImg)

    noiseScale = torch.linspace(0, 1, NOISE_STEP + 1)

    curImgs, curImgs = addNoise(originImg, noiseScale, NOISE_STEP)
    # showImage(curImgs)
    # showImage(curImgs)

if __name__ == '__main__':
    main()