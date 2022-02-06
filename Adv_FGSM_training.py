#FGSM和PGD训练,用于优化有目标攻击时呈现的V型防御效力。失败
from deeprobust.image.defense import fgsmtraining
from deeprobust.image.defense import pgdtraining
from deeprobust.image.defense import AWP
import os
import numpy as np
import argparse
from sklearn.neighbors import KernelDensity
import torch
import torchvision
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.utils.data as Data
from pathlib import Path

from scipy.interpolate import interpn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils_adv import ( get_noisy_samples, get_mc_predictions,
                         get_deep_representations, score_sampless, normalize,
                         train_lr, compute_roc)
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet50(num_classes=10).to(device)
model.load_state_dict(torch.load(Path("./transform_pth/resnet50_mn_10.pth")))
#
# net_eval = torchvision.models.resnet50(num_classes=20).to(device)
# net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_20_ls0.4.pth")))


train_data_mn = torchvision.datasets.MNIST(



    root='./mnist/',
    train=True,  # this is training data
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ]),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=False,
)
test_data_mn = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,  # this is training data
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ]),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=False,
)
dl_train_mn = Data.DataLoader(dataset=train_data_mn, batch_size=50, shuffle=False)
dl_test_mn = Data.DataLoader(dataset=test_data_mn, batch_size=50, shuffle=False)
# FT = AWP.AWP_AT(model=model,device=device)
test_loss = 0
correct = 0
test_loss_adv = 0
correct_adv = 0
for data, target in dl_test_mn:
    data, target = data.to(device), target.to(device)

    # print clean accuracy
    output = model(data)
    test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()

print('\nTest set: Clean loss: {:.3f}, Clean Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dl_test_mn.dataset),
            100. * correct / len(dl_test_mn.dataset)))
FT = pgdtraining.PGDtraining(model=model,device=device)
FT.generate(dl_train_mn,dl_test_mn,epsilon=0.3,save_name='mnist_pgdtraining_0.3',not_eval=True)

