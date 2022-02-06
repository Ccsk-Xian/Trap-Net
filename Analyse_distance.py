#用于使用wasserstein_distance,entropy,energy_distance进行模型内部数据分布的建模。用于前期理论探究
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.stats import energy_distance
import math
import os
import numpy as np
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchsnooper
import torchvision
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20                # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 512
LR = 0.005              # learning rate 0.001  12
DOWNLOAD_MNIST = False
DOWNLOAD_KMNIST = False
DOWNLOAD_FMNIST =False
DOWNLOAD_EMNIST = False
SAVE_PATH = "./transform_pth/"
DOWNLOAD_CIFAR10 = False
# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

if not (os.path.exists('./cifar10/')) or not os.listdir('./cifar10/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_CIFAR10 = True

# Mnist digits dataset
if not(os.path.exists('./FMNIST/')) or not os.listdir('./FMNIST/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True
# Mnist digits dataset
if not(os.path.exists('./KMNIST/')) or not os.listdir('./KMNIST/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

# Mnist digits dataset
if not(os.path.exists('./emnist/')) or not os.listdir('./emnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True


trans = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((28,28)),
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
])


train_data1 = torchvision.datasets.CIFAR10(
    root="./cifar10/",
    train=True,
    transform=trans,
    download=DOWNLOAD_CIFAR10,
)

ts_cf = torchvision.datasets.CIFAR10(
    root="./cifar10/",
    train=False,
    transform=trans,
    download=DOWNLOAD_CIFAR10,
)

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)


ts_mn = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,                                     # this is training data
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

train_data_EN = torchvision.datasets.EMNIST(
    root='./mnist/',
    split='letters',
    train=True,                                     # this is training data
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
test_data_EN = torchvision.datasets.EMNIST(
    root='./mnist/',
    split='letters',
    train=False,                                     # this is training data
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

tr_Fm = torchvision.datasets.FashionMNIST(
    root='./FMNIST/',
    train=True,                                     # this is training data
    transform=trans,   # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_FMNIST,
)

ts_Fm = torchvision.datasets.FashionMNIST(
    root='./FMNIST/',
    train=False,                                     # this is training data
    transform=trans,    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_FMNIST,
)

tr_Km = torchvision.datasets.KMNIST(
    root='./KMNIST/',
    train=True,                                     # this is training data
    transform=trans,   # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_KMNIST,
)

ts_Km = torchvision.datasets.KMNIST(
    root='./KMNIST/',
    train=False,                                     # this is training data
    transform=trans,    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_KMNIST,
)


mnist_loader = Data.DataLoader(dataset=ts_mn, batch_size=BATCH_SIZE, shuffle=True)
KM_loader = Data.DataLoader(dataset=ts_Km, batch_size=BATCH_SIZE, shuffle=True)
FM_loader = Data.DataLoader(dataset=ts_Fm, batch_size=BATCH_SIZE, shuffle=True)
EM_loader = Data.DataLoader(dataset=test_data_EN, batch_size=BATCH_SIZE, shuffle=True)
Cifar_loader = Data.DataLoader(dataset=ts_cf, batch_size=BATCH_SIZE, shuffle=True)

examples_mn = enumerate(mnist_loader)
examples_km = enumerate(KM_loader)
examples_fm = enumerate(FM_loader)
examples_em = enumerate(EM_loader)
examples_cf = enumerate(Cifar_loader)


batchIndex1, (mn_x, test_y1) = next(examples_mn)
batchIndex11, (mn_x1, test_y11) = next(examples_mn)
batchIndex2, (km_x, test_y2) = next(examples_km)
batchIndex3, (fm_x, test_y3) = next(examples_fm)
batchIndex4, (em_x, test_y4) = next(examples_em)
batchIndex5, (cf_x, test_y5) = next(examples_cf)
mk=0
mk_entropy=0
mk_energy = 0


for i in range(BATCH_SIZE):
    w_mk = wasserstein_distance(mn_x[i][0].flatten().cpu().detach().numpy(), km_x[i][0].flatten().cpu().detach().numpy())
    w_mk_entropy = entropy(mn_x[i][0].flatten().cpu().detach().numpy(),
                                km_x[i][0].flatten().cpu().detach().numpy())
    w_mk_energy = energy_distance(mn_x[i][0].flatten().cpu().detach().numpy(),
                                km_x[i][0].flatten().cpu().detach().numpy())

    mk=mk+w_mk
    mk_entropy+=w_mk_entropy
    mk_energy+=w_mk_energy
w_mk = mk/BATCH_SIZE
w_mk_entropy = mk_entropy/BATCH_SIZE
w_mk_energy = mk_energy/BATCH_SIZE

print(w_mk,'mk')
print(w_mk_entropy,'mk_entropy')
print(w_mk_energy,'mk_energy')

mf = 0
mf_entropy = 0
mf_energy = 0
for i in range(BATCH_SIZE):
    w_mf = wasserstein_distance(mn_x[i][0].flatten().cpu().detach().numpy(), fm_x[i][0].flatten().cpu().detach().numpy())
    w_mf_entropy = entropy(mn_x[i][0].flatten().cpu().detach().numpy(),
                                fm_x[i][0].flatten().cpu().detach().numpy())
    w_mf_energy = energy_distance(mn_x[i][0].flatten().cpu().detach().numpy(),
                                fm_x[i][0].flatten().cpu().detach().numpy())
    mf+=w_mf
    mf_entropy+=w_mf_entropy
    mf_energy+=w_mf_energy

w_mf = mf/BATCH_SIZE
w_mf_entropy = mf_entropy/BATCH_SIZE
w_mf_energy = mf_energy/BATCH_SIZE

print(w_mf,'mf')
print(w_mf_entropy,'mf_entropy')
print(w_mf_energy,'mf_energy')
me = 0
me_entropy = 0
me_energy = 0
for i in range(BATCH_SIZE):
    w_me = wasserstein_distance(mn_x[i][0].flatten().cpu().detach().numpy(), em_x[i][0].flatten().cpu().detach().numpy())
    w_me_entropy = entropy(mn_x[i][0].flatten().cpu().detach().numpy(),
                                em_x[i][0].flatten().cpu().detach().numpy())
    w_me_energy = energy_distance(mn_x[i][0].flatten().cpu().detach().numpy(),
                                em_x[i][0].flatten().cpu().detach().numpy())
    me+=w_mk
    me_entropy += w_me_entropy
    me_energy += w_me_energy
w_me = me/BATCH_SIZE
w_me_entropy = me_entropy / BATCH_SIZE
w_me_energy = me_energy / BATCH_SIZE
print(w_me,'me')
print(w_me_entropy,'me_entropy')
print(w_me_energy,'me_energy')

mc = 0
mc_entropy = 0
mc_energy = 0
for i in range(BATCH_SIZE):
    w_mc = wasserstein_distance(mn_x[i][0].flatten().cpu().detach().numpy(), cf_x[i][0].flatten().cpu().detach().numpy())
    w_mc_entropy = entropy(mn_x[i][0].flatten().cpu().detach().numpy(),
                                cf_x[i][0].flatten().cpu().detach().numpy())
    w_mc_energy = energy_distance(mn_x[i][0].flatten().cpu().detach().numpy(),
                                cf_x[i][0].flatten().cpu().detach().numpy())
    mc+=w_mc
    mc_entropy += w_mc_entropy
    mc_energy += w_mc_energy
w_mc = mc/BATCH_SIZE
w_mc_entropy = mc_entropy / BATCH_SIZE
w_mc_energy = mc_energy / BATCH_SIZE
print(w_mc,'mc')
print(w_mc_entropy,'mc_entropy')
print(w_mc_energy,'mc_energy')

mm = 0
mm_entropy = 0
mm_energy = 0
for i in range(BATCH_SIZE):
    w_mn= wasserstein_distance(mn_x[i][0].flatten().cpu().detach().numpy(), mn_x1[i][0].flatten().cpu().detach().numpy())
    w_mn_entropy = entropy(mn_x[i][0].flatten().cpu().detach().numpy(),
                                mn_x1[i][0].flatten().cpu().detach().numpy())
    w_mn_energy = energy_distance(mn_x[i][0].flatten().cpu().detach().numpy(),
                                mn_x1[i][0].flatten().cpu().detach().numpy())
    mm+=w_mn
    mm_entropy += w_mn_entropy
    mm_energy += w_mn_energy
w_mm = mm/BATCH_SIZE
w_mm_entropy = mm_entropy / BATCH_SIZE
w_mm_energy = mm_energy / BATCH_SIZE
print(w_mm,'mm')
print(w_mm_entropy,'w_mm_entropy')
print(w_mm_energy,'w_mm_energy')
# w_mk = wasserstein_distance(mn_x.cpu().detach().numpy(),km_x.cpu().detach().numpy())
# w_mf
# w_me
# w_mc