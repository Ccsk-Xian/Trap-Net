#测试对抗训练结果(陷阱式)
#用于FGSM和PGD训练
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
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

EPS = 1
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = torchvision.models.resnet50(num_classes=10).to(device)
# model.load_state_dict(torch.load(Path("./transform_pth/resnet50_mn_10.pth")))

net_eval = torchvision.models.resnet50(num_classes=20).to(device)
net_eval.load_state_dict(torch.load(Path("./defense_models_pgdtraining/defense_modelsmnist_pgdtraining_eval0.3_epoch5.pth")))
net_inil = torchvision.models.resnet50(num_classes=20).to(device)

net_inil.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_20_ls0.4.pth")))
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

net_eval.eval()
net_inil.eval()

correct = 0

correct_adv = 0
correct_inil = 0
correct_fgsm=0
correct_inil_fgsm=0
correct_pgd = 0
correct_pgd_inil = 0

eval_loss =0
inil_loss =0
eval_fgsm_loss =0
inil_fgsm_loss =0
eval_pgd_loss = 0
inil_pgd_loss = 0
count=0
count2=0
count3=0
for epoch,(data, target) in enumerate(dl_test_mn):
    data, target = data.to(device), target.to(device)
    print(epoch)

    with torch.no_grad():
        net_inil.eval()
        net_eval.eval()
        # print clean accuracy
        output = net_eval(data)
        output_inil = net_inil(data)
        pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
        pred_inil = output_inil.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_inil += pred_inil.eq(target.view_as(pred_inil)).sum().item()
        eval_loss += F.cross_entropy(output, target).item()
        inil_loss += F.cross_entropy(output_inil, target).item()
    # print adversarial accuracy
    x_fgm = fast_gradient_method(net_eval, data, eps=EPS, norm=np.inf, clip_min=-1, clip_max=1, targeted=True,
                                 y=(torch.ones_like(target) * 2).to(device))
    x_pgd = projected_gradient_descent(net_eval, data, EPS, 0.01, 40, np.inf, clip_min=-1, clip_max=1, targeted=True,
                                       y=(torch.ones_like(target) * 2).to(device))


    with torch.no_grad():
        net_inil.eval()
        net_eval.eval()
        # print clean accuracy
        output_fgsm = net_eval(x_fgm)
        output_inil_fgsm = net_inil(x_fgm)
        pred_fgsm = output_fgsm.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
        pred_inil_fgsm = output_inil_fgsm.argmax(dim=1, keepdim=True)
        pred_fgsm_numpy =pred_fgsm.detach().cpu().numpy()
        if np.where(pred_fgsm_numpy>9)[0].size>0:
            pred_fgsm[np.where(pred_fgsm_numpy>9)[0]]=target[np.where(pred_fgsm_numpy>9)[0]]
            count+=np.ones_like(np.where(pred_fgsm_numpy>9)[0]).sum().item()
        correct_fgsm += pred_fgsm.eq(target.view_as(pred_fgsm)).sum().item()
        correct_inil_fgsm += pred_inil_fgsm.eq(target.view_as(pred_inil_fgsm)).sum().item()
        eval_fgsm_loss += F.cross_entropy(output_fgsm, target).item()
        inil_fgsm_loss += F.cross_entropy(output_inil_fgsm, target).item()

    with torch.no_grad():
        net_inil.eval()
        net_eval.eval()
        # print clean accuracy
        output_pgd = net_eval(x_pgd)
        output_inil_pgd = net_inil(x_pgd)
        pred_pgd = output_pgd.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability

        pred_inil_pgd = output_inil_pgd.argmax(dim=1, keepdim=True)
        print(pred_pgd)
        print(pred_inil_pgd, "*")
        pred_pgd_numpy = pred_pgd.detach().cpu().numpy()
        if np.where(pred_pgd_numpy>9)[0].size>0:
            pred_pgd[np.where(pred_pgd_numpy > 9)[0]] = target[np.where(pred_pgd_numpy > 9)[0]]
            count2 += np.ones_like(np.where(pred_pgd_numpy > 9)[0]).sum().item()
        correct_pgd += pred_pgd.eq(target.view_as(pred_pgd)).sum().item()
        correct_pgd_inil += pred_inil_pgd.eq(target.view_as(pred_inil_pgd)).sum().item()
        eval_pgd_loss += F.cross_entropy(output_pgd, target).item()
        inil_pgd_loss += F.cross_entropy(output_inil_pgd, target).item()


eval_loss /= len(dl_test_mn.dataset)
inil_loss /= len(dl_test_mn.dataset)
eval_fgsm_loss /= len(dl_test_mn.dataset)
inil_fgsm_loss /= len(dl_test_mn.dataset)
eval_pgd_loss /= len(dl_test_mn.dataset)
inil_pgd_loss /= len(dl_test_mn.dataset)



print(' \nTest set: Clean loss: {:.3f}, eval Clean Accuracy: {}/{} ({:.0f}%)\n'.format(
     eval_loss,correct, len(dl_test_mn.dataset),
    100. * correct / len(dl_test_mn.dataset)))

print(' \nTest set: Clean loss: {:.3f},inil Clean Accuracy: {}/{} ({:.0f}%)\n'.format(
    inil_loss, correct_inil, len(dl_test_mn.dataset),
    100. * correct_inil / len(dl_test_mn.dataset)))



print('\nTest set: fgsm loss: {:.3f},Adv eval fgsm Accuracy: {}/{} ({:.0f}%)\n'.format(
     eval_fgsm_loss,correct_fgsm, len(dl_test_mn.dataset),
    100. * correct_fgsm / len(dl_test_mn.dataset)))

print('\nTest set: fgsm loss: {:.3f},Adv inil fgsm Accuracy: {}/{} ({:.0f}%)\n'.format(
     inil_fgsm_loss,correct_inil_fgsm, len(dl_test_mn.dataset),
    100. * correct_inil_fgsm / len(dl_test_mn.dataset)))


print('\nTest set: pgd loss: {:.3f},Adv eval pgd Accuracy: {}/{} ({:.0f}%)\n'.format(
     eval_pgd_loss,correct_pgd, len(dl_test_mn.dataset),
    100. * correct_pgd / len(dl_test_mn.dataset)))

print('\nTest set: fgsm loss: {:.3f},Adv inil fgsm Accuracy: {}/{} ({:.0f}%)\n'.format(
     inil_pgd_loss,correct_pgd_inil, len(dl_test_mn.dataset),
    100. * correct_pgd_inil / len(dl_test_mn.dataset)))
print(count)
print(count2)
