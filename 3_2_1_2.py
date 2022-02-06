#对应Trans_kmnist的测试。
#测试目标：不同输出类别数目对防御效力的影响


# library
# standard library
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
from pathlib import Path
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.manual_seed(1)    # reproducible
import torchvision.transforms as transforms
# Hyper Parameters
EPOCH = 20                # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 300
LR = 0.005              # learning rate 0.001  12
DOWNLOAD_MNIST = False
DOWNLOAD_KMNIST = False
SAVE_PATH = "./transform_pth/"
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

# Mnist digits dataset
if not(os.path.exists('./KMNIST/')) or not os.listdir('./KMNIST/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True


trans =transforms.Compose([
        transforms.CenterCrop((28,28)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
])

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

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes_target,classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls_target = classes_target
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()

            # #不为陷阱类分配分布，这两种尝试是因为我们并不太清楚哪种方法可以增大检测效率。可能因为数据分布本身不处于一个流形，我们需要先对其进行自编码处理
            # true_dist = torch.zeros((pred.size()[0], self.cls_target))
            # add_dist = torch.zeros((true_dist.size()[0], self.cls - 10))
            # true_dist.fill_(self.smoothing / (self.cls_target - 1))
            # true_dist = torch.cat((true_dist, add_dist), 1).to(device)


            # 为陷阱类分配分布
            true_dist = torch.zeros((pred.size()[0],self.cls_target))
            add_dist = torch.zeros((true_dist.size()[0],self.cls-10))
            add_dist.fill_(self.smoothing/(self.cls - 10))
            true_dist = torch.cat((true_dist,add_dist),1).to(device)

            #原方法
            # true_dist.fill_(self.smoothing / (self.cls - 1))
            # print(true_dist,'**')
            #我把这里的填补放在陷阱类里

            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim)),true_dist

trans = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((28,28)),
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
])



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

#分类
for j in range (0,10):
    X_class = []
    Y_class = []
    for i in range(59999):
        if tr_Km[i][1] == j:
            trans = tr_Km[i][0].numpy()
            X_class.append(trans)
            trans_y = tr_Km[i][1]
            Y_class.append(trans_y)
    if j == 0:
        X_10 = np.array(X_class)
        X_10 = torch.from_numpy(X_10)
        Y_10 = torch.ones(X_10.shape[0], dtype=int) * 10
    elif j == 1:
        X_11 = np.array(X_class)
        X_11 = torch.from_numpy(X_11)
        print(X_11.shape, '11')
        Y_11 = torch.ones(X_11.shape[0], dtype=int) * 11
    elif j == 2:
        X_12 = np.array(X_class)
        X_12 = torch.from_numpy(X_12)
        print(X_12.shape, '12')
        Y_12 = torch.ones(X_12.shape[0], dtype=int) * 12
    elif j == 3:
        X_13 = np.array(X_class)
        X_13 = torch.from_numpy(X_13)
        print(X_13.shape, '13')
        Y_13 = torch.ones(X_13.shape[0], dtype=int) * 13
    elif j == 4:
        X_14 = np.array(X_class)
        X_14 = torch.from_numpy(X_14)
        print(X_14.shape, '14')
        Y_14 = torch.ones(X_14.shape[0], dtype=int) * 14
    elif j == 5:
        X_15 = np.array(X_class)
        X_15 = torch.from_numpy(X_15)
        print(X_15.shape, '15')
        Y_15 = torch.ones(X_15.shape[0], dtype=int) * 15
    elif j == 6:
        X_16 = np.array(X_class)
        X_16 = torch.from_numpy(X_16)
        print(X_16.shape, '16')
        Y_16 = torch.ones(X_16.shape[0], dtype=int) * 16
    elif j == 7:
        X_17 = np.array(X_class)
        X_17 = torch.from_numpy(X_17)
        print(X_17.shape, '17')
        Y_17 = torch.ones(X_17.shape[0], dtype=int) * 17
    elif j == 8:
        X_18 = np.array(X_class)
        X_18 = torch.from_numpy(X_18)
        print(X_18.shape, '18')
        Y_18 = torch.ones(X_18.shape[0], dtype=int) * 18
    elif j == 9:
        X_19 = np.array(X_class)
        X_19 = torch.from_numpy(X_19)
        print(X_19.shape, '19')
        Y_19 = torch.ones(X_19.shape[0], dtype=int) * 19























test_mn = Data.DataLoader(dataset = ts_mn,batch_size=2000,shuffle = True)
test_km =  Data.DataLoader(dataset = ts_Km,batch_size=2000,shuffle = True)





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# cnn = torchvision.models.vgg16(num_classes=20).to(device)
examples_mn = enumerate(test_mn)
examples_km = enumerate(test_km)
batchIndex, (test_x_mn, test_y_mn) = next(examples_mn)
batchIndex, (test_x_km, test_y_km) = next(examples_km)
EPS=0.7
for i in range (10):
    net = torchvision.models.resnet50(num_classes=20).to(device)
    net.load_state_dict(torch.load(Path("./transform_pth/resnet50_mkls0.4"+str(i+11)+".pth")))
    test_x_mn = test_x_mn.to(device)
    test_y_mn = test_y_mn.to(device)

    test_x_km = test_x_km.to(device)
    test_y_km = test_y_km.to(device)

    x_fgm_mn = fast_gradient_method(net, test_x_mn, eps=EPS, norm=np.inf, targeted=True,)
    x_pgd_mn = projected_gradient_descent(net, test_x_mn, EPS, 0.01, 40, np.inf, targeted=True, )

    x_fgm_km = fast_gradient_method(net, test_x_km, eps=EPS, norm=np.inf, targeted=True, )
    x_pgd_km = projected_gradient_descent(net, test_x_km, EPS, 0.01, 40, np.inf, targeted=True, )
    net.eval()

    with torch.no_grad():
        _, y_pred_mn = net(test_x_mn).max(1)  # model prediction on clean examples
        _, y_pred_fgm_mn = net(x_fgm_mn).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd_mn = net(x_pgd_mn).max(
            1
        )  # model prediction on PGD adversarial examples

        _, y_pred_km = net(test_x_km).max(1)  # model prediction on clean examples
        _, y_pred_fgm_km = net(x_fgm_km).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd_km = net(x_pgd_km).max(
            1
        )  # model prediction on PGD adversarial examples
    accuracy_mn_ini = float((y_pred_mn == test_y_mn.data.cpu().numpy()).astype(int).sum()) / float(test_y_mn.size(0))
    accuracy_km_ini = float((y_pred_km == test_y_km.data.cpu().numpy()).astype(int).sum()) / float(test_y_km.size(0))
    print('count: ', i+10,'| accuracy_mn_ini accuracy: %.2f' % accuracy_mn_ini)
    print('count: ', i + 10, '| accuracy_km_ini accuracy: %.2f' % accuracy_km_ini)


