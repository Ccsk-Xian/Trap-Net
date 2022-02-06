#trap smoothing 3-D picture 图6
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
BATCH_SIZE = 300
LR = 0.005              # learning rate 0.001  12
DOWNLOAD_MNIST = False
DOWNLOAD_CIFAR10 = True
SAVE_PATH = "./transform_pth/"

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

# Mnist digits dataset
if not (os.path.exists('./cifar10/')) or not os.listdir('./cifar10/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_CIFAR10 = True


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
            print(true_dist.shape)
            print(pred.shape)
            print(torch.sum(-true_dist * pred, dim=self.dim).shape,"*****")
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim)),true_dist

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

test_cf = torchvision.datasets.CIFAR10(
    root="./cifar10/",
    train=False,
    transform=trans,
    download=DOWNLOAD_CIFAR10,
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
for j in range (10,20):
    X_class = []
    Y_class = []
    for i in range(50000):
        if train_data1[i][1] == j:
            trans = train_data1[i][0].numpy()
            X_class.append(trans)
            trans_y = train_data1[i][1]
            Y_class.append(trans_y)
    if j == 10:
        X_10 = np.array(X_class)
        X_10 = torch.from_numpy(X_10)
        Y_10 = np.array(Y_class)
        Y_10 = torch.from_numpy(Y_10)
    elif j == 11:
        X_11 = np.array(X_class)
        X_11 = torch.from_numpy(X_11)
        Y_11 = np.array(Y_class)
        Y_11 = torch.from_numpy(Y_11)
    elif j == 12:
        X_12 = np.array(X_class)
        X_12 = torch.from_numpy(X_12)
        Y_12 = np.array(Y_class)
        Y_12 = torch.from_numpy(Y_12)
    elif j == 13:
        X_13 = np.array(X_class)
        X_13 = torch.from_numpy(X_13)
        Y_13 = np.array(Y_class)
        Y_13 = torch.from_numpy(Y_13)
    elif j == 14:
        X_14 = np.array(X_class)
        X_14 = torch.from_numpy(X_14)
        Y_14 = np.array(Y_class)
        Y_14 = torch.from_numpy(Y_14)
    elif j == 15:
        X_15 = np.array(X_class)
        X_15 = torch.from_numpy(X_15)
        Y_15 = np.array(Y_class)
        Y_15 = torch.from_numpy(Y_15)
    elif j == 16:
        X_16 = np.array(X_class)
        X_16 = torch.from_numpy(X_16)
        Y_16 = np.array(Y_class)
        Y_16 = torch.from_numpy(Y_16)
    elif j == 17:
        X_17 = np.array(X_class)
        X_17 = torch.from_numpy(X_17)
        Y_17 = np.array(Y_class)
        Y_17 = torch.from_numpy(Y_17)
    elif j == 18:
        X_18 = np.array(X_class)
        X_18 = torch.from_numpy(X_18)
        Y_18 = np.array(Y_class)
        Y_18 = torch.from_numpy(Y_18)
    elif j == 19:
        X_19 = np.array(X_class)
        X_19 = torch.from_numpy(X_19)
        Y_19 = np.array(Y_class)
        Y_19 = torch.from_numpy(Y_19)

X = torch.cat((X_10,X_11,X_12,X_13,X_14,X_15,X_16,X_17,X_18,X_19),0)  #chage sample_number
Y = torch.cat((Y_10,Y_11,Y_12,Y_13,Y_14,Y_15,Y_16,Y_17,Y_18,Y_19),0)
print(X.shape)
print(Y.shape)

train_d = Data.TensorDataset(X,Y)





# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)


test = Data.DataLoader(dataset = ts_mn,batch_size=2000,shuffle = True)






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# cnn = torchvision.models.vgg16(num_classes=20).to(device)
net = torchvision.models.resnet50(num_classes=20).to(device)
net.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_20_ls0.4.pth")))
# net.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_20.pth")))
# net.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_20_mf_ls0.4.pth")))


# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels,epoch):
    # plt.cla()
    # X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    # for x, y, s in zip(X, Y, labels):  # 下边是画图和显示标签
    #     if s>9:
    #         c = cm.rainbow(int(255 * 1 / 9));
    #         plt.text(x, y, s, backgroundcolor=c, fontsize=6)
    #     else:
    #         c = cm.rainbow(int(255 * 9 / 9));
    #         plt.text(x, y, s, backgroundcolor=c, bbox=dict(facecolor='red', alpha=1) ,fontsize=6)
    # plt.xlim(X.min()-10, X.max()+10)
    # plt.ylim(Y.min()-10, Y.max()+10)
    # plt.title('Visualize the Last Hidden layer')
    # plt.show()
    # plt.pause(6)

    fig = plt.figure()
    ax = Axes3D(fig)
    plt.cla()
    X, Y,Z = lowDWeights[:, 0], lowDWeights[:, 1], lowDWeights[:, 2]
    for x, y, z, s in zip(X, Y, Z, labels):
        if s > 9:
            c = cm.rainbow(int(255 * 1 / 9));
            ax.text(x, y, z, s, backgroundcolor=c, fontsize=4)
        else:
            c = cm.rainbow(int(255 * 9 / 9));
            ax.text(x, y, z, s, backgroundcolor=c, bbox=dict(facecolor='red', alpha=1), fontsize=4)
    ax.set_xlim(X.min()-10, X.max()+10)
    ax.set_ylim(Y.min()-10, Y.max()+10)
    ax.set_zlim(Z.min()-10, Z.max()+10)
    # plt.savefig("-{}".format(epoch))
    plt.title('Visualize the Last Hidden layer');  plt.show(); plt.pause(10)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.cla()
    # X, Y,Z = lowDWeights[:, 0], lowDWeights[:, 1], lowDWeights[:, 2]
    # for x, y, z, s in zip(X, Y, Z, labels):
    #     c = cm.rainbow(int(255 * s / 20)); ax.text(x, y, z, s, backgroundcolor=c, fontsize=5)
    # ax.set_xlim(X.min(), X.max())
    # ax.set_ylim(Y.min(), Y.max())
    # ax.set_zlim(Z.min(), Z.max())
    # # plt.savefig("-{}".format(epoch))
    # plt.title('Visualize last layer');  plt.show(); plt.pause(10)


plt.ion()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        net.eval()
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        with torch.no_grad():
            output,last_layer = net(b_x)  # cnn output

        if HAS_SK:
            # Visualization of trained flatten layer (T-SNE)
            tsne = TSNE(perplexity=50, n_components=3, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
            labels = b_y.cpu().numpy()[:plot_only]
            plot_with_labels(low_dim_embs, labels,epoch)







    #     if step % 2199 == 0:
    #         examples = enumerate(test)
    #         batchIndex, (test_x, test_y) = next(examples)
    #         test_x = test_x.to(device)
    #         test_y = test_y.to(device)
    #         with torch.no_grad():
    #             test_output, last_layer = cnn(test_x)
    #         pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    #         # print(pred_y[:20], "pred_y")
    #         # print(test_y[:20], "test_y")
    #         accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
    #         print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
    #         if HAS_SK:
    #             # Visualization of trained flatten layer (T-SNE)
    #             tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
    #             plot_only = 1000
    #             low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
    #             labels = test_y.cpu().numpy()[:plot_only]
    #             plot_with_labels(low_dim_embs, labels)
    # print(step)

        # if step % 2199 == 0:
        #     test_x = test_x.to(device)
        #     test_y = test_y.to(device)
        #     with torch.no_grad():
        #         test_output, last_layer = cnn(test_x)
        #     pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        #     # print(pred_y[:20], "pred_y")
        #     # print(test_y[:20], "test_y")
        #     accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
        #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
        #     if HAS_SK:
        #         # Visualization of trained flatten layer (T-SNE)
        #         tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        #         plot_only = 1000
        #         low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
        #         labels = test_y.cpu().numpy()[:plot_only]
        #         plot_with_labels(low_dim_embs, labels)
    # print(step)
plt.ioff()

