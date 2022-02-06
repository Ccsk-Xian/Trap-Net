#3.2.2陷阱数据为CIFAR+Emnist,其中选择EMNIST中类别为2 3 11 6 5的类别标签数据作为陷阱数据
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

torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20                # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 512
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

#分类
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for j in range (10,20):
    X_class = []
    Y_class = []
    if j in [10,11,15,17,18]:
        for i in range(50000):
            if train_data1[i][1] == j:
                trans = train_data1[i][0].numpy()
                X_class.append(trans)
                trans_y = train_data1[i][1]
                Y_class.append(trans_y)
    if j == 10:
        X_10 = np.array(X_class)
        X_10 = torch.from_numpy(X_10)
        print(X_10.shape, '10t')
        Y_10 = torch.ones(X_10.shape[0], dtype=int) * 10
    elif j == 11:
        X_11 = np.array(X_class)
        X_11 = torch.from_numpy(X_11)
        print(X_11.shape, '11t')
        Y_11 = torch.ones(X_11.shape[0], dtype=int) * 11
    elif j == 15:
        X_12 = np.array(X_class)
        X_12 = torch.from_numpy(X_12)
        print(X_12.shape, '12t')
        Y_12 = torch.ones(X_12.shape[0], dtype=int) * 12
    elif j == 17:
        X_13 = np.array(X_class)
        X_13 = torch.from_numpy(X_13)
        print(X_13.shape, '13t')
        Y_13 = torch.ones(X_13.shape[0], dtype=int) * 13
    elif j == 18:
        X_14 = np.array(X_class)
        X_14 = torch.from_numpy(X_14)
        print(X_14.shape, '14t')
        Y_14 = torch.ones(X_14.shape[0], dtype=int) * 14

for j in range (1,27):
    X_class = []
    Y_class = []
    if j in [2,3,11,6,5]:
        for i in range(124799):
            if train_data_EN[i][1] == j:
                trans = train_data_EN[i][0].numpy()
                X_class.append(trans)
                trans_y = train_data_EN[i][1]
                Y_class.append(trans_y)
    if j == 2:
        X_15 = np.array(X_class)
        X_15 = torch.from_numpy(X_15)
        print(X_15.shape, '15t')
        Y_15 = torch.ones(X_15.shape[0], dtype=int) * 15
    elif j == 3:
        X_16 = np.array(X_class)
        X_16 = torch.from_numpy(X_16)
        print(X_16.shape, '16t')
        Y_16 = torch.ones(X_16.shape[0], dtype=int) * 16
    elif j == 6:
        X_17 = np.array(X_class)
        X_17 = torch.from_numpy(X_17)
        print(X_17.shape, '17t')
        Y_17 = torch.ones(X_17.shape[0], dtype=int) * 17
    elif j == 11:
        X_18 = np.array(X_class)
        X_18 = torch.from_numpy(X_18)
        print(X_18.shape, '18t')
        Y_18 = torch.ones(X_18.shape[0], dtype=int) * 18
    elif j == 5:
        X_19 = np.array(X_class)
        X_19 = torch.from_numpy(X_19)
        print(X_19.shape, '19t')
        Y_19 = torch.ones(X_19.shape[0], dtype=int) * 19
X_10 = X_18
Y_10 = torch.ones(X_10.shape[0], dtype=int) * 10
X_11 = X_17
Y_11 = torch.ones(X_11.shape[0], dtype=int) * 11
X_12 = X_16
Y_12 = torch.ones(X_12.shape[0], dtype=int) * 12
X = torch.cat((X_10,X_11,X_12,X_13,X_14,X_15),0)  #chage sample_number
Y = torch.cat((Y_10,Y_11,Y_12,Y_13,Y_14,Y_15),0)
print(X.shape)
print(Y.shape)

train_d = Data.TensorDataset(X,Y)


for j in range (10,20):
    X_class = []
    Y_class = []
    if j in [10,11,15,17,18]:
        for i in range(9999):
            if test_cf[i][1] == j:
                trans = test_cf[i][0].numpy()
                X_class.append(trans)
                trans_y = test_cf[i][1]
                Y_class.append(trans_y)
    if j == 10:
        X_10 = np.array(X_class)
        X_10 = torch.from_numpy(X_10)
        print(X_10.shape, '10t')
        Y_10 = torch.ones(X_10.shape[0], dtype=int) * 10
    elif j == 11:
        X_11 = np.array(X_class)
        X_11 = torch.from_numpy(X_11)
        print(X_11.shape, '11t')
        Y_11 = torch.ones(X_11.shape[0], dtype=int) * 11
    elif j == 15:
        X_12 = np.array(X_class)
        X_12 = torch.from_numpy(X_12)
        print(X_12.shape, '12t')
        Y_12 = torch.ones(X_12.shape[0], dtype=int) * 12
    elif j == 17:
        X_13 = np.array(X_class)
        X_13 = torch.from_numpy(X_13)
        print(X_13.shape, '13t')
        Y_13 = torch.ones(X_13.shape[0], dtype=int) * 13
    elif j == 18:
        X_14 = np.array(X_class)
        X_14 = torch.from_numpy(X_14)
        print(X_14.shape, '14t')
        Y_14 = torch.ones(X_14.shape[0], dtype=int) * 14

for j in range (1,27):
    X_class = []
    Y_class = []
    if j in [2,3,11,6,5]:
        for i in range(15000):
            if test_data_EN[i][1] == j:
                trans = test_data_EN[i][0].numpy()
                X_class.append(trans)
                trans_y = test_data_EN[i][1]
                Y_class.append(trans_y)
    if j == 2:
        X_15 = np.array(X_class)
        X_15 = torch.from_numpy(X_15)
        print(X_15.shape, '15t')
        Y_15 = torch.ones(X_15.shape[0], dtype=int) * 15
    elif j == 3:
        X_16 = np.array(X_class)
        X_16 = torch.from_numpy(X_16)
        print(X_16.shape, '16t')
        Y_16 = torch.ones(X_16.shape[0], dtype=int) * 16
    elif j == 6:
        X_17 = np.array(X_class)
        X_17 = torch.from_numpy(X_17)
        print(X_17.shape, '17t')
        Y_17 = torch.ones(X_17.shape[0], dtype=int) * 17
    elif j == 11:
        X_18 = np.array(X_class)
        X_18 = torch.from_numpy(X_18)
        print(X_18.shape, '18t')
        Y_18 = torch.ones(X_18.shape[0], dtype=int) * 18
    elif j == 5:
        X_19 = np.array(X_class)
        X_19 = torch.from_numpy(X_19)
        print(X_19.shape,'19t')
        Y_19 = torch.ones(X_19.shape[0], dtype=int) * 19
X_10 = X_18
Y_10 = torch.ones(X_10.shape[0], dtype=int) * 10
X_11 = X_17
Y_11 = torch.ones(X_11.shape[0], dtype=int) * 11
X_12 = X_16
Y_12 = torch.ones(X_12.shape[0], dtype=int) * 12
X = torch.cat((X_10,X_11,X_12,X_13,X_14,X_15),0)  #chage sample_number
Y = torch.cat((Y_10,Y_11,Y_12,Y_13,Y_14,Y_15),0)
print(X.shape)
print(Y.shape)

test_d = Data.TensorDataset(X,Y)


train_loader = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
train_loader1 = Data.DataLoader(dataset=train_data1,batch_size=BATCH_SIZE,shuffle=True)


test = Data.DataLoader(dataset = ts_mn,batch_size=2000,shuffle = True)






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# cnn = torchvision.models.vgg16(num_classes=20).to(device)
net = torchvision.models.resnet50(num_classes=16).to(device)

print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=0.0004)   # optimize all cnn parameters
loss_func = LabelSmoothingLoss(classes_target=10,classes=16, smoothing=0.4)                        # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels,epoch):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):  # 下边是画图和显示标签
        c = cm.rainbow(int(255 * s / 20));
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(10)

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
# training and testing
# with torchsnooper.snoop():
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        net.train()
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output,last_layer = net(b_x)  # cnn output
        loss = loss_func(output, b_y)[0]  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        #for cifar-10
        if step % 2199 == 0:
            examples = enumerate(test)
            batchIndex, (test_x, test_y) = next(examples)
            test_x = test_x.type(torch.FloatTensor)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            net.eval()
            with torch.no_grad():
                test_output,last_layer = net(test_x)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            print(pred_y)
            print(test_y.data)
            # print(pred_y[:20], "pred_y")
            # print(test_y[:20], "test_y")
            accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

            # if HAS_SK:
            #     # Visualization of trained flatten layer (T-SNE)
            #     tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
            #     plot_only = 1000
            #     low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
            #     labels = test_y.cpu().numpy()[:plot_only]
            #     plot_with_labels(low_dim_embs, labels,epoch)
    print(step)
    if accuracy >= 0.98:
        optimizer.param_groups[0]["lr"] = 0.0000000001
    elif accuracy >= 0.97:
        optimizer.param_groups[0]["lr"] = 0.0000000005
    elif accuracy >= 0.95:
        optimizer.param_groups[0]["lr"] = 0.000000001
    elif accuracy >= 0.88:
        optimizer.param_groups[0]["lr"] = 0.000001
    elif accuracy >= 0.90:
        optimizer.param_groups[0]["lr"] = 0.000005
    elif accuracy >= 0.88:
        optimizer.param_groups[0]["lr"] = 0.00001
    elif accuracy >= 0.86:
        optimizer.param_groups[0]["lr"] = 0.00005
    elif accuracy >= 0.83:
        optimizer.param_groups[0]["lr"] = 0.0001
    elif accuracy >= 0.80:
        optimizer.param_groups[0]["lr"] = 0.0005
    elif accuracy >= 0.75:
        optimizer.param_groups[0]["lr"] = 0.001


plt.ioff()

# print 10 predictions from test data
test_output,last_layer = net(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
print(pred_y, 'prediction number')
print(test_y[:10].cpu().numpy(), 'real number')

if not(os.path.exists(SAVE_PATH)):
    os.mkdir(SAVE_PATH)
torch.save(net.state_dict(), SAVE_PATH+"resnet50_ours_mec_label_16_ls0.4.pth")




