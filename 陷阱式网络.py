
#最初构建陷阱式网络，无陷阱式平滑损失函数
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
#构建基础陷阱模型，CIFAR-10
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 30                # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 160
LR = 0.0001              # learning rate
DOWNLOAD_MNIST = False
DOWNLOAD_CIFAR10 = True
SAVE_PATH = "./models/"

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

# Mnist digits dataset
if not (os.path.exists('./cifar10/')) or not os.listdir('./cifar10/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_CIFAR10 = True

trans = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((28,28)),
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
])
torchvision.models.resnet18()
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

#分类
for j in range (10,19):
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
X = torch.cat((X_10,X_11),0)
Y = torch.cat((Y_10,Y_11),0)
test_d = Data.TensorDataset(X,Y)
train_loader1 = Data.DataLoader(dataset=test_d,batch_size=BATCH_SIZE,shuffle=True)


# # for i in range(50000):        元组结构改不了
# #     (train_data1[i])[1] +=10
# data,label = train_data1[49999]
# print(type((train_data1[49999])))
# print(label)
# print(type(train_data1))
# print(type(train_data))
# print(type(data))
# print(data.size())
# print(type(train_data.train_data))
# print(data[0].shape)
# plt.imshow(data[0])
# plt.show()
# # plot one example
# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# print(train_data.train_data[0].shape)
# print(train_data.train_data.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data+test_d, batch_size=BATCH_SIZE, shuffle=True)
train_loader1 = Data.DataLoader(dataset=train_data1,batch_size=BATCH_SIZE,shuffle=True)
# pick 2000 samples to speed up testing
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# test_data1 = torchvision.datasets.MNIST(root='./cifar10/', train=False,download=DOWNLOAD_CIFAR10,)
# test_x1 = torch.unsqueeze(test_data1.test_data,dim=1).type(torch.FloatTensor)[:2000]

ts_cf = torchvision.datasets.CIFAR10(
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
for test_index in range(10,19):
    X_c = []
    Y_c = []
    for test_img_index in range(9999):
        if ts_cf[test_img_index][1] == test_index:
            trans_test = ts_cf[test_img_index][0].numpy()
            X_c.append(trans_test)
            trans_test_y = ts_cf[test_img_index][1]
            Y_c.append(trans_test_y)
    if test_index == 10:
        X_10 = np.array(X_c)
        X_10 = torch.from_numpy(X_10)
        Y_10 = np.array(Y_c)
        Y_10 = torch.from_numpy(Y_10)
    elif test_index == 11:
        X_11 = np.array(X_c)
        X_11 = torch.from_numpy(X_11)
        Y_11 = np.array(Y_c)
        Y_11 = torch.from_numpy(Y_11)
    elif test_index == 12:
        X_12 = np.array(X_c)
        X_12 = torch.from_numpy(X_12)
        Y_12 = np.array(Y_c)
        Y_12 = torch.from_numpy(Y_12)
    elif test_index == 13:
        X_13 = np.array(X_c)
        X_13 = torch.from_numpy(X_13)
        Y_13 = np.array(Y_c)
        Y_13 = torch.from_numpy(Y_13)
    elif test_index == 14:
        X_14 = np.array(X_c)
        X_14 = torch.from_numpy(X_14)
        Y_14 = np.array(Y_c)
        Y_14 = torch.from_numpy(Y_14)
    elif test_index == 15:
        X_15 = np.array(X_c)
        X_15 = torch.from_numpy(X_15)
        Y_15 = np.array(Y_c)
        Y_15 = torch.from_numpy(Y_15)
    elif test_index == 16:
        X_16 = np.array(X_c)
        X_16 = torch.from_numpy(X_16)
        Y_16 = np.array(Y_c)
        Y_16 = torch.from_numpy(Y_16)
    elif test_index == 17:
        X_17 = np.array(X_c)
        X_17 = torch.from_numpy(X_17)
        Y_17 = np.array(Y_c)
        Y_17 = torch.from_numpy(Y_17)
    elif test_index == 18:
        X_18 = np.array(X_c)
        X_18 = torch.from_numpy(X_18)
        Y_18 = np.array(Y_c)
        Y_18 = torch.from_numpy(Y_18)
    elif test_index == 19:
        X_19 = np.array(X_c)
        X_19 = torch.from_numpy(X_19)
        Y_19 = np.array(Y_c)
        Y_19 = torch.from_numpy(Y_19)
X = torch.cat((X_10,X_11),0)
Y = torch.cat((Y_10,Y_11),0)
test_d = Data.TensorDataset(X,Y)


test = Data.DataLoader(dataset = test_d+ts_mn,batch_size=2000,shuffle = True)





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# cnn = torchvision.models.vgg16(num_classes=20).to(device)
cnn = torchvision.models.resnet50(num_classes=12).to(device)

print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels,epoch):
    # view_data = train_data.train_data[:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    # encoded_data, _ = autoencoder(view_data)
    # fig = plt.figure();
    # ax = Axes3D(fig)
    # X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
    # values = train_data.train_labels[:200].numpy()
    # for x, y, z, s in zip(X, Y, Z, values):
    #     c = cm.rainbow(int(255 * s / 9));
    #     ax.text(x, y, z, s, backgroundcolor=c)
    # ax.set_xlim(X.min(), X.max());
    # ax.set_ylim(Y.min(), Y.max());
    # ax.set_zlim(Z.min(), Z.max())
    # plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    plt.cla()
    X, Y,Z = lowDWeights[:, 0], lowDWeights[:, 1], lowDWeights[:, 2]
    for x, y, z, s in zip(X, Y, Z, labels):
        c = cm.rainbow(int(255 * s / 20)); ax.text(x, y, z, s, backgroundcolor=c, fontsize=5)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    # plt.savefig("-{}".format(epoch))
    plt.title('Visualize last layer');  plt.show(); plt.pause(10)


plt.ion()
# training and testing
# with torchsnooper.snoop():
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        # print(b_x.shape)
        # print(b_y.shape)
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = cnn(b_x)[0]  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        #for cifar-10
        if step % 2199 == 0:
            examples = enumerate(test)
            batchIndex, (test_x, test_y) = next(examples)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            with torch.no_grad():
                test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            print(pred_y)
            print(test_y.data)
            # print(pred_y[:20], "pred_y")
            # print(test_y[:20], "test_y")
            accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
            rem_acc = accuracy
            if accuracy >= rem_acc*1.2:
                LR = LR/10
            elif accuracy >= 0.82 and accuracy >=rem_acc+0.03:
                LR = LR/2
            elif accuracy >= 0.86 and accuracy >=rem_acc+0.04:
                LR = LR/2
            elif accuracy >= 0.88 and accuracy >=rem_acc+0.02:
                LR = LR/2
            elif accuracy >= 0.90 and accuracy >=rem_acc+0.02:
                LR = LR/2
            elif accuracy <= rem_acc*0.8:
                LR = LR*10
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=50, n_components=3, init='pca', n_iter=5000)
                plot_only = 1000
                low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
                labels = test_y.cpu().numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels,epoch)
    print(step)


plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
print(pred_y, 'prediction number')
print(test_y[:10].cpu().numpy(), 'real number')

if not(os.path.exists(SAVE_PATH)):
    os.mkdir(SAVE_PATH)
torch.save(cnn.state_dict(), SAVE_PATH+"resnet_50_12.pth")
