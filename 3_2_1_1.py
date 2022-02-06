#联20个不同大小的输出模型
#用于测试输出类多少对结果的影响

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
import torchvision.transforms as transforms
# Hyper Parameters
EPOCH = 20                # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 300
LR = 0.005              # learning rate 0.001  12
DOWNLOAD_MNIST = False
DOWNLOAD_KMNIST = False
SAVE_PATH = "./transform_pth/"

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























test = Data.DataLoader(dataset = ts_mn,batch_size=200,shuffle = True)






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# cnn = torchvision.models.vgg16(num_classes=20).to(device)

for i in range (9,10):
    net = torchvision.models.resnet50(num_classes=i+11).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0004)  # optimize all cnn parameters
    loss_func = LabelSmoothingLoss(classes_target=10, classes=i+11, smoothing=0.4)

    if i ==0:
        train_d = Data.TensorDataset(X_10, Y_10)
        tl =  Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
    elif i == 1:
        X = torch.cat((X_10, X_11), 0)  # chage sample_number
        Y = torch.cat((Y_10, Y_11), 0)
        train_d = Data.TensorDataset(X, Y)
        tl = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
    elif i == 2:
        X = torch.cat((X_10, X_11, X_12), 0)  # chage sample_number
        Y = torch.cat((Y_10, Y_11, Y_12), 0)
        train_d = Data.TensorDataset(X, Y)
        tl = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
    elif i ==3:
        X = torch.cat((X_10, X_11, X_12, X_13), 0)  # chage sample_number
        Y = torch.cat((Y_10, Y_11, Y_12, Y_13), 0)
        train_d = Data.TensorDataset(X, Y)
        tl = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
    elif i ==4:

        X = torch.cat((X_10, X_11, X_12, X_13, X_14), 0)  # chage sample_number
        Y = torch.cat((Y_10, Y_11, Y_12, Y_13, Y_14), 0)
        train_d = Data.TensorDataset(X, Y)
        tl = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
    elif i ==5:
        X = torch.cat((X_10, X_11, X_12, X_13, X_14, X_15), 0)  # chage sample_number
        Y = torch.cat((Y_10, Y_11, Y_12, Y_13, Y_14, Y_15), 0)
        train_d = Data.TensorDataset(X, Y)
        tl = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
    elif i ==6:
        X = torch.cat((X_10, X_11, X_12, X_13, X_14, X_15, X_16), 0)  # chage sample_number
        Y = torch.cat((Y_10, Y_11, Y_12, Y_13, Y_14, Y_15, Y_16), 0)
        train_d = Data.TensorDataset(X, Y)
        tl = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
    elif i ==7:
        X = torch.cat((X_10, X_11, X_12, X_13, X_14, X_15, X_16, X_17), 0)  # chage sample_number
        Y = torch.cat((Y_10, Y_11, Y_12, Y_13, Y_14, Y_15, Y_16, Y_17), 0)
        train_d = Data.TensorDataset(X, Y)
        tl =Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
    elif i == 8:
        X = torch.cat((X_10, X_11, X_12, X_13, X_14, X_15, X_16, X_17, X_18), 0)  # chage sample_number
        Y = torch.cat((Y_10, Y_11, Y_12, Y_13, Y_14, Y_15, Y_16, Y_17, Y_18), 0)
        train_d = Data.TensorDataset(X, Y)
        tl = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)
    elif i == 9:
        X = torch.cat((X_10, X_11, X_12, X_13, X_14, X_15, X_16, X_17, X_18, X_19), 0)  # chage sample_number
        Y = torch.cat((Y_10, Y_11, Y_12, Y_13, Y_14, Y_15, Y_16, Y_17, Y_18, Y_19), 0)
        train_d = Data.TensorDataset(X, Y)
        tl = Data.DataLoader(dataset=train_data+train_d, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(tl):  # gives batch data, normalize x when iterate train_loader
            net.train()
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = net(b_x)  # cnn output
            loss = loss_func(output, b_y)[0]  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # for cifar-10
            if step % 2199 == 0:
                examples = enumerate(test)
                batchIndex, (test_x, test_y) = next(examples)
                test_x = test_x.to(device)
                test_y = test_y.to(device)
                net.eval()
                with torch.no_grad():
                    test_output = net(test_x)
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                print(pred_y)
                print(test_y.data)
                # print(pred_y[:20], "pred_y")
                # print(test_y[:20], "test_y")
                accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(),
                      '| test accuracy: %.2f' % accuracy)

                # if HAS_SK:
                #     # Visualization of trained flatten layer (T-SNE)
                #     tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
                #     plot_only = 1000
                #     low_dim_embs = tsne.fit_transform(last_layer.data.cpu().numpy()[:plot_only, :])
                #     labels = test_y.cpu().numpy()[:plot_only]
                #     plot_with_labels(low_dim_embs, labels,epoch)

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
    test_output = net(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
    print(pred_y, 'prediction number')
    print(test_y[:10].cpu().numpy(), 'real number')

    if not (os.path.exists(SAVE_PATH)):
        os.mkdir(SAVE_PATH)
    torch.save(net.state_dict(), SAVE_PATH + "resnet50_mkls0.4"+str(i+11)+".pth")








