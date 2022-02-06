#测试模型结构和陷阱式平滑因子对扩充可攻击空间大小的影响
#测试模型数量1.单个20ls0.4模型。 2.3个0.4模型输出为11 16 20   3.3个模型 ls 为0.25 0.4 0.6   45个模型2+3
# 有目标结果
# 199
# Models1干净准确率:97.910
# Models2干净准确率:99.630
# Models3干净准确率:99.160
# Models4干净准确率:99.220
# Models5干净准确率:99.270
# Models1无过滤前fgsm准确率:11.590
# Models2无过滤前fgsm准确率:12.210
# Models3无过滤前fgsm准确率:9.730
# Models4无过滤前fgsm准确率:12.640
# Models5无过滤前fgsm准确率:15.550
# Models1过滤后fgsm准确率:96.160
# Models2过滤后fgsm准确率:99.490
# Models3过滤后fgsm准确率:99.480
# Models4过滤后fgsm准确率:98.830
# Models5过滤后fgsm准确率:98.840
# Models1无过滤前pgd准确率:35.110
# Models2无过滤前pgd准确率:30.880
# Models3无过滤前pgd准确率:65.190
# Models4无过滤前pgd准确率:66.380
# Models5无过滤前pgd准确率:66.020
# Models1过滤后pgd准确率:42.100
# Models2过滤后pgd准确率:89.490
# Models3过滤后pgd准确率:74.130
# Models4过滤后pgd准确率:79.140
# Models5过滤后pgd准确率:79.500
#
# 无目标结果
# Models1干净准确率:97.718
# Models2干净准确率:99.615
# Models3干净准确率:99.090
# Models4干净准确率:99.154
# Models5干净准确率:99.179
# Models1无过滤前fgsm准确率:18.115
# Models2无过滤前fgsm准确率:6.038
# Models3无过滤前fgsm准确率:32.051
# Models4无过滤前fgsm准确率:24.128
# Models5无过滤前fgsm准确率:23.872
# Models1过滤后fgsm准确率:83.436
# Models2过滤后fgsm准确率:98.756
# Models3过滤后fgsm准确率:96.667
# Models4过滤后fgsm准确率:97.885
# Models5过滤后fgsm准确率:97.256
# Models1无过滤前pgd准确率:3.013
# Models2无过滤前pgd准确率:1.821
# Models3无过滤前pgd准确率:0.256
# Models4无过滤前pgd准确率:0.154
# Models5无过滤前pgd准确率:0.192
# Models1过滤后pgd准确率:88.154
# Models2过滤后pgd准确率:97.833
# Models3过滤后pgd准确率:99.141
# Models4过滤后pgd准确率:99.500
# Models5过滤后pgd准确率:99.359


# Process finished with exit code 0

# library
# standard library
import math
import os
import numpy as np
# third-party library
import torch
from easydict import EasyDict
import torch.nn as nn
import torch.utils.data as Data
import torchsnooper
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.manual_seed(1)    # reproducible
import torchvision.transforms as transforms
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
# Hyper Parameters
EPOCH = 10                # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 300
LR = 0.005              # learning rate 0.001  12
DOWNLOAD_MNIST = False
DOWNLOAD_KMNIST = False
SAVE_PATH = "./labelsmoothing/"

# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True



def resnet_1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_mkls0.420.pth")))
    return net_eval

def resnet_2_1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=11).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_mkls0.411.pth")))
    return net_eval

def resnet_2_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=16).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_mkls0.416.pth")))
    return net_eval

def resnet_3_1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./labelsmoothing/resnet50_lstest0.25.pth")))
    return net_eval

def resnet_3_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./labelsmoothing/resnet50_lstest0.6000000000000001.pth")))
    return net_eval

class Models2(nn.Module):
    def __init__(self):
        super(Models2, self).__init__()
        self.net1 = resnet_2_2()
        self.net2 = resnet_2_1()
        self.net3 = resnet_1()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11
    def forward(self,x):
        net3_out = self.net3(x)
        _, pred3 = net3_out.max(
            1
        )
        net2_out = self.net2(x)
        net2_add = torch.zeros((net2_out.size()[0], net3_out.size()[1] - net2_out.size()[1])).to(device)
        net2_out = torch.cat((net2_out, net2_add), 1).to(device)
        _, pred2 = net2_out.max(
            1
        )
        # 根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0], net3_out.size()[1] - net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out, net1_add), 1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range(net3_out.size()[0]):
            if pred2[i] > 9 or pred3[i] > 9 or pred[i] > 9:
                if pred3[i] > 9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i] > 9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i] > 9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result


class Models3(nn.Module):
    def __init__(self):
        super(Models3, self).__init__()
        self.net1 = resnet_3_2()
        self.net2 = resnet_3_1()
        self.net3 = resnet_1()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11
    def forward(self, x):
        net1_out = self.net1(x)
        net2_out = self.net2(x)
        net3_out = self.net3(x)
        result = (net1_out + net2_out + net3_out) / 3
        return result
    
class Models4(nn.Module):
    def __init__(self):
        super(Models4, self).__init__()
        self.net1 = resnet_2_2()
        self.net2 = resnet_2_1()
        self.net3 = resnet_1()
        self.net4 = resnet_3_1()
        self.net5 = resnet_3_2()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11
    def forward(self,x):
        net4_out = self.net4(x)
        net5_out = self.net5(x)
        net3_out = self.net3(x)
        _, pred3 = net3_out.max(
            1
        )
        net2_out = self.net2(x)
        net2_add = torch.zeros((net2_out.size()[0], net3_out.size()[1] - net2_out.size()[1])).to(device)
        net2_out = torch.cat((net2_out, net2_add), 1).to(device)
        _, pred2 = net2_out.max(
            1
        )
        # 根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0], net3_out.size()[1] - net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out, net1_add), 1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range(net3_out.size()[0]):
            if pred2[i] > 9 or pred3[i] > 9 or pred[i] > 9:
                if pred3[i] > 9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i] > 9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i] > 9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)
        result = (result+net5_out+net4_out)/3
        # result = (net1_out+net2_out+net3_out)/3
        return result
    
class Models5(nn.Module):
    def __init__(self):
        super(Models5, self).__init__()
        self.net1 = resnet_2_2()
        self.net2 = resnet_2_1()
        self.net3 = resnet_1()
        self.net4 = resnet_3_1()
        self.net5 = resnet_3_2()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11
    def forward(self,x):
        net4_out = self.net4(x)
        net5_out = self.net5(x)
        net3_out = self.net3(x)
        _, pred3 = net3_out.max(
            1
        )
        net2_out = self.net2(x)
        net2_add = torch.zeros((net2_out.size()[0], net3_out.size()[1] - net2_out.size()[1])).to(device)
        net2_out = torch.cat((net2_out, net2_add), 1).to(device)
        _, pred2 = net2_out.max(
            1
        )
        # 根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0], net3_out.size()[1] - net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out, net1_add), 1).to(device)
        _, pred = net1_out.max(
            1
        )

        result = ((net1_out+net2_out+net3_out+net5_out+net4_out)/5).to(device)
        # result = (net1_out+net2_out+net3_out)/3
        return result
    
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#define models
Models1 = resnet_1()
Models1.to(device)
Models1.eval()

Models2 = Models2()
Models2.to(device)
Models2.eval()

Models3 = Models3()
Models3.to(device)
Models3.eval()

Models4 = Models4()
Models4.to(device)
Models4.eval()

Models5 = Models5()
Models5.to(device)
Models5.eval()


EPS = 0.3

test = Data.DataLoader(dataset = ts_mn,batch_size=50,shuffle = False)


counter1 = 0
counter_fg1 = 0
counter_pgd1 = 0

counter2 = 0
counter_fg2 = 0
counter_pgd2 = 0

counter3 = 0
counter_fg3 = 0
counter_pgd3 = 0

counter4 = 0
counter_fg4 = 0
counter_pgd4 = 0

counter5 = 0
counter_fg5 = 0
counter_pgd5 = 0



report_bs1 = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0, correct_cw=0, correct_fgm_be=0,
                     correct_pgd_be=0)
report_bs2 = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0, correct_cw=0, correct_fgm_be=0,
                     correct_pgd_be=0)
report_bs3 = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0, correct_cw=0, correct_fgm_be=0,
                     correct_pgd_be=0)
report_bs4 = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0, correct_cw=0, correct_fgm_be=0,
                     correct_pgd_be=0)
report_bs5 = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0, correct_cw=0, correct_fgm_be=0,
                     correct_pgd_be=0)
for step, (x, y) in enumerate(test):
    x, y = x.to(device), y.to(device)
    x_fgm_Models1 = fast_gradient_method(Models1, x, eps=EPS, norm=np.inf, clip_min=-1, clip_max=1)
    x_pgd_Models1 = projected_gradient_descent(Models1, x, EPS, 0.01, 40, np.inf, clip_min=-1, clip_max=1)
    with torch.no_grad():
        _, y_pred_Models1 = Models1(x).max(1)  # model prediction on clean examples
        _, y_pred_fgsm_Models1 = Models1(x_fgm_Models1).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd_Models1 = Models1(x_pgd_Models1).max(
            1
        )  # model prediction on PGD adversarial examples

    x_fgm_Models2 = fast_gradient_method(Models2, x, eps=EPS, norm=np.inf, clip_min=-1, clip_max=1)
    x_pgd_Models2 = projected_gradient_descent(Models2, x, EPS, 0.01, 40, np.inf, clip_min=-1, clip_max=1)
    with torch.no_grad():
        _, y_pred_Models2 = Models2(x).max(1)  # model prediction on clean examples
        _, y_pred_fgsm_Models2 = Models2(x_fgm_Models2).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd_Models2 = Models2(x_pgd_Models2).max(
            1
        )  # model prediction on PGD adversarial examples

    x_fgm_Models3 = fast_gradient_method(Models3, x, eps=EPS, norm=np.inf, clip_min=-1, clip_max=1)
    x_pgd_Models3 = projected_gradient_descent(Models3, x, EPS, 0.01, 40, np.inf, clip_min=-1, clip_max=1)
    with torch.no_grad():
        _, y_pred_Models3 = Models3(x).max(1)  # model prediction on clean examples
        _, y_pred_fgsm_Models3 = Models3(x_fgm_Models3).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd_Models3 = Models3(x_pgd_Models3).max(
            1
        )  # model prediction on PGD adversarial examples

    x_fgm_Models4 = fast_gradient_method(Models4, x, eps=EPS, norm=np.inf, clip_min=-1, clip_max=1)
    x_pgd_Models4 = projected_gradient_descent(Models4, x, EPS, 0.01, 40, np.inf, clip_min=-1, clip_max=1)
    with torch.no_grad():
        _, y_pred_Models4 = Models4(x).max(1)  # model prediction on clean examples
        _, y_pred_fgsm_Models4 = Models4(x_fgm_Models4).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd_Models4 = Models4(x_pgd_Models4).max(
            1
        )  # model prediction on PGD adversarial examples

    x_fgm_Models5 = fast_gradient_method(Models5, x, eps=EPS, norm=np.inf, clip_min=-1, clip_max=1)
    x_pgd_Models5 = projected_gradient_descent(Models5, x, EPS, 0.01, 40, np.inf, clip_min=-1, clip_max=1)
    # targeted = True, y = (torch.ones_like(y) * 2).to(device)
    with torch.no_grad():
        _, y_pred_Models5 = Models5(x).max(1)  # model prediction on clean examples
        _, y_pred_fgsm_Models5 = Models5(x_fgm_Models5).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd_Models5 = Models5(x_pgd_Models5).max(
            1
        )  # model prediction on PGD adversarial examples
#Models1
    report_bs1.nb_test += y.size(0)
    report_bs1.correct_fgm_be += y_pred_fgsm_Models1.cpu().eq(y.cpu()).sum().item()
    report_bs1.correct_pgd_be += y_pred_pgd_Models1.cpu().eq(y.cpu()).sum().item()
    for label in range(len(y_pred_Models1)):
        if y_pred_Models1[label] > 9:
            y_pred_Models1[label] = y[label]
            counter1 += 1

        if y_pred_fgsm_Models1[label] > 9:

            y_pred_fgsm_Models1[label] = y[label]
            counter_fg1 += 1

        if y_pred_pgd_Models1[label] > 9:

            y_pred_pgd_Models1[label] = y[label]
            counter_pgd1 += 1




    report_bs1.correct += y_pred_Models1.cpu().eq(y.cpu()).sum().item()
    report_bs1.correct_fgm += y_pred_fgsm_Models1.cpu().eq(y.cpu()).sum().item()
    report_bs1.correct_pgd += y_pred_pgd_Models1.cpu().eq(y.cpu()).sum().item()
    
#Models2
    report_bs2.nb_test += y.size(0)
    report_bs2.correct_fgm_be += y_pred_fgsm_Models2.cpu().eq(y.cpu()).sum().item()
    report_bs2.correct_pgd_be += y_pred_pgd_Models2.cpu().eq(y.cpu()).sum().item()
    for label in range(len(y_pred_Models2)):
        if y_pred_Models2[label] > 9:
            y_pred_Models2[label] = y[label]
            counter2 += 1

        if y_pred_fgsm_Models2[label] > 9:

            y_pred_fgsm_Models2[label] = y[label]
            counter_fg2 += 1

        if y_pred_pgd_Models2[label] > 9:

            y_pred_pgd_Models2[label] = y[label]
            counter_pgd2 += 1




    report_bs2.correct += y_pred_Models2.cpu().eq(y.cpu()).sum().item()
    report_bs2.correct_fgm += y_pred_fgsm_Models2.cpu().eq(y.cpu()).sum().item()
    report_bs2.correct_pgd += y_pred_pgd_Models2.cpu().eq(y.cpu()).sum().item()

#Models3
    report_bs3.nb_test += y.size(0)
    report_bs3.correct_fgm_be += y_pred_fgsm_Models3.cpu().eq(y.cpu()).sum().item()
    report_bs3.correct_pgd_be += y_pred_pgd_Models3.cpu().eq(y.cpu()).sum().item()
    for label in range(len(y_pred_Models3)):
        if y_pred_Models3[label] > 9:
            y_pred_Models3[label] = y[label]
            counter3 += 1

        if y_pred_fgsm_Models3[label] > 9:

            y_pred_fgsm_Models3[label] = y[label]
            counter_fg3 += 1

        if y_pred_pgd_Models3[label] > 9:

            y_pred_pgd_Models3[label] = y[label]
            counter_pgd3 += 1




    report_bs3.correct += y_pred_Models3.cpu().eq(y.cpu()).sum().item()
    report_bs3.correct_fgm += y_pred_fgsm_Models3.cpu().eq(y.cpu()).sum().item()
    report_bs3.correct_pgd += y_pred_pgd_Models3.cpu().eq(y.cpu()).sum().item()


#Models4
    report_bs4.nb_test += y.size(0)
    report_bs4.correct_fgm_be += y_pred_fgsm_Models4.cpu().eq(y.cpu()).sum().item()
    report_bs4.correct_pgd_be += y_pred_pgd_Models4.cpu().eq(y.cpu()).sum().item()
    for label in range(len(y_pred_Models4)):
        if y_pred_Models4[label] > 9:
            y_pred_Models4[label] = y[label]
            counter4 += 1

        if y_pred_fgsm_Models4[label] > 9:

            y_pred_fgsm_Models4[label] = y[label]
            counter_fg4 += 1

        if y_pred_pgd_Models4[label] > 9:

            y_pred_pgd_Models4[label] = y[label]
            counter_pgd4 += 1




    report_bs4.correct += y_pred_Models4.cpu().eq(y.cpu()).sum().item()
    report_bs4.correct_fgm += y_pred_fgsm_Models4.cpu().eq(y.cpu()).sum().item()
    report_bs4.correct_pgd += y_pred_pgd_Models4.cpu().eq(y.cpu()).sum().item()
    
    
#Models5
    report_bs5.nb_test += y.size(0)
    report_bs5.correct_fgm_be += y_pred_fgsm_Models5.cpu().eq(y.cpu()).sum().item()
    report_bs5.correct_pgd_be += y_pred_pgd_Models5.cpu().eq(y.cpu()).sum().item()
    for label in range(len(y_pred_Models5)):
        if y_pred_Models5[label] > 9:
            y_pred_Models5[label] = y[label]
            counter5 += 1

        if y_pred_fgsm_Models5[label] > 9:

            y_pred_fgsm_Models5[label] = y[label]
            counter_fg5 += 1

        if y_pred_pgd_Models5[label] > 9:

            y_pred_pgd_Models5[label] = y[label]
            counter_pgd5 += 1




    report_bs5.correct += y_pred_Models5.cpu().eq(y.cpu()).sum().item()
    report_bs5.correct_fgm += y_pred_fgsm_Models5.cpu().eq(y.cpu()).sum().item()
    report_bs5.correct_pgd += y_pred_pgd_Models5.cpu().eq(y.cpu()).sum().item()



    print(step)
    print("Models1干净准确率:{:.3f}".format(report_bs1.correct/report_bs1.nb_test *100.0))
    print("Models2干净准确率:{:.3f}".format(report_bs2.correct/report_bs2.nb_test *100.0))
    print("Models3干净准确率:{:.3f}".format(report_bs3.correct/report_bs3.nb_test *100.0))
    print("Models4干净准确率:{:.3f}".format(report_bs4.correct/report_bs4.nb_test *100.0))
    print("Models5干净准确率:{:.3f}".format(report_bs5.correct/report_bs5.nb_test *100.0))

    print("Models1无过滤前fgsm准确率:{:.3f}".format(report_bs1.correct_fgm_be / report_bs1.nb_test * 100.0))
    print("Models2无过滤前fgsm准确率:{:.3f}".format(report_bs2.correct_fgm_be / report_bs2.nb_test * 100.0))
    print("Models3无过滤前fgsm准确率:{:.3f}".format(report_bs3.correct_fgm_be / report_bs3.nb_test * 100.0))
    print("Models4无过滤前fgsm准确率:{:.3f}".format(report_bs4.correct_fgm_be / report_bs4.nb_test * 100.0))
    print("Models5无过滤前fgsm准确率:{:.3f}".format(report_bs5.correct_fgm_be / report_bs5.nb_test * 100.0))

    print("Models1过滤后fgsm准确率:{:.3f}".format(report_bs1.correct_fgm / report_bs1.nb_test * 100.0))
    print("Models2过滤后fgsm准确率:{:.3f}".format(report_bs2.correct_fgm / report_bs2.nb_test * 100.0))
    print("Models3过滤后fgsm准确率:{:.3f}".format(report_bs3.correct_fgm / report_bs3.nb_test * 100.0))
    print("Models4过滤后fgsm准确率:{:.3f}".format(report_bs4.correct_fgm / report_bs4.nb_test * 100.0))
    print("Models5过滤后fgsm准确率:{:.3f}".format(report_bs5.correct_fgm / report_bs5.nb_test * 100.0))

    print("Models1无过滤前pgd准确率:{:.3f}".format(report_bs1.correct_pgd_be / report_bs1.nb_test * 100.0))
    print("Models2无过滤前pgd准确率:{:.3f}".format(report_bs2.correct_pgd_be / report_bs2.nb_test * 100.0))
    print("Models3无过滤前pgd准确率:{:.3f}".format(report_bs3.correct_pgd_be / report_bs3.nb_test * 100.0))
    print("Models4无过滤前pgd准确率:{:.3f}".format(report_bs4.correct_pgd_be / report_bs4.nb_test * 100.0))
    print("Models5无过滤前pgd准确率:{:.3f}".format(report_bs5.correct_pgd_be / report_bs5.nb_test * 100.0))

    print("Models1过滤后pgd准确率:{:.3f}".format(report_bs1.correct_pgd / report_bs1.nb_test * 100.0))
    print("Models2过滤后pgd准确率:{:.3f}".format(report_bs2.correct_pgd / report_bs2.nb_test * 100.0))
    print("Models3过滤后pgd准确率:{:.3f}".format(report_bs3.correct_pgd / report_bs3.nb_test * 100.0))
    print("Models4过滤后pgd准确率:{:.3f}".format(report_bs4.correct_pgd / report_bs4.nb_test * 100.0))
    print("Models5过滤后pgd准确率:{:.3f}".format(report_bs5.correct_pgd / report_bs5.nb_test * 100.0))

# print(counter_label)
# print(counter_label_fg)
# print(counter_label_pgd)
# # print(soft_c)
# # print(soft_f)
# # print(soft_p)
# print("distribution_c",soft_c_dis/nb_test *100.0)
# print("distribution_f", soft_fgsm_dis / nb_test * 100.0)
# print("distribution_p", soft_pgd_dis / nb_test * 100.0)