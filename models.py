import torch.nn as nn
import torch
import torchvision
from pathlib import Path
import torch.nn.functional as F
#所有测试的模型组合


# Target Model definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def resnet5019():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet5019mnist.pth")))
    return net_eval

def resnet5019_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet5019mnist_2.pth")))
    return net_eval

def resnet5019_3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet5019mnist_3.pth")))
    return net_eval


# 原模型基础上增加标签平滑
def resnet5019_label1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet5019mnist_label_2.pth")))
    return net_eval

def resnet5019_label2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet5019mnist_label_3.pth")))
    return net_eval

def resnet5019_label3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet5019mnist_label_4.pth")))
    return net_eval

def resnet1819_label():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet18(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet1819mnist_label_2.pth")))
    return net_eval

def resnet3419_label():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet34(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet3419mnist_label_2.pth")))
    return net_eval




def resnet3419():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet34(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet3419mnist.pth")))
    return net_eval

def resnet1819():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet18(num_classes=10).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet1819mnist.pth")))
    return net_eval



def resnet50_ours_16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=16).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_16.pth")))
    return net_eval

def resnet50_ours_18():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=18).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_18.pth")))
    return net_eval

def resnet50_ours_20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_20.pth")))
    return net_eval



def resnet50_ours_label_16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=16).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_16_ls0.4.pth")))
    return net_eval

def resnet50_ours_label_18():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=18).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_18_ls0.4.pth")))
    return net_eval


def resnet50_ours_label_20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_20_ls0.4.pth")))
    return net_eval

def resnet50_em_ours_label_16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=16).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_16_em_ls0.4.pth")))
    return net_eval

def resnet50_em_ours_label_18():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=18).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_18_em_ls0.4.pth")))
    return net_eval


def resnet50_em_ours_label_20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_20_em_ls0.4.pth")))
    return net_eval


def resnet50_emc_ours_label_16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=16).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_mec_label_16_ls0.4.pth")))
    return net_eval

def resnet50_emc_ours_label_18():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=18).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_mec_label_18_ls0.4.pth")))
    return net_eval


def resnet50_emc_ours_label_20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_mec_label_20_ls0.4.pth")))
    return net_eval

def resnet50_ek_ours_label_16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=16).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_16_ek_ls0.4.pth")))
    return net_eval

def resnet50_ek_ours_label_18():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=18).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_18_mk_ls0.4.pth")))
    return net_eval


def resnet50_ek_ours_label_20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_20_mk_ls0.4.pth")))
    return net_eval

def resnet50_ef_ours_label_16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=16).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_16_mf_ls0.4.pth")))
    return net_eval

def resnet50_ef_ours_label_18():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=18).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_18_mf_ls0.4.pth")))
    return net_eval


def resnet50_ef_ours_label_20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_label_20_mf_ls0.4.pth")))
    return net_eval

def resnet50_emkf_ours_label_16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=16).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_mkfe_label_16_ls0.4.pth")))
    return net_eval

def resnet50_emkf_ours_label_18():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=18).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_mkfe_label_18_ls0.4.pth")))
    return net_eval


def resnet50_emkf_ours_label_20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_mkfe_label_20_ls0.4.pth")))
    return net_eval

def resnet50_emkfc_ours_label_16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=16).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_mkfec_label_16_ls0.4.pth")))
    return net_eval

def resnet50_emkfc_ours_label_18():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=18).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_mkfec_label_18_ls0.4.pth")))
    return net_eval


def resnet50_emkfc_ours_label_20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_eval = torchvision.models.resnet50(num_classes=20).to(device)
    net_eval.load_state_dict(torch.load(Path("./transform_pth/resnet50_ours_mkfec_label_20_ls0.4.pth")))
    return net_eval


#基础的随机初始化集成网络
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.net1 = resnet5019()
        self.net2 = resnet5019_2()
        self.net3 = resnet5019_3()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11，可以试一下
    def forward(self,x):
        net1_out = self.net1(x)
        net2_out = self.net2(x)
        net3_out = self.net3(x)
        result = (net1_out+net2_out+net3_out)/3
        return result

#不同结构网络
class Baseline2(nn.Module):
    def __init__(self):
        super(Baseline2, self).__init__()
        self.net1 = resnet5019()
        self.net2 = resnet1819()
        self.net3 = resnet3419()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11
    def forward(self,x):
        net1_out = self.net1(x)
        net2_out = self.net2(x)
        net3_out = self.net3(x)
        result = (net1_out+net2_out+net3_out)/3
        return result

#随机初始化的平滑后。
class Baseline3(nn.Module):
    def __init__(self):
        super(Baseline3, self).__init__()
        self.net1 = resnet5019_label1()
        self.net2 = resnet5019_label2()
        self.net3 = resnet5019_label3()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11
    def forward(self,x):
        net1_out = self.net1(x)
        net2_out = self.net2(x)
        net3_out = self.net3(x)
        result = (net1_out+net2_out+net3_out)/3
        return result


#不同结构网络平滑后
class Baseline4(nn.Module):
    def __init__(self):
        super(Baseline4, self).__init__()
        self.net1 = resnet5019_label1()
        self.net2 = resnet1819_label()
        self.net3 = resnet3419_label()

#如果是我们自己的方法，那就需要多一列，如果判别是>正确类别的话，就要将后位全移到11
    def forward(self,x):
        net1_out = self.net1(x)
        net2_out = self.net2(x)
        net3_out = self.net3(x)
        result = (net1_out+net2_out+net3_out)/3
        return result

class OursNet(nn.Module):
    def __init__(self):
        super(OursNet, self).__init__()
        self.net1 = resnet50_ours_16()
        self.net2 = resnet50_ours_18()
        self.net3 = resnet50_ours_20()

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
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result


class OursNet_label(nn.Module):
    def __init__(self):
        super(OursNet_label, self).__init__()
        self.net1 = resnet50_ours_label_16()
        self.net2 = resnet50_ours_label_18()
        self.net3 = resnet50_ours_label_20()

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
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result

class OursNet_em_label(nn.Module):
    def __init__(self):
        super(OursNet_em_label, self).__init__()
        self.net1 = resnet50_em_ours_label_16()
        self.net2 = resnet50_em_ours_label_18()
        self.net3 = resnet50_em_ours_label_20()

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
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result


class OursNet_mec_label(nn.Module):
    def __init__(self):
        super(OursNet_mec_label, self).__init__()
        self.net1 = resnet50_emc_ours_label_16()
        self.net2 = resnet50_emc_ours_label_18()
        self.net3 = resnet50_emc_ours_label_20()

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
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result

class OursNet_mekf_label(nn.Module):
    def __init__(self):
        super(OursNet_mekf_label, self).__init__()
        self.net1 = resnet50_emkf_ours_label_16()
        self.net2 = resnet50_emkf_ours_label_18()
        self.net3 = resnet50_emkf_ours_label_20()

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
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result

class OursNet_mekfc_label(nn.Module):
    def __init__(self):
        super(OursNet_mekfc_label, self).__init__()
        self.net1 = resnet50_emkfc_ours_label_16()
        self.net2 = resnet50_emkfc_ours_label_18()
        self.net3 = resnet50_emkfc_ours_label_20()

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
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result

class OursNet_mk_label(nn.Module):
    def __init__(self):
        super(OursNet_mk_label, self).__init__()
        self.net1 = resnet50_ek_ours_label_16()
        self.net2 = resnet50_ek_ours_label_18()
        self.net3 = resnet50_ek_ours_label_20()

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
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result

class OursNet_mf_label(nn.Module):
    def __init__(self):
        super(OursNet_mf_label, self).__init__()
        self.net1 = resnet50_ef_ours_label_16()
        self.net2 = resnet50_ef_ours_label_18()
        self.net3 = resnet50_ef_ours_label_20()

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
        #根据最大输出类补充logits
        net1_out = self.net1(x)
        net1_add = torch.zeros((net1_out.size()[0],net3_out.size()[1]-net1_out.size()[1])).to(device)
        net1_out = torch.cat((net1_out,net1_add),1).to(device)
        _, pred = net1_out.max(
            1
        )
        result = torch.zeros(net3_out.size()).to(device)
        for i in range (net3_out.size()[0]):
            if pred2[i] >9 or pred3[i]>9 or pred[i]>9:
                if pred3[i]>9:
                    result[i] = net3_out[i].to(device)
                    # print('>9')
                elif pred2[i]>9:
                    result[i] = net2_out[i].to(device)
                    # print('>9')
                elif pred[i]>9:
                    result[i] = net1_out[i].to(device)
                    # print('>9')
            else:
                result[i] = ((net1_out[i] + net2_out[i] + net3_out[i]) / 3).to(device)

        # result = (net1_out+net2_out+net3_out)/3
        return result

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out