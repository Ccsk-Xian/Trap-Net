import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import resnet5019_3
import numpy as np
import matplotlib.pyplot as plt


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

use_cuda=True
image_nc=1
batch_size = 128
E=0.3

device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

target_model = resnet5019_3().to(device)
target_model.eval()

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())


# load the pretrained model
# pretrained_model = "./MNIST_target_model.pth"
# target_model = net().to(device)
# target_model.load_state_dict(torch.load(pretrained_model))


# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_10_resnet5019_3E0.1.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
mnist_dataset = torchvision.datasets.MNIST('./mnist/', train=False, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]), download=True)
train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False)
num_correct = 0
num_correct_true = 0

dd =enumerate(train_dataloader)
i, data = next(dd)
test_img, test_label = data
test_img, test_label = test_img.to(device), test_label.to(device)
perturbation = pretrained_G(test_img)
perturbation = torch.clamp(perturbation, -E, E)
adv_img = perturbation + test_img
adv_img = torch.clamp(adv_img, -1, 1)
pred_lab = torch.argmax(target_model(adv_img),1)

pred_true = torch.argmax(target_model(test_img), 1)
num_correct += torch.sum(pred_lab==test_label,0)
num_correct_true += torch.sum(pred_true == test_label, 0)



fig=plt.figure()
for i in range(1,10):
    plt.subplot(3,3,i)
    plt.tight_layout()
    plt.imshow(adv_img[i][0].cpu().detach(),plt.cm.gray,interpolation='none') # 10   11  15   18  17
    plt.title("{},{}".format(pred_lab[i],test_label[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
print('MNIST training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(test_label)))
print('accuracy of clean imgs in training set: %f\n'%(num_correct_true.item()/len(test_label)))
fig=plt.figure()
for i in range(1,10):
    plt.subplot(3,3,i)
    plt.tight_layout()
    plt.imshow(test_img[i][0].cpu().detach(),plt.cm.gray,interpolation='none') # 10   11  15   18  17
    plt.title("{},{}".format(pred_true[i],test_label[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

for i in range(len(test_label)):
    if pred_lab[i]>9:
        pred_lab[i] = test_label[i]
num_correct = 0
num_correct += torch.sum(pred_lab==test_label,0)
print('accuracy of ours imgs in training set: %f\n'%(num_correct.item()/len(test_label)))

# test adversarial examples in MNIST testing dataset
mnist_dataset_test = torchvision.datasets.MNIST('./mnist/', train=False, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]), download=False)
test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False)
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -E, E)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, -1, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))

