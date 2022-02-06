#main+advGAN+test_adversarial_examples。用于测试模型于advGAN攻击下的防御效力
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import resnet5019_3
import numpy as np

use_cuda=True
image_nc=1
epochs = 10
batch_size = 128
BOX_MIN = -1
BOX_MAX = 1
net = resnet5019_3
name = 'resnet5019_3'
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# pretrained_model = "./MNIST_target_model.pth"
targeted_model = net().to(device)
# targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# MNIST train dataset and dataloader declaration
mnist_dataset = torchvision.datasets.MNIST('./mnist/', train=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5],[0.5])
    ]), download=True)

dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX,
                       name)

advGAN.train(dataloader, epochs)
