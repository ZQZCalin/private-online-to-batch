import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import wandb

from models.resnet import ResNet18
from utils import save_json
from train import train, train_private


# ==> Arg Parsing

parser = argparse.ArgumentParser(description='private online-to-batch')
parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=50, type=int, help='number of epochs')

parser.add_argument('--lr', '-l', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', '-m', default=0.0, type=float, help='momentum')
parser.add_argument('--weight_decay', '-w', default=0.0, type=float, help='weight decay')

parser.add_argument('--scheduler', default='cosine', type=str, help='learning rate scheduler')

parser.add_argument('--privacy', action=argparse.BooleanOptionalAction, help='use private training')
parser.add_argument('--sigma', '-s', default=1.0, type=float, help='std of gaussian noise')
parser.add_argument('--beta_power', '-k', default=0, type=int, help='weight beta_t = t^k')

parser.add_argument('--dir', default='results', type=str, help='directory name')
parser.add_argument('--name', default='experiment', type=str, help='experiment name')
args = parser.parse_args()


# ==> Data Processing

print('=== Preparing Data.. ===')

# data transformation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

num_workers = 0
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=num_workers)


# ==> Build Model


print('=== Building model.. ===')

# model parameters
epochs = args.epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ADAM
# ...


# ==> Train Model
if __name__ == '__main__':
    print('=== Training model.. ===')

    wandb.init(project='private-online-to-batch', config=args, name=args.name)
    # wandb.watch(net)

    if not args.privacy:
        # standard non-private training
        stats = train(net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device)
    else:
        # private training
        stats = train_private(args.sigma, args.beta_power, net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device)
    stats.update(args)

    # save training statistics
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)
    save_json(stats, f'{args.dir}/{args.name}.json')