'''
Testing train_private() method.

For simplicity, we will make up a simple linear regression model.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import copy

from models.resnet import ResNet18
from utils import save_json
from train import train, train_private, test_step



class ToyDataset(Dataset):
    '''
    Prepares a toy dataset with y = (<e,x>+1) + eps,
    where x is uniformly chosen from [0,1)^d and eps~N(0,\sigma^2).
    '''
    def __init__(self, N, d, sigma):
        self.inputs = torch.rand((N, d))
        self.labels = self.inputs @ torch.ones((d, 1)) + 1
        self.labels += sigma*torch.randn_like(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# => Training

# parameters
sample_size = 5000
dimension = 1
batch_size = 1024
epochs = 10
device = 'cpu'

# dataset
train_data = ToyDataset(N=sample_size, d=dimension, sigma=0.1)
test_data = ToyDataset(N=10000, d=dimension, sigma=0.0)

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=100, shuffle=False)

# linear model
net = nn.Sequential(nn.Linear(in_features=dimension, out_features=1))

criterion = nn.MSELoss()
optimizer = optim.SGD(params=net.parameters(), lr=1)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow(epoch+1, -0.5))

privacy = True
debug = False

if not debug:
    if not privacy:
        train(net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device)
    else:
        train_private(0, 1, net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device)
else:
    OCO = net 
    OTB = copy.deepcopy(OCO)

    # weight function
    beta = lambda t: t

    # conversion dictionaries between OCO and OTB
    OCO_to_OTB = {p1: p2 for p1, p2 in zip(OCO.parameters(), OTB.parameters())}
    OTB_to_OCO = {p2: p1 for p1, p2 in zip(OCO.parameters(), OTB.parameters())}

    for epoch in range(epochs):
        print(f'\n=====training epoch {epoch+1}=====')
        OCO.train()
        OTB.train()

        # accumulated gradients g_t
        gradients = {p: torch.zeros_like(p) for p in OTB.parameters()}

        for batch, (inputs, labels) in enumerate(trainloader):
            t = batch+1
            print(f'\niteration {t}:')

            log_w_t = f'w_{t}: {[p.data for p in OCO.parameters()]}'
            log_x_prev = f'x_{t-1}: {[p.data for p in OTB.parameters()]}'

            # load data
            inputs, labels = inputs.to(device), labels.to(device)

            # 1) gradient at last iteration: \nabla\ell(x_{t-1}, z_t)
            # we need to compute it before we update x_t = x_{t-1} + (w_t-x_{t-1})*2/(t-1)
            OTB.zero_grad()
            if t > 1:
                outputs = OTB(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                # since we update g_t using:
                # g_t = g_{t-1} + t \nabla\ell(x_t, z_t) - (t-1) \nabla\ell(x_{t-1}, z_t),
                # we can subtract (t-1)*\nabla\ell(x_{t-1}, z_t) first.
                for p in OTB.parameters():
                    gradients[p] -= beta(t-1) * p.grad
            log_g_prev = f'g(x_{t-1},z_{t}): {[p.grad for p in OTB.parameters()]}'

            # 2) aggregate weight: x_t = x_{t-1} + (w_t-x_{t-1})*2/(t-1)
            for p in OTB.parameters():
                p_OCO = OTB_to_OCO[p]
                p.data += (p_OCO.data - p.data) * beta(t) / sum([beta(i+1) for i in range(t)])
                # error: a leaf Variable that requires grad is being used in an in-place operation.
                # p.add_((p_OCO.data - p.data)/t)
            log_x_t = f'x_{t}: {[p.data for p in OTB.parameters()]}'

            # 3) gradient at current iteration: \nabla\ell(x_t, z_t)
            OTB.zero_grad()
            outputs = OTB(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            log_g_current = f'g(x_{t},z_{t}): {[p.grad for p in OTB.parameters()]}'
            # now we add t*\nabla\ell(x_t, z_t), and we get new gradient g_t.
            for p in OTB.parameters():
                gradients[p] += beta(t) * p.grad

            # 4) optimize one step on OCO to update w_{t+1}
            optimizer.zero_grad()
            # we need to first copy the gradients g_t with noise to OCO
            for p in OCO.parameters():
                if not p.grad:
                    p.grad = torch.zeros_like(p.data)
                p.grad.copy_(gradients[OCO_to_OTB[p]])
                '''
                Caveat:
                The following line assigns the pointer of g_t to p.grad, so 
                each time we call `zero_grad()`, we set g_t to be zero.
                As a solution, we need to use in-place operation `.copy_()`
                to assign the value instead of the pointer.
                '''
                # p.grad = gradients[OCO_to_OTB[p]]
                # following eq. (6), variance is constant if we choose k = 1,
                # so we just sample i.i.d. gaussian with variance \sigma^2.
                # num_nodes = bin(t).count('1')
                # for _ in range(num_nodes):
                #     p.grad += sigma * torch.randn_like(p.grad)

                p.grad.div_(beta(t))
            log_g_t = f'g_{t}: {[p.grad for p in OCO.parameters()]}'
            optimizer.step()
            
            grad_error = sum([torch.norm(p_OCO.grad - OCO_to_OTB[p_OCO].grad)**2 for p_OCO in OCO.parameters()]) ** 0.5

            # debug logs
            print(log_w_t)
            print(log_x_prev)
            print(log_x_t)
            print(log_g_prev)
            print(log_g_current)
            print(log_g_t)
            print(f'grad diff: {grad_error}')

        # weight clone
        for p in OCO.parameters():
            p.data.copy_(OCO_to_OTB[p].data)

        # testing
        test_step(epoch, net, testloader, criterion, device)