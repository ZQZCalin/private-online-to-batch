import torch
import copy
import wandb
from tqdm import tqdm
from utils import dict_append


# =====================================
# Standard Training (Benchmark)
# =====================================

def train_step(epoch, net, trainloader, criterion, optimizer, device):
    '''
    Train single epoch.
    '''
    print(f'\nTraining epoch {epoch+1}..')
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(trainloader))
    for batch, (inputs, labels) in pbar:
        # load data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # forward and backward propagation
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # stat updates
        train_loss += (loss.item() - train_loss)/(batch+1)  # average train loss
        total += labels.size(0)                             # total predictions
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()        # correct predictions
        train_acc = 100*correct/total                       # average train acc

        pbar.set_description(f'epoch {epoch+1} batch {batch+1}: \
            train loss {train_loss:.2f}, train acc: {train_acc:.2f}')

    # return stats
    return train_loss, train_acc


def test_step(epoch, net, testloader, criterion, device):
    '''
    Test single epoch.
    '''
    print(f'\nEvaluating epoch {epoch+1}..')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(testloader))
    with torch.no_grad():
        for batch, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += (loss.item() - test_loss)/(batch+1)
            total += labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            # test_acc = 100*correct/total
            # pbar.set_description(f'test loss: {test_loss}, test acc: {test_acc}')

    test_acc = 100*correct/total
    print(f'test loss: {test_loss}, test acc: {test_acc}')
    return test_loss, test_acc


def train(net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device):
    stats = {
        'args': None,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(epoch, net, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test_step(epoch, net, testloader, criterion, device)
        scheduler.step()

        dict_append(stats, train_loss=train_loss, train_acc=train_acc, 
            test_loss=test_loss, test_acc=test_acc)
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc,
            'test_loss': test_loss, 'test_acc': test_acc})

    return stats



# =====================================
# Private Training (DP-OTB)
# =====================================
'''
Train with DP-OTB.

To implement DP-OTB, we need to networks: one original network (net) corresponding to the OCO algorithm and
one clone network (net_clone) corresponding to DP-OTB.
*At the end, we copy net <- net_clone because we want to output the aggregated weights.

We stick to the notation in the paper, with k = 1 (i.e., beta_t = t).
'''

# def beta(t):
#     '''
#     \beta_t = t^k. Here we choose k = 1.
#     '''
#     return t


def train_step_private(epoch, sigma, k, OCO, OTB, trainloader, criterion, optimizer, device):
    '''
    Train single epoch with differential privacy guarantees.
    '''
    print(f'\nTraining epoch {epoch+1}..')

    OCO.train()
    OTB.train()

    train_loss = 0
    correct = 0
    total = 0

    # beta_t
    beta = lambda t: pow(t, k)

    # conversion dictionaries between OCO and OTB
    OCO_to_OTB = {p1: p2 for p1, p2 in zip(OCO.parameters(), OTB.parameters())}
    OTB_to_OCO = {p2: p1 for p1, p2 in zip(OCO.parameters(), OTB.parameters())}

    # accumulated gradients g_t
    gradients = {p: torch.zeros_like(p) for p in OTB.parameters()}

    pbar = tqdm(enumerate(trainloader))
    for batch, (inputs, labels) in pbar:
        t = batch+1

        # load data
        inputs, labels = inputs.to(device), labels.to(device)

        # 1) gradient at last iteration: \nabla\ell(x_{t-1}, z_t)
        # we need to compute it before we update x_t = x_{t-1} + (w_t-x_{t-1})*2/(t-1)
        OTB.zero_grad()
        outputs = OTB(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # since we update g_t using:
        # g_t = g_{t-1} + t \nabla\ell(x_t, z_t) - (t-1) \nabla\ell(x_{t-1}, z_t),
        # we can subtract (t-1)*\nabla\ell(x_{t-1}, z_t) first.
        if t > 1:
            for p in OTB.parameters():
                gradients[p] -= beta(t-1) * p.grad

        # 2) aggregate weight: x_t = x_{t-1} + (w_t-x_{t-1})*2/(t-1)
        for p in OTB.parameters():
            p_OCO = OTB_to_OCO[p]
            p.data += (p_OCO.data - p.data) * beta(t) / sum([beta(i+1) for i in range(t)])
            # error: a leaf Variable that requires grad is being used in an in-place operation.
            # p.add_((p_OCO.data - p.data)/t)

        # 3) gradient at current iteration: \nabla\ell(x_t, z_t)
        OTB.zero_grad()
        outputs = OTB(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # now we add t*\nabla\ell(x_t, z_t), and we get new gradient g_t.
        for p in OTB.parameters():
            gradients[p] += beta(t) * p.grad

            # ***test mode only***
            # For test purpose, we turn off gradient difference aggregation 
            # and set gradient directly to be \beta_t\nabla\ell(x_t,z_t).
            # Uncomment this line for test mode.
            gradients[p] = beta(t) * p.grad

        # 4) optimize one step on OCO to update w_{t+1}
        optimizer.zero_grad()
        # we need to first copy the gradients g_t with noise to OCO
        for p in OCO.parameters():
            p.grad = gradients[OCO_to_OTB[p]]
            # following eq. (6), variance is constant if we choose k = 1,
            # so we just sample i.i.d. gaussian with variance \sigma^2.
            num_nodes = bin(t).count('1')
            for _ in range(num_nodes):
                p.grad += sigma * torch.randn_like(p.grad)
        optimizer.step()

        # stat updates
        train_loss += (loss.item() - train_loss)/(batch+1)  # average train loss
        total += labels.size(0)                             # total predictions
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()        # correct predictions
        train_acc = 100*correct/total                       # average train acc

        pbar.set_description(f'epoch {epoch+1} batch {batch+1}: \
            train loss {train_loss:.2f}, train acc: {train_acc:.2f}')

    # at the end, we need to clone OTB parameters to OCO
    for p in OCO.parameters():
        p.data = OCO_to_OTB[p].data
        # p.copy_(OCO_to_OTB[p])

    # return stats
    return train_loss, train_acc


def train_private(sigma, k, net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device):
    '''
    Main train loop for private online-to-batch.

    Recall that net corresponds to OCO and net_clone corresponds to OTB.
    '''

    stats = {
        'args': None,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    # DP-OTB model
    net_clone = copy.deepcopy(net)

    for epoch in range(epochs):
        train_loss, train_acc = train_step_private(epoch, sigma, k, net, net_clone, trainloader, criterion, optimizer, device)
        # note that we test the performance of OTB (net_clone)
        test_loss, test_acc = test_step(epoch, net_clone, testloader, criterion, device)
        scheduler.step()

        dict_append(stats, train_loss=train_loss, train_acc=train_acc, 
            test_loss=test_loss, test_acc=test_acc)
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc,
            'test_loss': test_loss, 'test_acc': test_acc})

    return stats