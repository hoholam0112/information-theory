""" Experiment for testing MINE approximation fidelity
    using equaility of mutual information when non-linear
    deterministic mapping is applied of a random variable """
import sys
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from mine import MINE

class StatisticsNetwork(nn.Module):
    def __init__(self, in_features1, in_features2):
        super(StatisticsNetwork, self).__init__()
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.activation = nn.ReLU()

        self.layer1 = nn.Linear(in_features1 + in_features2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.head = nn.Linear(128, 1)

    def forward(self, x, y):
        h = torch.cat([x, y], axis=1)
        h = self.activation(self.layer1(h))
        h = self.activation(self.layer2(h))
        h = self.activation(self.layer3(h))
        output = self.head(h)
        return output

def weight_init_fn(module):
    """ Initialize weights of a neural network """
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        torch.nn.init.zeros_(module.bias)

def train(statistics_network,
          optimizer,
          loader,
          epoch,
          device,
          ema_decay=0.9999):
    """ Approximate mutual information using MINE

    Args:
        statistics_network (nn.Module): neural network for estimating mutual information.
        optimizer (pytorch optimizer): optimizer
        loader (DataLoader): Dataloader loads
            mini-batch samples (x, z) from joint distribution p(x, z)
        epoch (int): The number of totol training epochs
        ema_decay (float): exponential decay rate for correcting biased gradient.
            default value is 0.9999.
    """
    mine_object = MINE(statistics_network,
                       ema_decay,
                       device)

    i = 0
    iterator = iter(loader)
    while i < epoch:
        try:
            while True:
                x, z = next(iterator)

                _, loss = mine_object.estimate_on_batch(x, z)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        except StopIteration:
            # Evaluate eMI
            eMI = mine_object.estimate_on_dataset(loader)
            print('{:d}/{:d} | eMI: {:.4f}\r'.format(i, epoch, eMI), end='')

            # Reset iterator for next epoch
            iterator = iter(loader)
            i += 1
    print()
    return eMI


def run(args):
    # Define argument 
    device = torch.device('cuda:{}'.format(args.gpu)
                    if torch.cuda.is_available() else 'cpu')

    opt_name = args.opt_name or 'rmsprop'
    init_lr = args.init_lr or 1e-4
    weight_decay = args.weight_decay or 0

    statistics_network = StatisticsNetwork(in_features1=2, in_features2=2)

    if opt_name == 'sgd':
        optimizer = optim.SGD(statistics_network.parameters(),
                              lr=init_lr,
                              momentum=0.9,
                              weight_decay=weight_decay,
                              nesterov=True)
    elif opt_name == 'rmsprop':
        optimizer = optim.RMSprop(statistics_network.parameters(),
                                  lr=init_lr,
                                  momentum=0.9,
                                  weight_decay=weight_decay)
    elif opt_name == 'adam':
        optimizer = optim.AdamW(statistics_network.parameters(),
                                lr=init_lr,
                                weight_decay=weight_decay)
    else:
        raise NotImplementedError('Unknown optimizer is passed: {}'.format(opt_name))

    init_state_dict = optimizer.state_dict()

    num_samples = 2000
    dim = 2
    rs = np.random.RandomState(seed=0)

    # Define input variables
    X = rs.uniform(-1, 1, [num_samples, dim])
    epsilon = rs.normal(0, 1, [num_samples, dim])

    X = torch.tensor(X, dtype=torch.float32).to(device)
    epsilon = torch.tensor(epsilon, dtype=torch.float32).to(device)
    mappings = {'identity' : lambda sigma: X + sigma * epsilon,
                'cubic' : lambda sigma: X**3 + sigma * epsilon,
                'sin' : lambda sigma: torch.sin(X) + sigma * epsilon}

    eMI = defaultdict(lambda: []) # estimated mutual information 
    for name, function in mappings.items():
        for sigma in np.arange(0.1, 1.1, 0.1):
            print('function: {}, sigma: {:.1f}'.format(name, sigma))

            Y = function(sigma)

            dataset = TensorDataset(X, Y)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)

            # Initialize neural network and optimizer 
            statistics_network.apply(weight_init_fn)
            optimizer.load_state_dict(init_state_dict)

            # Train MINE
            eMI_temp = train(statistics_network,
                             optimizer,
                             loader=loader,
                             epoch=100,
                             device=device)

            # Estimate MI
            eMI[name].append(eMI_temp.item())


    for key, emi_list in eMI.items():
        print(key)
        print(emi_list)


if __name__ == '__main__':
    statistics_network = StatisticsNetwork(in_features1=2, in_features2=2)

    for tensor in statistics_network.parameters():
        print(tensor)

    statistics_network.apply(weight_init_fn)

    for tensor in statistics_network.parameters():
        print(tensor)



