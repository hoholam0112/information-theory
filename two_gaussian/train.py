""" Estimate mutual information between two gaussian distribution """

import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import progressbar
import matplotlib.pyplot as plt

from mine import MINE

class StatisticsNetwork(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units

        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(2, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.head = nn.Linear(hidden_units, 1)

    def forward(self, x, y):
        #h = self.activation(self.layer1(x)) + self.activation(self.layer2(x))
        h = torch.cat([x, y], axis=1)
        h = self.activation(self.layer1(h))
        h = self.activation(self.layer2(h))
        output = self.head(h)
        return output

def run(args):
    """ train model """
    # Define argument 
    device = torch.device('cuda:{}'.format(args.gpu)
                    if torch.cuda.is_available() else 'cpu')

    hidden_units = 10
    var_x, var_z = 1.0, 1.0
    batch_size = 256
    total_steps = 5000

    # Build modules
    statistics_network = StatisticsNetwork(hidden_units)
    statistics_network.to(device)

    optimizer = optim.Adam(statistics_network.parameters(),
                          lr=0.001)

    mine_object = MINE(statistics_network,
                       ema_decay=0.999)

    # Define data loader
    analystic_mi = np.log(1 + var_x/var_z) / 2
    eMIs = np.zeros(total_steps, dtype=np.float32)

    statistics_network.train()
    pbar = progressbar.ProgressBar()
    for i in pbar(range(total_steps)):
        x = torch.randn(batch_size, 1) * np.sqrt(var_x)
        z = torch.randn(batch_size, 1) * np.sqrt(var_z)
        y = x + z

        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        eMI, loss = mine_object.estimate_on_batch(x, y)

        eMIs[i] = mine_object.eMI_ema

        optimizer.zero_grad()
        loss = -eMI
        loss.backward()
        optimizer.step()

    plt.figure()
    x = np.arange(total_steps)
    y_true = analystic_mi * np.ones_like(eMIs)
    plt.plot(x, y_true, 'r-', x, eMIs, 'b-')
    plt.show()


