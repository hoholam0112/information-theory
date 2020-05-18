""" Estimate mutual information between two gaussian distribution """

import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import progressbar
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from mine import MINE
from torchutils.logging import hyperparameter_string

class StatisticsNetwork(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_units):
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_units = hidden_units

        self.activation = nn.SELU()
        self.layer1 = nn.Linear(input_dim1 + input_dim2, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.head = nn.Linear(hidden_units, 1)

    def forward(self, x, y):
        #h = self.activation(self.layer1(x)) + self.activation(self.layer2(x))
        h = torch.cat([x, y], axis=1)
        h = self.activation(self.layer1(h))
        h = self.activation(self.layer2(h))
        output = self.head(h)
        return output

def get_covariance_matrix(dim, correlation):
    """ Compute mutual information analytically  """
    cov = np.zeros([2*dim, 2*dim])
    for i in range(dim):
        cov[i, i + dim] = correlation
        cov[i + dim, i] = correlation
        cov[i, i] = 1
        cov[i + dim, i + dim] = 1
    return cov

def compute_true_mutual_information(dim, correlation):
    """ Compute mutual information analytically  """
    cov = get_covariance_matrix(dim, correlation)
    det_cov = np.linalg.det(cov)
    if det_cov == 0:
        raise ValueError('determinant of covariance matrix is 0.')
    return -0.5 * np.log(det_cov)

def get_samples(dim, correlation, num_samples):
    data = np.random.multivariate_normal(
                mean=np.zeros(2*dim),
                cov=get_covariance_matrix(dim, correlation),
                size=num_samples)
    return data.astype(np.float32)

def run(args):
    """ train model """
    # Define argument 
    device = torch.device('cuda:{}'.format(args.gpu)
                    if torch.cuda.is_available() else 'cpu')

    hidden_units = 512
    dim = 20
    correlation = 0.9
    #sigma_x, sigma_z = 1.0, 0.09
    batch_size = 10000
    total_steps = 50000
    lr = 1e-3
    weight_decay = 0.0
    ema_decay = 0.999
    criterion = 'mine-d'

    # Define tensorboard summary writer
    save_dir = hyperparameter_string(hidden_units=hidden_units,
                                     dim=dim,
                                     correlation=correlation,
                                     batch_size=batch_size,
                                     total_steps=total_steps,
                                     lr=lr,
                                     weight_decay=weight_decay,
                                     ema_decay=ema_decay,
                                     criterion=criterion)
    writer = SummaryWriter('./train_logs/two_gaussian/{}'.format(save_dir))

    # Build modules
    statistics_network = StatisticsNetwork(
            dim, dim, hidden_units)
    statistics_network.to(device)

    optimizer = optim.Adam(statistics_network.parameters(),
                           lr=lr)

    mine_object = MINE(statistics_network,
                       criterion,
                       ema_decay)

    # Define data loader
    #true_mi = np.log(1 + sigma_x**2/sigma_z**2) / 2
    true_mi = compute_true_mutual_information(
            dim, correlation)
    eMIs = np.zeros(total_steps, dtype=np.float32)
    eMI_ema = None

    statistics_network.train()
    pbar = progressbar.ProgressBar()
    for i in pbar(range(total_steps)):
        #x = torch.randn(batch_size, dim) * sigma_x
        #z = torch.randn(batch_size, dim) * sigma_z
        #y = x + z

        #x = x.to(device)
        #y = y.to(device)
        #z = z.to(device)

        batch = get_samples(dim, correlation, batch_size)
        x = torch.from_numpy(batch[:, :dim])
        y = torch.from_numpy(batch[:, dim:])
        x = x.to(device)
        y = y.to(device)

        eMI, loss = mine_object.estimate_on_batch(x, y)

        #eMIs[i] = mine_object.eMI_ema
        eMIs[i] = eMI

        if eMI_ema is None:
            eMI_ema = eMI
        else:
            eMI_ema = eMI_ema*ema_decay + (1.0 - ema_decay) * eMI

        optimizer.zero_grad()
        #loss = -eMI
        loss.backward()
        optimizer.step()

        writer.add_scalars('estimates', {'eMI' : eMI,
                                         'true_mi' : true_mi,
                                         'eMI_ema' : eMI_ema}, i)
    writer.close()

    #plt.figure()
    #x = np.arange(total_steps)
    #y_true = true_mi * np.ones_like(eMIs)
    #plt.plot(x, y_true, 'r-', x, eMIs, 'b-')
    #plt.show()


