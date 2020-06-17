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
from two_gaussian.model import StatisticsNetwork

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

    dim = args.dim
    correlation = args.correlation

    hidden_units = args.hidden_units or 512
    #sigma_x, sigma_z = 1.0, 0.09
    batch_size = args.batch_size or 500
    total_steps = args.steps or 100000
    lr = args.lr or 1e-4
    weight_decay = args.weight_decay or 0.0
    ema_decay = args.ema_decay or 0.999
    criterion = args.criterion or 'mine-d'
    model_name = args.model_name or 'basic'
    activation = args.activation or 'relu'

    # Define tensorboard summary writer
    exp_dir = 'dim={}, correlation={}'.format(dim, correlation)
    summary_dir = hyperparameter_string(
                        hidden_units=hidden_units,
                        batch_size=batch_size,
                        lr=lr,
                        weight_decay=weight_decay,
                        ema_decay=ema_decay,
                        criterion=criterion,
                        model_name=model_name,
                        activation=activation,
                    )

    # Build modules
    statistics_network = StatisticsNetwork(
            dim, dim, hidden_units, model_name, activation)
    statistics_network.to(device)

    optimizer = optim.RMSprop(statistics_network.parameters(),
                              lr=lr,
                              weight_decay=weight_decay)

    mine_object = MINE(statistics_network,
                       criterion,
                       ema_decay)

    # Define data loader
    #true_mi = np.log(1 + sigma_x**2/sigma_z**2) / 2
    true_mi = compute_true_mutual_information(
            dim, correlation)
    eMIs = np.zeros(total_steps, dtype=np.float32)
    eMI_ema = None

    print('true_mi: {:.3f}'.format(true_mi))

    writer = SummaryWriter('./train_logs/two_gaussian/{}/{}'.format(exp_dir, summary_dir))
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

        if torch.isnan(eMI):
            raise RuntimeError('eMI produces NaN')

        writer.add_scalar('eMI', eMI, i)
        writer.add_scalar('eMI_ema', eMI_ema, i)
        writer.flush()

    writer.close()

    #plt.figure()
    #x = np.arange(total_steps)
    #y_true = true_mi * np.ones_like(eMIs)
    #plt.plot(x, y_true, 'r-', x, eMIs, 'b-')
    #plt.show()


