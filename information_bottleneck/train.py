""" Training information bottleneck model """

import os
from collections import defaultdict

import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
import progressbar

from information_bottleneck.model import StatisticsNetwork, Classifier
from data import get_dataset
from mine import MINE
from utils.torch.metrics import Accuracy, Mean

class GradientReversalLayer(Function):
    """ Negate gradient in backward pass """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -1.0 * grad_output, None

def run(args):
    """ train model """
    # Define argument 
    device = torch.device('cuda:{}'.format(args.gpu)
                    if torch.cuda.is_available() else 'cpu')

    dataset_name = args.dataset_name
    model_name = args.model_name
    tag = args.tag

    opt_name = args.opt_name or 'sgd'
    init_lr = args.init_lr or 0.005
    weight_decay = args.weight_decay or 0.0

    batch_size = args.batch_size or 128
    total_epochs = args.epochs or 1000

    bottleneck_dim = args.bottleneck_dim or 256
    beta = args.beta or 1e-3
    ema_decay = 0.999

    # Load checkpoint file
    checkpoint_dir = './train_logs/{}/{}'.format(dataset_name, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, '{}.tar'.format(tag))
    if os.path.exists(checkpoint_path):
        general_state_dict = torch.load(checkpoint_path)
        bottleneck_dim = general_state_dict['bottleneck_dim']
    else:
        general_state_dict = None

    # Build modules
    clf_name = dataset_name + '_' + model_name
    clf = Classifier(name=clf_name,
                     bottleneck_dim=bottleneck_dim)
    clf.to(device)

    if model_name == 'mine':
        statistics_network = StatisticsNetwork(name=dataset_name,
                                               bottleneck_dim=bottleneck_dim)
        statistics_network.to(device)

    # Build optimizer
    options = [
            {'params' : clf.parameters()},
        ]
    if model_name == 'mine':
        options.append({'params' : statistics_network.parameters()})

    if opt_name == 'sgd':
        optimizer = optim.SGD(options,
                              lr=init_lr,
                              weight_decay=weight_decay,
                              momentum=0.9)
    elif opt_name == 'rmsprop':
        optimizer = optim.RMSprop(options,
                                  lr=init_lr,
                                  momentum=0.9,
                                  weight_decay=weight_decay)
    elif opt_name == 'adam':
        optimizer = optim.Adam(options,
                               lr=init_lr,
                               weight_decay=weight_decay)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(options,
                                lr=init_lr,
                                weight_decay=weight_decay)
    else:
        raise NotImplementedError('Unknown optimizer is passed: {}'.format(opt_name))

    # Define data loader
    dataset = get_dataset(dataset_name)
    loader = {}
    for k, dset in dataset.items():
        shuffle = (k == 'train')
        loader[k] = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                shuffle=shuffle, pin_memory=True, num_workers=4)

    # Define loss function
    cross_entropy_loss = nn.CrossEntropyLoss()

    # Train a model
    clf.train()
    if model_name == 'mine':
        statistics_network.train()
        mine_object = MINE(statistics_network,
                           ema_decay)

    metric = {'train_error' : Accuracy(),
              'xent_loss' : Mean(),
              'ib_loss' : Mean(),
              'valid_error' : Accuracy()}

    if general_state_dict is not None:
        epoch = general_state_dict['epoch']
        best_val_error = general_state_dict['best_val_error']
        optimizer.load_state_dict(general_state_dict['optimizer'])
        clf.load_state_dict(general_state_dict['clf'])
        if model_name == 'mine':
            statistics_network.load_state_dict(
                    general_state_dict['statistics_network'])
    else:
        epoch = 0
        best_val_error = None

    epochs_phase_one = 20
    epochs_phase_two = 40

    while epoch < total_epochs:
        i = 1
        max_value = progressbar.UnknownLength
        with progressbar.ProgressBar(max_value) as pbar:
            # Train epoch
            metric['train_error'].reset_state()
            metric['xent_loss'].reset_state()
            metric['ib_loss'].reset_state()

            for x, y_true in loader['train']:
                x = x.to(device)
                y_true = y_true.to(device)

                # Forward pass
                y_pred, bottleneck = clf(x)
                loss = cross_entropy_loss(y_pred, y_true)
                metric['train_error'].update_state(y_pred, y_true)
                metric['xent_loss'].update_state(
                        loss.detach().cpu() * torch.ones(y_true.size(0), dtype=torch.float32))

                if (model_name == 'mine') and (epoch > epochs_phase_one):
                    if epoch <= epochs_phase_two:
                        bottleneck = bottleneck.detach()
                    bottleneck_rev = GradientReversalLayer.apply(bottleneck)
                    eMI, loss_ib = mine_object.estimate_on_batch(x, bottleneck_rev)
                    metric['ib_loss'].update_state(
                            loss_ib.detach().cpu() * torch.ones(y_true.size(0), dtype=torch.float32))
                    loss += beta * loss_ib

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(i)
                i += 1

        # Validation
        metric['valid_error'].reset_state()
        for x, y_true in loader['valid']:
            x = x.to(device)
            y_true = y_true.to(device)
            with torch.no_grad():
                y_pred, _ = clf(x)
            metric['valid_error'].update_state(y_pred, y_true)

        epoch += 1
        print('tag: {}'.format(tag))
        print('Epoch: {:d}/{:d}'.format(epoch, total_epochs), end='')
        if best_val_error is not None:
            print(', best_valid_error: {:.2f}'.format(best_val_error))
        else:
            print('')

        for k, v in metric.items():
            if k in ['valid_error', 'train_error']:
                result = (1.0 - v.result()) * 100
                print('{}: {:.2f}'.format(k, result))
            else:
                print('{}: {:.4f}'.format(k, v.result()))


        # Save model
        curr_val_error = (1.0 - metric['valid_error'].result()) * 100
        if best_val_error is None:
            best_val_error = curr_val_error
        else:
            if best_val_error > curr_val_error:
                best_val_error = curr_val_error

                general_state_dict = {'optimizer' : optimizer.state_dict(),
                                      'clf' : clf.state_dict(),
                                      'epoch' : epoch,
                                      'best_val_error' : best_val_error,
                                      'bottleneck_dim' : bottleneck_dim}
                if model_name == 'mine':
                    general_state_dict['statistics_network'] = statistics_network.state_dict()

                torch.save(general_state_dict, checkpoint_path)
                print('Model saved.')
