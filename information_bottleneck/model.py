""" Defining neural network architecture for information bottleneck experiment  """

import torch
import torch.nn as nn

# Weight initialization
def weight_init_fn(module):
    """ Initialize weights """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, std=0.01)
        torch.nn.init.zeros_(module.bias)

class GaussianAdditiveNoise(nn.Module):
    """ Gaussian additive noise layer """
    def __init__(self, mean=0.0, std=0.01):
        """ You can set mean and std of gaussian random noise """
        super().__init__()
        self.mean = mean
        self.std = std
        self.training = True

    def forward(self, x):
        """ Add gaussasin random noise when training.
        returns input itself when testing """
        if self.training:
            noise = torch.randn_like(x)
            noise = noise * self.std + self.mean
            return x + noise
        else:
            return x

class StatisticsNetwork(nn.Module):
    def __init__(self,
                 name,
                 bottleneck_dim,
                 noise='none'):
        super().__init__()

        self.name = name
        if name  == 'mnist':
            self.input_dim = 784
            self.bottleneck_dim = bottleneck_dim
            self.activation = nn.ELU()

            if noise == 'additive':
                self.add_noise_layer = GaussianAdditiveNoise(std=0.3)
                self.layer1 = nn.Sequential(
                        nn.Linear(self.input_dim + bottleneck_dim, 512)
                    )
                self.layer2 = nn.Sequential(
                        #GaussianAdditiveNoise(std=0.5),
                        nn.Linear(512, 512)
                    )
                self.head = nn.Sequential(
                        #GaussianAdditiveNoise(std=0.5),
                        nn.Linear(512, 1)
                    )
            elif noise == 'none':
                self.layer1 = nn.Linear(self.input_dim + bottleneck_dim, 512)
                self.layer2 = nn.Linear(512, 512)
                self.head = nn.Linear(512, 1)
            else:
                raise NotImplementedError('Invalid value for noise argument: {}'.format(noise))
        else:
            raise NotImplementedError('Unknown dataset name passed.')

    def forward(self, x, y):
        output = None
        if self.name == 'mnist':
            y = self.add_noise_layer(y)
            h = torch.cat([x, y], axis=1)
            h = self.activation(self.layer1(h))
            h = self.activation(self.layer2(h))
            output = self.head(h)
        return output


class Classifier(nn.Module):
    def __init__(self,
                 name,
                 bottleneck_dim=None):
        """ Build classifier by name """
        super().__init__()

        self.name = name
        self.activation = nn.ReLU()
        if name in ['mnist_mine', 'mnist_base']:
            assert bottleneck_dim is not None

            self.input_dim = 784
            self.output_dim = 10
            self.bottleneck_dim = bottleneck_dim

            # Define layers
            self.encoder = nn.Sequential(
                        nn.Linear(self.input_dim, 1024),
                        self.activation,
                        nn.Linear(1024, 1024),
                        self.activation,
                        nn.Linear(1024, bottleneck_dim),
                    )
            self.classifier = nn.Linear(bottleneck_dim, self.output_dim)

            # Initialize weights
            self.apply(weight_init_fn)
        #elif name == 'mnist_base':
        #    self.input_dim = 784
        #    self.output_dim = 10

        #    # Define layers
        #    self.encoder = nn.Sequential(
        #                nn.Linear(self.input_dim, 1024),
        #                self.activation,
        #                nn.Linear(1024, 1024),
        #            )
        #    self.classifier = nn.Linear(1024, self.output_dim)

        #    # Weight initialization
        #    self.apply(weight_init_fn)
        else:
            raise NotImplementedError('Unknown dataset name passed.')

    def forward(self, x):
        bottleneck = self.encoder(x)
        output = self.classifier(bottleneck)
        return output, bottleneck



