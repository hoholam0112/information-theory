""" MINE algorithm core part """

import sys
import torch
import numpy as np

class MINE:
    """ Mutual Information Neural Estimator class  """
    def __init__(self,
                 statistics_network,
                 ema_decay):
        """ Initialize MINE object

        Args:
            statistics_network (nn.Module): neural network f(x, z) -> R
            ema_decay (float): decay rate for exponential moving average
        """
        self.statistics_network = statistics_network
        self.ema_decay = ema_decay
        self.ema_denominator = None
        self.random_state = np.random.RandomState(0)

    def estimate_on_batch(self, x, z):
        """ Estimate mutual information and return loss function on mini-batch of samples """
        self.statistics_network.train()

        rand_indices = np.arange(z.size(0))
        self.random_state.shuffle(rand_indices)
        z_marg = z[rand_indices]

        statistics_joint = self.statistics_network(x, z)
        mean_statistics = torch.mean(statistics_joint)

        statistics_marginal = self.statistics_network(x, z_marg)
        denominator = torch.mean(torch.exp(statistics_marginal))

        # Correct biased gradient
        if self.ema_denominator is None:
            self.ema_denominator = denominator
        else:
            self.ema_denominator = ((1.0 - self.ema_decay) * denominator +
                                     self.ema_decay * self.ema_denominator).detach()

        eMI = (mean_statistics - torch.log(denominator)).detach()
        loss = -(mean_statistics -
                 denominator / self.ema_denominator)
        return eMI, loss

    def estimate_on_dataset(self, loader):
        """ Estimate mutual information between two distribution

        Args:
            loader (DataLoader): Dataloader loads
                mini-batch samples (x, z) from joint distribution p(x, z)
        """
        self.statistics_network.eval()
        iterator_joint = iter(loader)
        iterator_marginal = iter(loader)

        num_samples = 0.0
        term1, term2 = 0.0, 0.0

        try:
            while True:
                x, z = next(iterator_joint)
                _, z_marginal = next(iterator_marginal)

                with torch.no_grad():
                    statistics_joint = self.statistics_network(x, z)
                    statistics_marginal = self.statistics_network(x, z_marginal)

                    term1 += torch.sum(statistics_joint)
                    term2 += torch.sum(torch.exp(statistics_marginal))
                    num_samples += statistics_joint.size(0)
        except StopIteration:
            pass

        eMI = term1/num_samples - torch.log(term2/num_samples)
        return eMI

