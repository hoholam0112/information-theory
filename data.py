import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt

def imshow(image, title=None):
    image = image.numpy().transpose([1, 2, 0])
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def get_dataset(dataset_name):
    """ Get dataset torch.utils.data.DataSet object  """
    if dataset_name == 'mnist':
        training_set = torchvision.datasets.MNIST('./dataset', train=True, download=True)
        test_set = torchvision.datasets.MNIST('./dataset', train=False, download=True)

        x_train = training_set.data.numpy()
        x_train = np.reshape(x_train, [x_train.shape[0], -1])
        y_train = training_set.targets.numpy()

        #rs = np.random.RandomState(seed=0)
        #perm_index = rs.permutation(x_train.shape[0])
        #x_train = x_train[perm_index]
        #y_train = y_train[perm_index]

        x_valid = torch.tensor(x_train[50000:], dtype=torch.float32)
        y_valid = torch.tensor(y_train[50000:], dtype=torch.int64)
        x_train = torch.tensor(x_train[:50000], dtype=torch.float32)
        y_train = torch.tensor(y_train[:50000], dtype=torch.int64)

        train_set = torch.utils.data.TensorDataset(x_train, y_train)
        valid_set = torch.utils.data.TensorDataset(x_valid, y_valid)

        x_test = test_set.data.numpy()
        x_test = np.reshape(x_test, [x_test.shape[0], -1])
        x_test = torch.tensor(x_test)
        y_test = test_set.targets

        test_set = torch.utils.data.TensorDataset(x_test, y_test)

        dataset = {'train' : train_set,
                   'valid' : valid_set,
                   'test' : test_set}

    else:
        raise NotImplementedError('Unknown dataset_name passed: {}'.format(dataset_name))
    return dataset

if __name__ == '__main__':
    dataset = get_dataset('mnist')
    loader = {}
    for k, dset in dataset.items():
        shuffle = (k == 'train')
        loader[k] = torch.utils.data.DataLoader(dset, batch_size=4,
                shuffle=shuffle, pin_memory=True, num_workers=4)

    stop = 1
    for i, (x, y) in enumerate(loader['valid']):
        images = x.view(-1, 1, 28, 28)
        if i == stop:
            break

    print(y)
    image_grid = torchvision.utils.make_grid(images, nrow=2)

    plt.figure()
    imshow(image_grid)

    plt.show()
