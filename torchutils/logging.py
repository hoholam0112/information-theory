""" Utilities for logging training logs """

def hyperparameter_string(**hyperparams):
    """ convert set of hyperparameters to string """
    tokens = []
    for k, v in hyperparams.items():
        token = str(k) + '=' + str(v)
        tokens.append(token)
    hstring = '_'.join(tokens)
    return hstring

