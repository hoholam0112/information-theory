import torch.optim as optim

def build_optimizer(opt_name, per_param_options, global_option):
    """ Build pytorch optimizer from some options.

    Args:
        opt_name (string): Optimizer name.
        per_param_options (list of dict): List of dicts which specify per parameter option.
        global_option (dict): A Dictionary specifying global option.

    Returns:
        optimizer (torch.optim.Optimizer)
    """
    optimizer = {
            'sgd' : lambda: optim.SGD(per_param_options, **global_option),
            'rmsprop' : lambda: optim.RMSprop(per_param_options, **global_option),
            'adam' : lambda: optim.Adam(per_param_options, **global_option),
            'adamw': lambda: optim.AdamW(per_param_options, **global_option)
        }
    return optimizer[opt_name]()
