import sys, os, argparse, importlib

parser = argparse.ArgumentParser(description='MINE train command collection.')

subparser = parser.add_subparsers(help='subcommand parser')

# Parser for training domain adaptation model.
parser_eq = subparser.add_parser('equitability', help='MINE equitability experiment.')
parser_eq.set_defaults(experiment='equitability')

parser_ib = subparser.add_parser('information_bottleneck', help='MINE information bottleneck experiment.')
parser_ib.set_defaults(experiment='information_bottleneck')
parser_ib.add_argument('dataset_name', help='Dataset name.', type=str, choices=['mnist'])
parser_ib.add_argument('model_name', help='Model name.', type=str, choices=['mine', 'base'])
parser_ib.add_argument('tag', help='Tag name.', type=str)

# Optimizer argument
parser_ib.add_argument('--gpu', help='Which GPUs to be used for training.', type=int, required=True)
parser_ib.add_argument('--batch_size', help='batch size.', type=int)
parser_ib.add_argument('--epochs', help='The number of total training epochs.', type=int)
parser_ib.add_argument('--bottleneck_dim', help='Size of bottleneck dimension.', type=int)
parser_ib.add_argument('--beta', help='Weight for information bottleneck loss.', type=float)

group_opt = parser_ib.add_argument_group(title='Optimizer params',
        description='params for optimizer')
group_opt.add_argument('--opt_name', help='The name of optimizer.', type=str)
group_opt.add_argument('--init_lr', help='The initial learning rate.', type=float)
group_opt.add_argument('--weight_decay', help='weight decay rate.', type=float)

args = parser.parse_args()
module = importlib.import_module('{}.train'.format(args.experiment))
module.run(args)


