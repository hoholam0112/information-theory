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
group_opt.add_argument('--lr_clf', help='The initial learning rate for a classifier.', type=float)
group_opt.add_argument('--lr_mine', help='The initial learning rate for a statistics network.', type=float)
group_opt.add_argument('--weight_decay', help='weight decay rate.', type=float)

# Parser for mutual information between two gaussians
parser_tg = subparser.add_parser('two_gaussian', help='Two gaussian experiment.')
parser_tg.set_defaults(experiment='two_gaussian')
parser_tg.add_argument('dim', help='Dimension size of multivariate Gaussian random variables.', type=int)
parser_tg.add_argument('correlation', help='Correlation between two mutivariate Gaussian random variables.', type=float)

parser_tg.add_argument('--gpu', help='Which GPUs to be used for training.', type=int, required=True)
parser_tg.add_argument('--model_name', help='Model name.', type=str)
parser_tg.add_argument('--batch_size', help='Batch size.', type=int)
parser_tg.add_argument('--hidden_units', help='hidden units', type=int)
parser_tg.add_argument('--steps', help='steps', type=int)
parser_tg.add_argument('--lr', help='learning rate', type=float)
parser_tg.add_argument('--weight_decay', help='weight decay rate', type=float)
parser_tg.add_argument('--ema_decay', help='ema decay rate', type=float)
parser_tg.add_argument('--criterion', help='criterion name \'mine-d\' or \'mine-f\'.', type=str)
parser_tg.add_argument('--activation', help='activation function name default is \'relu\'.', type=str)
parser_tg.add_argument('--output_bound', help='output bound for statistics network.', type=float)

args = parser.parse_args()
module = importlib.import_module('{}.train'.format(args.experiment))
module.run(args)


