# -*- coding: utf-8 -*-
# Copyright 2024 Minggui Song.
# All rights reserved.

from sys import exit, version_info

from src import fullhelp_argumentparser
from src.sys_output import Output

# version control
if version_info[0] == 3 and version_info[1] >= 6:
    pass
else:
    output = Output()
    output.error('This program requires at least python3.6')
    exit()

def main() -> None:
    """Creat subcommands and execut."""
    PARSER = fullhelp_argumentparser.FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    DATA_PROCESS = fullhelp_argumentparser.DPArgs(
        SUBPARSER,
        'data_process',
        """.""")
    PRETRAIN = fullhelp_argumentparser.PreTrainArgs(
        SUBPARSER,
        'pre_train',
        """.""")
    TRAIN = fullhelp_argumentparser.TrainArgs(
        SUBPARSER,
        'train',
        """.""")
    PREDICT = fullhelp_argumentparser.PredictArgs(
        SUBPARSER,
        'predict',
        """.""")

    def bad_args(args) -> None:
        """Print help on bad arguments."""
        PARSER.print_help()
        exit()

    PARSER.set_defaults(func=bad_args)
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)

if __name__ == '__main__':
    main()