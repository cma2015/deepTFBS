# -*- coding: utf-8 -*-
# Copyright 2024 Minggui Song.
# All rights reserved.

from argparse import ArgumentParser, HelpFormatter
from importlib import import_module
from logging import getLogger
from os import getpid
from re import ASCII, compile
from sys import exit, stderr
from textwrap import wrap
from src.sys_output import Output

logger = getLogger(__name__)  # pylint: disable=invalid-name

class ScriptExecutor(object):
    """Loads the relevant script modules and executes the script.

    This class is initialised in each of the argparsers for the relevant
    command, then execute script is called within their set_default function.

    Attributes:
        - command (str): Full commands.
        - subparsers: Subparsers for each subcommand.
        - output: Output info, warning and error.

    """

    def __init__(self, command: str, subparsers=None) -> None:
        """Initialize ScriptExecutor.
        Args:
            - command (str): Full commands.
            - subparsers: Subparsers for each subcommand.

        """
        self.command = command.lower()
        self.subparsers = subparsers
        self.output = Output()

    def import_script(self):
        """Only import a script's modules when running that script."""
        # cmd = os.path.basename(sys.argv[0])
        src = 'src'
        mod = '.'.join((src, self.command.lower()))
        module = import_module(mod)
        script = getattr(module, self.command.title().replace('_', ''))
        return script

    def execute_script(self, arguments) -> None:
        """Run the script for called command."""
        self.output.info(f'Executing: {self.command}. PID: {getpid()}')
        logger.debug(f'Executing: {self.command}. PID: {getpid()}')

        try:
            script = self.import_script()
            process = script(arguments)
            process.process()
        except KeyboardInterrupt:  # pylint: disable=try-except-raise
            raise
        except SystemExit:
            pass
        except Exception:  # pylint: disable=broad-except
            logger.exception('Got Exception on main handler:')
            logger.critical(
                'An unexpected crash has occurred. '
                'Crash report written to logfile. '
                'Please verify you are running the latest version of *** '
                'before reporting.')
        finally:
            exit()

class FullHelpArgumentParser(ArgumentParser):
    """Identical to the built-in argument parser.

    On error it prints full help message instead of just usage information.
    """

    def error(self, message: str) -> None:
        """Print full help messages."""
        self.print_help(stderr)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, f'{self.prog}: error: {message}\n')

class SmartFormatter(HelpFormatter):
    """Smart formatter for allowing raw formatting.

    Mainly acting in help text and lists in the helptext.

    To use: prefix the help item with 'R|' to overide
    default formatting. List items can be marked with 'L|'
    at the start of a newline.

    Adapted from: https://stackoverflow.com/questions/3853722
    """
    def __init__(self, prog: str,
                indent_increment: int = 2,
                max_help_position: int = 24,
                width=None) -> None:
        """Initialize SmartFormatter.

        Args:
            - prog (str): Program name.
            - indent_increment (int): Indent increment. default 2.
            - max_help_position (int): Max help position. default 24.
            - width: Width.

        """
        super().__init__(prog, indent_increment, max_help_position, width)
        self._whitespace_matcher_limited = compile(r'[ \r\f\v]+', ASCII)

    def _split_lines(self, text: str, width) -> list:
        if text.startswith('R|'):
            text = self._whitespace_matcher_limited.sub(' ', text).strip()[2:]
            output = []
            for txt in text.splitlines():
                indent = ''
                if txt.startswith('L|'):
                    indent = '    '
                    txt = '  - {}'.format(txt[2:])
                output.extend(wrap(
                    txt, width, subsequent_indent=indent))
            return output
        return HelpFormatter._split_lines(self, text, width)

class DeepTFBSArgs(object):
    """deepTFBS argument parser functions.

    It is universal to all commands.
    Should be the parent function of all subsequent argparsers.

    Attributes:
        - global_arguments: Global arguments.
        - argument_list: Argument list.
        - optional_arguments: Optional arguments.
        - parser: Parser.

    """

    def __init__(self, subparser, command: str,
                description: str = 'default', subparsers=None) -> None:
        """Initialize PEAM6AArgs.

        Args:
            - subparser: Subparser.
            - command (str): Command.
            - description (str): Description. default 'default'.
            - subparsers: Subparsers.

        """
        self.global_arguments = self.get_global_arguments()
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        if not subparser:
            return
        self.parser = self.create_parser(subparser, command, description)
        self.add_arguments()
        script = ScriptExecutor(command, subparsers)
        self.parser.set_defaults(func=script.execute_script)

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        return argument_list

    @staticmethod
    def get_optional_arguments() -> list:
        """Put the arguments in a list so that they are accessible.

        This is used for when there are sub-children.
        (e.g. convert and extract) Override this for custom arguments.
        """
        argument_list = []
        return argument_list

    @staticmethod
    def get_global_arguments() -> list:
        """Arguments that are used in ALL parts of PEA-m6A.

        DO NOT override this!
        """
        global_args = []
        global_args.append({'opts': ('-v', '--version'),
                            'action': 'version',
                            'version': 'deepTFBS v0.0.1a'})
        return global_args

    @staticmethod
    def create_parser(subparser, command: str, description: str):
        """Create the parser for the selected command."""
        parser = subparser.add_parser(
            command,
            help=description,
            description=description,
            epilog='Questions and feedback: '
                ' https://github.com/Songmg-Nwafu/',
            formatter_class=SmartFormatter)
        return parser

    def add_arguments(self) -> None:
        """Parse the arguments passed in from argparse."""
        options = (self.global_arguments + self.argument_list +
                    self.optional_arguments)
        for option in options:
            args = option['opts']
            kwargs = {key: option[key]
                    for key in option.keys() if key != 'opts'}
            self.parser.add_argument(*args, **kwargs)

class DPArgs(DeepTFBSArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-i','--input'),
            'dest': 'input',
            'default': '/home/malab21/disk2/songmg/deepTFBS/tmp.fasta',
            'type': str,
            'help': 'Path to processed data directory.'})
        argument_list.append({
            'opts': ('-win','--window_length'),
            'dest': 'window_length',
            'default': 1000,
            'type': int,
            'help': 'Window length'})
        argument_list.append({
            'opts': ('-o','--output'),
            'dest': 'output',
            'default': '/home/malab21/disk2/songmg/deepTFBS/tmp',
            'type': str,
            'help': 'Path to output directory'})
        return argument_list
    
class PreTrainArgs(DeepTFBSArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-i','--TFBSDic'),
            'dest': 'TFBSDic',
            'default': '/home/malab9/Documents/project/03_m6ABoost/12_single_exon/02_zma_features_cdhit/03_weakrm',
            'type': str,
            'required': True,
            'help': 'Path for processed data directory.'})
        argument_list.append({
            'opts': ('-tf','--TF'),
            'dest': 'TF',
            'type': str,
            'required': True,
            'help': 'TF name for processed data.'})
        argument_list.append({
            'opts': ('-eval','--eval_after_train'),
            'dest': 'eval_after_train',
            'default': True,
            'type': bool,
            'help': 'Eval after train'})
        argument_list.append({
            'opts': ('-lr','--lr_init'),
            'dest': 'lr_init',
            'default': 0.00005,
            'type': float,
            'help': 'Initial learning rate'})
        argument_list.append({
            'opts': ('-e','--epsilon'),
            'dest': 'epsilon',
            'default': 1e-8,
            'type': float,
            'help': 'Initial learning rate'})
        argument_list.append({
            'opts': ('-ep','--n_epochs'),
            'dest': 'n_epochs',
            'default': 200,
            'type': int,
            'help': 'Initial pretraining epochs'})
        argument_list.append({
            'opts': ('-bs','--batch_size'),
            'dest': 'batch_size',
            'default': 512,
            'type': int,
            'help': 'Initial batch size'})
        argument_list.append({
            'opts': ('-o','--resDic'),
            'dest': 'resDic',
            'default': '/home/malab9/Documents/project/03_m6ABoost/12_single_exon/02_zma_features_cdhit/05_checkpoint',
            'type': str,
            'required': True,
            'help': 'Path for checkpoint directory'})
        return argument_list
    
class TrainArgs(DeepTFBSArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-i','--TFBSDic'),
            'dest': 'TFBSDic',
            'default': '/home/malab9/Documents/project/03_m6ABoost/12_single_exon/02_zma_features_cdhit/03_weakrm',
            'type': str,
            'required': True,
            'help': 'Path for processed data directory.'})
        argument_list.append({
            'opts': ('-pm','--pretrain_model'),
            'dest': 'pretrain_model',
            'default': '/home/malab21/disk2/deepTFBS/1.performance.evaluation/2.models_tf2/1.raw_peak.hdf5',
            'type': str,
            'required': True,
            'help': 'name for pretrain model.'})
        argument_list.append({
            'opts': ('-tf','--TF'),
            'dest': 'TF',
            'type': str,
            'required': True,
            'help': 'TF name for processed data.'})
        argument_list.append({
            'opts': ('-eval','--eval_after_train'),
            'dest': 'eval_after_train',
            'default': True,
            'type': bool,
            'help': 'Eval after train'})
        argument_list.append({
            'opts': ('-lr','--lr_init'),
            'dest': 'lr_init',
            'default': 0.00005,
            'type': float,
            'help': 'Initial learning rate'})
        argument_list.append({
            'opts': ('-e','--epsilon'),
            'dest': 'epsilon',
            'default': 1e-8,
            'type': float,
            'help': 'Initial learning rate'})
        argument_list.append({
            'opts': ('-ep','--n_epochs'),
            'dest': 'n_epochs',
            'default': 200,
            'type': int,
            'help': 'Initial pretraining epochs'})
        argument_list.append({
            'opts': ('-bs','--batch_size'),
            'dest': 'batch_size',
            'default': 512,
            'type': int,
            'help': 'Initial batch size'})
        argument_list.append({
            'opts': ('-o','--resDic'),
            'dest': 'resDic',
            'default': '/home/malab9/Documents/project/03_m6ABoost/12_single_exon/02_zma_features_cdhit/05_checkpoint',
            'type': str,
            'required': True,
            'help': 'Path for checkpoint directory'})
        return argument_list

class PredictArgs(DeepTFBSArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-i','--input_npz'),
            'dest': 'input_npz',
            'default': '/home/malab21/disk2/deepTFBS/1.performance.evaluation/2.models_tf2/1.raw_peak.hdf5',
            'type': str,
            'required': True,
            'help': 'File path for predicting.'})
        argument_list.append({
            'opts': ('-tf','--TF'),
            'dest': 'TF',
            'type': str,
            'required': True,
            'help': 'TF name for processed data.'})
        argument_list.append({
            'opts': ('-lr','--lr_init'),
            'dest': 'lr_init',
            'default': 0.00005,
            'type': float,
            'help': 'Initial learning rate'})
        argument_list.append({
            'opts': ('-e','--epsilon'),
            'dest': 'epsilon',
            'default': 1e-8,
            'type': float,
            'help': 'Initial learning rate'})
        argument_list.append({
            'opts': ('-bs','--batch_size'),
            'dest': 'batch_size',
            'default': 512,
            'type': int,
            'help': 'Initial batch size'})
        argument_list.append({
            'opts': ('-d','--resDic'),
            'dest': 'resDic',
            'default': '/home/malab9/Documents/project/03_m6ABoost/12_single_exon/02_zma_features_cdhit/05_checkpoint',
            'type': str,
            'required': True,
            'help': 'Path for checkpoint directory'})
        argument_list.append({
            'opts': ('-o','--output'),
            'dest': 'output',
            'default': '/home/malab9/Documents/project/03_m6ABoost/12_single_exon/02_zma_features_cdhit/05_checkpoint',
            'type': str,
            'required': True,
            'help': 'Path for output filename'})
        
        return argument_list