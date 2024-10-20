__all__ = ['CLIArgumentParser']

import datetime
import textwrap
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from dataclasses import dataclass
from pathlib import Path
from typing import  Sequence, Union
import ast

def str_to_list(input_str: str):
    """
    Convert a string representation of a list into an actual list of numbers.
    Example: '[0,5,8,9,80]' -> [0, 5, 8, 9, 80]
    """
    # Use `ast.literal_eval` to safely evaluate the string into a Python list
    return ast.literal_eval(input_str)

@dataclass
class CLIArgumentParser:
    def __init__(self, argv: Union[Sequence[str] , None] = None) -> None:
        """
            A class to parse and store command-line arguments for mortgage analysis.

            Attributes
            ----------
            banks_info_path : Path
                Path to the file containing information about banks.
            mortgage_amount_nis : str
                Mortgage amount in NIS (must be a float number greater than zero).
            years : str
                Duration of the mortgage in years (float number).
            bank_name : str
                Name of the bank (e.g., Mizrachi, Leumi, Hapoalim, etc.).
            loan_types_weights : list
                Weights for the five types of loans (must be a list of five float numbers).
            num_years_per_loan_type : list
                Number of years corresponding to each loan type (must be a list of five float numbers).
            save_file_path : Path
                Path where the output file will be saved.
            save_file_name : str
                Name of the output file.
            save_file_fmt : str
                Format of the output file (e.g., 'parquet', 'csv', 'yaml').
            plot_results : bool
                Flag to indicate whether to plot the results.
            n_jobs : int
                Number of parallel jobs for processing (-1 to use all available cores).

            Methods
            -------
            _build_parser() -> ArgumentParser
                Builds and returns the argument parser.

            Raises
            ------
            FileNotFoundError
                If the specified banks information path or save path does not exist.
        """
        self._argv = argv
        self._parser = self._build_parser()
        self._args: Namespace = self._parser.parse_args(self._argv)

        self.banks_info_path = Path(self._args.banks_info_path)
        self.mortgage_amount_nis = self._args.mortgage_amount_nis
        self.years: str = self._args.years
        self.bank_name: str = self._args.bank_name
        self.loan_types_weights: str = self._args.loan_types_weights
        self.num_years_per_loan_type: str = self._args.num_years_per_loan_type
        self.save_file_path = Path(self._args.save_file_path)
        self.save_file_name: str = self.banks_info_path.name  # self._args.save_file_name
        self.save_file_fmt: str = self._args.save_file_fmt
        self.plot_results: bool = self._args.plot_results
        self.n_jobs: int = self._args.n_jobs

        if not self.banks_info_path.exists():
            raise FileNotFoundError("Specified recording path does not exist")
        if not self.save_file_path.exists():
            try:
                self.save_file_path.mkdir(parents=True)
            except:
                raise FileNotFoundError("Specified save path does not exist")

    def _build_parser(self) -> ArgumentParser:
        parser = ArgumentParser(
            prog='mortgage_analysis_parser',
            description=textwrap.dedent(f"""
                MORTGAGE ANALYSIS
                ===========
                mortgage analysis simulation, convert to dataframes, and manipulate data
                """),
            formatter_class=RawTextHelpFormatter)

        parser.add_argument(
           'banks_info_path',
            help="banks information path")
        parser.add_argument(
            '-m', '--mortgage_amount_nis',
            help="mortgage amount in nis (float number greater than zero)",
            type=str, default='')
        parser.add_argument('-y', '--years',
                            help='Duration of all mortgages in years (float number greater than zero)',
                            type=str, default='')
        parser.add_argument('-b', '--bank_name',
                            help='name of the bank - strings option: Mizrachi, Leomi, Hapoalim, Diskont, Benleomi, Marcantil, Jerusalem, Others',
                            type=str, default='')
        parser.add_argument('-w', '--loan_types_weights',
                            help='weigth the loan types  - 5 float numbers (5 types: 1: const intrest not index linked, 2: change intrest not index linked prime, \
                                3: change intrest not index linked, 4: change intrest index linked, 5: const intrest index linked)',
                            type=str_to_list, required=True)
        parser.add_argument('-yl', '--num_years_per_loan_type',
                            help='number of years per loan types  - 5 float numbers (5 types: 1: years for const intrest not index linked, 2: years for change intrest not index linked prime, \
                                3: years for change intrest not index linked, 4: years for change intrest index linked, 5: years for const intrest index linked)',
                            type=str_to_list, required=True)
        parser.add_argument('-p', '--save_file_path', type=str, metavar='PATH',
                            help='path to the output file, default: %(default)s',
                            default=f"{(Path.cwd() / 'data').as_posix()}")
        parser.add_argument('-n', '--save_file_name', type=str, metavar='FILENAME',
                            help='name of the output file, default: %(default)s',
                            default=f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        parser.add_argument('-f', '--save_file_fmt', type=str, metavar='FORMAT',
                            help='file format of the output file, choices: {%(choices)s}, default: %(default)s',
                            choices=('parquet', 'csv', 'yaml'),
                            default='parquet')
        parser.add_argument('-j', '--n_jobs', type=int,
                            help='# of parallel jobs (-1 to use all available cores for multi-processing) default: %(default)s',
                            default=-1)
        parser.add_argument('-plot', '--plot_results',
                            action='store_true',
                            help='plot results')
        return parser
