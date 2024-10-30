__all__ = ['main_user', 'main_cli']

from pathlib import Path
from mortgage_toolkit.mortgage_calculator import DataLoader, SingleLoanType, Mortgage
import warnings
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from utills import create_table_for_bank_and_risk, import_package
from typing import Union, Sequence, List, Optional
from my_argparser import ArgumentParserCLI
from plot_utils import plot_tables_and_figures
import matplotlib.pyplot as plt

# import pickle
# import os
# import tabloo
# import itertools

# blog_parser = import_package(PACKAGE_NAME="blog_parser",
                                #    PACKAGE_PATH=".venv\\Lib\\site-packages\\blog_parser")

def main_user(banks_info_path: Path, mortgage_amount_nis: float = 100000, years: float = 15, loan_types_weights: Union[List[float] , None] = None, 
              num_years_per_loan_type: Union[List[float] , None] = None, bank_name_for_plot: str = 'Benleomi', 
              extenal_plot: bool = False) ->  Optional[List[Union[go.Figure, List[go.Figure]]]]:
    try:
        figs_tmp = []
        traces_payback = []
        traces_payback_rate = []
        bank_info_instance = DataLoader(path=banks_info_path)
        loan_types_per_risk_per_bank = {}
        mortgage_per_risk_per_bank = {}
        for risk_level, mix_loan_types in bank_info_instance.combination_dict.items(): 
            bank_names_list = []
            total_payback_shpizer_per_bank = []
            total_payback_keren_shava_per_bank = []
            total_payback_rate_keren_shava_per_bank = []
            total_payback_rate_shpizer_per_bank     = []

            for bank_name in bank_info_instance.unique_bank_names: 
                bank_names_list.append(bank_name)
                if risk_level not in loan_types_per_risk_per_bank:
                   loan_types_per_risk_per_bank[risk_level] = {}

                loan_types_per_risk_per_bank[risk_level][bank_name] = SingleLoanType.create_new_instances(partial_mortgage_amount_nis= mortgage_amount_nis / len(mix_loan_types),
                                                                                                          years= years, 
                                                                                                          banks_info_data=bank_info_instance,
                                                                                                          bank_name= bank_name,
                                                                                                          mix_loan_types= mix_loan_types)

                mortgage_instance = Mortgage(mortgage_amount_nis=mortgage_amount_nis, 
                                             loan_types=loan_types_per_risk_per_bank[risk_level][bank_name],
                                             years=years, 
                                             num_years_per_loan_type=num_years_per_loan_type, 
                                             loan_types_weights=loan_types_weights)
                
                if risk_level not in mortgage_per_risk_per_bank:
                   mortgage_per_risk_per_bank[risk_level] = {}
                
                mortgage_per_risk_per_bank[risk_level][bank_name] = mortgage_instance

                total_payback_dict = mortgage_instance.calc_total_payback_for_all_mortgage()
                total_payback_shpizer_per_bank.append(total_payback_dict["shpizer"])
                total_payback_keren_shava_per_bank.append(total_payback_dict["keren_shava"])
                
                total_payback_rate_dict = mortgage_instance.calc_total_payback_rate_for_all_mortgage()
                total_payback_rate_shpizer_per_bank.append(total_payback_rate_dict["shpizer"])
                total_payback_rate_keren_shava_per_bank.append(total_payback_rate_dict["keren_shava"])
                
                # loan_table = create_table_for_bank_and_risk(risk_level=risk_level, 
                                                    # bank_name=bank_name, 
                                                    # loan_types_per_risk_per_bank=loan_types_per_risk_per_bank,
                                                    # mortgage_amount_nis= mortgage_amount_nis)
                # loan_table.show()
            trace = go.Scatter(x=bank_names_list,    
                                y=total_payback_shpizer_per_bank,         
                                mode='lines+markers',    
                                name=f'Payback in NIS - {risk_level}- Shpizer Method'  # Ensure unique trace names
                               )
            traces_payback.append(trace)
            trace = go.Scatter(x=bank_names_list,    
                                y=total_payback_keren_shava_per_bank,         
                                mode='lines+markers',    
                                name=f'Payback in NIS - {risk_level} - Keren-Shava Method'  # Ensure unique trace names
                               )
            traces_payback.append(trace)
            trace = go.Scatter(x=bank_names_list,    
                                y=total_payback_rate_shpizer_per_bank,         
                                mode='lines+markers',    
                                name=f'Payback Rate - {risk_level}- Shpizer Method'  # Ensure unique trace names
                               )
            traces_payback_rate.append(trace)
            trace = go.Scatter(x=bank_names_list,    
                                y=total_payback_rate_keren_shava_per_bank,         
                                mode='lines+markers',    
                                name=f'Payback Rate - {risk_level} - Keren-Shava Method'  # Ensure unique trace names
                               )
            traces_payback_rate.append(trace)
                
            [loan_table, table_shpizer, table_keren_shava, figs] = plot_tables_and_figures(traces_payback, 
                                                                                       traces_payback_rate, 
                                                                                       extenal_plot, 
                                                                                       bank_name_for_plot, 
                                                                                       mortgage_per_risk_per_bank, 
                                                                                       mortgage_amount_nis)
        return [loan_table, table_shpizer, table_keren_shava, figs]

    except KeyboardInterrupt:
        print("Exiting...")
        return None

def main_cli(argv: Union[Sequence[str] , None] = None) -> Optional[List[Union[go.Figure, List[go.Figure]]]]:
    """Command Line Interface for mortgage analysis. This function processes the mortgage data, 
    calculates payback amounts and rates for different loan types and risk levels, and generates 
    plots to visualize the results.


    Parameters
    ----------
    argv :  Sequence[str] | None, optional
        Command-line arguments passed to the function, by default None. If not provided, the default
        command-line arguments will be used.

    Command Line Arguments
    ----------------------
     banks_info_path : Path
        The file path containing information about banks, including combinations of loan types 
        and risk levels.
    mortgage_amount_nis : float
        Total mortgage amount in NIS.
    years : str
        Duration of the mortgage in years.
    bank_name : str
        Name of the bank, chosen from options like Mizrachi, Leumi, Hapoalim, Discount, etc.
    loan_types_weights : list[float]
        Weights for each of the five loan types. The weights should represent 5 float values 
        corresponding to types such as constant interest, variable interest, index-linked, etc.
    num_years_per_loan_type : list[float]
        Number of years allocated to each of the five loan types.
    save_file_path : Path
        Directory path where the output file will be saved.
    save_file_name : str
        Name of the output file.
    save_file_fmt : str
        Format of the output file. Options include 'parquet', 'csv', and 'yaml'.
    plot_results : bool
        Whether to generate plots for the payback results.
    n_jobs : int
        Number of parallel jobs to use. If -1, all available CPUs are used.
        
    Returns
    -------
    results : pd.DataFrame | None
        A DataFrame containing the parsed mortgage data or None if an error occurs.

    Raises
    ------
    FileNotFoundError
        Raised if the provided banks information file or the save path does not exist.

    Notes
    -----
    The function processes bank and loan type information for different risk levels and performs 
    the following tasks:
    - Calculates total payback amounts using different loan repayment methods (Shpizer, Keren-Shava).
    - Calculates the payback rates for each method.
    - Generates visualizations for the payback amounts and rates across banks and risk levels.
    - Displays summary tables for specific mortgage types and repayment methods.

    """
    try:
        args = ArgumentParserCLI(argv)
        traces_payback = []
        traces_payback_rate = []
        bank_info_instance = DataLoader(path=args.banks_info_path)
        loan_types_per_risk_per_bank = {}
        mortgage_per_risk_per_bank = {}
        for risk_level, mix_loan_types in bank_info_instance.combination_dict.items(): 
            bank_names_list = []
            total_payback_shpizer_per_bank = []
            total_payback_keren_shava_per_bank = []
            total_payback_rate_keren_shava_per_bank = []
            total_payback_rate_shpizer_per_bank     = []

            for bank_name in bank_info_instance.unique_bank_names: 
                bank_names_list.append(bank_name)
                if risk_level not in loan_types_per_risk_per_bank:
                   loan_types_per_risk_per_bank[risk_level] = {}

                loan_types_per_risk_per_bank[risk_level][bank_name] = SingleLoanType.create_new_instances(partial_mortgage_amount_nis= float(args.mortgage_amount_nis) / len(mix_loan_types),
                                                                                                          years= float(args.years), 
                                                                                                          banks_info_data=bank_info_instance,
                                                                                                           bank_name= bank_name,
                                                                                                           mix_loan_types= mix_loan_types)
                
                num_years_per_loan_type = args.num_years_per_loan_type
                loan_types_weights      = args.loan_types_weights

                mortgage_instance = Mortgage(mortgage_amount_nis=float(args.mortgage_amount_nis), loan_types=loan_types_per_risk_per_bank[risk_level][bank_name],
                                              years=float(args.years), 
                                              num_years_per_loan_type=num_years_per_loan_type, 
                                              loan_types_weights=loan_types_weights)
                
                if risk_level not in mortgage_per_risk_per_bank:
                   mortgage_per_risk_per_bank[risk_level] = {}
                
                mortgage_per_risk_per_bank[risk_level][bank_name] = mortgage_instance

                total_payback_dict = mortgage_instance.calc_total_payback_for_all_mortgage()
                total_payback_shpizer_per_bank.append(total_payback_dict["shpizer"])
                total_payback_keren_shava_per_bank.append(total_payback_dict["keren_shava"])
                
                total_payback_rate_dict = mortgage_instance.calc_total_payback_rate_for_all_mortgage()
                total_payback_rate_shpizer_per_bank.append(total_payback_rate_dict["shpizer"])
                total_payback_rate_keren_shava_per_bank.append(total_payback_rate_dict["keren_shava"])
                
                # loan_table = create_table_for_bank_and_risk(risk_level=risk_level, 
                                                    # bank_name=bank_name, 
                                                    # loan_types_per_risk_per_bank=loan_types_per_risk_per_bank,
                                                    # mortgage_amount_nis= mortgage_amount_nis)
                # loan_table.show()

            trace = go.Scatter(x=bank_names_list,    
                                y=total_payback_shpizer_per_bank,         
                                mode='lines+markers',    
                                name=f'Payback in NIS - {risk_level}- Shpizer Method'  # Ensure unique trace names
                               ) 

            traces_payback.append(trace)

            trace = go.Scatter(x=bank_names_list,    
                                y=total_payback_keren_shava_per_bank,         
                                mode='lines+markers',    
                                name=f'Payback in NIS - {risk_level} - Keren-Shava Method'  # Ensure unique trace names
                               )
            traces_payback.append(trace)
            
            trace = go.Scatter(x=bank_names_list,    
                                y=total_payback_rate_shpizer_per_bank,         
                                mode='lines+markers',    
                                name=f'Payback Rate - {risk_level}- Shpizer Method'  # Ensure unique trace names
                               ) 

            traces_payback_rate.append(trace)

            trace = go.Scatter(x=bank_names_list,    
                                y=total_payback_rate_keren_shava_per_bank,         
                                mode='lines+markers',    
                                name=f'Payback Rate - {risk_level} - Keren-Shava Method'  # Ensure unique trace names
                               )
            traces_payback_rate.append(trace)

        layout_payback = go.Layout(
           title="Shpizer/Keren-Shava Payback vs Bank",
           xaxis_title="Bank",
           yaxis_title="Payback Cash [NIS]",
           hovermode='closest',
           showlegend=True,  # Explicitly show the legend
           legend=dict(
           orientation='v',  # Vertical orientation
           x=0,  # Position on the left
           xanchor='left',  # Anchor to the left
           y=1,  # Position at the top
           yanchor='top'  # Anchor to the top
           )
        )
        
        layout_payback_rate = go.Layout(
           title="Shpizer/Keren-Shava Payback Rate vs Bank",
           xaxis_title="Bank",
           yaxis_title="Payback Rate",
           hovermode='closest',
           showlegend=True,  # Explicitly show the legend
           legend=dict(
           orientation='v',  # Vertical orientation
           x=0,  # Position on the left
           xanchor='left',  # Anchor to the left
           y=1,  # Position at the top
           yanchor='top'  # Anchor to the top
           )
        )
        
        fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["Shpizer/Keren-Shava Payback Cash [NIS] vs Bank",
                                        "Shpizer/Keren-Shava Payback Rate vs Bank"],
                        horizontal_spacing=0.05,
                        vertical_spacing=0.1,
                        shared_xaxes="all")  # type: ignore
        
        for trace in traces_payback:
            fig.add_trace(trace, row=1, col=1)

        for trace in traces_payback_rate:
            fig.add_trace(trace, row=2, col=1)

        fig.update_xaxes(title_text="Bank", row=1, col=1)
        fig.update_xaxes(title_text="Bank", row=2, col=1)
        fig.update_yaxes(title_text="Payback [nis]", row=1, col=1)
        fig.update_yaxes(title_text="Payback Rate", row=2, col=1)  
        
        if args.plot_results:
            fig.show()

            fig = go.Figure(data=traces_payback, layout=layout_payback)
            fig.show()

            fig = go.Figure(data=traces_payback_rate, layout=layout_payback_rate)
            fig.show()
        
        loan_table, table_shpizer, table_keren_shava, figs = create_table_for_bank_and_risk(risk_level='Our_Mortgage', 
                                                    bank_name =args.bank_name, 
                                                    mortgage_per_risk_per_bank = mortgage_per_risk_per_bank,
                                                    mortgage_amount_nis = float(args.mortgage_amount_nis))
        if args.plot_results:
            loan_table.show()
            for fig in figs:
                fig.show()

            table_shpizer.show()
            table_keren_shava.show()
        

        return [loan_table, table_shpizer, table_keren_shava, figs]
    
    except KeyboardInterrupt:
        print("Exiting...")
        return None

if __name__ == '__main__':
    if len(sys.argv) == 1:
        path_file_banks_info = Path(r"mortgage_israel_bank_info.xlsx")
        warnings.simplefilter('always', RuntimeWarning)
        loan_types_weights = [0, 0, 0, 0, 100] 
        num_years_per_loan_type = [5, 15, 15, 5, 15]
        results = main_user(banks_info_path=path_file_banks_info, 
                            mortgage_amount_nis = 1200000, 
                            years = 20,
                            loan_types_weights=loan_types_weights,
                            extenal_plot = True,
                            num_years_per_loan_type = num_years_per_loan_type)
                
        # results = main_cli(
        #         [r"mortgage_israel_bank_info.xlsx"] +
        #         "-m 1200000 -y 15 -b Benleomi -w [0,0,0,0,100] -yl [10,10,10,10,15] -plot".split())
    else:
        main_cli()
    


# dist\mortgage_analysis_cli.exe mortgage_israel_bank_info.xlsx -m 1200000 -y 15 -b Benleomi -w "[0,0,0,0,100]" -yl "[10,10,10,10,15]" -plot
# pyinstaller --onefile --name mortgage_analysis_cli --hidden-import=pmdarima --hidden-import=tqdm --hidden-import=pickle --hidden-import=typing --add-data "mortgage_toolkit;mortgage_toolkit" --add-data "AI;AI" --add-data "payback_methods;payback_methods" --add-data "plot_utils.py;." --add-data "utills.py;." --add-data "my_argparser.py;." main.py
