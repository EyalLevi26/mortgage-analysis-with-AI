__all__ = ['main_user']

from pathlib import Path
from format_obj.classes_utills import DataLoader, SingleLoanType, Mortgage

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from utills import create_table_for_bank_and_risk
from typing import Union

# import pickle
# import os
# import tabloo
# import itertools


def main_user(banks_info_path: Path, mortgage_amount_nis: float = 100000, years: float = 15) ->  Union[pd.DataFrame, None]:
    try:
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
                if risk_level == 'Our_Mortgage':
                    if len(mix_loan_types) == 3:
                        # [const_intrest_not_index_linked, change_intrest_not_index_linked, change_intrest_not_index_linked_prime]
                        num_years_per_loan_type = [15, 15, 15]
                        loan_types_weights      = [33.3333, 50, 16.6667]
                    elif len(mix_loan_types) == 4:
                         # [const_intrest_not_index_linked, change_intrest_not_index_linked, change_intrest_not_index_linked_prime, const_intrest_index_linked]
                         num_years_per_loan_type = [5, 15, 15, 5]
                        #  loan_types_weights      = [33.3333, 1, 16.6667, 49]
                        # loan_types_weights      = [33.3333, 25, 16.6667, 25]
                         loan_types_weights      = [0, 0, 0, 100]
                    else:
                        num_years_per_loan_type = None
                        loan_types_weights = None
                else:
                    num_years_per_loan_type = None
                    loan_types_weights = None

                mortgage_instance = Mortgage(mortgage_amount_nis=mortgage_amount_nis, loan_types=loan_types_per_risk_per_bank[risk_level][bank_name],
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

        # fig.show()
        # 
        # fig = go.Figure(data=traces_payback, layout=layout_payback)
        # fig.show()
        # 
        # fig = go.Figure(data=traces_payback_rate, layout=layout_payback_rate)
        # fig.show()
        
        loan_table, table_shpizer, table_keren_shava = create_table_for_bank_and_risk(risk_level='Our_Mortgage', 
                                                    bank_name ='Benleomi', 
                                                    mortgage_per_risk_per_bank = mortgage_per_risk_per_bank,
                                                    mortgage_amount_nis = mortgage_amount_nis)
        loan_table.show()
        table_shpizer.show()
        table_keren_shava.show()

        return None

    except KeyboardInterrupt:
        print("Exiting...")
        return None


if __name__ == '__main__':
    path_file_banks_info = Path(r"C:\Users\DELL\Documents\mortgage\MortgageAnalysis\mortgage_israel_bank_info.xlsx")
    results = main_user(banks_info_path=path_file_banks_info)
    a=1


