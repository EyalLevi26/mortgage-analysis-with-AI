__all__ = ['create_table_for_bank_and_risk', 'cpi_growth','import_package']

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from payback_methods.methods import shpitzer_payment

import sys
import subprocess
from importlib.util import find_spec
from types import ModuleType

# import plotly.offline as pyo  # Import the offline mode

# Initialize offline mode
# pyo.init_notebook_mode(connected=False)

# Function to create table for a specific risk and bank
def create_table_for_bank_and_risk(risk_level, bank_name, mortgage_per_risk_per_bank, mortgage_amount_nis):
    mortgage_instance_per_risk_per_bank = mortgage_per_risk_per_bank[risk_level][bank_name] 
    loan_types = mortgage_instance_per_risk_per_bank.loan_types
    loan_type_names = [loan.loan_type_name for loan in loan_types]
    years = [loan.years for loan in loan_types]
    percentage_mortgage = [loan.partial_mortgage_amount_nis / mortgage_amount_nis for loan in loan_types]
    interest_rates = [loan.annual_interest_rate for loan in loan_types]
    
    keren_shava_list_per_loan = []
    shpizer_per_loan = []
    header_values = ["Month"]
    for i_loan, loan in enumerate(loan_types): 
        header_values.append(loan.loan_type_name)
        keren_shava_list_per_loan.append(loan.calc_keren_shava())
        shpizer_per_loan.append(loan.calc_shpizer())
    
    header_values.append("Total")
    
    max_len = max(len(sublist) for sublist in keren_shava_list_per_loan)
    padded_lists = [sublist + [0] * (max_len - len(sublist)) for sublist in keren_shava_list_per_loan]
    sum_combined_keren_shava = [sum(elements) for elements in zip(*padded_lists)]
    
    cell_values_keren_shava = [np.arange(max_len) + 1]
    for i_loan in range(len(keren_shava_list_per_loan)):
        cell_values_keren_shava.append(keren_shava_list_per_loan[i_loan])
    cell_values_keren_shava.append(sum_combined_keren_shava)

    table_keren_shava = go.Figure(data=[go.Table(
        header=dict(values= header_values,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=cell_values_keren_shava,
                   fill_color='lavender',
                   align='left'))
    ])
    table_keren_shava.update_layout(title_text=f"Kern Shava Payback per Month {bank_name} - {risk_level} Risk Level")
    
    max_len = max(len(sublist) for sublist in shpizer_per_loan)
    padded_lists = [sublist + [0] * (max_len - len(sublist)) for sublist in shpizer_per_loan]
    sum_combined_shpizer = [sum(elements) for elements in zip(*padded_lists)]
    
    cell_values_shpizer = [np.arange(max_len) + 1]
    for i_loan in range(len(shpizer_per_loan)):
        cell_values_shpizer.append(shpizer_per_loan[i_loan])
    cell_values_shpizer.append(sum_combined_shpizer)

    table_shpizer = go.Figure(data=[go.Table(
        header=dict(values= header_values,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=cell_values_shpizer,
                   fill_color='lavender',
                   align='left'))
    ])
    table_shpizer.update_layout(title_text=f"Shpizer Payback per Month {bank_name} - {risk_level} Risk Level <br> Loan Cost: \
                                 {sum(sum_combined_shpizer) - mortgage_instance_per_risk_per_bank.mortgage_amount_nis} NIS")

    # Create a table using go.Table
    table = go.Figure(data=[go.Table(
        header=dict(values=["Loan Type", "Number of Years", "Percentage of Mortgage", "Interest Rate"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[loan_type_names, years, percentage_mortgage, interest_rates],
                   fill_color='lavender',
                   align='left'))
    ])
    
    # Add title for the table
    table.update_layout(title_text=f"Loan Details for {bank_name} - {risk_level} Risk Level")
    figs = []
    for i_loan, loan in enumerate(loan_types): 
        if loan.partial_mortgage_amount_nis == 0: continue
        breakdown = loan.calculate_interest_and_principal_breakdown()
        months = [entry['Month'] for entry in breakdown]
        interest_payments = [entry['Interest Payment'] for entry in breakdown]
        principal_payments = [entry['Principal Payment'] for entry in breakdown]
        total_payments = [entry['Total Payment'] for entry in breakdown]
        principal_percentage = [entry['Principal Percentage'] for entry in breakdown]
        interest_percentage = [entry['Interest Percentage'] for entry in breakdown]
        remaining_balances = [entry['Remaining Balance'] for entry in breakdown]
        annual_cpi_index_growth_percentage = [entry['Annual CPI Index Growth Percentage'] for entry in breakdown]
        
        annual_cpi_index_growth_percentage = [round(payment, 2) for payment in annual_cpi_index_growth_percentage]
        
        # Convert data to 2 decimal points for display purposes
        interest_payments    = [round(payment, 2) for payment in interest_payments]
        principal_payments   = [round(payment, 2) for payment in principal_payments]
        remaining_balances   = [round(balance, 2) for balance in remaining_balances]
        total_payments       = [round(balance, 2) for balance in total_payments]
        principal_percentage = [round(balance, 2) for balance in principal_percentage]
        interest_percentage  = [round(balance, 2) for balance in interest_percentage]
        
        total_payment_all_loan = round(sum(total_payments), 2)
        loan_cost_nis = round(sum(total_payments) - loan.partial_mortgage_amount_nis, 2)
       
        # Define the header values for the table
        header_values = ["Month", "Interest Payment (NIS)", "Principal Payment (NIS)", "Toatl Payment (NIS)", 
                         "Principal Percentage", "Interest Percentage", "Remaining Balance (NIS)"]
        # Define the cell values for the table
        cell_values = [months, interest_payments, principal_payments, total_payments, principal_percentage, interest_percentage, remaining_balances]
        
        #####################
        fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Payment Breakdown (Shpitzer Method) - Bank: {bank_name} <br> {loan.loan_type_name} \
                        <br>Total Payment All: {total_payment_all_loan} (NIS) - Loan Cost: {loan_cost_nis} (NIS)",
                                        "Remaining Balance (NIS) vs Months", "Interest/Principal Payment (NIS) vs Months",
                                        "Interest/Principal Percentage vs Months"],
        row_heights=[0.5, 0.5],  
        vertical_spacing=0.1,  # Space between the plots
        horizontal_spacing=0.1,
        specs=[[{"type": "table"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "scatter"}]]  # First row is table, second is scatter plot
        )

        # Add the table in the first row
        fig.add_trace(go.Table(
            header=dict(values=header_values,
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=cell_values,
                       fill_color='lavender',
                       align='left'))
            , row=1, col=1
        )
        
        annual_cpi_index_info = [f"Annual CPI Index Growth: {annual_cpi_index_growth_percentage[i]:.2f} %" for i in range(len(remaining_balances))]
        # Add the plot of remaining_balances vs months in the second row
        fig.add_trace(go.Scatter(
            x=months, 
            y=remaining_balances, 
            mode='lines+markers', 
            name='Remaining Balance',
            line=dict(color='royalblue', width=2),
            text=annual_cpi_index_info,  # Custom text for additional info
            hovertemplate='Month: %{x}<br>Remaining Balance: %{y:.2f} NIS<br>%{text}<extra></extra>' 
        ), row=1, col=2)
        
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Remaining Balance (NIS)", row=1, col=2)
        fig.update_layout(showlegend=False)
        
        remaining_balance_info = [f"Remaining Balance: {remaining_balances[i]:.2f} NIS" for i in range(len(interest_payments))]
        remaining_balance_info = remaining_balance_info[1:]

        fig.add_trace(go.Scatter(
            x=months[1:], 
            y=interest_payments[1:], 
            mode='lines+markers', 
            name='Interest Payment (NIS)',
            line=dict(color='red', width=2),
            text=remaining_balance_info,  # Custom text for additional info
            hovertemplate='Month: %{x}<br>Interest Payment: %{y:.2f} NIS<br>%{text}<extra></extra>' 
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=months[1:], 
            y=principal_payments[1:], 
            mode='lines+markers', 
            name='Principal Payment (NIS)',
            line=dict(color='royalblue', width=2),
            text=remaining_balance_info,  # Custom text for additional info
            hovertemplate='Month: %{x}<br>Principal Payment: %{y:.2f} NIS<br>%{text}<extra></extra>' 
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=months[1:], 
            y=total_payments[1:], 
            mode='lines+markers', 
            name='Total Payment (NIS)',
            line=dict(color='purple', width=2),
            text=remaining_balance_info,  # Custom text for additional info
            hovertemplate='Month: %{x}<br>Total Payment: %{y:.2f} NIS<br>%{text}<extra></extra>' 
        ), row=2, col=1)

        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Interest/Principal/Total Payment (NIS)", row=2, col=1)
        fig.update_layout(showlegend=True)
        
        fig.add_trace(go.Scatter(
            x=months[1:], 
            y=interest_percentage[1:], 
            mode='lines+markers', 
            name='Interest Percentage',
            line=dict(color='red', width=2),
            text=remaining_balance_info,  # Custom text for additional info
            hovertemplate='Month: %{x}<br>Interest Percentage: %{y:.2f} %<br>%{text}<extra></extra>'  # Include custom text for remaining balance
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=months[1:], 
            y=principal_percentage[1:], 
            mode='lines+markers', 
            name='Principal Percentage',
            line=dict(color='royalblue', width=2),
            text=remaining_balance_info,  # Custom text for additional info
            hovertemplate='Month: %{x}<br>Principal Percentage: %{y:.2f} %<br>%{text}<extra></extra>'  # Include custom text for remaining balance
        ), row=2, col=2)
    
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Interest/Principal Percentage [%] ", row=2, col=2)
        fig.update_layout(showlegend=True)

        # fig.show()
        figs.append(fig)
    return table, table_shpizer, table_keren_shava, figs

def cpi_growth(years, growth_pattern_monthly_term: bool =True):
    cpi = [100]  # Starting CPI value at 100 in 2024

    years_vec = np.arange(2024, 2024 + years)
    
    growth_pattern = []
    for i_year, _ in enumerate(years_vec):
        if i_year > 15:
            growth_rate_rand = np.random.uniform(1.01, 1.02)
        else:
            growth_rate_rand = np.random.uniform(1.04, 1.06)

        growth_pattern.append(growth_rate_rand)
        if not growth_pattern_monthly_term:
            cpi.append(cpi[-1] * growth_rate_rand)

    growth_pattern_np = np.array(growth_pattern) - 1
    growth_pattern_np = np.insert(growth_pattern_np, 0, 0)
    
    if growth_pattern_monthly_term:
       growth_pattern_np_monthly = [0]
       for i_year in range(len(growth_pattern_np) - 1):
           for i_month in range(12):
               if i_year == 0 and i_month == 0: continue
               growth_pattern_np_monthly.append(growth_pattern_np[i_year+1] / 12) 
               cpi.append(cpi[-1] * (growth_pattern_np[i_year+1] / 12))
        #    if i_year == len(growth_pattern_np) - 2:
        #       for i_month in range(12):
        #           growth_pattern_np_monthly.append(growth_pattern_np[i_year] / 12)  

       growth_pattern_np_monthly = np.array(growth_pattern_np_monthly)
       return np.array(cpi), growth_pattern_np_monthly

    return np.array(cpi), growth_pattern_np

def import_package(PACKAGE_NAME: str,
                   PACKAGE_PATH: str) -> ModuleType:
    """Import package if it is not installed, otherwise return the package

    Parameters
    ----------
    PACKAGE_NAME : str
        package name
    PACKAGE_PATH : str
        package path

    Returns
    -------
    package : ModuleType
        package module

    Raises
    ------
    ImportError
        Failed to load or install package
    """
    if find_spec(PACKAGE_NAME) is not None:  # package is already installed
        loader = getattr(find_spec(PACKAGE_NAME), "loader")
        if loader is not None:
            package = loader.load_module()
        else:
            raise ImportError(f"Failed to load {PACKAGE_NAME} module")
        return package
    else:  # package is not installed
        COMMAND = [sys.executable, "-m", "pip", "install", PACKAGE_PATH]
        subprocess.check_call(COMMAND)
        if find_spec(PACKAGE_NAME) is not None:
            loader = getattr(find_spec(PACKAGE_NAME), "loader")
            if loader is not None:
                package = loader.load_module()
            else:
                raise ImportError(f"Failed to load {PACKAGE_NAME} module")
        else:
            raise ImportError(f"Failed to install {PACKAGE_NAME} module")
        return package


if __name__ == '__main__':
    # Generate years from 2024 to 2054
    years = 15
    years_vec = np.arange(2024, 2024 + years)
    # Get the CPI values
    cpi_values, growth_pattern = cpi_growth(years)

    # Plot the CPI over the years
    plt.figure(1) 
    plt.plot(years_vec, cpi_values, label="CPI Growth", marker='o')
    plt.xlabel("Year")
    plt.ylabel("CPI Value")
    plt.title("Consumer Price Index Growth in Israel (2024-2054)")
    plt.grid(True)
    plt.show(block=False)

    # Plot growth pattern directly as percentage
    plt.figure(2) 
    plt.plot(years_vec, 100 * growth_pattern, label="CPI Growth Rate [%]", marker='o')
    plt.xlabel("Year")
    plt.ylabel("Growth Rate [%]")
    plt.title("Consumer Price Index Growth Rate (2024-2054)")
    plt.grid(True)
    plt.show(block=True)

    
    partial_amount_mortgage_nis = 400000
    annual_interest_rate = 2.77
    monthly_interest_rate = (annual_interest_rate / 100) / 12
    growth_pattern = growth_pattern - 1
    for i_year in range(years):
        if i_year > 0:
            monthly_cpi_index = growth_pattern[i_year - 1]
        else:
            monthly_cpi_index = 0

        monthly_payback_shpizer = shpitzer_payment(mortgage_amount_nis=partial_amount_mortgage_nis,
                                                   years=years, 
                                                   annual_interest_rate=annual_interest_rate,
                                                   monthly_cpi_index=monthly_cpi_index)


