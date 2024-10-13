__all__ = ['keren_shava', 'shpitzer_payment']

import pandas as pd
import numpy as np

def keren_shava(mortgage_amount_nis: float, years: float, annual_interest_rate: float):
    monthly_interest_rate = (annual_interest_rate / 100) / 12
    
    total_months = int(years * 12)  # Ensure total_months is an integer
    payback_per_month = []
    remaining_loan_principal = mortgage_amount_nis
    const_payback_per_month = mortgage_amount_nis / total_months

    for i in range(total_months):
        monthly_payment = const_payback_per_month + (monthly_interest_rate * remaining_loan_principal)
        payback_per_month.append(monthly_payment)
        remaining_loan_principal -= const_payback_per_month

    return payback_per_month

def shpitzer_payment(mortgage_amount_nis: float, years: float, annual_interest_rate: float, monthly_cpi_index: float = 0):
    """
    Calculate the monthly payment for a mortgage using the Shpitzer method.

    Parameters:
    mortgage_amount_nis (float): Total mortgage amount in NIS.
    years (int): Number of years for the mortgage.
    annual_interest_rate (float): Annual interest rate as a percentage (e.g., 3.5 for 3.5%).

    Returns:
    float: Monthly payment amount in NIS.
    """
    # Convert annual interest rate to monthly interest rate (in decimal form)
    monthly_interest_rate = (annual_interest_rate / 100) / 12
    
    # Total number of months
    total_months = years * 12
    
    # Apply the annuity formula to calculate the monthly payment
    if monthly_interest_rate > 0:
        monthly_payment = ((1 + monthly_cpi_index) * mortgage_amount_nis * monthly_interest_rate) / \
                          (1 - (1 + monthly_interest_rate) ** -total_months)
    else:
        # If the interest rate is 0, divide the mortgage amount by total months
        monthly_payment = mortgage_amount_nis / total_months
    
    return monthly_payment
