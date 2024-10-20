__all__ = ['DataLoader', 'SingleLoanType', 'Mortgage']

from payback_methods.methods import keren_shava, shpitzer_payment
import pandas as pd
import numpy as np
from pathlib import Path
from typing import (Callable, Dict, List, Literal, Optional, Tuple, 
                    Union)
import math
from utills import cpi_growth

class DataLoader:
    def __init__(self, path: Path) -> None:
        self.path_to_bank_info = path
        self.data_bank_info: Union[pd.DataFrame , None] = None
        self.unique_bank_names = None
        self.unique_loan_type  =  None
        self.not_index_linked_types = None

        self.data_bank_info = self._load_data_frame()
        if not self.data_bank_info.empty:
           self.unique_bank_names = self.data_bank_info["Bank"].unique()
           self.unique_loan_type  = self.data_bank_info["loan_type"].unique()
           self.not_index_linked_types = self.unique_loan_type[np.char.find(self.unique_loan_type.astype(str), 'not index-linked') != -1]
           self.very_low_risk_type     = self.unique_loan_type[np.char.find(self.unique_loan_type.astype(str), 'Constant interest rate and not index-linked') != -1]
           self.all_types              = self.unique_loan_type[:-1]
        #    self.combination_dict       = {"const_intrest_not_index_linked": self.very_low_risk_type, "middle_risk": self.not_index_linked_types, "all": self.all_types,
        #                                   "change_intrest_not_index_linked_prime": [self.all_types[1]], "change_intrest_not_index_linked": [self.all_types[2]],
        #                                   "change_intrest_index_linked": [self.all_types[3]], "const_intrest_index_linked": [self.all_types[4]]}
        #    self.combination_dict       = {"const_intrest_not_index_linked": self.very_low_risk_type, "change_intrest_not_index_linked_prime": [self.all_types[1]], "change_intrest_not_index_linked": [self.all_types[2]],
                                        #   "change_intrest_index_linked": [self.all_types[3]], "const_intrest_index_linked": [self.all_types[4]]}
           self.combination_dict       = {"Our_Mortgage": [self.all_types[0], self.all_types[2], self.all_types[1], self.all_types[4]]}

    def _load_data_frame(self) -> Union[pd.DataFrame, None]:
        try:
            banks_info = pd.read_excel(self.path_to_bank_info)
            return banks_info
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        

class SingleLoanType:
    def __init__(self, partial_mortgage_amount_nis: float, years: float, annual_interest_rate: float, loan_type_name: str) -> None:
        self.partial_mortgage_amount_nis = partial_mortgage_amount_nis
        self.years = years
        self.annual_interest_rate = annual_interest_rate
        self.monthly_interest_rate = (annual_interest_rate / 100) / 12
        self.total_months = int(years * 12)  # Ensure total_months is an integer
        self.loan_type_name = loan_type_name    
        self._calc_keren_shava = keren_shava
        self._calc_shpizer = shpitzer_payment
        self.is_index_linked = self._check_if_index_linked_type()
    
        _, growth_pattern = cpi_growth(years=self.years)
        if self.is_index_linked:
            self.monthly_cpi_index_est = growth_pattern
        else:
           self.monthly_cpi_index_est =  np.zeros(growth_pattern.shape)

    def _check_if_index_linked_type(self) -> bool:
        return "index-linked" in self.loan_type_name and "not index-linked" not in self.loan_type_name

    def update_partial_mortgage_amount_nis(self, partial_mortgage_amount_nis: float) -> None:
        self.partial_mortgage_amount_nis = partial_mortgage_amount_nis

    def update_years_or_months(self, years: Union[float, None] = None, total_months: Union[float, None]=None) -> None:
        if years:
           self.years = years 
           self.total_months = years * 12
        if total_months:
           self.total_months =  total_months
           self.years = total_months / 12
        
        _, growth_pattern = cpi_growth(years=self.years)
        if self.is_index_linked:
            self.monthly_cpi_index_est = growth_pattern
        else:
           self.monthly_cpi_index_est =  np.zeros(growth_pattern.shape)
           
    def calc_shpizer(self):
        monthly_payment_list = []
        remaining_balance = self.partial_mortgage_amount_nis
        for i_month, monthly_cpi_index_est in enumerate(self.monthly_cpi_index_est):
            monthly_payment_num = self._calc_shpizer(
                mortgage_amount_nis=remaining_balance, 
                years= self.years - i_month / 12, 
                annual_interest_rate= self.annual_interest_rate,
                monthly_cpi_index=monthly_cpi_index_est
            )
            monthly_payment_list.append(monthly_payment_num)

            if self.is_index_linked and self.monthly_cpi_index_est[i_month] > 0:
                remaining_balance = remaining_balance + remaining_balance * self.monthly_cpi_index_est[i_month]

            # Interest payment for the month
            interest_payment = remaining_balance * self.monthly_interest_rate

            # Principal payment for the month
            principal_payment = monthly_payment_num - interest_payment

            # Update the remaining balance
            remaining_balance -= principal_payment
        return monthly_payment_list
    
    def calc_keren_shava(self):
        monthly_payment_list = self._calc_keren_shava(
            mortgage_amount_nis=self.partial_mortgage_amount_nis, 
            years=self.years, 
            annual_interest_rate=self.annual_interest_rate
        )
        return monthly_payment_list
    
    def calc_total_payback(self):
        # Call the methods correctly using self
        monthly_payment_list_shpizer = self.calc_shpizer()
        monthly_payment_list_keren_shava = self.calc_keren_shava()
        
        return {
            "shpizer": sum(monthly_payment_list_shpizer), 
            "keren_shava": sum(monthly_payment_list_keren_shava)
        }
    
    def calc_payback_rate(self):
        # Call the methods correctly using self
        monthly_payment_list = self.calc_shpizer()
        monthly_payment_list = self.calc_keren_shava()
        
        return {
            "shpizer": sum(monthly_payment_list) / self.partial_mortgage_amount_nis, 
            "keren_shava": sum(monthly_payment_list) / self.partial_mortgage_amount_nis
        }
    
    def calculate_interest_and_principal_breakdown(self):
        """
        Calculate the interest and principal breakdown for each month in a Shpitzer mortgage.

        Parameters:
        self.partial_mortgage_amount_nis (float): Total mortgage amount in NIS.
        self.years (int): Number of years for the mortgage.
        self.annual_interest_rate (float): Annual interest rate as a percentage (e.g., 3.5 for 3.5%).

        Returns:
        List[dict]: A list of dictionaries containing the interest, principal, and remaining balance per month.
        """           
        # Get the fixed monthly payment
        monthly_payment_list = self.calc_shpizer()

        # Convert annual interest rate to monthly interest rate (in decimal form)
        monthly_interest_rate = self.monthly_interest_rate

        # Initialize variables for the breakdown
        remaining_balance = self.partial_mortgage_amount_nis
        breakdown = []
        
        breakdown.append({
                'Month': 0,
                'Interest Payment': 0,
                'Principal Payment': 0,
                'Total Payment': 0,
                'Principal Percentage': 0,
                'Interest Percentage': 0,
                'Annual CPI Index Growth Percentage': 0,
                'Remaining Balance': remaining_balance 
                })
        
        # Calculate the breakdown for each month
        for month in range(1, self.total_months + 1):
            if self.is_index_linked and self.monthly_cpi_index_est[month-1] > 0:
                remaining_balance = remaining_balance + remaining_balance * self.monthly_cpi_index_est[month-1]

            # Interest payment for the month
            interest_payment = remaining_balance * monthly_interest_rate

            # Principal payment for the month
            principal_payment = monthly_payment_list[month-1] - interest_payment

            # Update the remaining balance
            remaining_balance -= principal_payment

            # Store the breakdown
            breakdown.append({
                'Month': month,
                'Interest Payment': interest_payment,
                'Principal Payment': principal_payment,
                'Total Payment': monthly_payment_list[month-1],
                'Principal Percentage': (principal_payment / monthly_payment_list[month-1]) * 100,
                'Interest Percentage': (interest_payment / monthly_payment_list[month-1]) * 100,
                'Annual CPI Index Growth Percentage': self.monthly_cpi_index_est[month-1] * 12 * 100,
                'Remaining Balance': remaining_balance if remaining_balance > 0 else 0  # Avoid negative balance at the end
            })

        return breakdown

    @classmethod
    def create_new_instances(cls, partial_mortgage_amount_nis: float, years: float, banks_info_data: pd.DataFrame, bank_name: str, mix_loan_types: List[str]):
        # Ensure `banks_info_data` is a DataFrame, not DataLoader unless you handle DataLoader properly.
        bank_data = banks_info_data.data_bank_info.loc[banks_info_data.data_bank_info['Bank'] == bank_name]

        loan_types = []
        for single_loan_type in mix_loan_types:
            # Get interest rate; ensure there are rows that match the loan type.
            filtered_data = bank_data.loc[bank_data["loan_type"] == single_loan_type]

            if not filtered_data.empty:
                interest_rate = filtered_data["Interest Rate"].values[0]
                loan_types.append(cls(
                    partial_mortgage_amount_nis=partial_mortgage_amount_nis, 
                    years=years, 
                    annual_interest_rate=interest_rate, 
                    loan_type_name=single_loan_type
                ))
            else:
                raise ValueError(f"No data found for loan type {single_loan_type} at {bank_name}")
        
        return loan_types
    

class Mortgage():
    def __init__(self, 
                 mortgage_amount_nis: float, 
                 years: float = 15, 
                 loan_types: Union[List['SingleLoanType'] , None] = None, 
                 num_years_per_loan_type: Union[List[float] , None] = None,
                 loan_types_weights: Union[List[float] , None] = None
                 ) -> None:
        
        self.mortgage_amount_nis = mortgage_amount_nis
        self.max_num_of_years_all_types = years
        self.loan_types = loan_types or []
        self.num_of_types = len(self.loan_types)
        self.loan_types_weights = loan_types_weights
        self.num_years_per_loan_type = num_years_per_loan_type       
        self._initialize(loan_types=self.loan_types, 
                         num_years_per_loan_type=self.num_years_per_loan_type, 
                         loan_types_weights=self.loan_types_weights)
    
    def _initialize(self, 
                    loan_types: Union[List['SingleLoanType'] , None] = None, 
                    num_years_per_loan_type: Union[List[float] , None] = None, 
                    loan_types_weights: Union[List[float] , None] = None):
        loan_types = loan_types or []
        self.num_of_types = len(loan_types)
        
        if loan_types_weights:
            if math.isclose(sum(loan_types_weights), 1) and len(loan_types_weights) == self.num_of_types:
                self.loan_types_weights = loan_types_weights
            elif math.isclose(sum(loan_types_weights), 100) and len(loan_types_weights) == self.num_of_types:
                self.loan_types_weights = [w / 100 for w in loan_types_weights]   
            else:
                self.loan_types_weights = [1 / self.num_of_types for _ in range(self.num_of_types)]    
        elif self.num_of_types > 0:
            self.loan_types_weights = [1 / self.num_of_types for _ in range(self.num_of_types)] 
        else: 
            self.loan_types_weights = []
        
        if not num_years_per_loan_type:
            if self.num_of_types > 0:
                self.num_years_per_loan_type = [self.max_num_of_years_all_types for _ in range(self.num_of_types)]
            else:
                self.num_years_per_loan_type = []
        else:
            self.num_years_per_loan_type = num_years_per_loan_type
        
        if self.num_years_per_loan_type:
            for i_loan, years in enumerate(self.num_years_per_loan_type): 
                self.loan_types[i_loan].update_years_or_months(years=years)
        
        if self.loan_types_weights:
           for i_loan, weight in enumerate(self.loan_types_weights): 
               self.loan_types[i_loan].update_partial_mortgage_amount_nis(partial_mortgage_amount_nis = weight * self.mortgage_amount_nis)

    def add_loan_type(self, 
                      partial_mortgage_amount_nis: float, 
                      years: float, 
                      annual_interest_rate: float, 
                      loan_type_name: str,
                      num_years_per_loan_type: Union[float , None] = None):
         loan_type2add = SingleLoanType(
             partial_mortgage_amount_nis=partial_mortgage_amount_nis, 
             years=years, 
             annual_interest_rate=annual_interest_rate, 
             loan_type_name=loan_type_name
         )
         self.loan_types.append(loan_type2add)
         self._initialize(
             loan_types=self.loan_types, 
             num_years_per_loan_type=self.num_years_per_loan_type, 
             loan_types_weights=None
         )
    
    def calc_total_payback_for_all_mortgage(self):
        total_payback = {"shpizer": 0, "keren_shava": 0}
        for single_loan_type in self.loan_types:
            total_payback_per_type_dict = single_loan_type.calc_total_payback()
            total_payback["shpizer"] += total_payback_per_type_dict["shpizer"]
            total_payback["keren_shava"] += total_payback_per_type_dict["keren_shava"]

        return total_payback
        
    def calc_total_payback_rate_for_all_mortgage(self):
        total_payback = self.calc_total_payback_for_all_mortgage()
        payback_rate = {
            "shpizer": total_payback["shpizer"] / self.mortgage_amount_nis, 
            "keren_shava": total_payback["keren_shava"] / self.mortgage_amount_nis
        }
        return payback_rate
    
# class Mortgage():
#     def __init__(self, 
#                  mortgage_amount_nis: float = 1200000, 
#                  years: float = 15, 
#                  loan_types: List[SingleLoanType] | None = None, 
#                  num_years_per_loan_type: List[float] | None = None,
#                  loan_types_weights: List[float] | None = None
#                  ) -> None:
        
#         self.mortgage_amount_nis = mortgage_amount_nis
#         self.max_num_of_years_all_types = years
#         self.loan_types = loan_types
#         self.num_of_types = len(loan_types)
#         self.loan_types_weights = loan_types_weights
#         self.num_years_per_loan_type = num_years_per_loan_type       
#         self._initialize(loan_types=loan_types, num_years_per_loan_type=num_years_per_loan_type, loan_types_weights= loan_types_weights)
    
#     def _initialize(self, loan_types: List[SingleLoanType] | None = [], num_years_per_loan_type: List[float] | None = [], loan_types_weights: List[float] | None = None):
#         self.loan_types = loan_types
#         self.num_of_types = len(loan_types)
        
#         if not loan_types_weights and self.num_of_types > 0:
#             if math.isclose(sum(loan_types_weights), 1)  and  math.isclose(len(loan_types_weights), self.num_of_types):
#                self.loan_types_weights = loan_types_weights
#             elif math.isclose(sum(loan_types_weights), 100) and  math.isclose(len(loan_types_weights), self.num_of_types):
#                self.loan_types_weights = loan_types_weights / 100   
#             else:
#                self.loan_types_weights = [1 / self.num_of_types for i in range(self.num_of_types)]    

#         elif self.num_of_types > 0:
#             self.loan_types_weights = [1 / self.num_of_types for i in range(self.num_of_types)] 
#         else: 
#             self.loan_types_weights = []
        
#         if not num_years_per_loan_type:
#             num_years_per_loan_type_tmp = [min(num_years, max(num_years_per_loan_type)) for num_years in num_years_per_loan_type]
#             if not self.loan_types_weights and math.isclose(len(num_years_per_loan_type), self.num_of_types):
#                self.num_years_per_loan_type = num_years_per_loan_type_tmp
#             elif not self.loan_types_weights:
#                 self.num_years_per_loan_type = [max(num_years_per_loan_type_tmp) for ii in range(self.num_of_types)]
#             else:
#                self.num_years_per_loan_type = []    
#         else:
#             self.num_years_per_loan_type = []
    

#     def add_loan_type(self, 
#                       partial_mortgage_amount_nis: float, 
#                       years: float, 
#                       annual_interest_rate: float, 
#                       loan_type_name: str,
#                       num_years_per_loan_type: float |None = None):
#          loan_type2add = SingleLoanType(partial_mortgage_amount_nis= partial_mortgage_amount_nis, 
#                                         years=years, 
#                                         annual_interest_rate=annual_interest_rate, 
#                                         loan_type_name=loan_type_name)
#          self.loan_types.append(loan_type2add)
#          self._initialize(loan_types= self.loan_types, 
#                           num_years_per_loan_type=num_years_per_loan_type, 
#                           loan_types_weights = None)
    
#     def calc_total_payback_for_all_mortgage(self):
#         total_payback = {"shpizer": 0, "keren_shava": 0}
#         for single_loan_type in self.loan_types:
#             total_payback_pew_type_dict= single_loan_type.calc_total_payback()
#             total_payback["shpizer"] += total_payback_pew_type_dict["shpizer"]
#             total_payback["keren_shava"] += total_payback_pew_type_dict["keren_shava"]

#         return total_payback
        
#     def calc_total_payback_rate_for_all_mortgage(self):
#         total_payback = calc_total_payback_for_all_mortgage()
#         payback_rate = {"shpizer": total_payback["shpizer"] / self.mortgage_amount_nis, "keren_shava": total_payback["keren_shava"] / self.mortgage_amount_nis}
            
#         return payback_rate
