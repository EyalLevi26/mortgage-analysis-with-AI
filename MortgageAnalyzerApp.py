import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from main import main_user
from mortgage_toolkit.mortgage_calculator import DataLoader
from plotly import graph_objects as go
from typing import List, Tuple, Optional


class MortgageAnalysisTool:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mortgage Analysis Tool")
        self.root.geometry("800x600")

        # Initialize variables
        self.bank_name_var = tk.StringVar(root)
        self.loan_type_var = tk.StringVar(root)
        self.total_mortgage_amount_nis = tk.StringVar(root)
        self.loan_types: List[Tuple[str, int, float]] = []

        self.default_path = "mortgage_israel_bank_info.xlsx"

        # Create UI
        self.create_widgets()
        self.update_bank_dropdown(filepath=self.default_path)
        
    def create_widgets(self) -> None:
        tk.Label(self.root, text="Bank Info Path:").pack()
        self.entry_banks_info_path = tk.Entry(self.root, width=50)
        self.entry_banks_info_path.insert(0, self.default_path)
        self.entry_banks_info_path.pack()
        tk.Button(self.root, text="Browse", command=self.select_file).pack()

        tk.Label(self.root, text="Select Bank:").pack()
        self.bank_dropdown = tk.OptionMenu(self.root, self.bank_name_var, "")
        self.bank_dropdown.pack()

        tk.Label(self.root, text="Select Loan Type:").pack()
        self.loan_type_dropdown = tk.OptionMenu(self.root, self.loan_type_var, "")
        self.loan_type_dropdown.pack()

        tk.Label(self.root, text="Total Mortgage Amount [NIS]").pack()
        self.entry_mortgage_amount = tk.Entry(self.root, textvariable=self.total_mortgage_amount_nis, width=50)
        self.entry_mortgage_amount.pack()

        tk.Label(self.root, text="Loan Years:").pack()
        self.entry_loan_years = tk.Entry(self.root, width=50)
        self.entry_loan_years.pack()

        tk.Label(self.root, text="Loan Weight:").pack()
        self.entry_loan_weight = tk.Entry(self.root, width=50)
        self.entry_loan_weight.pack()

        tk.Button(self.root, text="Add Loan", command=self.add_loan).pack()
        tk.Label(self.root, text="Added Loans:").pack()
        self.loan_listbox = tk.Listbox(self.root, width=80)
        self.loan_listbox.pack()
        tk.Button(self.root, text="Reset Loans", command=self.reset_loan_listbox).pack()

        tk.Button(self.root, text="Run Analysis", command=self.run_analysis).pack()

    def select_file(self) -> None:
        filepath = filedialog.askopenfilename(title="Select Bank Info File")
        self.entry_banks_info_path.delete(0, tk.END)
        self.entry_banks_info_path.insert(0, filepath)
        self.update_bank_dropdown(filepath)

    def update_bank_dropdown(self, filepath: str) -> None:
        try:
            bank_info_instance = DataLoader(path=Path(filepath))
            bank_names = bank_info_instance.unique_bank_names.tolist()
            loan_types_from_file = bank_info_instance.unique_loan_types.tolist()[:-1]

            menu = self.bank_dropdown["menu"]
            menu.delete(0, "end")

            for bank_name in bank_names:
                menu.add_command(label=bank_name, command=lambda value=bank_name: self.bank_name_var.set(value))

            if bank_names:
                self.bank_name_var.set(bank_names[0])
            else:
                self.bank_name_var.set("No Banks Found")

            self.update_loan_type_dropdown(loan_types_from_file, first_call=True)

        except Exception as e:
            messagebox.showerror("Error", f"Could not read bank info: {e}")

    def update_loan_type_dropdown(self, loan_types: List[str], first_call: bool = False) -> None:
        if first_call:
            self.all_loan_types        = loan_types
            self.unselected_loan_types = loan_types
            self.selected_loan_types   = []
            
        loan_type_menu = self.loan_type_dropdown["menu"]
        loan_type_menu.delete(0, "end")

        for loan_type in loan_types:
            loan_type_menu.add_command(label=loan_type, command=lambda value=loan_type: self.loan_type_var.set(value))

        if loan_types:
            self.loan_type_var.set(loan_types[0])
        else:
            self.loan_type_var.set("No Loan Types Found")

    def add_loan(self) -> None:
        loan_type = self.loan_type_var.get()
        years = self.entry_loan_years.get()
        weight = self.entry_loan_weight.get()

        if loan_type and years and weight:
            try:
                years = int(years)
                weight = float(weight)
                self.loan_types.append((loan_type, years, weight))
                self.update_loan_listbox()
                self.clear_loan_entries()
                self.selected_loan_types.append(loan_type)
                
                if loan_type in self.unselected_loan_types:
                    index = self.unselected_loan_types.index(loan_type)
                    self.unselected_loan_types = self.unselected_loan_types[:index] + self.unselected_loan_types[index + 1:]
                    self.update_loan_type_dropdown(self.unselected_loan_types)
            except ValueError:
                messagebox.showerror("Error", "Years must be an integer and Weight must be a float.")
        else:
            messagebox.showerror("Error", "Please fill in all fields.")

    def clear_loan_entries(self) -> None:
        self.entry_loan_years.delete(0, tk.END)
        self.entry_loan_weight.delete(0, tk.END)

    def update_loan_listbox(self) -> None:
        self.loan_listbox.delete(0, tk.END)
        for loan_type, years, weight in self.loan_types:
            self.loan_listbox.insert(tk.END, f"Type: {loan_type}, Years: {years}, Weight: {weight:.2f}")

    def reset_loan_listbox(self) -> None:
        self.loan_listbox.delete(0, tk.END)
        self.loan_types.clear()
        self.selected_loan_types = []
        self.unselected_loan_types = self.all_loan_types
        self.update_loan_type_dropdown(self.all_loan_types)

    def run_analysis(self) -> None:
        try:
            banks_info_path = Path(self.entry_banks_info_path.get())
            bank_name = self.bank_name_var.get()
            mortgage_amount_nis = float(self.total_mortgage_amount_nis.get())

            if mortgage_amount_nis <= 0:
                messagebox.showinfo("Total Mortgage Amount Error", "The total mortgage amount must be greater than zero.")
                return

            num_years_per_loan_type, loan_types_weights = self.get_loan_type_data()

            if sum(loan_types_weights) > 100 or sum(loan_types_weights) < 100:
                messagebox.showinfo("Weight Error", "The total weight must equal 100 or 1.")
                return  

            results = main_user(
                banks_info_path=banks_info_path,
                mortgage_amount_nis=mortgage_amount_nis,
                years=max(num_years_per_loan_type),
                loan_types_weights=loan_types_weights,
                num_years_per_loan_type=num_years_per_loan_type,
                bank_name_for_plot=bank_name,
                extenal_plot=False
            )

            if results:
                loan_table, table_shpizer, table_keren_shava, figs = results
                self.display_results(loan_table, table_shpizer, table_keren_shava, figs)
                messagebox.showinfo("Analysis Complete", "Analysis and plots generated successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def get_loan_type_data(self) -> Tuple[List[int], List[float]]:
        num_years_per_loan_type = [0 for _ in range(5)]
        loan_types_weights = [0 for _ in range(5)]

        for loan in self.loan_types:
            loan_type, years, weight = loan
            if loan_type == "Constant interest rate and not index-linked":
                num_years_per_loan_type[0] = years
                loan_types_weights[0] = weight
            elif loan_type == "change interest rate and not index-linked - prime":
                num_years_per_loan_type[1] = years
                loan_types_weights[1] = weight
            elif loan_type == "change interest rate and not index-linked - not prime":
                num_years_per_loan_type[2] = years
                loan_types_weights[2] = weight
            elif loan_type == "change interest rate and index-linked":
                num_years_per_loan_type[3] = years
                loan_types_weights[3] = weight
            elif loan_type == "Constant interest rate and index-linked":
                num_years_per_loan_type[4] = years
                loan_types_weights[4] = weight

        if sum(loan_types_weights) <= 1:
            loan_types_weights = [weight * 100 for weight in loan_types_weights]

        return num_years_per_loan_type, loan_types_weights

    def display_results(self, loan_table, table_shpizer, table_keren_shava, figs: List[go.Figure]) -> None:
        loan_table.show()
        table_shpizer.show()
        table_keren_shava.show()
        for fig in figs:
            fig.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = MortgageAnalysisTool(root)
    root.mainloop()
