import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from main import main_user
from format_obj.classes_utills import DataLoader
import plotly.graph_objects as go
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
from plotly import offline as pyo
import plotly.io as pio
from PIL import Image, ImageTk
import io
from tkinterweb import HtmlFrame

# Turn off interactive mode for Matplotlib to suppress backend messages
plt.ioff()

# Initialize main window
root = tk.Tk()
root.title("Mortgage Analysis Tool")
root.geometry("800x600")

# Variable to store bank names and reference to OptionMenu
bank_name_var = tk.StringVar(root)
loan_types = []  # List to hold multiple loan types with years and weights
bank_dropdown = None  # Placeholder for dropdown widget

# Function to browse and select file path
def select_file():
    filepath = filedialog.askopenfilename(title="Select Bank Info File")
    entry_banks_info_path.delete(0, tk.END)
    entry_banks_info_path.insert(0, filepath)
    update_bank_dropdown(filepath)

# Function to update bank dropdown based on file contents
def update_bank_dropdown(filepath):
    try:
        bank_info_instance = DataLoader(path=Path(filepath))
        bank_names = bank_info_instance.unique_bank_names.tolist()  # Assuming this provides a list of banks
        loan_types_from_file = bank_info_instance.unique_loan_types.tolist()[:-1]  # Assuming this provides a list of loan types

        # Clear previous dropdown options
        menu = bank_dropdown["menu"]
        menu.delete(0, "end")

        # Insert new bank names into dropdown
        for bank_name in bank_names:
            menu.add_command(label=bank_name, command=lambda value=bank_name: bank_name_var.set(value))

        # Set default dropdown value if there are banks
        if bank_names:
            bank_name_var.set(bank_names[0])
        else:
            bank_name_var.set("No Banks Found")

        # Update loan type dropdown
        update_loan_type_dropdown(loan_types_from_file, first_call = True)

    except Exception as e:
        messagebox.showerror("Error", f"Could not read bank info: {e}")

# Function to update loan type dropdown based on file contents
def update_loan_type_dropdown(loan_types, first_call = False):
    global all_loan_types 
    if first_call:
        all_loan_types = []

    # Clear previous loan type dropdown options
    loan_type_menu = loan_type_dropdown["menu"]
    loan_type_menu.delete(0, "end")
    

    # Insert new loan types into dropdown
    for loan_type in loan_types:
        loan_type_menu.add_command(label=loan_type, command=lambda value=loan_type: loan_type_var.set(value))
        if first_call:
            all_loan_types.append(loan_type)
        
    # Set default dropdown value if there are loan types
    if loan_types:
        loan_type_var.set(loan_types[0])
    else:
        loan_type_var.set("No Loan Types Found")

# Function to run the main_user function
def run_analysis():
    try:
        # Fetch parameters from user inputs
        banks_info_path = Path(entry_banks_info_path.get())
        bank_name = bank_name_var.get()

        # Prepare loan details for analysis
        num_years_per_loan_type = [0 for _ in range(5)]
        loan_types_weights      = [0 for _ in range(5)]
        for loan in loan_types:
            loan_type, years, weight = loan
            if loan_type == "Constant interest rate and not index-linked":
                num_years_per_loan_type[0] = years
                loan_types_weights[0]      = weight
            elif loan_type == "change interest rate and not index-linked - prime":
                num_years_per_loan_type[1] = years
                loan_types_weights[1]      = weight
            elif loan_type == "change interest rate and not index-linked - not prime":
                num_years_per_loan_type[2] = years
                loan_types_weights[2]      = weight
            elif loan_type == "change interest rate and index-linked":
                num_years_per_loan_type[3] = years
                loan_types_weights[3]      = weight
            elif loan_type == "Constant interest rate and  index-linked":
                num_years_per_loan_type[4] = years
                loan_types_weights[4]      = weight
        
        if sum(loan_types_weights) <= 1:
            update_loan_listbox(is_weight_percentage = False)
            loan_types_weights = [weight * 100 for weight in loan_types_weights]
            
        if sum(loan_types_weights) > 100 or sum(loan_types_weights) < 100:
           messagebox.showinfo("Weight Error", "The total weight must equal 100 or 1. Please adjust the weights for each loan type accordingly.")
           reset_loan_listbox()
           return  
        
        mortgage_amount_nis = float(total_mortgage_amount_nis.get())
        if mortgage_amount_nis <= 0:
           messagebox.showinfo("Total Mortgage Amount Error", "The total mortgage amount must be greater than zero. Please adjust the total mortgage amount [NIS] accordingly.")
           total_mortgage_amount_nis.delete(0, tk.END)
           return
           
        # Run analysis
        # results = main_user(banks_info_path, loan_details, bank_name_for_plot=bank_name)
        results = main_user(banks_info_path= banks_info_path, 
                            mortgage_amount_nis= mortgage_amount_nis, 
                            years = max(num_years_per_loan_type), 
                            loan_types_weights = loan_types_weights, 
                            num_years_per_loan_type = num_years_per_loan_type, 
                            bank_name_for_plot = bank_name, 
                            extenal_plot = False)

        # Check and display results
        if results:
            loan_table, table_shpizer, table_keren_shava, figs = results
            display_results(loan_table, table_shpizer, table_keren_shava, figs)
            messagebox.showinfo("Analysis Complete", "Analysis and plots generated successfully.")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def display_results(loan_table, table_shpizer, table_keren_shava, figs):
    
    loan_table.show()
    table_shpizer.show()
    table_keren_shava.show()

    for fig in figs:
        fig.show()
      
# Function to clear all items in the Added Loans list box
def reset_loan_listbox():
    loan_listbox.delete(0, tk.END)  # Deletes all items in the list box
    loan_types.clear()              # Clears the loan_types list
    update_loan_type_dropdown(all_loan_types)
    
# Function to add loan entry
def add_loan():
    loan_type = loan_type_var.get()
    years = entry_loan_years.get()
    weight = entry_loan_weight.get()

    if loan_type and years and weight:
        try:
            years = int(years)
            weight = float(weight)

            # Add the loan details to the list
            loan_types.append((loan_type, years, weight))

            # Update the display
            update_loan_listbox()
            clear_loan_entries()
            if loan_type in all_loan_types:
                index = all_loan_types.index(loan_type)
                loan_types_updated = all_loan_types[:index] + all_loan_types[index + 1:]
                update_loan_type_dropdown(loan_types_updated)
        except ValueError:
            messagebox.showerror("Error", "Years must be an integer and Weight must be a float.")
    else:
        messagebox.showerror("Error", "Please fill in all fields.")

# Function to clear loan entry fields
def clear_loan_entries():
    entry_loan_years.delete(0, tk.END)
    entry_loan_weight.delete(0, tk.END)
    
# Function to update the listbox with loan types
def update_loan_listbox(is_weight_percentage:bool=True):
    # Update loan_types weights based on is_weight_percentage
    global loan_types
    updated_loan_types = []
    for loan_type, years, weight in loan_types:
        # Convert to percentage or decimal form based on the flag
        updated_weight = weight if is_weight_percentage else weight * 100
        updated_loan_types.append((loan_type, years, updated_weight))
    
    # Update the loan_types with the new values
    loan_types = updated_loan_types

    # Refresh the listbox display
    loan_listbox.delete(0, tk.END)
    for loan_type, years, weight in loan_types:
        loan_listbox.insert(tk.END, f"Type: {loan_type}, Years: {years}, Weight: {weight:.2f}")
        
    # loan_listbox.delete(0, tk.END)
    # for loan_type, years, weight in loan_types:
        # if is_weight_percentage:
            # weight *= 100
        # loan_listbox.insert(tk.END, f"Type: {loan_type}, Years: {years}, Weight: {weight}")

# Entry fields and browse button
tk.Label(root, text="Bank Info Path:").pack()
entry_banks_info_path = tk.Entry(root, width=50)
default_path = "C:\\Users\\DELL\\Documents\\mortgage\\MortgageAnalysis\\mortgage_israel_bank_info.xlsx"
entry_banks_info_path.insert(0, default_path)
entry_banks_info_path.pack()
tk.Button(root, text="Browse", command=select_file).pack()

tk.Label(root, text="Select Bank:").pack()
bank_name_var.set("Select Bank")
bank_dropdown = tk.OptionMenu(root, bank_name_var, "")
bank_dropdown.pack()

# Dropdown for loan type selection, initialized as empty
tk.Label(root, text="Select Loan Type:").pack()
loan_type_var = tk.StringVar(root)
loan_type_var.set("Select Loan Type")
loan_type_dropdown = tk.OptionMenu(root, loan_type_var, "")
loan_type_dropdown.pack()

# Entry fields for loan years and weight
tk.Label(root, text="Total Mortgage Amount [NIS]").pack()
total_mortgage_amount_nis = tk.Entry(root, width=50)
total_mortgage_amount_nis.pack()

# Entry fields for loan years and weight
tk.Label(root, text="Loan Years:").pack()
entry_loan_years = tk.Entry(root, width=50)
entry_loan_years.pack()

tk.Label(root, text="Loan Weight:").pack()
entry_loan_weight = tk.Entry(root, width=50)
entry_loan_weight.pack()

# Button to add loan entry
tk.Button(root, text="Add Loan", command=add_loan).pack()

# Listbox to display added loans
tk.Label(root, text="Added Loans:").pack()
loan_listbox = tk.Listbox(root, width=80)
loan_listbox.pack()

# Add a "Reset Loans" button to clear the list box
tk.Button(root, text="Reset Loans", command=reset_loan_listbox).pack()

# Initialize the dropdowns with the default file path
update_bank_dropdown(default_path)

# Run button to trigger analysis
tk.Button(root, text="Run Analysis", command=run_analysis).pack()

root.mainloop()
