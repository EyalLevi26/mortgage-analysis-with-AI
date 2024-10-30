# **Mortgage Analysis with AI**
AI-powered mortgage data analysis using time series forecasting (ARIMA/SARIMAX), stationarity tests, and model optimization. Includes data preprocessing and visualization for predictive insights.

## **Quick Start Guide**

To set up the project and install necessary dependencies, follow these steps:

1. **Navigate to your desired directory:**
   ```bash
   cd path/to/clone/project
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/EyalLevi26/mortgage-analysis-with-AI.git
   ```

3. **Enter the project directory:**
   ```bash
   cd mortgage-analysis-with-AI
   ```

4. **Create a virtual environment with Python 3.8.10:**
   ```bash
   C:\Users\DELL\AppData\Local\Programs\Python\Python38\python.exe -m venv myenv
   ```

5. **Activate the virtual environment:**
   ```bash
   myenv\Scripts\activate.bat
   ```

6. **Install dependencies from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

This will prepare the project for execution and install all required packages in a virtual environment.

## **Running the Mortgage Analysis**
You have three options to run the code and simulate the mortgage:

### **Option 1**: Run `main_user` in `main.py` using Python
You can directly execute the `main_user` function in `main.py` by running:

```bash
python main.py
```

This option allows you to run the code directly through Python, which can be useful for debugging or customizing the mortgage analysis in code.

### **Option 2**: Run the GUI with `MortgageAnalyzerApp.py`
To interact with a graphical interface, you can run the `MortgageAnalyzerApp.py` file:

```bash
python MortgageAnalyzerApp.py
```

This will open a user-friendly GUI, where you can input mortgage details, choose options, and visualize results without needing to use the command line.

### **Option 3**: Run with the CLI

## **Running the Mortgage Analysis CLI**
To use the Mortgage Analysis CLI, you’ll need to create an executable file from the `main.py` script using **PyInstaller**. This ensures that all dependencies and required files are included in the executable. Follow these steps:

### **Step 1**: **Building the Executable**
Run the following **PyInstaller** command to package `main.py` as an executable:

```bash
pyinstaller --onefile --name mortgage_analysis_cli --hidden-import=pmdarima --hidden-import=tqdm --hidden-import=pickle --hidden-import=typing --add-data "mortgage_toolkit;mortgage_toolkit" --add-data "AI;AI" --add-data "payback_methods;payback_methods" --add-data "plot_utils.py;." --add-data "utills.py;." --add-data "my_argparser.py;." main.py
```

**Explanation of flags used:**

- `--onefile`: Packages everything into a single executable file.
- `--name mortgage_analysis_cli`: Names the output executable as `mortgage_analysis_cli`.
- `--hidden-import=...`: Ensures that specific modules (`pmdarima`, `tqdm`, `pickle`, and `typing`) are included, which may not be detected automatically.
- `--add-data`: Adds essential data and module folders, as well as helper scripts, to the executable.

This command will generate `mortgage_analysis_cli.exe` in the `dist` folder within your project directory.

### **Step 2**: **Running the CLI**
Once the executable is created, you can run it using the following command:

```bash
dist\mortgage_analysis_cli.exe mortgage_israel_bank_info.xlsx -m 1200000 -y 15 -b Benleomi -w "[0,0,0,0,100]" -yl "[10,10,10,10,15]" -plot
```

**Explanation of arguments:**

- **`mortgage_israel_bank_info.xlsx`**: Path to the Excel file containing mortgage bank information.
- **`-m`**: Specifies the mortgage amount in NIS.
- **`-y`**: Defines the number of years for the mortgage.
- **`-b`**: Bank name (e.g., **Benleomi**).
- **`-w`**: Sets weights for each loan type in the form of a list (e.g., `"[0,0,0,0,100]"`).
- **`-yl`**: Specifies the number of years per loan type as a list (e.g., `"[10,10,10,10,15]"`).
- **`-plot`**: Optional flag to generate visualizations of the mortgage analysis.

By following these steps, you’ll be able to run the mortgage analysis through the command line interface (CLI) and generate loan estimations and visualizations.

### **Argument Details for -w and -yl**
The **`-w`** and **`-yl`** arguments allow you to define weights and the number of years for different types of loans in your mortgage analysis. Each index in these arguments corresponds to a specific loan type or configuration. Below are the detailed mappings for each index:

- **`-w` Argument: Loan Type Weights**

   - **Index 0**: `const_interest_not_index_linked` - Represents the weight for loans with constant interest rates that are not linked to the CPI.
   - **Index 1**: `change_interest_not_index_linked_prime` - Represents the weight for loans with changing interest rates that are not linked to the CPI (prime).
   - **Index 2**: `change_interest_not_index_linked` - Represents the weight for loans with changing interest rates that are not linked to the CPI.
   - **Index 3**: `change_interest_index_linked` - Represents the weight for loans with changing interest rates that are linked to the CPI.
   - **Index 4**: `const_interest_index_linked` - Represents the weight for loans with constant interest rates that are linked to the CPI.

- **`-yl` Argument: Loan Duration Years**

   - Each index in the `-yl` argument corresponds to the same loan types as above, specifying the duration in years for each loan type.

