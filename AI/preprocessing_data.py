
__all__ = ['DataLoader', 'SingleLoanType', 'Mortgage']

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats                   import gaussian_kde, norm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools     import kpss
from statsmodels.stats.diagnostic  import acorr_ljungbox
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.stattools import adfuller


class LoadTable4CPI:
    def __init__(self, file_path: Path):
         self.path_to_blumb_data = file_path
         self.cpi_df = None
         self.cpi_percentage_change_per_month = None 
         self._intialize()
    def _intialize(self):
        try:
            df_raw = pd.read_excel(self.path_to_blumb_data, header=None)  # Read the file without specifying headers yet
        except FileNotFoundError:
               print(f"Error: The file {self.path_to_blumb_data} was not found.")
               return 
        except Exception as e:
               print(f"Error reading the file: {e}")
               return
        
        try:
            # Adaptive approach: find the starting row of the table
            header_row = None
            for i, row in df_raw.iterrows():
                if 'TIME_PERIOD|Time period' in row.values:  # Look for the row containing the 'TIME_PERIOD|Time period' header
                    header_row = i
                    break        
            if header_row is None:
               raise ValueError("Header row with 'TIME_PERIOD|Time period' not found.")      
            
            # Now read the data starting from the header row
            self.cpi_df = pd.read_excel(self.path_to_blumb_data, skiprows=header_row, usecols=[1, 2])
            DateTimeCol = self.cpi_df.columns[0]
            AccumulateCpiCol = self.cpi_df.columns[1]
            self.cpi_df[DateTimeCol] = pd.to_datetime(self.cpi_df[DateTimeCol], format='%Y-%m')
        except ValueError as ve:
              print(f"Value error: {ve}")
              return
        except Exception as e:
               print(f"Error processing data: {e}")
               return
        
        try:
            self.cpi_percentage_change_per_month = pd.DataFrame({
                DateTimeCol: self.cpi_df[DateTimeCol][1:].reset_index(drop=True),  # Shifted time periods
                'CPI_PERCETAGE_CHANGE_PER_MONTH': -(self.cpi_df[AccumulateCpiCol].diff(-1).iloc[:-1]).reset_index(drop=True)  # Calculate percentage change
            })
        except Exception as e:
               print(f"Error calculating CPI percentage change: {e}")
        
         
if __name__ == '__main__':
   file_path = Path(r"C:\Users\eyall\Desktop\mortgage_eyal_ortal_levi\data_cpi_influation.xlsx")   
   cpi_data_obj = LoadTable4CPI(file_path=file_path)
   
   diff_series = cpi_data_obj.cpi_percentage_change_per_month['CPI_PERCETAGE_CHANGE_PER_MONTH'].diff().dropna()
   # Differencing the series to make it stationary
   
   # Perform ADF test
   adf_result = adfuller(diff_series)

   # Print ADF statistic and p-value
   print(f'ADF Statistic: {adf_result[0]}')
   print(f'p-value: {adf_result[1]}')


   vXcpi    = cpi_data_obj.cpi_df[cpi_data_obj.cpi_df.columns[1]].values
   Dates    = cpi_data_obj.cpi_df[cpi_data_obj.cpi_df.columns[0]].dt.strftime('%Y-%m-%d').values
   Ncpi     = len(vXcpi)
   
   vIdx    = np.arange(Ncpi)
   Ntrain  = 185
   vTrainX = vXcpi[:Ntrain]
   DatesTrain = Dates[:Ntrain]
   vTestX  = vXcpi[Ntrain:]
   
   #-- Plotting data
   plt.figure(figsize=(10, 3))
   plt.plot  (vIdx[:Ntrain], vTrainX, 'b',      label='train')
   plt.plot  (vIdx[Ntrain:], vTestX,  'orange', label='test')
   plt.title (f"Monthly CPI Growth in Percetage from {Dates[0]}")
   plt.xticks(vIdx[::50], Dates[::50])
   plt.legend(loc='upper left')
   plt.grid  ()
   plt.show  ()

   vXcpiChange    = cpi_data_obj.cpi_percentage_change_per_month[cpi_data_obj.cpi_percentage_change_per_month.columns[1]].values
   DatesChange    = cpi_data_obj.cpi_percentage_change_per_month[cpi_data_obj.cpi_percentage_change_per_month.columns[0]].dt.strftime('%Y-%m-%d').values
   NcpiChange     = len(vXcpiChange)
   vIdxChange     = np.arange(NcpiChange)
   Nchange        = len(vIdxChange)
   NtrainChange   = 230
   vTrainXChange = vXcpiChange[:NtrainChange]
   vTestXChange  = vXcpiChange[NtrainChange:]

   plt.figure(figsize=(10, 3))
   plt.plot  (vIdxChange[:NtrainChange], vTrainXChange, 'b',      label='train')
   plt.plot  (vIdxChange[NtrainChange:], vTestXChange,  'orange', label='test')
   plt.title (f"Monthly CPI Growth in Percetage from Last Month")
   plt.xticks(vIdxChange[::50], Dates[::50])
   plt.legend(loc='upper left')
   plt.grid  ()
   plt.show  ()
   

   S      = 12
   vXcpiS = vTrainX[S:] - vTrainX[:-S]
   pVal   = kpss(vXcpiS, nlags='auto')[1] #-- stationarity test

   plt.figure(figsize=(10, 3))
   plt.plot  (vIdx[S:Ntrain], vXcpiS, 'b', label='$\log(x_n)$')
   plt.title (f'Stationarity test (KPSS): $p = {pVal}$')
   plt.xticks(np.arange(S, Ntrain, 50), Dates[::50])
   plt.legend(loc=2)
   plt.grid  ()
   

   #-- Grid search:
vp = np.arange(7)
vq = np.arange(7)
vP = np.arange(2)
vQ = np.arange(2)
vS = np.array([3, 6, 12])
T  = pd.DataFrame(columns=['p', 'q', 'P', 'Q', 'AIC', 'BIC'])
for pp in vp:
    for qq in vq:
        for PP in vP:
            for QQ in vQ:
                for S in vS:
                    if qq == QQ and qq > 0:
                        continue  # Skip this iteration

                    oModel        = sarimax.SARIMAX(vTrainXChange, order=(pp,0,qq), seasonal_order=(PP,0,QQ,S), trend='c').fit(maxiter=100)
                    T.loc[len(T)] = [pp, qq, PP, QQ, oModel.aic, oModel.bic] 
T.sort_values(by='AIC')
   
p,d,q = 0,0,6
P,D,Q,S = 0,0,1,12

oTrainModel = sarimax.SARIMAX(
    vTrainXChange,
    order          = (0,0,6),
    seasonal_order = (0,0,1,12),
    trend          = 'c'
    )
oTrainModel = oTrainModel.fit(maxiter=1000)
oTrainModel.summary()

oTestModel = sarimax.SARIMAX(vXcpiChange, order=(p,d,q), seasonal_order=(P,D,Q,S), trend='c')   
oTestModel = oTestModel.filter(oTrainModel.params)

start = 22
Np = 20
oPred = oTestModel.get_prediction(start=start, end=Ncpi - 1 + Np, dynamic=False)  # Adjusted end parameter
vHatX = oPred.predicted_mean  # No parentheses needed here
pred_ci = oPred.conf_int(alpha=0.05)
vPredIdx = np.arange(start, Ncpi + Np)

plt.figure      (figsize=(16, 5))
plt.plot        (vIdxChange[:NtrainChange], vTrainXChange, 'b',     lw=1.5, label='train')
plt.plot        (vIdxChange[NtrainChange:], vTestXChange,  'orange',lw=1.5, label='test')
plt.plot        (vPredIdx,      vHatX,   'r',     lw=1.5, label='$\hat{x}_n$')
plt.fill_between(vPredIdx, pred_ci[:,0], pred_ci[:,1], color='r', alpha=.15, label = '95% CI')
plt.title       ('SARIMA($p=%d,d=%d,q=%d$)($P=%d,D=%d,Q=%d)_{%d}$ forecast' % (p,d,q,P,D,Q,S))
plt.xticks      (vIdxChange[6::12], DatesChange[6::12])
plt.xlim        ([Nchange/2, Nchange+Np])
plt.legend      (loc='upper left')
plt.grid        ()
plt.show        ()

   #-- Plotting data
   plt.figure(figsize=(10, 3))
   plt.plot  (vIdxChange[:NtrainChange], vTrainXChange, 'b', label='$\log(x_n)$')
   plt.title (f'Stationarity test (KPSS): $p = {pVal}$')
   plt.xticks(vIdxChange[::50], DatesChange[::50])
   plt.legend(loc=2)
   plt.grid  ()





