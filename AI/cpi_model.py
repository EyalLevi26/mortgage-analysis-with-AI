
__all__ = ['LoadTable4CPI', 'ProgressCallback', 'MySARIMAX']

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import time
import warnings

from scipy.stats                   import gaussian_kde, norm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools     import kpss
from statsmodels.stats.diagnostic  import acorr_ljungbox
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import pmdarima as pm
from tqdm import tqdm
import pickle
from typing import Union

class LoadTable4CPI:
    """
    The LoadTable4CPI class loads and processes Consumer Price Index (CPI) data from an Excel file.

    Attributes:
    ----------
    path_to_blumb_data : Path
        The file path of the Excel file containing the CPI data.
    cpi_values_df : pandas.DataFrame or None
        A DataFrame that holds the CPI data, including time periods and CPI values. 
        Initialized to None and populated once the data is successfully loaded and processed.

    Methods:
    -------
    __init__(self, file_path: Path)
        Initializes the LoadTable4CPI object with the path to the Excel file and triggers the 
        data loading process by calling the _initialize method.
    
    _initialize(self)
        Reads and processes the Excel file. It attempts to find the starting row of the CPI data, 
        processes the data into a DataFrame, and formats the date column for further use.
    """
    def __init__(self, file_path: Path):
         self.path_to_blumb_data = file_path
         self.cpi_values_df = None
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
            self.cpi_values_df = pd.read_excel(self.path_to_blumb_data, skiprows=header_row, usecols=[1, 2])
            DateTimeCol = self.cpi_values_df.columns[0]
            AccumulateCpiCol = self.cpi_values_df.columns[1]
            self.cpi_values_df[DateTimeCol] = pd.to_datetime(self.cpi_values_df[DateTimeCol], format='%Y-%m').dt.strftime('%Y-%m')
        except ValueError as ve:
              print(f"Value error: {ve}")
              return
        except Exception as e:
               print(f"Error processing data: {e}")
               return
# Custom callback class to track time and progress
class ProgressCallback:
    def __init__(self, total_models):
        self.total_models = total_models
        self.start_time = time.time()
        self.model_count = 0
        self.progress_bar = tqdm(total=total_models, desc="Fitting models", unit="model")

    def on_fit(self):
        self.model_count += 1
        elapsed_time = time.time() - self.start_time
        average_time = elapsed_time / self.model_count
        remaining_models = self.total_models - self.model_count
        estimated_remaining_time = remaining_models * average_time
        
        # Update progress bar
        self.progress_bar.update(1)
        if self.model_count % 200 == 0 or self.model_count == self.total_models:
           print(f"Fitting model {self.model_count}/{self.total_models}")
           print(f"Elapsed time: {elapsed_time:.2f} seconds")
           print(f"Estimated time remaining: {estimated_remaining_time:.2f} seconds\n")

    def close(self):
        self.progress_bar.close()

class MySARIMAX:
      def __init__(self, cpi_data_obj: Union[LoadTable4CPI , None] = None, partial_train_percentage: float = 85, apply_model_fitting: bool =False,  use_plotly: bool=True):
         if cpi_data_obj is None:
            file_path    = Path(r"C:\Users\DELL\Documents\mortgage\MortgageAnalysis\data_cpi.xlsx")   
            cpi_data_obj = LoadTable4CPI(file_path=file_path)
        
         self.cpi_values                         = cpi_data_obj.cpi_values_df[cpi_data_obj.cpi_values_df.columns[1]].values
         self.dates_cpi_values                   = cpi_data_obj.cpi_values_df[cpi_data_obj.cpi_values_df.columns[0]].values
         self.n_cpi_values                       = len(self.cpi_values)
         self.vIdx_cpi_values                    = np.arange(self.n_cpi_values)

         self.use_plotly                  = use_plotly         
         self.valid_for_train             = False
         self.partial_train_percentage    = partial_train_percentage
         self.data_train_vX               = None
         self.data_test_vX                = None
         self.data_train_v_dates          = None
         self.data_test_v_dates           = None  

         self.split_train_test()
         
         self.p_d_q   = np.array([2, 2, 3])
         self.P_D_Q_S = np.array([1, 0, 1, 12])

         if apply_model_fitting:
             self.model_fitting(auto_fit_model=True)

         self.oTrainModel = None  # Attribute to store the trained model
         self.oTestModel  = None

         self.adf_stas = None
         self.p_val = None

      def check_adf(self, print_result: bool = False):
          adf_result = adfuller(self.data_train_vX) 
          self.adf_stas = adf_result[0]
          self.p_val    = adf_result[1]
          if print_result:
             print(f"p-value is {self.p_val}, and ADF Statistic is {self.adf_stas}")  

      def model_fitting(self, auto_fit_model=True, plot_acf_vs_diff = False):
          if plot_acf_vs_diff:
             # Original Series
             fig, axes = plt.subplots(3, 2, sharex=True)
             axes[0, 0].plot(np.log(self.cpi_values)); axes[0, 0].set_title('Original Series')
             plot_acf(np.log(self.cpi_values), ax=axes[0, 1])
             
             # 1st Differencing
             axes[1, 0].plot(np.diff(np.log(self.cpi_values))); axes[1, 0].set_title('1st Order Differencing')
             plot_acf(np.diff(np.log(self.cpi_values)), ax=axes[1, 1])
             
             # 2nd Differencing
             axes[2, 0].plot(np.diff(np.diff(np.log(self.cpi_values)))); axes[2, 0].set_title('2nd Order Differencing')
             plot_acf(np.diff(np.diff(np.log(self.cpi_values))), ax=axes[2, 1])
             
             plt.show(block=True)
          
          self._preprocess()
          self.check_adf(print_result=True)

          if auto_fit_model:
             total_models = 3587
             progress_callback = ProgressCallback(total_models)
             def fit_auto_arima(y):
                 model = pm.auto_arima(y=y, 
                                       start_p =1, 
                                       start_q= 1, 
                                       m=12,
                                       maxiter =100, 
                                       error_action='ignore', 
                                       verbose=True, 
                                       stepwise = False,
                                       callback= lambda n_jobs, *args, **kwargs: progress_callback.on_fit())
                 return model
             model = fit_auto_arima(self.data_train_vX)
             progress_callback.close()             
             print(model.summary())
             p, d, q = model.order
             P, D, Q, S = model.seasonal_order
             self.p_d_q     = np.array([p, d, q])
             self.P_D_Q_S   = np.array([P, D, Q, S])
             self._inverse_preprocess()
             best_model_aic = {'p':p, 'd':d, 'q':q, 'P':P, 'D': D, 'Q':Q, 'S':S}
             return best_model_aic


          vD = np.arange(4)
          vd = np.arange(4)
          vp = np.arange(4)
          vq = np.arange(7)
          vP = np.arange(2)
          vQ = np.arange(2)
          vS = np.array([12])
          T  = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D','Q', 'S', 'AIC', 'BIC'])
          total_models = len(vD)*len(vd)*len(vp)*len(vq)*len(vP)*len(vQ)*len(vS)

          progress_callback = ProgressCallback(total_models)
          for pp in vp:
              for qq in vq:
                  for dd in vd:
                      for PP in vP:
                          for DD in vD:
                              for QQ in vQ:
                                  for S in vS:
                                      try:
                                          warnings.filterwarnings("ignore", category=UserWarning, message=".*Maximum Likelihood optimization failed to converge.*")
                                          warnings.filterwarnings("ignore", category=UserWarning, message=".*Non-invertible starting seasonal moving average.*")

                                          oModel = sarimax.SARIMAX(self.data_train_vX, 
                                                                   order=(pp, dd, qq), 
                                                                   seasonal_order=(PP, DD, QQ, S), 
                                                                   trend='c').fit(maxiter=100, disp=False)  # Increased maxiter and changed method
                                          T.loc[len(T)] = [pp, dd, qq, PP, DD, QQ, S, oModel.aic, oModel.bic]
                                          progress_callback.on_fit()
                                      except (ValueError, np.linalg.LinAlgError) as e:
                                          print(f"Skipped combination (p={pp}, q={qq}, P={PP}, Q={QQ}, S={S}) due to error: {e}")
          progress_callback.close()
          best_model_aic = T.sort_values(by='AIC').iloc[0]
          self.p_d_q     = np.array([best_model_aic["p"], best_model_aic["d"], best_model_aic["q"]])
          self.P_D_Q_S   = np.array([best_model_aic["P"], best_model_aic["D"], best_model_aic["Q"], best_model_aic["S"]])

        #   self.p_d_q     = np.array([best_model_aic["p"], 0, best_model_aic["q"]])
        #   self.P_D_Q_S   = np.array([best_model_aic["P"], 0, best_model_aic["Q"], best_model_aic["S"]])
          self._inverse_preprocess()
          return best_model_aic
      
      def split_train_test(self):
        """
        Split cpi_values and corresponding dates into training and testing sets based on self.partial_train_percentage.
        """
        # Calculate the number of training samples
        n_train = int(self.n_cpi_values * self.partial_train_percentage / 100)
        
        # Split the data into training and test sets
        self.data_train_vX = self.cpi_values[:n_train]
        self.data_test_vX = self.cpi_values[n_train:]

        # Split the dates correspondingly
        self.data_train_v_dates = self.dates_cpi_values[:n_train]
        self.data_test_v_dates = self.dates_cpi_values[n_train:]

        self.valid_for_train = True  # Indicate that the split is successful

        print(f"Training set size: {len(self.data_train_vX)}, Testing set size: {len(self.data_test_vX)}")
     
      def train(self, maxiter=1000, print_summary: bool = True):
        """
        Trains the SARIMAX model with given default parameters.
        :param p: Non-seasonal AR order
        :param d: Non-seasonal differencing order
        :param q: Non-seasonal MA order
        :param P: Seasonal AR order
        :param D: Seasonal differencing order
        :param Q: Seasonal MA order
        :param S: Length of the seasonal cycle
        :param maxiter: Maximum number of iterations for optimization
        :return: Trained SARIMAX model
        """
            # Check if the model is valid for training
        if not self.valid_for_train:
            raise ValueError("The model is not valid for training. Ensure the data is ready for training.")
        
        self._preprocess()

        p, d, q = self.p_d_q
        P,D,Q,S = self.P_D_Q_S

        # Training the model
        self.oTrainModel = sarimax.SARIMAX(
            self.data_train_vX,
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            trend='c'
        )
        self.oTrainModel = self.oTrainModel.fit(maxiter=maxiter)
        if print_summary:
            print(self.oTrainModel.summary())

        self._inverse_preprocess()
    
        return self.oTrainModel

      def _preprocess(self):
        """Apply log transformation to training and testing data."""
        # Apply log transformation to training data
        if self.data_train_vX is not None:
            self.data_train_vX = np.log(self.data_train_vX)

      def _inverse_preprocess(self):
        """Inverse the log transformation."""
        # Inverse for training data
        if self.data_train_vX is not None:
            self.data_train_vX = np.exp(self.data_train_vX)  # Inverse log transformation
                  
      def predict(self, start=None, end=None, steps_ahead=None, return_conf_int=True, plot_results = False, manual_predict = False):
        """
        Generates predictions from the trained SARIMAX model.

        :param start: The start index for predictions. If None, defaults to the beginning of the test set.
        :param steps_ahead: The number of steps to predict beyond the training data (out-of-sample). 
                            If None, defaults to predicting the test set length.
        :param return_conf_int: Whether to return confidence intervals along with the predictions.
        :return: Predictions (and confidence intervals if return_conf_int is True)
        """
        if self.oTrainModel is None:
            raise ValueError("Model has not been trained yet. Call the `train` method first.")
        p, d, q = self.p_d_q
        P,D,Q,S = self.P_D_Q_S

        oTestModel = sarimax.SARIMAX(np.log(self.cpi_values), order=(p, d, q), seasonal_order=(P,D,Q,S), trend='c')
        oTestModel = oTestModel.filter(self.oTrainModel.params)
        self.oTestModel = oTestModel

        # Get predictions from the trained SARIMAX model
        if not manual_predict:
            # Default the start index to the beginning of the test data
            if start is None or isinstance(start, str):
                start = 0 # int(len(self.data_train_vX)/20)

            # Default steps_ahead to cover the entire test set if not specified
            if steps_ahead is None or isinstance(steps_ahead, str):
                steps_ahead = 0 # int(len(self.data_test_vX) / 20)
                
            prediction = oTestModel.get_prediction(start, self.n_cpi_values - 1 + steps_ahead, len(self.data_train_vX) - start)
            
            pred_mean  = prediction.predicted_mean
            pred_mean  = np.exp(pred_mean)
            vPredIdx   = np.arange(start, self.n_cpi_values + steps_ahead) 

            dates = self.dates_cpi_values[np.arange(0, self.n_cpi_values)]

            dates_datetime = pd.to_datetime(dates, format='%Y-%m')
            additional_dates = [dates_datetime[-1] + pd.DateOffset(months=i) for i in range(1, steps_ahead + 1)]
            if additional_dates:
                all_dates = np.concatenate([dates_datetime, additional_dates])
            else:
                all_dates = dates_datetime

            all_dates = pd.to_datetime(all_dates)
            all_dates_pred = all_dates.strftime('%Y-%m').values
        
        else:
            if isinstance(start, (int, float)) and start < len(self.dates_cpi_values) - 1:
                start = self.dates_cpi_values[start]
                
            index_start = np.where(self.dates_cpi_values == start)[0][0]
            
            if end is None:
                index_end = index_start + 1
            elif isinstance(end, (int, float)) and end < len(self.dates_cpi_values):
                end = self.dates_cpi_values[end]
                                
            index_end = np.where(self.dates_cpi_values == end)[0][0]
                
            prediction = oTestModel.get_prediction(index_start, index_end, len(self.data_train_vX) - index_start)
            
            pred_mean  = prediction.predicted_mean
            pred_mean  = np.exp(pred_mean)
            vPredIdx   = np.arange(index_start, index_end+1) - index_start
            if index_end < len(self.dates_cpi_values):
                dates      = self.dates_cpi_values[np.arange(index_start, index_end+1)]
            else:
                dates      = self.dates_cpi_values[np.arange(index_start, len(self.dates_cpi_values))]
            
            dates_datetime = pd.to_datetime(dates, format='%Y-%m')
            additional_dates = [dates_datetime[-1] + pd.DateOffset(months=i) for i in range(1, index_end - len(self.dates_cpi_values) + 1)]
            if additional_dates:
                all_dates = np.concatenate([dates_datetime, additional_dates])
            else:
                all_dates = dates_datetime
                
            all_dates = pd.to_datetime(all_dates)
            all_dates_pred = all_dates.strftime('%Y-%m').values    
                
        if return_conf_int:
            pred_conf_int = prediction.conf_int(alpha=0.05)
            
            pred_conf_int[:,0] = np.exp(pred_conf_int[:,0])
            pred_conf_int[:,1] = np.exp(pred_conf_int[:,1])
            if plot_results:  
               plot_data = {
                            'pred_mean': pred_mean,
                            'vPredIdx': vPredIdx,
                            'all_dates_pred':all_dates_pred,
                            'pred_conf_int': pred_conf_int
                            }
               self.visualize_results(plot_data)  

            return pred_mean, all_dates_pred, pred_conf_int
        else:
            if plot_results:  
               plot_data = {
                            'pred_mean': pred_mean,
                            'vPredIdx': vPredIdx,
                            'all_dates_pred':all_dates_pred,
                            'pred_conf_int': None
                            }
               self.visualize_results(plot_data)  
            return pred_mean, all_dates_pred
      
      def visualize_results(self, plot_data):
           pred_mean      = plot_data['pred_mean']
           pred_conf_int  = plot_data['pred_conf_int']
           vPredIdx       = plot_data['vPredIdx']
           all_dates_pred = plot_data['all_dates_pred']

           n_train = len(self.data_train_vX)
           p, d, q    = self.p_d_q
           P, D, Q, S = self.P_D_Q_S

           if not self.use_plotly:
              plt.figure      (figsize=(16, 5))
              plt.plot        (self.vIdx_cpi_values[:n_train], self.data_train_vX, 'b',     lw=1.5, label='train')
              plt.plot        (self.vIdx_cpi_values[n_train:], self.data_test_vX,  'orange',lw=1.5, label='test')
              plt.plot        (vPredIdx,      pred_mean,   'r',     lw=1.5, label='$\hat{x}_n$')

              if pred_conf_int is not None:
                 plt.fill_between(vPredIdx, pred_conf_int[:,0], pred_conf_int[:,1], color='r', alpha=.15, label = '95% CI')
              plt.title       ('SARIMA($p=%d,d=%d,q=%d$)($P=%d,D=%d,Q=%d)_{%d}$ forecast' % (p,d,q,P,D,Q,S))
              plt.xticks      (self.vIdx_cpi_values[6::80], self.dates_cpi_values[6::80])
              plt.legend(loc='upper left')
              plt.grid        ()
              plt.show        (block=True)
           else:
              fig = go.Figure() 

              # Plot train data
              fig.add_trace(go.Scatter(
                  x=self.dates_cpi_values[self.vIdx_cpi_values[:n_train]],
                  y=self.data_train_vX,
                  mode='lines',
                  name='train',
                  hovertemplate='Year-Month: %{x}<br>Index Value: %{y:.2f}<extra></extra>',
                  line=dict(color='blue', width=1.5)
              ))
              
              # Plot test data
              fig.add_trace(go.Scatter(
                  x=self.dates_cpi_values[self.vIdx_cpi_values[n_train:]],
                  y=self.data_test_vX,
                  mode='lines',
                  name='test',
                  hovertemplate='Year-Month: %{x}<br>Index Value: %{y:.2f}<extra></extra>',
                  line=dict(color='orange', width=1.5)
              ))
              
              # Plot prediction data
              fig.add_trace(go.Scatter(
                  x=all_dates_pred[vPredIdx],
                  y=pred_mean,
                  mode='lines',
                  name='$\hat{x}_n$',
                  hovertemplate='Year-Month: %{x}<br>Index Value: %{y:.2f}<extra></extra>',
                  line=dict(color='red', width=1.5)
              ))
              
              if pred_conf_int is not None:
                 # Add fill for confidence interval (95% CI)
                 fig.add_trace(go.Scatter(
                     x=np.concatenate([all_dates_pred[vPredIdx], all_dates_pred[vPredIdx[::-1]]]),
                     y=np.concatenate([pred_conf_int[:, 0], pred_conf_int[:, 1][::-1]]),
                     fill='toself',
                     fillcolor='rgba(255, 0, 0, 0.15)',
                     line=dict(color='rgba(255,255,255,0)'),
                     hoverinfo="skip",
                     showlegend=True,
                     name='95% CI'
                 ))
              
               # Title and axis labels
              fig.update_layout(
                  title=f"SARIMA(p={p},d={d},q={q})(P={P},D={D},Q={Q})_{{{S}}} forecast",
                  title_x=0.5,  # Center the title horizontally
                  title_y=0.9,  # Adjust the vertical position of the title
                  xaxis_title='Date',
                  yaxis_title='Index Value',
                  xaxis=dict(
                      tickvals=all_dates_pred[vPredIdx[0::50]],  # Adjust xticks as per original
                      ticktext=all_dates_pred[vPredIdx[0::50]]
                  ),
                  legend=dict(
                      orientation="v",
                      yanchor="top",
                      y=1,
                      xanchor="left",
                      x=0
                  ),
                  autosize=True,
                #   width=1000,
                #   height=1000,
                #   margin=dict(
                #               l=500,  # Adjust left margin
                #               r=0,  # Adjust right margin
                #               t=200,  # Adjust top margin
                #               b=0   # Adjust bottom margin
                #              ),
                  template="ggplot2"
              )
           
              # Show the plot
              fig.show(block=False)
              start = 80
              prediction_train_test = self.oTestModel.get_prediction(start=start, end = self.n_cpi_values - 1, dynamic=self.n_cpi_values - start)
              pred_mean_train_test  = prediction_train_test.predicted_mean
      
              #-- Residuals:
              vEn    = np.log(self.cpi_values[start:]) - pred_mean_train_test             
              fig = PlotResidual(vEn)
              fig.show    (block=True)
              
      def calc_cpi_growth_in_percentage(self, years = 30, plot_results = False):
          """
            Calculate the CPI growth percentage over a specified period and optionally plot the results.

            This method calculates the percentage growth in the Consumer Price Index (CPI) values over a given number of years,
            using a rolling monthly approach. Additionally, it generates predictions over the same time period and calculates 
            the predicted growth percentage. If `plot_results` is set to True, it displays a line plot comparing actual growth
            percentages with predicted values.

            Parameters
            ----------
            years : int, optional
                The number of years to consider for calculating CPI growth percentage. Defaults to 30 if not provided.
            plot_results : bool, optional
                If True, generates a Plotly figure with actual and predicted CPI growth percentages over time. Default is False.

            Returns
            -------
            cpi_growth_percentage : numpy.ndarray
                The actual CPI growth percentages calculated over the specified period.
            dates : numpy.ndarray
                The corresponding dates for each calculated growth percentage.
            cpi_pred_growth_percentage : numpy.ndarray
                The predicted CPI growth percentages over the same period.

            Plot
            ----
            If `plot_results` is True, displays a Plotly figure with:
                - Actual CPI growth percentage (labeled "Ground Truth")
                - Predicted CPI growth percentage (labeled "Prediction")

            Example
            -------
            cpi_growth, dates, pred_growth = obj.calc_cpi_growth_in_percentage(years=10, plot_results=True)
        """

          if years == None:
              years = 30
          mounths = years * 12
          start_index = len(self.cpi_values) - mounths - 1
          cpi_growth_percentage = 100 * (self.cpi_values[start_index + 1: ] / self.cpi_values[start_index: -1]) - 100
          
          pred_mean, dates = self.predict(start=start_index, end=len(self.cpi_values)-1, return_conf_int=False, plot_results = False, manual_predict = True)
          cpi_pred_growth_percentage = 100 * (pred_mean[1: ] / pred_mean[:-1]) - 100
          
          if plot_results:
             # Create the figure
             fig = go.Figure()
             
             fig.add_trace(go.Scatter(
                 x=dates,
                 y=cpi_growth_percentage,
                 mode='lines+markers',
                 name='Ground Truth'
             ))
             
             fig.add_trace(go.Scatter(
                 x=dates,
                 y=cpi_pred_growth_percentage,
                 mode='lines+markers',
                 name='Prediction'
             ))
             
             fig.update_layout(
                               title="Growth Percentage and Prediction over Time",
                               xaxis_title="Dates",
                               yaxis_title="Monthly Growth Change (%)",
                               legend=dict(
                               title="Legend",
                               orientation="h",
                               yanchor="bottom",
                               y=1.02,
                               xanchor="center",
                               x=0.5)
                              )
             
             fig.show()
          
          return cpi_growth_percentage, dates, cpi_pred_growth_percentage
      
      @classmethod
      def load_model(self):
          model_directory = Path(r"C:\Users\DELL\Documents\mortgage\MortgageAnalysis\AI\models")
          try:
              model_file_path = model_directory / 'mysarimax_instance.pkl'
              with open(model_file_path, 'rb') as f:
                  MySarimaxModel = pickle.load(f)
              print("Model loaded successfully.")
              return MySarimaxModel
          except FileNotFoundError:
              print("Model file not found. Please ensure the file exists.")
          except Exception as e:
              print(f"Error loading model: {e}")   
      
      @classmethod
      def save_model(self, model_file_path):
          try:
              model_file_path = model_directory / 'mysarimax_instance.pkl'
              with open(model_file_path, 'wb') as f:
                  pickle.dump(MySarimaxModel, f)
              print("Model saved successfully.")
          except Exception as e:
              print(f"Error saving model: {e}")
                  
def PlotResidual(vEn):
    std     = np.std(vEn)
    oKde    = gaussian_kde(vEn)
    pVal = sm.stats.acorr_ljungbox(vEn, lags=[40], return_df=False)["lb_pvalue"].item()  # White noise test
    
    # Create a 2-row subplot structure (1 column in first row, 2 columns in second row)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],  # First row spans both columns
               [{}, {}]],               # Second row has 2 subplots
        subplot_titles=(
            f'$\mu_e = {np.mean(vEn):.3f}$, $\sigma_e = {std:.3f}$',
            "Density Estimation",
            f"White noise test: $p = {pVal:.5f}$"
        )
    )

    # Plot error signal (stem plot approximation using scatter+lines) in (1,1)
    x = np.arange(len(vEn))
    fig.add_trace(go.Scatter(
        x=x, 
        y=vEn, 
        mode='markers+lines', 
        marker=dict(symbol='circle', size=6, color='blue'),
        line=dict(color='blue'),
        name='$e_n$'
    ), row=1, col=1)

    # Plot histogram and KDE on the second row, left subplot (2,1)
    xlim = (-3 * std, 3 * std)
    x_hist = np.linspace(xlim[0], xlim[1], 100)

    fig.add_trace(go.Histogram(
        x=vEn, 
        histnorm='probability density', 
        marker=dict(color='cyan', line=dict(color='black', width=1)),
        name='Hist',
        showlegend=True
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x_hist, 
        y=oKde(x_hist), 
        mode='lines', 
        name='KDE',
        line=dict(color='blue')
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x_hist,
        y=norm.pdf(x_hist, 0, std),
        mode='lines',
        name='$\mathcal{N}(0,\sigma^2)$',
        line=dict(color='red')
    ), row=2, col=1)

    # Plot auto-correlation function (ACF) on the second row, right subplot (2,2)
    acf_vals, confint = sm.tsa.acf(vEn, nlags=20, alpha=0.05)
    lags = np.arange(len(acf_vals))

    fig.add_trace(go.Scatter(
        x=lags, 
        y=acf_vals, 
        mode='lines+markers', 
        name='$\\hat{\\rho}_{e}[k]$',
        line=dict(color='green'),
        marker=dict(symbol='circle')
    ), row=2, col=2)

    # Add the confidence intervals as a filled area
    fig.add_trace(go.Scatter(
        x=np.concatenate([lags, lags[::-1]]),
        y=np.concatenate([confint[:, 0], confint[::-1, 1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ), row=2, col=2)

    # Update layout for the entire figure
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Residual Analysis",
        legend=dict(x=1, y=1),
        margin=dict(l=100, r=100, t=100, b=100)
    )

    # Customize individual axes for better readability
    fig.update_xaxes(title_text="$n$", row=1, col=1)
    fig.update_yaxes(title_text="Error Signal", row=1, col=1)
    
    fig.update_xaxes(title_text="Residual Value", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    
    fig.update_xaxes(title_text="$k$", row=2, col=2)
    fig.update_yaxes(title_text="ACF", row=2, col=2, range=[-1, 1])

    return fig

 
if __name__ == '__main__':
    save_model = True
    load_model = False
    
    if save_model: load_model = False     
    if load_model: save_model = False
        
    if not load_model:
       file_path = Path(r"C:\Users\DELL\Documents\mortgage\MortgageAnalysis\data_cpi.xlsx")   
       cpi_data_obj = LoadTable4CPI(file_path=file_path)
       
       MySarimaxModel =  MySARIMAX(cpi_data_obj=cpi_data_obj, partial_train_percentage=98, apply_model_fitting=False, use_plotly=True)
       MySarimaxModel.train()
       MySarimaxModel.predict(manual_predict = True, start='2004-08', end='2024-08', plot_results=False)
       MySarimaxModel.calc_cpi_growth_in_percentage(years = 20, plot_results=False)
    
    model_directory = Path(r"C:\Users\DELL\Documents\mortgage\MortgageAnalysis\AI\models")
    model_directory.mkdir(exist_ok=True)
    
    if save_model:
        try:
            model_file_path = model_directory / 'mysarimax_instance.pkl'
            with open(model_file_path, 'wb') as f:
                pickle.dump(MySarimaxModel, f)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        try:
            model_file_path = model_directory / 'mysarimax_instance.pkl'
            with open(model_file_path, 'rb') as f:
                MySarimaxModel = pickle.load(f)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model file not found. Please ensure the file exists.")
        except Exception as e:
            print(f"Error loading model: {e}")   
        
        MySarimaxModel.predict(manual_predict = True, start='2004-08', end='2024-08', plot_results=True)
        cpi_growth_percentage, dates, cpi_pred_growth_percentage = MySarimaxModel.calc_cpi_growth_in_percentage(years = 1, plot_results=True)




