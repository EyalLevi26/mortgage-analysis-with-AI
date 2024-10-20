
__all__ = ['LoadTable4CPI', 'MySARIMAX']

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

import plotly.graph_objects as go

from scipy.stats                   import gaussian_kde, norm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools     import kpss
from statsmodels.stats.diagnostic  import acorr_ljungbox
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


class LoadTable4CPI:
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
            self.cpi_values_df[DateTimeCol] = pd.to_datetime(self.cpi_values_df[DateTimeCol], format='%Y-%m')
        except ValueError as ve:
              print(f"Value error: {ve}")
              return
        except Exception as e:
               print(f"Error processing data: {e}")
               return
        
class MySARIMAX:
      def __init__(self, cpi_data_obj: LoadTable4CPI, partial_train_percentage: float = 85, apply_model_fitting: bool =False,  use_plotly: bool=True):
         self.cpi_values                         = cpi_data_obj.cpi_values_df[cpi_data_obj.cpi_values_df.columns[1]].values
         self.dates_cpi_values                   = cpi_data_obj.cpi_values_df[cpi_data_obj.cpi_values_df.columns[0]].dt.strftime('%Y-%m-%d').values
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
             self.model_fitting()

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

      def model_fitting(self):
          self._preprocess()
          self.check_adf(print_result=True)

          vD = np.arange(4)
          vd = np.arange(4)
          vp = np.arange(4)
          vq = np.arange(7)
          vP = np.arange(2)
          vQ = np.arange(2)
          vS = np.array([12])
          T  = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D','Q', 'S', 'AIC', 'BIC'])
          for pp in vp:
              for qq in vq:
                  for dd in vd:
                      for PP in vP:
                          for DD in vD:
                              for QQ in vQ:
                                  for S in vS:
                                      try:
                                          oModel = sarimax.SARIMAX(self.data_train_vX, 
                                                                   order=(pp, dd, qq), 
                                                                   seasonal_order=(PP, DD, QQ, S), 
                                                                   trend='c').fit(maxiter=100)  # Increased maxiter and changed method
                                          T.loc[len(T)] = [pp, dd, qq, PP, DD, QQ, S, oModel.aic, oModel.bic]
                                      except (ValueError, np.linalg.LinAlgError) as e:
                                          print(f"Skipped combination (p={pp}, q={qq}, P={PP}, Q={QQ}, S={S}) due to error: {e}")

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
                  
      def predict(self, start=None, steps_ahead=None, return_conf_int=True, plot_results = True):
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
        # Default the start index to the beginning of the test data
        if start is None:
            start = int(len(self.data_train_vX)/20)

        # Default steps_ahead to cover the entire test set if not specified
        if steps_ahead is None:

            steps_ahead = int(len(self.data_test_vX) / 20)

        # Get predictions from the trained SARIMAX model
        prediction = oTestModel.get_prediction(start, self.n_cpi_values - 1 + steps_ahead, len(self.data_train_vX) + start)
        pred_mean  = prediction.predicted_mean
        pred_mean  = np.exp(pred_mean)
        vPredIdx   = np.arange(start, self.n_cpi_values + steps_ahead) 

        if return_conf_int:
            pred_conf_int = prediction.conf_int(alpha=0.05)
            
            pred_conf_int[:,0] = np.exp(pred_conf_int[:,0])
            pred_conf_int[:,1] = np.exp(pred_conf_int[:,1])
            if plot_results:  
               plot_data = {
                            'pred_mean': pred_mean,
                            'vPredIdx': vPredIdx,
                            'pred_conf_int': pred_conf_int
                            }
               self.visualize_results(plot_data)  

            return pred_mean, pred_conf_int
        else:
            if plot_results:  
               plot_data = {
                            'pred_mean': pred_mean,
                            'vPredIdx': vPredIdx,
                            'pred_conf_int': None
                            }
               self.visualize_results(plot_data)  
            return pred_mean
      
      def visualize_results(self, plot_data):
           pred_mean     = plot_data['pred_mean']
           pred_conf_int = plot_data['pred_conf_int']
           vPredIdx      = plot_data['vPredIdx']
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
                  x=self.vIdx_cpi_values[:n_train],
                  y=self.data_train_vX,
                  mode='lines',
                  name='train',
                #   hovertemplate='Month: %{x}<br>Index Value: %{y:.2f}<extra></extra>' ,
                  line=dict(color='blue', width=1.5)
              ))
              
              # Plot test data
              fig.add_trace(go.Scatter(
                  x=self.vIdx_cpi_values[n_train:],
                  y=self.data_test_vX,
                  mode='lines',
                  name='test',
                  line=dict(color='orange', width=1.5)
              ))
              
              # Plot prediction data
              fig.add_trace(go.Scatter(
                  x=vPredIdx,
                  y=pred_mean,
                  mode='lines',
                  name='$\hat{x}_n$',
                  line=dict(color='red', width=1.5)
              ))
              
              if pred_conf_int is not None:
                 # Add fill for confidence interval (95% CI)
                 fig.add_trace(go.Scatter(
                     x=np.concatenate([vPredIdx, vPredIdx[::-1]]),
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
                      tickvals=self.vIdx_cpi_values[6::80],  # Adjust xticks as per original
                      ticktext=self.dates_cpi_values[6::80]
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
   file_path = Path(r"C:\Users\DELL\Documents\mortgage\MortgageAnalysis\data_cpi.xlsx")   
   cpi_data_obj = LoadTable4CPI(file_path=file_path)
   
   MyModel =  MySARIMAX(cpi_data_obj=cpi_data_obj, partial_train_percentage=85, apply_model_fitting=False, use_plotly=True)
   MyModel.train()
   MyModel.predict()

   a=1




