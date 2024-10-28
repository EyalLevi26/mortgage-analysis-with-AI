__all__ = ['plot_tables_and_figures']

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utills import create_table_for_bank_and_risk
import matplotlib.pyplot as plt

def plot_tables_and_figures(traces_payback, 
                            traces_payback_rate, 
                            extenal_plot, 
                            bank_name_for_plot, 
                            mortgage_per_risk_per_bank, 
                            mortgage_amount_nis):
    
    figs_tmp = []   
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

    figs_tmp.append(fig)

    if extenal_plot:
        fig.show()

    fig = go.Figure(data=traces_payback, layout=layout_payback)
    figs_tmp.append(fig)

    if extenal_plot:
        fig.show()

    fig = go.Figure(data=traces_payback_rate, layout=layout_payback_rate)
    figs_tmp.append(fig)

    if extenal_plot:
        fig.show()

    loan_table, table_shpizer, table_keren_shava, figs = create_table_for_bank_and_risk(risk_level='Our_Mortgage', 
                                                    bank_name = bank_name_for_plot, 
                                                    mortgage_per_risk_per_bank = mortgage_per_risk_per_bank,
                                                    mortgage_amount_nis = mortgage_amount_nis)
    if extenal_plot:
        loan_table.show()
        for fig in figs:
            fig.show()
        table_shpizer.show()
        table_keren_shava.show()

    figs = figs_tmp + figs
    return [loan_table, table_shpizer, table_keren_shava, figs]