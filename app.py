import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/flatly/bootstrap.min.css"
app = dash.Dash(external_stylesheets=[BS])
server = app.server

app.title = 'Proforma'

# Read the data
income = pd.read_csv('data/Olive_Devaud_Income.csv')
cost = pd.read_csv('data/Olive_Devaud_Cost.csv')

# Prepare the data
def lease_income(monthly_rent, units, occupancy=0.95):
    return monthly_rent * units * occupancy * 12

income['Yearly_income'] = lease_income(income.Monthly_lease, income.Units, income.Occupancy)
income = income.sort_values('Yearly_income', ascending=True)
income['Total'] = 'Annual Income'

grouped_cost = cost.groupby(by=['Categories']).sum()
grouped_cost = grouped_cost.sort_values('Cost')
grouped_cost = grouped_cost[grouped_cost['Cost'] > 0]

cost['Total'] = 'Total Cost'

# Make the plots
fig_income_bar = go.Figure(data=[go.Bar(
    y=income.Asset, x=income.Yearly_income,
    text=income.Yearly_income,
    textposition='auto',
    orientation='h'
    )])
# fig_income_bar.update_layout(title='Annual Income')
fig_income_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20))
fig_income_bar.update_xaxes(title='Income')

fig_income_sunburst = px.sunburst(income, path=['Total', 'Asset'], values='Yearly_income')
fig_income_sunburst.update_traces(textinfo='label+value+percent entry')

fig_grouped_cost_bar = go.Figure(data=[go.Bar(
    y=grouped_cost.index, x=grouped_cost.Cost,
    text=grouped_cost.Cost,
    textposition='auto',
    orientation='h'
    )])
# fig_grouped_cost_bar.update_layout(title='Cost')
fig_grouped_cost_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20))
fig_grouped_cost_bar.update_xaxes(title='Cost')

def plotly_sub_cost(df, cat):
    df_sub = df[df['Categories'] == cat].sort_values('Cost')
    fig = go.Figure(data=[go.Bar(
            y=df_sub.Details, x=df_sub.Cost,
            text=df_sub.Cost,
            textposition='auto',
            orientation='h'
        )])
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=20))
    fig.update_xaxes(title=f'Cost of {cat}')
    return fig

fig_cost_sunburst = px.sunburst(cost, path=['Total', 'Categories', 'Details'], values='Cost')
fig_cost_sunburst.update_traces(textinfo='label+value+percent entry')
fig_cost_sunburst.update_layout(title='Total Cost')

app.layout = html.Div([
    html.Img(src="https://i.ibb.co/qyddfCX/aaa7b154-f309-4610-a845-24d833c35a1e-200x200.png", 
            width='10%'),
    dbc.Col(width=2),
    dbc.Col([
        dbc.Row([
            dbc.Col(width=1.5),
            dbc.Col([
                html.H1('Investor Bridge', style={'textAlign': 'center'})
            ])
        ]),
        dbc.Row([
            # Need to auto
            dbc.Col([
                html.H3('Total Revenue:'),
                html.H3('23M')
            ]),
            dbc.Col([
                html.H3('Return:'),
                html.H3('230%')
            ]),
            dbc.Col([
                html.H3('Units:'),
                html.H3('90')
            ])
        ]),
        dbc.Row([
            # For the left buttons
            dbc.Col([
                dbc.Row(style={'height': '10vh'}),
                dbc.Row([
                    html.H5('Cost')
                ]),
                dbc.Row([
                    html.H5('Revenue')
                ]),
                dbc.Row([
                    html.H5('Building Plan')
                ]),
                dbc.Row([
                    html.H5('Other facts')
                ]),
                dbc.Row(style={'height': '40vh'}),
                dbc.Row([
                    html.H5('Upload data')
                ])
            ],width=2),
            dbc.Col([
                dbc.Row([
                    dcc.Graph(figure=fig_income_bar),
                ], style={'height': '25vh'}),
                dbc.Row([
                    dcc.Graph(figure=fig_grouped_cost_bar)
                ], style={'height': '25vh'}),
                dbc.Row([
                    dcc.Graph(figure=plotly_sub_cost(cost, 'Consultants'))
                ], style={'height': '25vh'})
            ], width=4.5),
            dbc.Col([
                dbc.Row([
                    dcc.Graph(figure=fig_cost_sunburst)
                ], style={'height': '85vh'})
            ], width=4)
        ])
    ]) 
])

if __name__ == '__main__':
    app.run_server(debug=True)