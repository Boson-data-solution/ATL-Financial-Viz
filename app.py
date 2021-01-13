import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import base64
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

BS = "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/flatly/bootstrap.min.css"
app = dash.Dash(external_stylesheets=[BS])
app.config.suppress_callback_exceptions = True
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
# fig_income_sunburst.update_layout(title='Total Income')
fig_income_sunburst.update_layout(margin=dict(l=20, r=20, t=10, b=20))
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
# fig_cost_sunburst.update_layout(title='Total Cost')
fig_cost_sunburst.update_layout(margin=dict(l=20, r=20, t=10, b=20))

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

upload_style = {
    'width': '200px',
    'height': '40px',
    'lineHeight': '40px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px'
    }


app.layout = html.Div([
    dbc.Col(width=2),
    dbc.Col([
        dbc.Row([
            dbc.Col([
                html.Img(src="https://i.ibb.co/qyddfCX/aaa7b154-f309-4610-a845-24d833c35a1e-200x200.png", 
            width='15%')
            ]),
            dbc.Col([
                html.H1('Investor Bridge', style={'textAlign': 'left'})
            ])
        ]),
        dbc.Row([
            # Need to auto
            dbc.Col([
                html.H3('Total Revenue:'),
                html.H3('$23M')
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
                dbc.Row([html.H5('Choose one:')]),
                dbc.Row([
                    dbc.RadioItems(
                        options=[
                            {'label': 'Revenue', 'value': 'Revenue'},
                            {'label': 'Cost', 'value': 'Cost'},
                            {'label': 'Building Plan', 'value': 'Plan'},
                            {'label': 'Other facts', 'value': 'Other'}
                        ],
                        value='Revenue', id='radioitem'
                    )
                ]),
                dbc.Row(style={'height': '10vh'}),
                dbc.Row([
                    html.H5('Upload data:'),
                    dcc.Upload(
                        id='upload-income',
                        children=html.Div([
                            'Drop or ',
                            html.A('Select Income')
                        ]),
                        style=upload_style
                    ),
                    dcc.Upload(
                        id='upload-cost',
                        children=html.Div([
                            'Drop or ',
                            html.A('Select Cost')
                        ]),
                        style=upload_style
                    ),
                    dcc.Upload(
                        id='upload-plan',
                        children=html.Div([
                            'Drop or ',
                            html.A('Select Building Plan')
                        ]),
                        style=upload_style
                    ),
                    dcc.Upload(
                        id='upload-other',
                        children=html.Div([
                            'Drop or ',
                            html.A('Select Other Facts')
                        ]),
                        style=upload_style
                    )
                ])
            ],width=2),
            dbc.Col(width=4.5, id='col2'),
            dbc.Col(width=4, id='col3')
        ])
    ]) 
])


@app.callback(Output('col2', 'children'),
              [Input('radioitem', 'value')])
def render_col2(value):
    if value == 'Revenue':
        col = dbc.Col([
                dbc.Row([
                    dcc.Graph(figure=fig_income_bar),
                ], style={'height': '70vh'}, id='income_bar')
            ], width=4.5)
    elif value == 'Cost':
        col = dbc.Col([
                dbc.Row([
                    dcc.Graph(figure=fig_grouped_cost_bar, style={'height': '35vh'}),
                ], style={'height': '35vh'}),
                dbc.Row([
                    dcc.Graph(figure=plotly_sub_cost(cost, 'Consultants')),
                ], style={'height': '35vh'})
            ], width=4.5, id='cost_bar')
    elif value == 'Plan':
        col = dbc.Col(width=4.5)
    else:
        col = dbc.Col(width=4.5)
    return col


@app.callback(Output('col3', 'children'),
              [Input('radioitem', 'value')])
def render_col3(value):
    if value == 'Revenue':
        col = dbc.Col([
            dbc.Row([
                dcc.Graph(figure=fig_income_sunburst)
                ], style={'height': '65vh'}, id='income_sunburst')
                ], width=4)
    elif value == 'Cost':
        col = dbc.Col([
            dbc.Row([
                dcc.Graph(figure=fig_cost_sunburst)
                ], style={'height': '65vh'}, id='cost_sunburst')
                ], width=4)
    elif value == 'Plan':
        col = dbc.Col(width=4)
    else:
        col = dbc.Col(width=4)
    return col

@app.callback(Output('income_bar', 'children'),
              Input('upload-income', 'contents'),
              State('upload-income', 'filename'))
def update_income_bar(contents, filename):
    if contents:
        income = parse_contents(contents, filename)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    income['Yearly_income'] = lease_income(income.Monthly_lease, income.Units, income.Occupancy)
    income = income.sort_values('Yearly_income', ascending=True)
    income['Total'] = 'Annual Income'

    fig_income_bar = go.Figure(data=[go.Bar(
        y=income.Asset, x=income.Yearly_income,
        text=income.Yearly_income,
        textposition='auto',
        orientation='h'
    )])
    fig_income_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20))
    fig_income_bar.update_xaxes(title='Income')

    return dbc.Row([
        dcc.Graph(figure=fig_income_bar)
        ], style={'height': '70vh'})

@app.callback(Output('income_sunburst', 'children'),
              Input('upload-income', 'contents'),
              State('upload-income', 'filename'))
def update_income_sunburst(contents, filename):
    if contents:
        income = parse_contents(contents, filename)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    income['Yearly_income'] = lease_income(income.Monthly_lease, income.Units, income.Occupancy)
    income = income.sort_values('Yearly_income', ascending=True)
    income['Total'] = 'Annual Income'

    fig_income_sunburst = px.sunburst(income, path=['Total', 'Asset'], values='Yearly_income')
    fig_income_sunburst.update_layout(margin=dict(l=20, r=20, t=10, b=20))
    fig_income_sunburst.update_traces(textinfo='label+value+percent entry')

    return dbc.Row([
        dcc.Graph(figure=fig_income_sunburst)
        ], style={'height': '65vh'})

@app.callback(Output('cost_bar', 'children'),
              Input('upload-cost', 'contents'),
              State('upload-cost', 'filename'))
def update_cost_bar(contents, filename):
    if contents:
        cost = parse_contents(contents, filename)
    else:
        cost = pd.read_csv('data/Olive_Devaud_Cost.csv')

    grouped_cost = cost.groupby(by=['Categories']).sum()
    grouped_cost = grouped_cost.sort_values('Cost')
    grouped_cost = grouped_cost[grouped_cost['Cost'] > 0]
    cost['Total'] = 'Total Cost'

    fig_grouped_cost_bar = go.Figure(data=[go.Bar(
        y=grouped_cost.index, x=grouped_cost.Cost,
        text=grouped_cost.Cost,
        textposition='auto',
        orientation='h'
        )])
    fig_grouped_cost_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20))
    fig_grouped_cost_bar.update_xaxes(title='Cost')

    return dbc.Col([
                dbc.Row([
                    dcc.Graph(figure=fig_grouped_cost_bar, style={'height': '35vh'}),
                ], style={'height': '35vh'}),
                dbc.Row([
                    dcc.Graph(figure=plotly_sub_cost(cost, 'Consultants')),
                ], style={'height': '35vh'})
            ], width=4.5)

@app.callback(Output('cost_sunburst', 'children'),
              Input('upload-cost', 'contents'),
              State('upload-cost', 'filename'))
def update_cost_bar(contents, filename):
    if contents:
        cost = parse_contents(contents, filename)
    else:
        cost = pd.read_csv('data/Olive_Devaud_Cost.csv')

    grouped_cost = cost.groupby(by=['Categories']).sum()
    grouped_cost = grouped_cost.sort_values('Cost')
    grouped_cost = grouped_cost[grouped_cost['Cost'] > 0]
    cost['Total'] = 'Total Cost'

    fig_cost_sunburst = px.sunburst(cost, path=['Total', 'Categories', 'Details'], values='Cost')
    fig_cost_sunburst.update_traces(textinfo='label+value+percent entry')
    fig_cost_sunburst.update_layout(margin=dict(l=20, r=20, t=10, b=20))

    return dbc.Row([
        dcc.Graph(figure=fig_cost_sunburst)
        ], style={'height': '65vh'})


if __name__ == '__main__':
    app.run_server(debug=True)