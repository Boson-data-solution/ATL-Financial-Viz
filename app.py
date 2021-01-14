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

# Prepare the data
def lease_income(monthly_rent, units, occupancy=0.95):
    return monthly_rent * units * occupancy * 12


def prepare_income(income):
    income['Yearly_income'] = lease_income(income.Monthly_lease, income.Units, income.Occupancy)
    income = income.sort_values('Yearly_income', ascending=True)
    income['Total'] = 'Annual Income'
    return income


def prepare_group_cost(cost):
    grouped_cost = cost.groupby(by=['Categories']).sum()
    grouped_cost = grouped_cost.sort_values('Cost')
    grouped_cost = grouped_cost[grouped_cost['Cost'] > 0]
    return grouped_cost


def prepare_cost(cost):
    cost['Total'] = 'Total Cost'
    return cost


# Make the plots
def plot_income_bar(income):
    fig_income_bar = go.Figure(data=[go.Bar(
        y=income.Asset, x=income.Yearly_income,
        text=income.Yearly_income,
        textposition='auto',
        orientation='h'
        )])
    fig_income_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20))
    fig_income_bar.update_xaxes(title='Income')
    return fig_income_bar


def plot_income_sunburst(income):
    fig_income_sunburst = px.sunburst(income, path=['Total', 'Asset'], values='Yearly_income')
    fig_income_sunburst.update_layout(margin=dict(l=20, r=20, t=10, b=20))
    fig_income_sunburst.update_traces(textinfo='label+value+percent entry')
    return fig_income_sunburst


def plot_cost_bar(grouped_cost):
    fig_grouped_cost_bar = go.Figure(data=[go.Bar(
        y=grouped_cost.index, x=grouped_cost.Cost,
        text=grouped_cost.Cost,
        textposition='auto',
        orientation='h'
        )])
    fig_grouped_cost_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20))
    fig_grouped_cost_bar.update_xaxes(title='Cost')
    return fig_grouped_cost_bar


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


def plot_cost_sunburst(cost):
    fig_cost_sunburst = px.sunburst(cost, path=['Total', 'Categories', 'Details'], values='Cost')
    fig_cost_sunburst.update_traces(textinfo='label+value+percent entry')
    fig_cost_sunburst.update_layout(margin=dict(l=20, r=20, t=10, b=20))
    return fig_cost_sunburst


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

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 240,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#c9bdb9",
}

jumbotron = dbc.Jumbotron(
    [
        dbc.Container(
            [dbc.Row([
               dbc.Col([html.Img(src="https://i.ibb.co/qyddfCX/aaa7b154-f309-4610-a845-24d833c35a1e-200x200.png", 
            width='15%')]),
                dbc.Col([
                html.P("Investor Bridge", className="display-4"),
                html.P("Visualized Proforma",className="display-5")
                ]),
                dbc.Col(width=4)]),
                  
            ],
            fluid=True,
        )
    ],
    fluid=False,
)

app.layout = html.Div([
    jumbotron,
    dbc.Col([
        dbc.Row([
            # For the left buttons
            dbc.Col([
                dbc.Col([html.H5('Choose one:')]),
                dbc.Col([
                    dbc.RadioItems(
                        options=[
                            {'label': 'Revenue', 'value': 'Revenue'},
                            {'label': 'Cost', 'value': 'Cost'},
                            {'label': 'Other facts', 'value': 'Other'}
                        ],
                        value='Revenue', id='radioitem'
                    )
                ]),
                dbc.Col(style={'height': '10vh'}),
                dbc.Col([
                    html.H5('Upload data:\n'),
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
                        id='upload-other',
                        children=html.Div([
                            'Drop or ',
                            html.A('Select Other Facts')
                        ]),
                        style=upload_style
                    )
                ]),
                dbc.Col([
                    # Need to auto
                    dbc.Col([
                        html.P('Estimated Asset:',style={'color': 'blue'}),
                        dbc.Row(id='total_asset')
                    ]),
                    dbc.Col([
                        html.P('Return:',style={'color': 'blue'}),
                        dbc.Row(id='total_return')
                    ]),
                    dbc.Col([
                        html.P('Units:',style={'color': 'blue'}),
                        dbc.Row(id='total_units')
                    ])
                ]),
            ],width=200),
            dbc.Col( id='col2'),
            dbc.Col( id='col3')
        ])
    ]) 
])


@app.callback(Output('col2', 'children'),
              [Input('radioitem', 'value')])
def render_col2(value):
    if value == 'Revenue':
        col = dbc.Col([dbc.Row(id='income_bar')], width=4.5)
    elif value == 'Cost':
        col = dbc.Col(id='cost_bar')
    elif value == 'Plan':
        col = dbc.Col(width=4.5)
    else:
        col = dbc.Col(width=4.5)
    return col


@app.callback(Output('col3', 'children'),
              [Input('radioitem', 'value')])
def render_col3(value):
    if value == 'Revenue':
        col = dbc.Col([dbc.Row(id='income_sunburst')], width=4)
    elif value == 'Cost':
        col = dbc.Col([
            dbc.Row(id='cost_sunburst')
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
    income = prepare_income(income)

    fig_income_bar = plot_income_bar(income)

    return dbc.Row([dcc.Graph(figure=fig_income_bar)], style={'height': '70vh'})


@app.callback(Output('income_sunburst', 'children'),
              Input('upload-income', 'contents'),
              State('upload-income', 'filename'))
def update_income_sunburst(contents, filename):
    if contents:
        income = parse_contents(contents, filename)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    income = prepare_income(income)

    fig_income_sunburst = plot_income_sunburst(income)

    return dbc.Row([dcc.Graph(figure=fig_income_sunburst)], style={'height': '65vh'})


@app.callback(Output('cost_bar', 'children'),
              Input('upload-cost', 'contents'),
              State('upload-cost', 'filename'))
def update_cost_bar(contents, filename):
    if contents:
        cost = parse_contents(contents, filename)
    else:
        cost = pd.read_csv('data/Olive_Devaud_Cost.csv')

    grouped_cost = prepare_group_cost(cost)
    cost = prepare_cost(cost)

    fig_grouped_cost_bar = plot_cost_bar(grouped_cost)

    col = dbc.Col([
            dbc.Row([
                dcc.Graph(figure=fig_grouped_cost_bar, style={'height': '35vh'}),
                ], style={'height': '35vh'}),
            dbc.Row([
                dcc.Graph(figure=plotly_sub_cost(cost, 'Consultants')),
                ], style={'height': '35vh'})
            ], width=4.5)
    return col


@app.callback(Output('cost_sunburst', 'children'),
              Input('upload-cost', 'contents'),
              State('upload-cost', 'filename'))
def update_cost_bar(contents, filename):
    if contents:
        cost = parse_contents(contents, filename)
    else:
        cost = pd.read_csv('data/Olive_Devaud_Cost.csv')

    cost = prepare_cost(cost)

    fig_cost_sunburst = plot_cost_sunburst(cost)

    return dbc.Row([dcc.Graph(figure=fig_cost_sunburst)], style={'height': '65vh'})


@app.callback(Output('total_asset', 'children'),
              Input('upload-income', 'contents'),
              State('upload-income', 'filename'))
def update_income_sunburst(contents, filename, interest_rate=0.08):
    if contents:
        income = parse_contents(contents, filename)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    income = prepare_income(income)
    total_asset = (income.Yearly_income * income.Revenue_rate).sum() / interest_rate

    return html.H5(f'${round(total_asset / 1000000, 2)}M')


@app.callback(Output('total_units', 'children'),
              Input('upload-income', 'contents'),
              State('upload-income', 'filename'))
def update_income_sunburst(contents, filename):
    if contents:
        income = parse_contents(contents, filename)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    living = ['Independent_living', 'Assited_living']
    total_units = income[income['Asset'].isin(living)].Units.sum()

    return html.H5(str(int(total_units)))

@app.callback(Output('total_return', 'children'),
              Input('upload-income', 'contents'),
              Input('upload-cost', 'contents'),
              State('upload-income', 'filename'),
              State('upload-cost', 'filename'))
def update_income_sunburst(contents1, contents2, filename1, filename2, interest_rate=0.08):
    if contents1:
        income = parse_contents(contents1, filename1)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    if contents2:
        cost = parse_contents(contents2, filename2)
    else:
        cost = pd.read_csv('data/Olive_Devaud_Cost.csv')
    
    income = prepare_income(income)
    total_asset = (income.Yearly_income * income.Revenue_rate).sum() / interest_rate

    total_cost = cost.Cost.sum()
    
    return html.H5(f'{int((total_asset - total_cost) / total_cost * 100)}%')


if __name__ == '__main__':
    app.run_server(debug=True)
