import dash
import dash_table
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


def prepare_income(income, occupancy=0.95):
    income['Annual_income'] = lease_income(income.Monthly_lease, income.Units, occupancy)
    income = income.sort_values('Annual_income', ascending=True)
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
        y=income.Asset, x=income.Annual_income,
        text=income.Annual_income,
        textposition='auto',
        orientation='h'
        )])
    fig_income_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20))
    fig_income_bar.update_xaxes(title='Income')
    return fig_income_bar


def plot_income_sunburst(income):
    fig_income_sunburst = px.sunburst(income, path=['Total', 'Asset'], values='Annual_income',
    title="Income Breakdown")
    fig_income_sunburst.update_layout(margin=dict(l=20, r=20, t=40, b=20))
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
    fig_cost_sunburst = px.sunburst(cost, path=['Total', 'Categories', 'Details'], values='Cost',
    title="Cost Breakdown")
    fig_cost_sunburst.update_traces(textinfo='label+value+percent entry')
    fig_cost_sunburst.update_layout(margin=dict(l=20, r=20, t=40, b=20))
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
    'width': '180px',
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
                html.P("—  Visualized Proforma",className="display-5")
                ]),
                dbc.Col(width=4)]),
            dbc.Row([
                dbc.Col([
                    html.H4('Asset Value:',style={'color': 'black'}),
                    dbc.Row(id='total_asset')
                ]),
                dbc.Col([
                    html.H4('Return on Cost:',style={'color': 'black'}),
                    dbc.Row(id='total_return')
                ]),
                dbc.Col([
                    html.H4('Total Units:',style={'color': 'black'}),
                    dbc.Row(id='total_units')
                ])
            ], style={'color': '#22a40c'})
            ],
            fluid=True,
        )
    ],
    fluid=False, style={'padding': '0.5rem 0.5rem'}
)

app.layout = html.Div([
    jumbotron,
    dbc.Col(width=2),
    dbc.Col([
        # Sliders
        dbc.Row([
            dbc.Col([
                html.H5('Building Occupancy:'),
                dcc.Slider(
                    id='occupancy_slider',
                    min=0.5,
                    max=1.00001,
                    step=0.01,
                    value=0.95,
                    marks={occupancy: f'{round(occupancy*100)}%' for occupancy in np.arange(0.5, 1.1, 0.1)}
                ),
                html.H6(id='occupancy_output')
            ]),
            dbc.Col([
                html.H5('Profit Rate:'),
                dcc.Slider(
                    id='profit_rate_slider',
                    min=0.3,
                    max=0.8001,
                    step=0.01,
                    value=0.5,
                    marks={rate: f'{int(rate*100)}%' for rate in np.arange(0.3, 0.9, 0.1)}
                ),
                html.H6(id='profit_rate_output')
            ]),
            dbc.Col([
                html.H5('Estimate factor:'),
                dcc.Slider(
                    id='interest_rate_slider',
                    min=0.03,
                    max=0.1,
                    step=0.001,
                    value=0.08,
                    marks={rate: f'{round(rate*100,1)}%' for rate in np.arange(0.03, 0.11, 0.01)}
                ),
                html.H6(id='interest_rate_output')
            ],style={'height': '18vh'})
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Row([html.H5('Insight for:')]),
                dbc.Row([
                    dbc.RadioItems(
                        options=[
                            {'label': 'Revenue', 'value': 'Revenue'},
                            {'label': 'Cost', 'value': 'Cost'},
                            {'label': 'Other facts', 'value': 'Other'}
                        ],
                        value='Revenue', id='radioitem'
                    )
                ]),
                dbc.Row(style={'height': '15vh'}),
                dbc.Col([
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


@app.callback(
    Output('occupancy_output', 'children'),
    Input('occupancy_slider', 'value'))
def update_occupancy(occupancy):
    return f'You have selected {int(occupancy*100)}%.'


@app.callback(
    Output('profit_rate_output', 'children'),
    Input('profit_rate_slider', 'value'))
def update_profit_rate(rate):
    return f'You have selected {int(rate*100)}%.'


@app.callback(
    Output('interest_rate_output', 'children'),
    Input('interest_rate_slider', 'value'))
def update_interest_rate(rate):
    return f'You have selected {round(rate*100, 1)}%.'


@app.callback(Output('col2', 'children'),
              [Input('radioitem', 'value')])
def render_col2(value):
    if value == 'Revenue':
        col = dbc.Col([dbc.Row(id='income_bar')], width=4)
    elif value == 'Cost':
        col = dbc.Col(id='cost_bar')
    else:
        col = dbc.Col([
            dbc.Row(style={'height': '5vh'}),
            dbc.Row([
                dbc.Col(html.Br(), width=3),
                dbc.Col(id='other_fact_table')
            ])
        ])
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
    else:
        col = dbc.Col()
    return col


@app.callback(Output('income_bar', 'children'),
              Input('upload-income', 'contents'),
              Input('occupancy_slider', 'value'),
              State('upload-income', 'filename'))
def update_income_bar(contents, occupancy, filename):
    if contents:
        income = parse_contents(contents, filename)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    income = prepare_income(income, occupancy)

    fig_income_bar = plot_income_bar(income)

    return dbc.Row([dcc.Graph(figure=fig_income_bar)], style={'height': '70vh'})


@app.callback(Output('income_sunburst', 'children'),
              Input('upload-income', 'contents'),
              Input('occupancy_slider', 'value'),
              State('upload-income', 'filename'))
def update_income_sunburst(contents, occupancy, filename):
    if contents:
        income = parse_contents(contents, filename)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    income = prepare_income(income, occupancy)

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


@app.callback(Output('other_fact_table', 'children'),
              Input('upload-other', 'contents'),
              State('upload-other', 'filename'))
def update_income_bar(contents, filename):
    if contents:
        other = parse_contents(contents, filename)
    else:
        other = pd.read_csv('data/other_fact.csv')
    
    t = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in other.columns],
            data=other.to_dict('records'),
            style_cell={
                'minWidth': '300px', 
                'font_size': '20px'}
        )
    
    return t


@app.callback(Output('total_asset', 'children'),
              Input('upload-income', 'contents'),
              Input('occupancy_slider', 'value'),
              Input('profit_rate_slider', 'value'),
              Input('interest_rate_slider', 'value'),
              State('upload-income', 'filename'))
def update_total_asset(contents, occupancy, profit_rate, interest_rate, filename):
    if contents:
        income = parse_contents(contents, filename)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    income = prepare_income(income, occupancy)
    total_asset = (income.Annual_income * profit_rate).sum() / interest_rate

    return html.H5(f'${round(total_asset / 1000000, 2)}M')


@app.callback(Output('total_units', 'children'),
              Input('upload-income', 'contents'),
              State('upload-income', 'filename'))
def update_total_units(contents, filename):
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
              Input('occupancy_slider', 'value'),
              Input('profit_rate_slider', 'value'),
              Input('interest_rate_slider', 'value'),
              State('upload-income', 'filename'),
              State('upload-cost', 'filename'))
def update_total_return(contents1, contents2,  occupancy, profit_rate, interest_rate, filename1, filename2):
    if contents1:
        income = parse_contents(contents1, filename1)
    else:
        income = pd.read_csv('data/Olive_Devaud_Income.csv')
    if contents2:
        cost = parse_contents(contents2, filename2)
    else:
        cost = pd.read_csv('data/Olive_Devaud_Cost.csv')
    
    income = prepare_income(income, occupancy)
    total_asset = (income.Annual_income * profit_rate).sum() / interest_rate

    total_cost = cost.Cost.sum()
    
    return html.H5(f'{int((total_asset - total_cost) / total_cost * 100)}%')


if __name__ == '__main__':
    app.run_server(debug=True)
