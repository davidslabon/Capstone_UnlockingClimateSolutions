import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash
from dash.dependencies import Input, Output

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
server = app.server

logo = "https://www.flaticon.com/svg/static/icons/svg/3749/3749877.svg"
cs = [[0, "rgb(207, 219, 206)"],[1, "rgb(53, 66, 52)"]]

# ------------------------------------------------------------------------------
# Import and clean data (importing pkl into pandas)
df = pd.read_pickle("dummy_df.pkl")


# main dataframe 
df = df.groupby(["country", "alpha3code", "type", "year"])[[
                                                            'total', 
                                                            's_score', 
                                                            'c_score', 
                                                            'o_score', 
                                                            'r_score', 
                                                            'e_score', 
                                                            'e2_score']
                                                            ].mean()
df.reset_index(inplace=True)
print(df.info())

# dashtable dataframe 
tdf = pd.read_pickle("dummy_df.pkl")
tdf = tdf.loc[:,["entity", "s_score", "c_score", "o_score", "r_score", "e_score", "year", "country", "type", "total"]]
tdf.sort_values(by="total", ascending=False, inplace=True)
utdf = tdf.copy()

# radar datamframe
rdf = pd.read_pickle("dummy_df.pkl")
rdf = rdf.groupby("e_score")[[
                            's_score', 
                            'c_score', 
                            'o_score', 
                            'r_score', 
                            'e2_score']
                            ].mean()
rdf.reset_index(inplace=True)
print(rdf.info())

# ------------------------------------------------------------------------------
# App layout


app.layout = html.Div([
    dbc.Navbar([
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row([
                dbc.Col(html.Img(src=logo, height="40px")),
                dbc.Col(dbc.NavbarBrand("Social Collaborative Opportunity Risk Engagement Index", className="ml-2")),
            ],
                align="left",
                no_gutters=True,
            ),
            href="https://www.kaggle.com/c/cdp-unlocking-climate-solutions/",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Row([
                    dbc.Col(dbc.Input(type="search", placeholder="Search")),
                    dbc.Col(
                        dbc.Button("Search", color="secondary", className="btn btn-primary disabled"),
                        width="auto",
                    ),
                ],
                no_gutters=True,
                className="ml-auto flex-nowrap mt-3 mt-md-0",
                align="left",
                ),
                id="navbar-collapse", 
                navbar=True
            ),
    ],
        color="beige",
        light=True,
    ),
    dbc.Row([
        html.Br(),
        html.Div(children = """This interaktiv dashboards allows users to explore the 
                            capstone results of Felix, Olaf, Tobi and David. For further 
                            information please visit us on GitHub: 
                            https://github.com/davidslabon/Capstone_UnlockingClimateSolutions."""),
        html.Br()
    ]),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='slct_type', 
            placeholder='Select a type',
            options=[{'label': 'Cities', 'value': 'cid'},
                    {'label': 'Corporates', 'value': 'cod'}]
            ),
        width={'size': 5, "offset": 0, 'order': 1}
        ),
        dbc.Col(dcc.Dropdown(
            id="slct_country", 
            placeholder='Select a country',
            options=[{"label":x, "value":x} for x in df.country.unique()],
            multi=False,
            value="",
            ),
        width={'size': 5, "offset": 0, 'order': 2}
        ),
        dbc.Col(dcc.Dropdown(
            id="slct_year", 
            placeholder='Select a year',
            options=[{"label": "2018", "value": "2018"},
                    {"label": "2019", "value": "2019"},
                    {"label": "2020", "value": "2020"}],
            multi=False,
            value="",
            ),
        width={'size': 2,  "offset": 0, 'order': 3}
        ),
    ], 
    no_gutters=True
    ),
    dbc.Row(
        html.Br()
    ),
    dbc.Row([    
        dbc.Col([
            html.H4(children="Score Distribution"),
            dcc.Graph(id='score_bar', figure={}),
                    ],
                    width=5, lg={'size': 5,  "offset": 0, 'order': 'first'}
            ),
        dbc.Col([
            html.H4(children="TOP 10 entities"),
            html.Br(),
            html.Br(),
            html.Div([
                    dash_table.DataTable(
                        id='datatable-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": False, "selectable": True, "hideable": True}
                            if i == "alpha3code" or i == "year" or i == "type"
                            else {"name": i.title().replace("_"," "), "id": i, "deletable": True, "selectable": True}
                            for i in ["entity", "s_score", "c_score", "o_score", "r_score", "e_score"]
                        ],
                        data=tdf.to_dict('records'),  # the contents of the table
                        editable=True,              # allow editing of data inside all cells
                        #filter_action="native",     # allow filtering of data by user ('native') or not ('none')
                        sort_action="native",       # enables data to be sorted per-column by user or not ('none')
                        #sort_mode="single",         # sort across 'multi' or 'single' columns
                        column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                        #row_selectable="multi",     # allow users to select 'multi' or 'single' rows
                        #row_deletable=True,         # choose if user can delete a row (True) or not (False)
                        selected_columns=[],        # ids of columns that user selects
                        selected_rows=[],           # indices of rows that user selects
                        page_action="native",       # all data is passed to the table up-front or not ('none')
                        page_current=0,             # page number that user is on
                        page_size=10,                # number of rows visible per page
                        style_cell={                # ensure adequate header width when text is shorter than cell's text
                            'minWidth': 10, 
                            'maxWidth': 95, 
                            'width': 40, 
                            "textAlign": "left",
                            'font-family':'sans-serif'
                        },
                        style_cell_conditional=[    # align text columns to left. By default they are aligned to right
                            {
                            'if': {'column_id': c},
                            'minWidth': 30, 
                            'maxWidth': 95, 
                            'width': 95, 
                            'textAlign': 'left'
                            } for c in ['entity']
                        ],
                        style_data={                # overflow cells' content into multiple lines
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'margin':
                            {"r":"30"}
                        },
                        style_as_list_view=True,
                    ),
                ]),
        ],
        width=7, lg={'size': 7,  "offset": 0}
        ),
    ]),
    dbc.Row(
        dbc.Col([
            html.H4(children="Average Score per Country"),
            dcc.Graph(id='avg_score_map', figure={}),
        ],
        width=12, lg={'size': 12,  "offset": 0, 'order': 'first'}
        ),
    ),
    dbc.Row([
        dbc.Col([
            html.H4(children="Engagement Score"),
            dcc.Slider(id="slct_elvl", 
                min=1,
                max=5,
                value=3,
                step=1,
                marks={
                    1: {'label': 'not engaged', 'style': {'color': "rgb(53, 66, 52)"}},
                    3: {'label': 'balanced', 'style': {'color': "rgb(53, 66, 52)"}},
                    5: {'label': 'very engaged', 'style': {'color': "rgb(53, 66, 52)"}}
                    },
                included=False
            ),
        ],
        width=4, lg={'size': 4,  "offset": 0, 'order': 'first'}
        ),
        dbc.Col([
                html.H4(children="Score Radar"),
                dcc.Graph(id="score_radar", figure={}),
            ],
            width=8, lg={'size': 8,  "offset": 0, 'order': 'last'}
            )
    ])
])




# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='avg_score_map', component_property='figure'),
     Output(component_id='score_bar', component_property='figure')],
    [Input(component_id='slct_type', component_property='value'),
     Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_year', component_property='value')]
)

def update_graph(*option_slctd):
    print(option_slctd)
    # print(type(option_slctd))

 
    dff = df.copy()
    if option_slctd[2]:
        dff = dff[dff["year"] == option_slctd[2]] 
    if option_slctd[1]:
        dff = dff[dff["country"] == option_slctd[1]] 
    if option_slctd[0]:
        dff = dff[dff["type"] == option_slctd[0]] 

    # plotly go
    fig_map = go.Figure(
        data=[go.Choropleth(
            locations=dff["alpha3code"],
            z=dff["total"],
            colorscale=cs,
        )],
    )

    fig_map.update_layout(
        margin=dict(
                    l=0,
                    r=0,
                    b=10,
                    t=10,
                    pad=10
                    ),
            #paper_bgcolor="LightSteelBlue",
        #colorscale="viridis"
    )

    fig_bar = go.Figure(
        data=[
            go.Bar(name="S", x=dff["year"], y=dff["s_score"], marker_color="rgb(207, 219, 206)"),
            go.Bar(name="C",x=dff["year"], y=dff["c_score"], marker_color="rgb(169, 181, 168)"),
            go.Bar(name="O",x=dff["year"], y=dff["o_score"], marker_color="rgb(121, 135, 120)"),
            go.Bar(name="R",x=dff["year"], y=dff["r_score"], marker_color="rgb(89, 105, 88)"),
            go.Bar(name="E",x=dff["year"], y=dff["e_score"], marker_color="rgb(64, 79, 63)"),
            go.Bar(name="E2", x=dff["year"], y=dff["e2_score"], marker_color="rgb(53, 66, 52)"),
        ],
    )

    fig_bar.update_layout(
        barmode="group",
        margin=dict(
                    l=10,
                    r=0,
                    b=70,
                    t=50,
                    pad=10
                    ),
        #paper_bgcolor="#f1f5eb",
        template="simple_white"
     )
    return fig_map, fig_bar

@app.callback(
    Output(component_id='datatable-interactivity', component_property='data'),
    [Input(component_id='slct_type', component_property='value'),
     Input(component_id='slct_country', component_property='value'),
     Input(component_id='slct_year', component_property='value')]
)

def update_rows(*option_slctd):
    utdf = tdf.copy()
    if option_slctd[2]:
        utdf = utdf[utdf["year"] == option_slctd[2]] 
    if option_slctd[1]:
        utdf = utdf[utdf["country"] == option_slctd[1]] 
    if option_slctd[0]:
        utdf = utdf[utdf["type"] == option_slctd[0]] 
    
    return utdf.to_dict("records")

@app.callback(
    Output(component_id='score_radar', component_property='figure'),
    Input(component_id='slct_elvl', component_property='value')
)

def update_radar(option_slctd):

    rdff = rdf.copy()
    rdff = rdff[rdff["e_score"] == option_slctd]   

    fig_radar = go.Figure(data=go.Scatterpolar(
        r = rdff.values.flatten(),
        theta=['Engagement', 'Social','Collaboration','Opportunities', 'Risks',
                'Emissions'],
        fill='toself',
        fillcolor="rgb(207, 219, 206)",
        line={"color":"rgb(53, 66, 52)"}
        ))

    fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[1,5]
        ),
    ),
    showlegend=False
    )

    return fig_radar





# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)