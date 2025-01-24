from state_codes import state_id_map
import dash
import dash_leaflet as dl
from dash.dependencies import Input, Output, ALL
from dash import html, Input, Output, dash_table, State, dcc
import pandas as pd
from extract_coords import get_state_boundaries
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import json
import datetime
import plotly.express as px


# Load the data
df = pd.read_csv("filtered_plus_severity.csv", low_memory=False)

# Ensure the DATE column is in datetime format
df["DATE"] = pd.to_datetime(df["DATE"], format='%Y-%m-%d')

# Constants, beginning year of DATA, end year of DATA
START_YEAR = 2003  # as in 2003
END_YEAR = 2024    # as in 2024

# Safety limit for the maximal amount of points allowed to be rendered
MAX_POINTS_LIMIT = 4000

# Global variables to hold state of SELECTIONS
SELECTED_STATE = None  # index
SELECTED_TIME_FROM = datetime.datetime(year=START_YEAR, month=1, day=1)  # datetime
SELECTED_TIME_UNTIL = datetime.datetime(year=END_YEAR + 1, month=1, day=1)   # datetime
SELECTED_FILTER = None  # attribute name
SELECTED_VALUE = None   # value of set attribute
INTERSECTED_DF = df.copy()
INTERSECTED_DF_STATELESS = df.copy()
STATE_COLORS = {}

# Group by State
grouped = INTERSECTED_DF.groupby("STATE")

# Build frequency-dictionary key: State, value: nr. of accidents in set state
freq_map = grouped.size().to_dict()

# Normalize frequencies for color mapping
max_incidents = max(freq_map.values())
min_incidents = min(freq_map.values())

#Detect initial call
INITIAL_CALL = True

#Computes the resulting dataframe, that is the intersection of selected values, starting from the original dataframe
def compute_and_update_dataframe():
    global SELECTED_STATE, SELECTED_TIME_FROM, SELECTED_TIME_UNTIL, SELECTED_FILTER, SELECTED_VALUE, INTERSECTED_DF, INTERSECTED_DF_STATELESS
    filtered_df = df.copy()
    stateless_filtered_df = df.copy()
    if SELECTED_STATE is not None:
        filtered_df = filtered_df[filtered_df["STATE"] == SELECTED_STATE]
    if SELECTED_TIME_FROM is not None:
        filtered_df = filtered_df[filtered_df["DATE"] > SELECTED_TIME_FROM]
        stateless_filtered_df = stateless_filtered_df[stateless_filtered_df["DATE"] > SELECTED_TIME_FROM]
    if SELECTED_TIME_UNTIL is not None:
        filtered_df = filtered_df[filtered_df["DATE"] < SELECTED_TIME_UNTIL]
        stateless_filtered_df = stateless_filtered_df[stateless_filtered_df["DATE"] < SELECTED_TIME_UNTIL]
    if SELECTED_FILTER is not None and SELECTED_VALUE is not None:
        filtered_df = filtered_df[filtered_df[SELECTED_FILTER] == SELECTED_VALUE]
        stateless_filtered_df = stateless_filtered_df[stateless_filtered_df[SELECTED_FILTER] == SELECTED_VALUE]
    INTERSECTED_DF = filtered_df.copy()
    INTERSECTED_DF_STATELESS = stateless_filtered_df.copy()
    compute_color()

#Scatter plot matrix generation
def generate_scatter_matrix(selected_df):
    if not selected_df.empty:
        categories = ['VISIBLTY', 'TRNSPD', 'TEMP', 'WEATHER', 'severity', 'TYPTRK']

        fig = px.scatter_matrix(
            selected_df,
            dimensions=categories,
            labels={col: f"{col}" for col in categories},
            title="Scatter Matrix of Accident Variables"
        )

        fig.update_layout(
            height=800,
            width=800,
            margin={'t': 50, 'b': 50, 'l': 50, 'r': 50}
        )
    else:
        fig = {
            "data": [],
            "layout": {
                "title": "No Data Available",
                "height": 300,
                "margin": {'t': 30, 'b': 50, 'l': 50, 'r': 50},
            },
        }

    return fig

#Radar graph generation
def generate_radar_graph(selected_df):
    """For demonstration, reusing the same plot structure as the scatter plot.
       You can replace it with your own radar logic."""
    if not selected_df.empty:
        categories = ['VISIBLTY', 'TRNSPD', 'TEMP', 'WEATHER', 'severity', 'TONS']
        #categories = ['ALCOHOL', 'DRUG', 'ENGRS', 'FIREMEN', 'CONDUCTR', 'BRAKEMEN', 'TIMEHR']
        normalized_df = selected_df[categories].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        data = {
            'type': 'scatterpolar',
            'r': normalized_df.mean().values.tolist() + [normalized_df.mean().values[0]],
            'theta': categories + [categories[0]],
            'fill': 'toself'
        }

        layout = {
            'title': 'Accident Overview',
            'polar': {
                'radialaxis': {
                    'visible': True,
                    'range': [0, 1]
                }
            },
            'showlegend': False,
            'height': 450,
            'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50}
        }

        fig = {'data': [data], 'layout': layout}
    else:
        fig = {
            'data': [],
            'layout': {
                'title': 'No Data Available',
                'height': 300,
                'margin': {'t': 30, 'b': 50, 'l': 50, 'r': 50},
            },
        }

    return fig


def concat_narrative(row):
    """Joins together the strings from NARR1 to NARR6, if non-empty."""
    i = 1
    name = "NARR" + str(i)
    message = ""
    while type(row[name]) == str and i <= 6:
        message += row[name]
        i += 1
        name = "NARR" + str(i)
    return message


def generate_markers_for_df(selected_df):
    """Generate markers for a specific state's data."""
    if len(selected_df) > MAX_POINTS_LIMIT:
        return []
    return [
        dl.Marker(
            position=[row['Latitude'], row['Longitud']],
            children=[dl.Popup(html.Div(f"{concat_narrative(row)}"))],
            id={"type": "marker", "index": index}
        ) for index, row in selected_df.iterrows()
    ]


# Get state boundaries
state_boundaries = get_state_boundaries().copy()

#Compute color per state and save it in dictionary format
def compute_color():
    global INTERSECTED_DF_STATELESS, max_incidents, min_incidents, STATE_COLORS
    # Group by State
    grouped = INTERSECTED_DF_STATELESS.groupby("STATE")

    # Build frequency-dictionary key: State, value: nr. of accidents in set state
    freq_map = grouped.size().to_dict()

    # Normalize frequencies for color mapping
    max_incidents = max(freq_map.values())
    min_incidents = min(freq_map.values())

    for state_id in state_id_map.keys():
        if state_id in freq_map.keys():
            count = freq_map[state_id]
            log_count = np.log1p(count)
            log_min = np.log1p(min_incidents)
            log_max = np.log1p(max_incidents)
            if log_max > log_min:
                norm = (log_count - log_min) / (log_max - log_min)
            else:
                norm = 0.5
            STATE_COLORS[state_id] = mcolors.to_hex(cm.plasma(norm))
        else:
            STATE_COLORS[state_id] = mcolors.to_hex(cm.plasma(0.0))

compute_color()
# Define a color scale, for the heatmap
def get_color(state):
    global INTERSECTED_DF_STATELESS, max_incidents, min_incidents, STATE_COLORS
    if state == 15:
        # Dummy check if you have an unused state code
        return mcolors.to_hex(cm.plasma(0.0))
    return STATE_COLORS[state]

# Create a gradient legend for the heatmap
legend = html.Div(
    style={
        "border": "1px solid black",
        "border-radius": "5px",
        "background": "white",
        "padding": "10px",
        "font-size": "12px",
        "marginBottom": "10px"
    },
    children=[
        html.Div("Incident Legend", style={"fontWeight": "bold", "textAlign": "center"}),
        html.Div(
            style={
                "height": "20px",
                "width": "100%",
                "background": "linear-gradient(to right, " +
                              ", ".join([mcolors.to_hex(cm.plasma(v)) for v in np.linspace(0, 1, 100)]) +
                              ")",
                "marginBottom": "5px"
            }
        ),
        html.Div(
            style={"display": "flex", "justifyContent": "space-between"},
            children=[
                html.Span(f"{int(0):,}"),  # Min value
                html.Span(f"{int(max_incidents):,}")   # Max value
            ]
        )
    ]
)

# Build dropdown-menu options for categorical attributes and weather
dropdown_options = [
    {"label": col, "value": col}
    for col in df.columns
    if df[col].dtype in [np.object_, "category"] or col == "WEATHER"
]

#Creates the dash app
app = dash.Dash(__name__)

#The dash app layout
app.layout = html.Div(
    style={
        "height": "87vh",     
        "margin": "0",
        "padding": "0",
        "display": "flex",
        "flexDirection": "column",
        "fontFamily": "Arial, sans-serif"
    },
    children=[

        # Row 1: Map + Right Column (Legend, Filter, Clear-state Button) 
        html.Div(
            style={
                "flex": "0 0 49%",        
                "display": "flex",
                "flexDirection": "row",
                "padding": "2px"
            },
            children=[
                # The map takes the left portion
                html.Div(
                    style={
                        "flex": "1",
                        "position": "relative",
                        "border": "2px solid black",
                        "borderRadius": "5px",
                        "marginRight": "5px",
                        "overflow": "hidden"
                    },
                    children=[
                        dl.Map(
                            style={"width": "100%", "height": "100%"},
                            center=[40.2, -98.5],  # Centered on the USA
                            zoom=4.7,
                            children=[
                                dl.TileLayer(),
                                *[
                                    dl.Polygon(
                                        positions=state_boundaries[state],
                                        color=get_color(state),
                                        fillOpacity=0.7,
                                        weight=1,
                                        id={"type": "polygon", "index": state}
                                    )
                                    for state in state_boundaries
                                ],
                                dl.MarkerClusterGroup(id="cluster-layer")
                            ]
                        ),
                    ]
                ),

                # The right portion for legend, filter dropdown, reset button
                html.Div(
                    style={
                        "width": "300px",
                        "display": "flex",
                        "flexDirection": "column"
                    },
                    children=[
                        legend,
                        html.Div(
                            style={
                                "border": "1px solid black",
                                "border-radius": "5px",
                                "padding": "10px",
                                "marginBottom": "10px"
                            },
                            children=[
                                html.Div("Filter by Category:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="filter-dropdown",
                                    options=dropdown_options,
                                    placeholder="Select category...",
                                    multi=False,
                                    style={"marginBottom": "10px"}
                                ),
                                dcc.Dropdown(
                                    id="value-dropdown",
                                    placeholder="Select value...",
                                    multi=False
                                ),
                            ]
                        ),
                        html.Button(
                            "Unselect-State",
                            id="reset-button",
                            n_clicks=0,
                            style={
                                "padding": "10px",
                                "border": "1px solid black",
                                "borderRadius": "5px",
                                "backgroundColor": "#ddd",
                                "cursor": "pointer"
                            }
                        )
                    ]
                ),
            ]
        ),

        # Row 2: The Two Graphs Side-by-Side
        html.Div(
            style={
                "flex": "0 0 35%",         
                "display": "flex",
                "flexDirection": "row",
                "padding": "2px"
            },
            children=[
                # Scatter Plot
                html.Div(
                    style={
                        "flex": "1",
                        "border": "2px solid black",
                        "borderRadius": "5px",
                        "marginRight": "5px",
                        "backgroundColor": "white",
                        "overflow": "hidden"
                    },
                    children=[
                        dcc.Graph(
                            id="scatter-plot",
                            style={"width": "100%", "height": "100%"}
                        )
                    ]
                ),
                # Radar Graph
                html.Div(
                    style={
                        "flex": "1",
                        "border": "2px solid black",
                        "borderRadius": "5px",
                        "backgroundColor": "white",
                        "overflow": "hidden"
                    },
                    children=[
                        dcc.Graph(
                            id="radar-graph",
                            style={"width": "100%", "height": "100%"}
                        )
                    ]
                )
            ]
        ),

        #Time Slider
        html.Div(
        [
            html.H5("Select Time Range:"), 
            dcc.RangeSlider(
                id="time-slider",
                min=START_YEAR,  # Minimum value
                max=END_YEAR,  # Maximum value
                step=1,                # Step size for the slider
                marks={year: str(year) for year in range(START_YEAR, END_YEAR+1)},  # Labels
                value=[START_YEAR, END_YEAR],    # Default selected range
            ),
        ],
        style={
            "padding": "1px",               
            "backgroundColor": "#f9f9f9",    
            "margin": "0px",                
            "width": "99.5%",                  
        }
    )

    ]
)

# The dash callbacks for updating placeholder outputs, depending on interactive inputs
@app.callback(
    [
        Output("scatter-plot", "figure"),
        Output("radar-graph", "figure"),
        Output("cluster-layer", "children"),
        Output("value-dropdown", "options"),
        Output("value-dropdown", "value"),
        Output({"type": "polygon", "index": ALL}, "color") 
    ],
    [
        Input({"type": "polygon", "index": ALL}, "click_lat_lng"),
        Input("filter-dropdown", "value"),
        Input("value-dropdown", "value"),
        Input("time-slider", "value"),
        Input("reset-button", "n_clicks")
    ]
)
def update_logic(state_click, filter_column, filter_value, time_range, button_click):
    global SELECTED_STATE, SELECTED_TIME_FROM, SELECTED_TIME_UNTIL
    global SELECTED_FILTER, SELECTED_VALUE, INTERSECTED_DF, INITIAL_CALL

    if INITIAL_CALL: #Case 0, initial generation
        INITIAL_CALL = False
        return (
            generate_scatter_matrix(INTERSECTED_DF),
            generate_radar_graph(INTERSECTED_DF),
            generate_markers_for_df(INTERSECTED_DF),
            dash.no_update,
            dash.no_update,
            [get_color(state) for state in state_boundaries]
        )

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, [get_color(state) for state in state_boundaries]

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Case A: State-click
    if "polygon" in triggered_id:
        click_info_dict = json.loads(triggered_id)
        SELECTED_STATE = click_info_dict["index"]
        compute_and_update_dataframe()
        return (
            generate_scatter_matrix(INTERSECTED_DF),
            generate_radar_graph(INTERSECTED_DF),
            generate_markers_for_df(INTERSECTED_DF),
            dash.no_update,
            dash.no_update,
            [get_color(state) for state in state_boundaries]
        )

    # Case B: filter-dropdown
    if "filter-dropdown" in triggered_id:
        if not filter_column:
            SELECTED_FILTER = None
            SELECTED_VALUE = None
            compute_and_update_dataframe()
            return (
                generate_scatter_matrix(INTERSECTED_DF),
                generate_radar_graph(INTERSECTED_DF),
                generate_markers_for_df(INTERSECTED_DF),
                dash.no_update,
                None,
                [get_color(state) for state in state_boundaries]
            )
        SELECTED_FILTER = filter_column
        compute_and_update_dataframe()
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            [{"label": val, "value": val} for val in df[filter_column].dropna().unique()],
            dash.no_update,
            [get_color(state) for state in state_boundaries]
        )

    # Case C: value-dropdown
    if "value-dropdown" in triggered_id:
        if (not filter_column or not filter_value) and (SELECTED_FILTER is not None and SELECTED_VALUE is not None):
            if filter_column is None:
                SELECTED_FILTER = None
            if filter_value is None:
                SELECTED_VALUE = None
            compute_and_update_dataframe()
            return (
                generate_scatter_matrix(INTERSECTED_DF),
                generate_radar_graph(INTERSECTED_DF),
                generate_markers_for_df(INTERSECTED_DF),
                dash.no_update,
                dash.no_update,
                [get_color(state) for state in state_boundaries]
            )
        SELECTED_VALUE = filter_value
        compute_and_update_dataframe()
        return (
            generate_scatter_matrix(INTERSECTED_DF),
            generate_radar_graph(INTERSECTED_DF),
            generate_markers_for_df(INTERSECTED_DF),
            dash.no_update,
            dash.no_update,
            [get_color(state) for state in state_boundaries]
        )

    # Case D: Time-slider
    if "time-slider" in triggered_id:
        year_from, year_until = time_range
        if year_from != SELECTED_TIME_FROM.year or year_until != SELECTED_TIME_UNTIL.year:
            SELECTED_TIME_FROM = datetime.datetime(year=year_from, month=1, day=1)
            SELECTED_TIME_UNTIL = datetime.datetime(year=year_until, month=1, day=1)
            compute_and_update_dataframe()
            return (
                generate_scatter_matrix(INTERSECTED_DF),
                generate_radar_graph(INTERSECTED_DF),
                generate_markers_for_df(INTERSECTED_DF),
                dash.no_update,
                dash.no_update,
                [get_color(state) for state in state_boundaries]
            )
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, [get_color(state) for state in state_boundaries]

    # Case E: Reset-button
    if "reset-button" in triggered_id:
        SELECTED_STATE = None
        compute_and_update_dataframe()
        return (
            generate_scatter_matrix(INTERSECTED_DF),
            generate_radar_graph(INTERSECTED_DF),
            generate_markers_for_df(INTERSECTED_DF),
            dash.no_update,
            dash.no_update,
            [get_color(state) for state in state_boundaries]
        )

#Run the app
if __name__ == '__main__': 
    app.run_server(debug=False)
