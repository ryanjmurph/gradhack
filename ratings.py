import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
def get_data_from_mongo():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['gradhack']
    collection = db['feedback']
    data = list(collection.find())
    return data

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    style={'font-family': 'Arial', 'background-color': '#f8f9fa', 'padding': '20px'},
    children=[
        html.H1("Feedback Ratings Dashboard", style={'text-align': 'center', 'color': '#2c3e50'}),
        dcc.Graph(id='bar-chart'),
        html.Div(id='output-container', style={'text-align': 'center', 'margin-top': '20px', 'color': '#2c3e50'})
    ]
)

# Callback to update the bar chart
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('bar-chart', 'id')]
)
def update_bar_chart(input_id):
    data = get_data_from_mongo()
    df = pd.DataFrame(data)
    rating_counts = df['rating'].value_counts().sort_index()
    
    fig = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values, labels={'x':'Score', 'y':'Frequency'}, title='Ratings Frequency Bar Chart')

    # Update layout to match the desired style
    fig.update_traces(marker_color='rgba(0,0,255,0.5)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title='Frequency',
            titlefont_size=14,
            tickfont_size=10
        ),
        titlefont=dict(size=18),
        plot_bgcolor='white',
        bargap=0.2
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
