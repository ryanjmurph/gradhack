import dash
from dash import dcc #import dash_core_components as dcc
from dash import html #import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Sample data
data = {
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [10, 20, 30, 40]
}

# Create a pie chart
fig = px.pie(data, names='Category', values='Values', title='Interactive Pie Chart')

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='pie-chart', figure=fig),
    html.Div(id='output-container')
])

# Callback to update the output container based on click data
@app.callback(
    Output('output-container', 'children'),
    [Input('pie-chart', 'clickData')]
)
def display_click_data(clickData):
    if clickData is None:
        return "Click on a section of the pie chart to see more information"
    else:
        return f"You clicked on: {clickData['points'][0]['label']} with value: {clickData['points'][0]['value']}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)