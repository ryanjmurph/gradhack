import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from pymongo import MongoClient

# Connect to MongoDB
def get_data_from_mongo():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['gradhack']
    collection = db['interactions']
    data = list(collection.find())
    return data

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    style={'font-family': 'Arial', 'background-color': '#f8f9fa', 'padding': '20px'},
    children=[
        html.H1("Silicon Slipper Insights", style={'text-align': 'center', 'color': '#2c3e50'}),
        dcc.Graph(id='pie-chart'),
        html.Div(id='output-container', style={'text-align': 'center', 'margin-top': '20px', 'color': '#2c3e50'})
    ]
)

@app.callback(
    [Output('pie-chart', 'figure'),
     Output('output-container', 'children')],
    [Input('pie-chart', 'clickData')]
)

def update_chart(clickData):
    data = get_data_from_mongo()
    categories = ['Savings', 'Investment Plan', 'Spending Habits', 'Discovery Card Plans', 'Other']
    category_count = {category: 0 for category in categories}

    for item in data:
        category = item.get('type', 'Other')
        if category not in category_count:
            category = 'Other'
        category_count[category] += 1

    fig = px.pie(
        names=list(category_count.keys()), 
        values=list(category_count.values()), 
        title='Question Types Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    if clickData is None:
        return fig, "Click on a section of the pie chart to see more information"
    else:
        label = clickData['points'][0]['label']
        questions = [item['question'] for item in data if item.get('type') == label]
        questions_html = html.Ul([html.Li(q) for q in questions])
        return fig, html.Div([
            html.H3(f"Questions for {label}:", style={'color': '#2c3e50'}),
            questions_html
        ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
