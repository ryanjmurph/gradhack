import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from pymongo import MongoClient
import pandas as pd

def get_feedback_data():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['gradhack']
    collection = db['feedback']
    data = list(collection.find())
    client.close()
    return data

def get_interactions_data():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['gradhack']
    collection = db['interactions']
    data = list(collection.find())
    client.close()
    return data

app = dash.Dash(__name__)

app.layout = html.Div(
    style={'font-family': 'Arial', 'background-color': '#f8f9fa', 'padding': '20px'},
    children=[
        html.H1("Financial Dashboard", style={'text-align': 'center', 'color': '#2c3e50'}),
        dcc.Tabs(id="tabs", value='tab-2', children=[
            dcc.Tab(label='Financial Questions', value='tab-2'),
            dcc.Tab(label='Feedback Ratings', value='tab-1'),
        ]),
        html.Div(id='tabs-content')
    ]
)

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Graph(id='bar-chart'),
            html.Div(id='reviews-container', style={'text-align': 'center', 'margin-top': '20px', 'color': '#2c3e50'})
        ])
    elif tab == 'tab-2':
        return html.Div([
            dcc.Graph(id='pie-chart'),
            html.Div(id='output-container', style={'text-align': 'center', 'margin-top': '20px', 'color': '#2c3e50'})
        ])

@app.callback(
    [Output('bar-chart', 'figure'),
     Output('reviews-container', 'children')],
    Input('tabs', 'value')
)
def update_bar_chart(tab):
    if tab == 'tab-1':
        data = get_feedback_data()
        df = pd.DataFrame(data)
        rating_counts = df['rating'].value_counts().sort_index()
        
        fig = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values, labels={'x':'Score', 'y':'Frequency'}, title='Ratings Frequency Bar Chart')

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

        reviews = []
        for index, row in df.iterrows():
            review = row.get('comment', 'No Review')
            if review  == "":
                review = "No Review"
            reviews.append(html.Div(f"Rating: {row['rating']} Comment:{review}"))

        return fig, reviews
    return dash.no_update, dash.no_update

@app.callback(
    [Output('pie-chart', 'figure'),
     Output('output-container', 'children')],
    [Input('pie-chart', 'clickData'),
     Input('tabs', 'value')]
)
def update_pie_chart(clickData, tab):
    if tab == 'tab-2':
        data = get_interactions_data()
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
    return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
