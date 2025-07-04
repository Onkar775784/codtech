import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("diabetes.csv")

# Create Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Diabetes Dashboard"),
    
    dcc.Dropdown(
        id='feature',
        options=[{'label': col, 'value': col} for col in df.columns[:-1]],
        value='Glucose'
    ),

    dcc.Graph(id='feature-dist'),

    dcc.Graph(figure=px.box(df, x="Outcome", y="BMI", color="Outcome", title="BMI by Outcome"))
])

# Callback
@app.callback(
    dash.dependencies.Output('feature-dist', 'figure'),
    [dash.dependencies.Input('feature', 'value')]
)
def update_graph(selected_feature):
    fig = px.histogram(df, x=selected_feature, color='Outcome', barmode='overlay', nbins=30)
    fig.update_layout(title=f'Distribution of {selected_feature}')
    return fig

# Run server
if __name__ == '__main__':
    app.run(debug=True)

