import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from collections import Counter

# Import our recommender system
sys.path.append('.')  # Ensure we can import from this directory
from recommender import GameRecommender, load_data, load_models, load_recommender

# Load data and models
print("Loading data and models...")
df = load_data()
recommender = load_recommender()

# If recommender couldn't be loaded, create a new one
if recommender is None:
    models = load_models()
    recommender = GameRecommender(df, models)

# Get unique game IDs for dropdown
game_ids = sorted(df['product_id'].unique())

# Extract LDA topics if available
def get_lda_topic_data():
    try:
        # Check if the LDA data is already available in the dataset
        if 'lda_topics' in df.columns:
            print("LDA topics found in dataframe")
            return df
        
        # If not available directly, try to use the loaded model to extract topics
        elif hasattr(recommender, 'models') and 'lda_model' in recommender.models:
            print("Extracting LDA topics from model...")
            lda_model = recommender.models['lda_model']
            lda_dictionary = recommender.models.get('lda_dictionary')
            
            # Function to get the dominant topic for a document
            def get_dominant_topic(text):
                try:
                    if not isinstance(text, str) or not text.strip():
                        return None
                    
                    # Convert text to bag of words
                    bow = lda_dictionary.doc2bow(text.split())
                    
                    # Get topic distribution
                    topics = lda_model[bow]
                    
                    # Return the most probable topic
                    if topics:
                        return max(topics, key=lambda x: x[1])[0]
                    return None
                except:
                    return None
            
            # Apply to dataframe
            df['dominant_topic'] = df['cleaned_text'].apply(get_dominant_topic)
            return df
        else:
            print("No LDA model found, using random topics for demonstration")
            # Create fake topic data for demonstration if no LDA model is available
            np.random.seed(42)
            df['dominant_topic'] = np.random.randint(0, 5, size=len(df))
            return df
    except Exception as e:
        print(f"Error getting LDA topic data: {e}")
        # Fallback
        df['dominant_topic'] = 0
        return df

# Get LDA topic data
df_with_topics = get_lda_topic_data()

# Create aggregate data by game
game_data = df.groupby('product_id').agg(
    review_count=('username', 'count'),
    avg_hours=('hours', lambda x: x.mean()),
    avg_tokens=('tokens_count', lambda x: x.mean() if 'tokens_count' in df.columns else 0)
).reset_index()

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1("Game Recommender Dashboard", style={'textAlign': 'center'}),
    
    # Game selection section
    html.Div([
        html.H3("Select a Game:"),
        dcc.Dropdown(
            id='game-dropdown',
            options=[{'label': f"Game {game_id}", 'value': game_id} for game_id in game_ids],
            value=game_ids[0] if game_ids else None,
            style={'width': '100%'}
        ),
        html.Br(),
        html.Div([
            html.Label("Recommendation Method:"),
            dcc.RadioItems(
                id='recommendation-method',
                options=[
                    {'label': 'Content-Based', 'value': 'content'},
                    {'label': 'Collaborative', 'value': 'collaborative'},
                    {'label': 'Hybrid', 'value': 'hybrid'}
                ],
                value='hybrid',
                inline=True
            ),
        ]),
        html.Br(),
        html.Button('Get Recommendations', id='recommendation-button', n_clicks=0),
        html.Div(id='game-info', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ddd'})
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    
    # Visualization section
    html.Div([
        html.H3("Game Recommendations:"),
        dcc.Graph(id='recommendation-graph'),
        html.Div(id='recommendation-details', style={'marginTop': '20px'})
    ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    
    # LDA Topic Visualization Section
    html.Div([
        html.H3("Topic Distribution for Games:"),
        dcc.Graph(id='topic-distribution-graph'),
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    
    # Additional Visualization Section
    html.Div([
        html.H3("Review Analysis:"),
        dcc.Graph(id='review-analysis-graph'),
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    
    # Hours Played vs Review Count Scatter Plot
    html.Div([
        html.H3("Hours Played vs Review Count:"),
        dcc.Graph(id='hours-review-graph'),
    ], style={'width': '100%', 'padding': '20px'}),
    
    # Store component to save the current recommendations
    dcc.Store(id='current-recommendations'),
])

# Callback to display game information
@app.callback(
    Output('game-info', 'children'),
    [Input('game-dropdown', 'value')]
)
def update_game_info(game_id):
    if not game_id:
        return html.P("Please select a game")
    
    # Get game info
    game_info = recommender.get_game_info(game_id)
    
    # Get review samples
    game_reviews = df[df['product_id'] == game_id]
    review_sample = "No reviews available"
    if len(game_reviews) > 0:
        review_sample = game_reviews.iloc[0]['text']
        if isinstance(review_sample, str) and len(review_sample) > 100:
            review_sample = review_sample[:100] + "..."
    
    # Get dominant topic if available
    topic_info = ""
    if 'dominant_topic' in df_with_topics.columns:
        game_topics = df_with_topics[df_with_topics['product_id'] == game_id]['dominant_topic']
        if not game_topics.empty and pd.notna(game_topics.iloc[0]):
            topic_counts = Counter(game_topics.dropna())
            if topic_counts:
                dominant_topic = topic_counts.most_common(1)[0][0]
                topic_info = html.P(f"Dominant Topic: {dominant_topic}")
    
    return [
        html.H4(f"Game {game_id}"),
        html.P(f"Number of Reviews: {game_info.get('review_count', 'N/A')}"),
        html.P(f"Average Hours Played: {game_info.get('avg_hours', 0):.1f}"),
        topic_info,
        html.H5("Sample Review:"),
        html.P(review_sample, style={'fontStyle': 'italic'})
    ]

# Callback to get recommendations and store them
@app.callback(
    [Output('current-recommendations', 'data')],
    [Input('recommendation-button', 'n_clicks')],
    [State('game-dropdown', 'value'),
     State('recommendation-method', 'value')]
)
def get_recommendations(n_clicks, game_id, method):
    if n_clicks == 0 or not game_id:
        return [None]
    
    # Get recommendations
    recommendations = recommender.get_recommendations(game_id, n=10, method=method)
    
    # Store them for other callbacks
    return [recommendations]

# Callback to update recommendation graph
@app.callback(
    Output('recommendation-graph', 'figure'),
    [Input('current-recommendations', 'data')]
)
def update_recommendation_graph(recommendations):
    if not recommendations:
        # Return empty figure
        return go.Figure()
    
    # Create bar chart of recommendations
    game_ids = [rec['product_id'] for rec in recommendations]
    scores = [rec['similarity_score'] for rec in recommendations]
    
    fig = px.bar(
        x=scores, 
        y=game_ids, 
        orientation='h',
        labels={'x': 'Similarity Score', 'y': 'Game ID'},
        title='Recommended Games',
        color=scores,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

# Callback to update recommendation details
@app.callback(
    Output('recommendation-details', 'children'),
    [Input('recommendation-graph', 'clickData'),
     Input('current-recommendations', 'data')]
)
def update_recommendation_details(click_data, recommendations):
    if not click_data or not recommendations:
        return html.P("Click on a game in the recommendation chart to see details")
    
    # Get the clicked game
    game_index = click_data['points'][0]['pointIndex']
    game_data = recommendations[game_index]
    
    return [
        html.H4(f"Game {game_data['product_id']}"),
        html.P(f"Similarity Score: {game_data['similarity_score']:.4f}"),
        html.P(f"Number of Reviews: {game_data.get('review_count', 'N/A')}"),
        html.P(f"Average Hours Played: {game_data.get('avg_hours', 0):.1f}"),
        html.H5("Sample Review:"),
        html.P(game_data.get('sample_review', 'No sample review available'), style={'fontStyle': 'italic'})
    ]

# Callback to update topic distribution graph
@app.callback(
    Output('topic-distribution-graph', 'figure'),
    [Input('current-recommendations', 'data'),
     Input('game-dropdown', 'value')]
)
def update_topic_distribution(recommendations, selected_game):
    # Create a list of games to include in the visualization
    game_ids = [selected_game] if selected_game else []
    
    if recommendations:
        # Add recommended games to the list
        game_ids.extend([rec['product_id'] for rec in recommendations])
    
    # Only keep unique games
    game_ids = list(set(game_ids))
    
    if not game_ids:
        return go.Figure()
    
    # Calculate topic distributions for these games
    topic_data = []
    
    for game_id in game_ids:
        game_reviews = df_with_topics[df_with_topics['product_id'] == game_id]
        
        if not game_reviews.empty and 'dominant_topic' in game_reviews.columns:
            topic_counts = Counter(game_reviews['dominant_topic'].dropna())
            total = sum(topic_counts.values())
            
            if total > 0:
                for topic, count in topic_counts.items():
                    if pd.notna(topic):
                        topic_data.append({
                            'game_id': game_id,
                            'topic': f'Topic {int(topic)}',
                            'percentage': (count / total) * 100
                        })
    
    if topic_data:
        # Create DataFrame for visualization
        topic_df = pd.DataFrame(topic_data)
        
        # Create sunburst chart
        fig = px.sunburst(
            topic_df, 
            path=['game_id', 'topic'], 
            values='percentage',
            title='Topic Distribution by Game',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
        return fig
    else:
        # If no topic data is available
        return go.Figure()

# Callback to update review analysis graph
@app.callback(
    Output('review-analysis-graph', 'figure'),
    [Input('current-recommendations', 'data'),
     Input('game-dropdown', 'value')]
)
def update_review_analysis(recommendations, selected_game):
    # Create a list of games to include in the visualization
    game_ids = [selected_game] if selected_game else []
    
    if recommendations:
        # Add top 5 recommended games to the list
        game_ids.extend([rec['product_id'] for rec in recommendations[:5]])
    
    # Only keep unique games
    game_ids = list(set(game_ids))
    
    if not game_ids:
        return go.Figure()
    
    # Prepare data for visualization
    analysis_data = []
    
    for game_id in game_ids:
        game_reviews = df[df['product_id'] == game_id]
        
        if not game_reviews.empty:
            # Calculate average hours
            avg_hours = game_reviews['hours'].mean() if 'hours' in game_reviews.columns else 0
            
            # Calculate tokens count if available
            avg_tokens = 0
            if 'tokens_count' in game_reviews.columns:
                avg_tokens = game_reviews['tokens_count'].mean()
            
            # Count funny reviews
            funny_count = 0
            if 'found_funny' in game_reviews.columns:
                funny_count = game_reviews['found_funny'].sum()
            
            # Count reviews
            review_count = len(game_reviews)
            
            analysis_data.append({
                'game_id': game_id,
                'avg_hours': avg_hours,
                'avg_tokens': avg_tokens,
                'funny_count': funny_count,
                'review_count': review_count
            })
    
    if analysis_data:
        # Create DataFrame for visualization
        analysis_df = pd.DataFrame(analysis_data)
        
        # Create radar chart
        categories = ['avg_hours', 'avg_tokens', 'funny_count', 'review_count']
        
        fig = go.Figure()
        
        for i, row in analysis_df.iterrows():
            # Normalize data for better visualization
            values = [
                row['avg_hours'] / analysis_df['avg_hours'].max() if analysis_df['avg_hours'].max() > 0 else 0,
                row['avg_tokens'] / analysis_df['avg_tokens'].max() if analysis_df['avg_tokens'].max() > 0 else 0,
                row['funny_count'] / analysis_df['funny_count'].max() if analysis_df['funny_count'].max() > 0 else 0,
                row['review_count'] / analysis_df['review_count'].max() if analysis_df['review_count'].max() > 0 else 0,
            ]
            
            # Add a closed polygon
            values.append(values[0])
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill='toself',
                name=f'Game {row["game_id"]}'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Game Metrics Comparison',
            showlegend=True
        )
        
        return fig
    else:
        return go.Figure()

# Callback to update hours vs review scatter plot
@app.callback(
    Output('hours-review-graph', 'figure'),
    [Input('current-recommendations', 'data'),
     Input('game-dropdown', 'value')]
)
def update_hours_review_graph(recommendations, selected_game):
    # Use all game data for this visualization but highlight selected games
    highlight_games = [selected_game] if selected_game else []
    
    if recommendations:
        highlight_games.extend([rec['product_id'] for rec in recommendations])
    
    # Create scatter plot data
    fig = px.scatter(
        game_data, 
        x='review_count', 
        y='avg_hours',
        size='avg_tokens' if 'avg_tokens' in game_data.columns else None,
        color=[1 if g in highlight_games else 0 for g in game_data['product_id']],
        color_discrete_map={0: 'lightgrey', 1: 'red'},
        hover_name='product_id',
        title='Hours Played vs Review Count',
        labels={'review_count': 'Number of Reviews', 'avg_hours': 'Average Hours Played'}
    )
    
    fig.update_layout(
        xaxis_title='Number of Reviews',
        yaxis_title='Average Hours Played',
        coloraxis_showscale=False
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)