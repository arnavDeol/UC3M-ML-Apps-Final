import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares
import pickle
import os
from gensim.models import Doc2Vec, Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
# Load the preprocessed reviews
def load_data():
    try:
        df = pd.read_csv('processed_data/final_preprocessed_data.csv')
        print(f"Loaded {len(df)} preprocessed reviews")
        return df
    except FileNotFoundError:
        print("Preprocessed data not found. Please run the preprocessing pipeline first.")
        return None

# Load the models created during preprocessing
def load_models():
    models = {}
    try:
        # Load TF-IDF model (both vectorizer and matrix)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_data = pickle.load(f)
            # Handle different saving formats
            if isinstance(tfidf_data, tuple) and len(tfidf_data) == 2:
                # If saved as a tuple (vectorizer, matrix)
                models['tfidf_vectorizer'], models['tfidf_matrix'] = tfidf_data
            elif isinstance(tfidf_data, dict):
                # If saved as a dictionary with keys
                models['tfidf_vectorizer'] = tfidf_data.get('vectorizer')
                models['tfidf_matrix'] = tfidf_data.get('matrix')
            else:
                # If saved in another format, try direct assignment
                models['tfidf_model'] = tfidf_data
        print("TF-IDF model loaded")
    except Exception as e:
        print(f"Error loading TF-IDF model: {e}")
    
    try:
        # Load Word2Vec model
        models['word2vec'] = Word2Vec.load('models/word2vec_model.model')
        print("Word2Vec model loaded")
    except Exception as e:
        print(f"Word2Vec model not found: {e}")
    
    try:
        # Load Doc2Vec model
        models['doc2vec'] = Doc2Vec.load('models/doc2vec_model.model')
        print("Doc2Vec model loaded")
    except Exception as e:
        print(f"Doc2Vec model not found: {e}")
        
    try:
        # Load LDA model and dictionary
        with open('models/lda_model.model', 'rb') as f:
            models['lda_model'] = pickle.load(f)
        with open('models/lda_dictionary.dict', 'rb') as f:
            models['lda_dictionary'] = pickle.load(f)
        print("LDA model loaded")
    except Exception as e:
        print(f"LDA model not found: {e}")
        
    return models

df = load_data()
models = load_models()
class ContentBasedRecommender:
    def __init__(self, df, models):
        self.df = df
        self.models = models
        self.product_ids = df['product_id'].unique()
        self.product_id_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        self.idx_to_product_id = {i: pid for i, pid in enumerate(self.product_ids)}
        self.prepare_data()
        
    def prepare_data(self):
        """Aggregate reviews by product_id and prepare game profiles"""
        # Group by product_id to aggregate reviews for each game
        self.game_reviews = self.df.groupby('product_id')
        
        print("Creating game profiles...")
        self.game_profiles = {}
        
        # Prepare document vectors if available
        if 'doc2vec' in self.models and self.models['doc2vec'] is not None:
            print("Using Doc2Vec model for content-based recommendations")
            self.prepare_doc2vec_profiles()
        elif 'word2vec' in self.models and self.models['word2vec'] is not None:
            print("Using Word2Vec model for content-based recommendations")
            self.prepare_word2vec_profiles()
        else:
            print("Using review text for content-based recommendations")
            self.prepare_text_profiles()
            
        # Calculate similarity matrix
        self.calculate_similarity_matrix()
    
    def prepare_doc2vec_profiles(self):
        """Create game profiles using Doc2Vec vectors"""
        model = self.models['doc2vec']
        
        # For each game, infer a vector from all its reviews
        for pid in self.product_ids:
            game_reviews = self.df[self.df['product_id'] == pid]
            if len(game_reviews) == 0:
                continue
                
            # Get all tokens from reviews for this game
            # Make sure each review is properly tokenized
            review_vectors = []
            
            for _, row in game_reviews.iterrows():
                try:
                    # Ensure we have a list of tokens, not a string
                    if isinstance(row['cleaned_text'], str):
                        tokens = row['cleaned_text'].split()
                        # Only use if there are actual tokens
                        if tokens:
                            vec = model.infer_vector(tokens)
                            review_vectors.append(vec)
                except Exception as e:
                    print(f"Error inferring vector for game {pid}: {e}")
                    continue
            
            # Only create a profile if we have at least one valid review vector
            if review_vectors:
                game_vector = np.mean(review_vectors, axis=0)
                self.game_profiles[pid] = game_vector
    
    def prepare_word2vec_profiles(self):
        """Create game profiles using Word2Vec vectors"""
        model = self.models['word2vec']
        
        # For each game, create a vector from all its reviews
        for pid in self.product_ids:
            try:
                game_reviews = self.df[self.df['product_id'] == pid]
                if len(game_reviews) == 0:
                    continue
                    
                # Get all tokens from reviews for this game
                all_word_vectors = []
                
                for _, row in game_reviews.iterrows():
                    if isinstance(row['cleaned_text'], str):
                        tokens = row['cleaned_text'].split()
                        for token in tokens:
                            try:
                                if token in model.wv:
                                    all_word_vectors.append(model.wv[token])
                            except:
                                # Skip tokens not in vocabulary
                                continue
                
                # Create vector for game using word vectors
                if all_word_vectors:
                    game_vector = np.mean(all_word_vectors, axis=0)
                    self.game_profiles[pid] = game_vector
            except Exception as e:
                print(f"Error processing game {pid}: {e}")
                continue
    
    def prepare_text_profiles(self):
        """Create game profiles using raw text frequency"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Aggregate all reviews for each game
            game_texts = {}
            for pid in self.product_ids:
                game_reviews = self.df[self.df['product_id'] == pid]
                if len(game_reviews) > 0:
                    cleaned_texts = [text for text in game_reviews['cleaned_text'].tolist() if isinstance(text, str)]
                    if cleaned_texts:
                        game_texts[pid] = " ".join(cleaned_texts)
            
            # Create TF-IDF vectors for each game
            if game_texts:
                vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
                tfidf_matrix = vectorizer.fit_transform(list(game_texts.values()))
                
                # Store profiles
                for i, pid in enumerate(game_texts.keys()):
                    self.game_profiles[pid] = tfidf_matrix[i].toarray().flatten()
            else:
                print("No valid game texts found for TF-IDF vectorization")
        except Exception as e:
            print(f"Error in text profile creation: {e}")
            # Fallback to simple word count if TF-IDF fails
            self._create_simple_profiles()
    
    def _create_simple_profiles(self):
        """Create simple word count profiles as fallback"""
        from collections import Counter
        
        for pid in self.product_ids:
            game_reviews = self.df[self.df['product_id'] == pid]
            if len(game_reviews) > 0:
                # Create a simple word count vector
                word_counts = Counter()
                for text in game_reviews['cleaned_text']:
                    if isinstance(text, str):
                        word_counts.update(text.split())
                
                if word_counts:
                    # Convert to a simple normalized vector
                    total = sum(word_counts.values())
                    if total > 0:
                        profile = np.zeros(100)  # Simple fixed-size profile
                        for i, (word, count) in enumerate(word_counts.most_common(100)):
                            profile[i] = count / total
                        self.game_profiles[pid] = profile
    
    def calculate_similarity_matrix(self):
        """Calculate cosine similarity between game profiles"""
        print("Calculating game similarity matrix...")
        
        # Extract vectors in consistent order
        profile_vectors = []
        self.profile_indices = {}
        i = 0
        
        for pid in self.product_ids:
            if pid in self.game_profiles:
                profile_vectors.append(self.game_profiles[pid])
                self.profile_indices[pid] = i
                i += 1
        
        if profile_vectors:
            # Convert to array and calculate similarity
            try:
                profile_matrix = np.vstack(profile_vectors)
                self.similarity_matrix = cosine_similarity(profile_matrix)
            except Exception as e:
                print(f"Error calculating similarity matrix: {e}")
                # Create an empty matrix as fallback
                self.similarity_matrix = np.zeros((len(profile_vectors), len(profile_vectors)))
        else:
            print("No game profiles created")
            self.similarity_matrix = np.array([])
    
    def get_similar_games(self, product_id, n=10):
        """Get n most similar games to the given product_id"""
        try:
            if product_id not in self.profile_indices:
                print(f"Product ID {product_id} not found in profiles")
                return []
                
            game_idx = self.profile_indices[product_id]
            
            # Get similarity scores for this game
            similarity_scores = self.similarity_matrix[game_idx]
            
            # Get indices of most similar games (excluding self)
            similar_indices = np.argsort(similarity_scores)[::-1][1:n+1]
            
            # Map indices back to product IDs and scores
            similar_games = []
            profile_pid_map = {v: k for k, v in self.profile_indices.items()}
            
            for idx in similar_indices:
                pid = profile_pid_map[idx]
                score = similarity_scores[idx]
                similar_games.append((pid, score))
                
            return similar_games
        except Exception as e:
            print(f"Error getting similar games: {e}")
            return []
class CollaborativeRecommender:
    def __init__(self, df, factors=50, iterations=10):  # Reduced complexity for stability
        self.df = df
        self.factors = factors
        self.iterations = iterations
        self.model = None
        self.user_to_idx = None
        self.idx_to_user = None
        self.game_to_idx = None
        self.idx_to_game = None
        self.user_game_matrix = None
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare user-game interaction matrix for collaborative filtering"""
        print("Preparing user-game matrix for collaborative filtering...")
        
        try:
            # Create user and game id mappings
            users = self.df['username'].unique()
            games = self.df['product_id'].unique()
            
            self.user_to_idx = {user: i for i, user in enumerate(users)}
            self.idx_to_user = {i: user for i, user in enumerate(users)}
            self.game_to_idx = {game: i for i, game in enumerate(games)}
            self.idx_to_game = {i: game for i, game in enumerate(games)}
            
            # Build interaction data - we'll use hours played as the interaction strength
            user_indices = []
            game_indices = []
            values = []
            
            for _, row in self.df.iterrows():
                user_idx = self.user_to_idx.get(row['username'])
                game_idx = self.game_to_idx.get(row['product_id'])
                
                if user_idx is not None and game_idx is not None:
                    try:
                        # Use hours played as interaction strength (with some minimum value)
                        hours = float(row['hours']) if pd.notna(row['hours']) else 0.1
                        hours = max(hours, 0.1)  # Ensure positive value
                        
                        user_indices.append(user_idx)
                        game_indices.append(game_idx)
                        values.append(float(hours))
                    except (ValueError, TypeError):
                        # Skip if hours can't be converted to float
                        continue
            
            # Create sparse matrix
            self.user_game_matrix = sparse.coo_matrix(
                (values, (user_indices, game_indices)), 
                shape=(len(users), len(games))
            ).tocsr()
            
            print(f"Created matrix with {len(users)} users and {len(games)} games")
        except Exception as e:
            print(f"Error preparing data for collaborative filtering: {e}")
            # Create an empty matrix as fallback
            self.user_game_matrix = sparse.csr_matrix((1, 1))
        
    def train_model(self):
        """Train the collaborative filtering model"""
        print("Training collaborative filtering model...")
        
        try:
            # Check if we have enough data
            if self.user_game_matrix.nnz < 10:
                print("Not enough data for collaborative filtering")
                return False
                
            # Initialize ALS model with error handling
            self.model = AlternatingLeastSquares(
                factors=self.factors,
                iterations=self.iterations,
                calculate_training_loss=True,
                random_state=42
            )
            
            # Train on the user-game matrix
            self.model.fit(self.user_game_matrix)
            print("Model training complete")
            return True
        except Exception as e:
            print(f"Error training collaborative model: {e}")
            return False
        
    def get_similar_games(self, product_id, n=10):
        """Get n most similar games to the given product_id based on collaborative filtering"""
        try:
            if self.model is None:
                print("Collaborative model not trained")
                return []
                
            if product_id not in self.game_to_idx:
                print(f"Product ID {product_id} not found in collaborative model")
                return []
                
            # Get the item factors for the game
            game_idx = self.game_to_idx[product_id]
            
            # Find similar games
            similar_indices, similar_scores = self.model.similar_items(game_idx, n+1)
            
            # Map indices back to product IDs and scores (exclude the first one which is itself)
            similar_games = []
            for idx, score in zip(similar_indices[1:], similar_scores[1:]):
                pid = self.idx_to_game[idx]
                similar_games.append((pid, float(score)))
                
            return similar_games
        except Exception as e:
            print(f"Error getting collaborative recommendations: {e}")
            return []
class GameRecommender:
    def __init__(self, df, models, content_weight=0.5):
        self.df = df
        self.models = models
        self.content_weight = content_weight
        print("Initializing content-based recommender...")
        self.content_recommender = ContentBasedRecommender(df, models)
        print("Initializing collaborative recommender...")
        self.collaborative_recommender = CollaborativeRecommender(df)
        self.collaborative_trained = self.collaborative_recommender.train_model()
        self.game_info = self._prepare_game_info()
        
    def _prepare_game_info(self):
        """Prepare additional information about games for recommendations"""
        game_info = {}
        
        try:
            # Aggregate data for each game
            for pid in self.df['product_id'].unique():
                game_reviews = self.df[self.df['product_id'] == pid]
                if len(game_reviews) > 0:
                    # Calculate average hours and sample review text
                    hours = [h for h in game_reviews['hours'] if pd.notna(h)]
                    avg_hours = np.mean(hours) if hours else 0
                    
                    # Get a sample review text
                    sample_review = ""
                    for text in game_reviews['text']:
                        if isinstance(text, str) and len(text) > 10:
                            sample_review = text[:100] + "..."  # First 100 chars
                            break
                    
                    game_info[pid] = {
                        'product_id': pid,
                        'review_count': len(game_reviews),
                        'avg_hours': avg_hours,
                        'sample_review': sample_review
                    }
        except Exception as e:
            print(f"Error preparing game info: {e}")
        
        return game_info
    
    def get_game_info(self, product_id):
        """Get information about a specific game"""
        return self.game_info.get(product_id, {'product_id': product_id, 'not_found': True})
    
    def get_recommendations(self, product_id, n=10, method='hybrid'):
        """Get game recommendations using the specified method"""
        try:
            if method == 'content':
                return self._get_content_recommendations(product_id, n)
            elif method == 'collaborative' and self.collaborative_trained:
                return self._get_collaborative_recommendations(product_id, n)
            elif method == 'hybrid' and self.collaborative_trained:
                return self._get_hybrid_recommendations(product_id, n)
            else:
                # Default to content-based if collaborative not available
                print(f"Using content-based as fallback for {method} method")
                return self._get_content_recommendations(product_id, n)
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def _get_content_recommendations(self, product_id, n=10):
        """Get content-based recommendations"""
        similar_games = self.content_recommender.get_similar_games(product_id, n)
        return self._format_recommendations(similar_games)
    
    def _get_collaborative_recommendations(self, product_id, n=10):
        """Get collaborative filtering recommendations"""
        similar_games = self.collaborative_recommender.get_similar_games(product_id, n)
        return self._format_recommendations(similar_games)
    
    def _get_hybrid_recommendations(self, product_id, n=10):
        """Get hybrid recommendations by combining content and collaborative approaches"""
        try:
            # Get recommendations from both systems
            content_games = self.content_recommender.get_similar_games(product_id, n*2)
            collab_games = self.collaborative_recommender.get_similar_games(product_id, n*2)
            
            # Create dictionaries for easier merging
            content_dict = {pid: score for pid, score in content_games}
            collab_dict = {pid: score for pid, score in collab_games}
            
            # Merge recommendations
            all_games = set(content_dict.keys()) | set(collab_dict.keys())
            hybrid_scores = {}
            
            for game in all_games:
                # Default scores of 0 if not in either list
                content_score = content_dict.get(game, 0)
                collab_score = collab_dict.get(game, 0)
                
                # Weighted combination
                hybrid_scores[game] = (self.content_weight * content_score + 
                                      (1 - self.content_weight) * collab_score)
            
            # Sort by hybrid score and take top n
            top_games = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return self._format_recommendations(top_games)
        except Exception as e:
            print(f"Error in hybrid recommendations: {e}")
            # Fallback to content-based
            return self._get_content_recommendations(product_id, n)
    
    def _format_recommendations(self, similar_games):
        """Format the recommendation results with additional game info"""
        recommendations = []
        
        for pid, score in similar_games:
            game_data = self.get_game_info(pid)
            game_data['similarity_score'] = score
            recommendations.append(game_data)
            
        return recommendations
    
    def visualize_recommendations(self, product_id, n=10, method='hybrid'):
        """Visualize the recommendations"""
        try:
            recommendations = self.get_recommendations(product_id, n, method)
            
            if not recommendations:
                print(f"No recommendations found for product ID {product_id}")
                return []
            
            # Get the source game info
            source_game = self.get_game_info(product_id)
            if 'not_found' in source_game:
                print(f"Source game with product ID {product_id} not found")
                return []
                
            # Create a plot
            plt.figure(figsize=(10, 6))
            
            # Plot similarity scores
            game_ids = [str(rec['product_id']) for rec in recommendations]
            scores = [rec['similarity_score'] for rec in recommendations]
            
            plt.barh(range(len(game_ids)), scores, align='center')
            plt.yticks(range(len(game_ids)), game_ids)
            plt.xlabel('Similarity Score')
            plt.ylabel('Product ID')
            plt.title(f'Games Similar to {product_id} ({method.capitalize()} Method)')
            plt.tight_layout()
            
            # Save the visualization
            os.makedirs('recommendations', exist_ok=True)
            plt.savefig(f'recommendations/game_{product_id}_{method}_recommendations.png')
            plt.close()
            
            # Also create a text report
            report = f"Recommendations for Game {product_id}\n"
            report += f"Method: {method.capitalize()}\n"
            report += f"Source Game: Product ID {product_id}, {source_game['review_count']} reviews, "
            report += f"Avg Hours: {source_game.get('avg_hours', 'N/A'):.1f}\n\n"
            
            report += "Recommended Games:\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. Product ID: {rec['product_id']}, "
                report += f"Score: {rec['similarity_score']:.4f}, "
                report += f"Reviews: {rec.get('review_count', 'N/A')}, "
                report += f"Avg Hours: {rec.get('avg_hours', 'N/A'):.1f}\n"
                report += f"   Sample Review: {rec.get('sample_review', 'N/A')}\n"
            
            with open(f'recommendations/game_{product_id}_{method}_recommendations.txt', 'w') as f:
                f.write(report)
                
            print(f"Recommendations for game {product_id} saved to file")
            return recommendations
        except Exception as e:
            print(f"Error visualizing recommendations: {e}")
            return []
    
def test_recommender():
    """Test the recommender system with sample games"""
    df = load_data()
    models = load_models()
    
    if df is None or not models:
        print("Cannot test recommender: data or models missing")
        return
    
    # Initialize the recommender
    recommender = GameRecommender(df, models)
    
    # Get some sample game IDs to test
    sample_games = df['product_id'].value_counts().head(5).index.tolist()
    
    for game_id in sample_games:
        print(f"\nTesting recommendations for game {game_id}:")
        
        # Try different recommendation methods
        for method in ['content', 'collaborative', 'hybrid']:
            print(f"\n{method.capitalize()} Recommendations:")
            recommendations = recommender.visualize_recommendations(game_id, n=5, method=method)
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. Game {rec['product_id']}, Score: {rec['similarity_score']:.4f}")

def save_recommender(recommender, filename='models/game_recommender.pkl'):
    """Save the trained recommender system"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(recommender, f)
    print(f"Recommender saved to {filename}")

def load_recommender(filename='models/game_recommender.pkl'):
    """Load a trained recommender system"""
    try:
        with open(filename, 'rb') as f:
            recommender = pickle.load(f)
        print("Recommender loaded successfully")
        return recommender
    except:
        print("No saved recommender found. Training a new one...")
        df = load_data()
        models = load_models()
        recommender = GameRecommender(df, models)
        save_recommender(recommender)
        return recommender

def get_game_recommendations(product_id, n=10, method='hybrid'):
    """Get recommendations for a game - function to be used by Dash"""
    recommender = load_recommender()
    return recommender.get_recommendations(product_id, n, method)

def get_all_game_ids():
    """Get all available game IDs - function to be used by Dash"""
    df = load_data()
    return df['product_id'].unique().tolist()

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    models = load_models()
    
    if df is not None and models:
        # Create and save recommender
        recommender = GameRecommender(df, models)
        save_recommender(recommender)
        
        # Test with a few examples
        test_recommender()
    else:
        print("Cannot create recommender: data or models missing")