from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from wordcloud import WordCloud
from src.model_evaluator import ModelEvaluator
from src.time_series_analyzer import TimeSeriesAnalyzer

class NewsAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        self.model = LogisticRegression(random_state=42)
        self.feature_names = None
        self.evaluator = ModelEvaluator()
        self.time_analyzer = TimeSeriesAnalyzer()
        
    def prepare_data(self, news_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the news data for analysis using time-based split
        """
        # Combine all news columns into one
        news_df['Combined_News'] = news_df.filter(like='Top').apply(
            lambda x: ' '.join(str(text) for text in x), axis=1
        )
        
        # Split data by time first
        train_df, test_df = self.time_analyzer.split_by_time(news_df)

        # Convert news headlines to TF-IDF features
        X_train = self.vectorizer.fit_transform(train_df['Combined_News'])
        X_test = self.vectorizer.transform(test_df['Combined_News'])
        y_train = train_df['Label'].values
        y_test = test_df['Label'].values

        # Store training and test data for later access by the app
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = self.vectorizer.get_feature_names_out()

        return X_train, X_test, y_train, y_test
        
    def train_model(self, X_train, y_train):
        """
        Train the model on the prepared data
        """
        self.model.fit(X_train, y_train)
        
    def analyze_performance(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Analyze model performance using AUC and other metrics
        """
        # Get prediction probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics using ModelEvaluator
        metrics = self.evaluator.calculate_metrics(y_test, y_pred_proba)
        
        # Perform cross-validation
        cv_mean, cv_std = self.evaluator.perform_cross_validation(self.model, X_train, y_train)
        
        # Plot ROC curve
        self.evaluator.plot_roc_curve(y_test, y_pred_proba)
        
        # Plot confusion matrix
        self.evaluator.plot_confusion_matrix(y_test, y_pred)
        
        return {
            'metrics': metrics,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
    
    def get_most_influential_words(self, n=10):
        """
        Get the most influential words/phrases for market movement with their frequencies
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model hasn't been trained yet")
            
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        # Get positive and negative influencers
        pos_indices = coefficients.argsort()[-n:][::-1]
        neg_indices = coefficients.argsort()[:n]
        
        # Get term frequencies
        term_freq = np.asarray(self.X_train.sum(axis=0)).flatten()
        
        positive_words = [(feature_names[i], coefficients[i], term_freq[i]) for i in pos_indices]
        negative_words = [(feature_names[i], coefficients[i], term_freq[i]) for i in neg_indices]
        
        return positive_words, negative_words
        
    def analyze_topic_distribution(self):
        """
        Analyze the distribution of topics/themes in the news
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model hasn't been trained yet")
            
        feature_names = self.vectorizer.get_feature_names_out()
        term_freq = np.asarray(self.X_train.sum(axis=0)).flatten()
        
        # Create topic buckets
        topics = defaultdict(list)
        
        # Common topic keywords
        topic_keywords = {
            'economy': ['economy', 'market', 'stocks', 'fed', 'rates', 'growth'],
            'politics': ['president', 'congress', 'government', 'election', 'policy'],
            'international': ['china', 'russia', 'europe', 'global', 'trade'],
            'technology': ['tech', 'apple', 'google', 'software', 'cyber'],
            'finance': ['bank', 'investment', 'trading', 'shares', 'profit']
        }
        
        # Categorize terms
        for idx, term in enumerate(feature_names):
            for topic, keywords in topic_keywords.items():
                if any(keyword in term.lower() for keyword in keywords):
                    topics[topic].append((term, term_freq[idx]))
                    break
        
        # Sort terms by frequency within each topic
        for topic in topics:
            topics[topic].sort(key=lambda x: x[1], reverse=True)
            
        return dict(topics)
    
    def create_word_cloud(self, words_with_scores: List[Tuple[str, float, float]], 
                          title: str) -> None:
        """
        Create and display a word cloud visualization
        """
        # Create word frequency dict with impact scores as weights
        word_scores = {word: abs(score) for word, score, _ in words_with_scores}
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=50,
        ).generate_from_frequencies(word_scores)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
        
    def plot_word_influence(self, n: int = 10) -> None:
        """
        Enhanced visualization of the most influential words
        """
        positive_words, negative_words = self.get_most_influential_words(n)
        
        # Create word clouds
        self.create_word_cloud(positive_words, 'Positive Impact Words Cloud')
        self.create_word_cloud(negative_words, 'Negative Impact Words Cloud')
        
        # Create detailed bar plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot positive words with enhanced styling
        words, values, freqs = zip(*positive_words)
        ax1.barh(range(len(words)), values, 
                color=plt.cm.YlGn(np.linspace(0.3, 0.9, len(words))))
        ax1.set_yticks(range(len(words)))
        ax1.set_yticklabels([f"{w} ({f:,.0f})" for w, f in zip(words, freqs)])
        ax1.set_title('Top Positive Impact Words (with frequencies)', pad=20)
        ax1.set_xlabel('Impact Score')
        
        # Plot negative words with enhanced styling
        words, values, freqs = zip(*negative_words)
        ax2.barh(range(len(words)), values,
                color=plt.cm.YlOrRd_r(np.linspace(0.3, 0.9, len(words))))
        ax2.set_yticks(range(len(words)))
        ax2.set_yticklabels([f"{w} ({f:,.0f})" for w, f in zip(words, freqs)])
        ax2.set_title('Top Negative Impact Words (with frequencies)', pad=20)
        ax2.set_xlabel('Impact Score')
        
        plt.tight_layout()
        plt.savefig('word_influence.png')
        plt.close()