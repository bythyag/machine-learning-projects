import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# For NLP processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# For topic modeling
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Path to LinkedIn post JSON files
data_dir = "machine-learning-projects/ml-projects/linkedin-post-sentiment-analysis/linkedin-post"
output_path = "machine-learning-projects/ml-projects/linkedin-post-sentiment-analysis/final-dataset.csv"

# 1. Load and consolidate JSON files
def load_json_data(directory):
    posts = []

    for file in os.listdir(directory):
        if not file.endswith('.json'):
            continue

        path = os.path.join(directory, file)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            posts.append({
                'company': file.rsplit('.', 1)[0],
                'text': entry.get('text', ''),
                'published_time': pd.to_datetime(entry.get('postedAtTimestamp', 0) / 1000, unit='s'),
                'num_likes': entry.get('numLikes', 0),
                'num_comments': entry.get('numComments', 0),
                'num_shares': entry.get('numShares', 0)
            })

    return pd.DataFrame(posts)

# 2. Clean and preprocess text data
def clean_text(text):
    """Clean and normalize text data"""
    if not isinstance(text, str) or text.strip() == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens
    return ' '.join(tokens)

# 3. Perform sentiment analysis
def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    if not text:
        return {'polarity': 0, 'sentiment': 'neutral'}
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = 'positive'
    elif polarity < -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
        
    return {'polarity': polarity, 'sentiment': sentiment}

# 4. Perform topic modeling
def perform_topic_modeling(texts, n_topics=5, n_top_words=10):
    """Extract topics from a collection of texts using LDA"""
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.95,  # Ignore terms that appear in more than 95% of docs
        min_df=2,     # Ignore terms that appear in fewer than 2 docs
        stop_words='english'
    )
    
    # Remove empty strings
    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        print("Warning: No valid texts for topic modeling")
        return [], []
    
    X = vectorizer.fit_transform(valid_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Create and fit LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='online'
    )
    lda.fit(X)
    
    # Extract top words for each topic
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-n_top_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
    
    # Get topic distribution for each document
    topic_distribution = lda.transform(X)
    
    return topics, topic_distribution

# 5. Main analysis pipeline
def analyze_linkedin_posts():
    """Main function to analyze LinkedIn posts"""
    print("Loading and combining JSON data...")
    df = load_json_data(data_dir)
    
    if df.empty:
        print("No data found. Please check the data directory.")
        return
    
    print(f"Loaded {len(df)} posts from {df['company'].nunique()} companies")
    
    # Convert date formats
    print("Processing timestamps...")
    df['published_time'] = pd.to_datetime(df['published_time'], errors='coerce')
    df['year_month'] = df['published_time'].dt.strftime('%Y-%m')
    
    # Clean text
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Analyze sentiment
    print("Performing sentiment analysis...")
    sentiment_results = df['text'].apply(analyze_sentiment)
    df['polarity'] = sentiment_results.apply(lambda x: x['polarity'])
    df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
    
    # Perform topic modeling
    print("Extracting topics...")
    topics, topic_distribution = perform_topic_modeling(df['cleaned_text'].tolist())
    
    # Add dominant topic to each document
    if len(topic_distribution) > 0:
        df['dominant_topic'] = np.argmax(topic_distribution, axis=1)
    
    # Save processed data
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    
    return df, topics

# 6. Visualize results
def visualize_results(df, topics, output_dir='machine-learning-projects/ml-projects/linkedin-post-sentiment-analysis/plots'):
    """Generate visualizations and save them to a specified directory"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sentiment distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=df)
    plt.title('Distribution of Post Sentiments')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()

    # 2. Average engagement by sentiment
    plt.figure(figsize=(10, 6))
    engagement = df.groupby('sentiment').agg({
        'num_likes': 'mean',
        'num_comments': 'mean',
        'num_shares': 'mean'
    }).reset_index()
    melted = pd.melt(
        engagement,
        id_vars='sentiment',
        value_vars=['num_likes', 'num_comments', 'num_shares'],
        var_name='Engagement Type',
        value_name='Average Count'
    )
    sns.barplot(x='sentiment', y='Average Count', hue='Engagement Type', data=melted)
    plt.title('Average Engagement by Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'engagement_by_sentiment.png'))
    plt.close()

    # 3. Topics
    plt.figure(figsize=(12, 8))
    for i, topic_words in enumerate(topics):
        plt.subplot(2, (len(topics) + 1) // 2, i + 1)
        y = np.arange(len(topic_words))
        plt.barh(y, range(len(topic_words), 0, -1))
        plt.yticks(y, topic_words)
        plt.title(f'Topic {i+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topics.png'))
    plt.close()

    # 4. Sentiment by company
    sentiment_pct = pd.crosstab(df['company'], df['sentiment']).div(df['company'].value_counts(), axis=0) * 100
    sentiment_pct.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Sentiment Distribution by Company')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'company_sentiment.png'))
    plt.close()

    # 5. Monthly post volume
    plt.figure(figsize=(14, 7))
    monthly = df.groupby('year_month').size().reset_index(name='count')
    sns.lineplot(x='year_month', y='count', data=monthly, marker='o')
    plt.title('Monthly Post Activity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_trend.png'))
    plt.close()

# 7. Print analysis summary
def print_analysis_summary(df, topics):
    """Print summary of the analysis results"""
    
    # Overall statistics
    print("\n==== LinkedIn Post Analysis Summary ====\n")
    print(f"Total posts analyzed: {len(df)}")
    print(f"Companies included: {', '.join(df['company'].unique())}")
    print(f"Date range: {df['published_time'].min().date()} to {df['published_time'].max().date()}")
    
    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment.capitalize()}: {count} posts ({count/len(df)*100:.1f}%)")
    
    # Engagement statistics
    print("\nAverage Engagement by Sentiment:")
    engagement = df.groupby('sentiment').agg({
        'num_likes': 'mean',
        'num_comments': 'mean',
        'num_shares': 'mean'
    })
    print(engagement.round(2))
    
    # Top topics
    print("\nDetected Topics:")
    for i, topic_words in enumerate(topics):
        print(f"  Topic {i+1}: {', '.join(topic_words[:5])}")
    
    # Top companies by engagement
    print("\nCompany Engagement Ranking (by average likes):")
    company_engagement = df.groupby('company')['num_likes'].mean().sort_values(ascending=False)
    for i, (company, avg_likes) in enumerate(company_engagement.items(), 1):
        print(f"  {i}. {company}: {avg_likes:.1f} likes per post")
    
    print("\n==== Analysis Complete ====\n")

# Run the analysis pipeline
df, topics = analyze_linkedin_posts()
if df is not None:
    visualize_results(df, topics)
    print_analysis_summary(df, topics)