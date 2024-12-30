
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter


def overview(data):
    """Prints basic information and statistics of the DataFrame."""
    
    print("Dataset Overview:")
    print(data.info())
    print('-' * 50)

    eda_overview = pd.DataFrame({
        "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
        "Value": [
            len(data),
            len(data.columns),
            data.isnull().sum().sum(),
            data.duplicated().sum()
        ]
    })
    print(eda_overview)
    print('-' * 100)


def clean_text(text):
    
    filter_words = [
    
        'game', 'twitter', 'twitch', 'youtube', 'google','pic',
        'com', 'instagram', 'reddit', 'tv'

        # Positive
        "borderlands", "gta", "xbox", "amazon",

        # Neutral
        "facebook", "johnson", "microsoft", "amazon", "verizon",

        # Negative
        "facebook", "microsoft", "pubg", "fortnite",

        # Irrelevant
        "facebook", "fifa", "bfbd", "csgo"
    ]

    # Preprocessing steps
    text = emoji.demojize(text) # Convert emojis to text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove digits
    
    # Convert to lowercase and tokenize
    text = text.lower()
    words_in_text = word_tokenize(text)
    cleaned_text = ' '.join([word for word in words_in_text if word not in filter_words and len(word) > 2])
    return cleaned_text


def common_words(data):

    all_words = data['cleaned_text'].str.split().explode()
    word_freq = Counter(all_words)
    common_words = word_freq.most_common(20)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), orient='h')
    plt.title("Top 20 Most Common Words in Training Dataset")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.show()


def text_length_distribution(data, data_test):

    data['text_length'] = data['cleaned_text'].str.len()
    data_test['text_length'] = data_test['cleaned_text'].str.len()

    # Plotting text length distribution
    # Visualize text lengths for both train and validation datasets
    plt.figure(figsize=(10, 6))
    sns.histplot(data['text_length'], bins=50, kde=True, color='blue', label='Train')
    sns.histplot(data_test['text_length'], bins=50, kde=True, color='orange', label='Validation')
    plt.title("Text Length Distribution")
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def classes_distribution(data):

    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment', data=data, order=data['sentiment'].value_counts().index)
    plt.title("Training Dataset Sentiment Distribution")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Count")
    plt.show()


def word_cloud(data):

    sentiments = data['sentiment'].unique()
    for sentiment in sentiments:
        sentiment_text = data[data['sentiment'] == sentiment]['cleaned_text'].str.cat(sep=' ')
        sentiment_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
        plt.figure(figsize=(10, 6))
        plt.imshow(sentiment_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for Sentiment: {sentiment}")
        plt.show()


def vectorize_train_text(train_texts, vectorizer):
    """Vectorizes train text"""
    X_train = vectorizer.fit_transform(train_texts)
    joblib.dump(vectorizer, "vectorizer.pkl")  # Save the fitted vectorizer
    return X_train


def vectorize_test_text(test_texts):
    """Vectorizes test text."""
    vectorizer = joblib.load("vectorizer.pkl")  # Load the same vectorizer used for training
    X_test = vectorizer.transform(test_texts)
    return X_test


def train_random_forest(X_train, y_train):
    """Trains a Random Forest model using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_svm(X_train, y_train):
    """Trains an SVM model using GridSearchCV."""
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'degree': [2, 3, 4],
    }
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def predict_sentiment(sentence, model, vectorizer):
    """Predicts sentiment for a given sentence using the trained model."""
    dic = {0: 'Negative', 1: 'Positive'}
    
    cleaned_sentence = clean_text(sentence)
    sentence_vectorized = vectorizer.transform([cleaned_sentence])
    prediction = model.predict(sentence_vectorized)
    return dic[prediction[0]]


def save_model(model, filename):
    """Saves the trained model to a file."""
    joblib.dump(model, filename, compress=("gzip", 3))
    print(f"Model saved to {filename}")