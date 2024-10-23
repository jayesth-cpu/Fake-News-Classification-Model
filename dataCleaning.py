import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

true_news['label'] = 1
fake_news['label'] = 0

all_news = pd.concat([true_news, fake_news], ignore_index=True)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # For lemmatization


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


all_news['clean_text'] = all_news['text'].apply(clean_text)

# Lemmatization
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


all_news['lemmatized_text'] = all_news['clean_text'].apply(lemmatize_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(all_news['clean_text'])
y = all_news['label']


# Evaluation Function
def evaluate_models():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'true_values': y_test,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        print(f"\nResults for {name}:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    return results, X_train, X_test, y_train, y_test


# Run the evaluation
results, X_train, X_test, y_train, y_test = evaluate_models()


# Model Comparison Function
def compare_models(results):
    comparison = {}
    for name, result in results.items():
        metrics = result['report']
        comparison[name] = {
            'Accuracy': metrics['accuracy'],
            'Precision (Fake)': metrics['0']['precision'],
            'Recall (Fake)': metrics['0']['recall'],
            'F1-score (Fake)': metrics['0']['f1-score'],
            'Precision (Real)': metrics['1']['precision'],
            'Recall (Real)': metrics['1']['recall'],
            'F1-score (Real)': metrics['1']['f1-score']
        }
    comparison_df = pd.DataFrame(comparison).round(3)
    print("\nModel Comparison:")
    print(comparison_df)
    best_model = max(results.items(), key=lambda x: x[1]['report']['accuracy'])
    print(f"\nBest performing model: {best_model[0]}")
    return comparison_df


# Compare models
comparison_df = compare_models(results)


# Prediction Testing Function
def test_prediction_system():
    best_model_name = comparison_df.idxmax().mode()[0]
    best_model = results[best_model_name]['model']

    def predict_news_article(text):
        cleaned_text = clean_text(text)
        text_vector = vectorizer.transform([cleaned_text])
        prediction = best_model.predict(text_vector)[0]
        probability = best_model.predict_proba(text_vector)[0]
        return prediction, probability

    test_articles = [

        "Zero UK growth in 2023"
        ,

        "Trump's team accuses Labour of election interference as Starmer tries to play down row"
        ,
    ]

    print("\nTesting Prediction System:")
    print(f"Using model: {best_model_name}")
    print("\nResults:")

    for i, article in enumerate(test_articles, 1):
        prediction, probability = predict_news_article(article)
        print(f"\nTest Article {i}:")
        print(f"Prediction: {'Real' if prediction == 1 else 'Fake'}")
        print(f"Confidence: {max(probability) * 100:.2f}%")
        print("-" * 50)


# Run the prediction test
test_prediction_system()
