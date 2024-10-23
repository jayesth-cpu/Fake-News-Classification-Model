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
from sklearn.model_selection import learning_curve


true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

true_news['label'] = 1
fake_news['label'] = 0

all_news = pd.concat([true_news, fake_news], ignore_index=True)


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet') #for lemmatization

def clean_text(text):

    #convert to lowercase
    text = text.lower()

    #remove html tags
    text = re.sub(r"<.*?", " ", text)

    #remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]','',text)

    #tokenize the text
    tokens = word_tokenize(text)

    #remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)

all_news['clean_text'] = all_news['text'].apply(clean_text)
print(all_news[['text','clean_text']].head())

all_news.to_csv('preprocessed_news.csv',index=False)


#Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens =[lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

all_news['lemmatized_text'] = all_news['clean_text'].apply(lemmatize_text)

#feature extraction
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(all_news['clean_text']).toarray()

# Add the features to the DataFrame
all_news['tfidf_features'] = list(X)

X = vectorizer.fit_transform(all_news['clean_text'])
y = all_news['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train multiple models
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print(f"\nResults for {model_name}:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


# Initialize models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Train and evaluate each model
trained_models = {}
for name, model in models.items():
    trained_models[name] = train_and_evaluate_model(model, name, X_train, X_test, y_train, y_test)


def evaluate_models():
    # Prepare features (X) and labels (y)
    X = vectorizer.fit_transform(all_news['clean_text'])
    y = all_news['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary to store results
    results = {}

    # Initialize models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Store results
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'true_values': y_test,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        # Print results
        print(f"\nResults for {name}:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    return results, X_train, X_test, y_train, y_test


# Run the evaluation
results, X_train, X_test, y_train, y_test = evaluate_models()


def compare_models(results):
    # Create a comparison DataFrame
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

    # Determine best model based on accuracy
    best_model = max(results.items(), key=lambda x: x[1]['report']['accuracy'])
    print(f"\nBest performing model: {best_model[0]}")
    return comparison_df


# Compare models
comparison_df = compare_models(results)


# Function to make predictions on new text
def predict_news(text, model, vectorizer):
    # Clean and preprocess the text
    cleaned_text = clean_text(text)
    # Transform the text using the same vectorizer
    text_vector = vectorizer.transform([cleaned_text])
    # Make prediction
    prediction = model.predict(text_vector)
    probability = model.predict_proba(text_vector)

    return prediction[0], probability[0]


# Save the best model and vectorizer (let's say we choose Logistic Regression)
import pickle

best_model = trained_models["Logistic Regression"]  # You can change this based on performance
model_filename = 'fake_news_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

# Save the model and vectorizer
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)
with open(vectorizer_filename, 'wb') as file:
    pickle.dump(vectorizer, file)

# Example of how to use the prediction function
example_text = """
SAN JOSE (Reuters) - Women who were sexually tortured by Mexican security forces over a decade ago testified Thursday before the Inter-American Court of Human Rights, asking for an investigation into the case that happened in the state once run by President Enrique Pena Nieto. Several of the 11 victims told judges at the Costa-Rica based court about the abuse they suffered after they were detained following a protest in May 2006 at the town of San Salvador de Atenco, about 25 miles (40 km) northeast of Mexico City. The town is located in the State of Mexico, which rings the capital.   We ve come here to speak out. In Mexico, justice has not been done,  said Maria Cristina Sanchez, 50, who detailed how she was beaten, sexually abused and how the case languished in Mexico s criminal justice system with no resolution for years. The women, known as the  Women of Atenco , say they were thrown into a police bus, raped and tortured following a two-day protest by a group of flower sellers who had negotiated a labor agreement that allowed them to set up stalls in a nearby downtown area. The women were initially accused of illegally blocking public access, but later acquitted of the charges. Before being elected president in 2012, Pena Nieto was governor of the State of Mexico, heading the country s most populous state from in late 2005 till 2011.  The Mexican government reiterates the recognition of its international responsibility... and its sincere will to fully repair the human rights violations in this case,  said Uriel Salas, an attorney representing Mexico in the case. In the years since the abuses were committed, some police were accused of crimes, but there have been no convictions.  We deserve the recognition that we re telling the truth, that the chain of command be investigated, not only so that justice is done, but so that these events never happen again,  said Norma Jimenez, 33, at the end of her testimony. The court is expected to continue hearing testimony through Friday, while a final ruling in the case could take months.
"""

prediction, probability = predict_news(example_text, best_model, vectorizer)
print("\nExample Prediction:")
print(f"Prediction: {'Real' if prediction == 1 else 'Fake'}")
print(f"Confidence: {max(probability) * 100:.2f}%")

# Add some basic error handling
try:
    # Calculate and print some additional statistics
    print("\nDataset Statistics:")
    print(f"Total number of articles: {len(all_news)}")
    print(f"Number of real news: {len(all_news[all_news['label'] == 1])}")
    print(f"Number of fake news: {len(all_news[all_news['label'] == 0])}")

    # Calculate average text length
    all_news['text_length'] = all_news['clean_text'].str.len()
    print(f"\nAverage text length:")
    print(f"Real news: {all_news[all_news['label'] == 1]['text_length'].mean():.2f}")
    print(f"Fake news: {all_news[all_news['label'] == 0]['text_length'].mean():.2f}")

except Exception as e:
    print(f"An error occurred while calculating statistics: {str(e)}")


import pickle


def get_user_input():
    print("\nEnter your news article (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

article = get_user_input()
custom_articles = [article]

# article_from_input = get_user_input()
#
# # Option A: Directly in Code
# articles_from_code = [
#     """NASA has successfully launched a new satellite to study climate change...""",
#     """A new study shows that drinking herbal tea can cure COVID-19, which has been debunked by experts..."""
# ]
#
# # You can now choose which set of articles to use for prediction
# custom_articles = articles_from_code

# Load model and vectorizer
def load_model_and_vectorizer():
    with open('fake_news_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


# Function to make predictions
def predict_custom_news(articles, model, vectorizer):
    cleaned_articles = [clean_text(article) for article in articles]
    article_vectors = vectorizer.transform(cleaned_articles)
    predictions = model.predict(article_vectors)
    probabilities = model.predict_proba(article_vectors)

    results = []
    for i, article in enumerate(articles):
        results.append({
            "article": article,
            "prediction": "Real" if predictions[i] == 1 else "Fake",
            "confidence": max(probabilities[i]) * 100
        })

    return pd.DataFrame(results)


# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Predict on custom articles
results_df = predict_custom_news(custom_articles, model, vectorizer)

# Print results
for i, row in results_df.iterrows():
    print(f"\nArticle {i + 1}:")
    print(f"Prediction: {row['prediction']}")
    print(f"Confidence: {row['confidence']:.2f}%")


# Add this code at the end of your script

def visualize_results():
    print("\nGenerating Visualizations...")

    # 1. Model Performance Comparison
    plt.figure(figsize=(12, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    model_names = list(results.keys())

    performance_data = {model: [] for model in model_names}
    for name in model_names:
        report = results[name]['report']
        performance_data[name] = [
            report['accuracy'],
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            report['weighted avg']['f1-score']
        ]

    x = np.arange(len(metrics))
    width = 0.25

    for i, (model, scores) in enumerate(performance_data.items()):
        plt.bar(x + i * width, scores, width, label=model)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.show()

    # 2. Dataset Balance Visualization
    plt.figure(figsize=(8, 6))
    labels = ['Real News', 'Fake News']
    sizes = [len(all_news[all_news['label'] == 1]),
             len(all_news[all_news['label'] == 0])]
    colors = ['lightgreen', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Distribution of Real vs Fake News')
    plt.axis('equal')
    plt.show()

    # 3. Text Length Distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='label', y='text_length', data=all_news)
    plt.title('Text Length Distribution by News Type')
    plt.xlabel('News Type (0: Fake, 1: Real)')
    plt.ylabel('Text Length')
    plt.show()

    # 4. Model Performance Over Time (Training History)
    def plot_learning_curve(model, X, y, title):
        train_sizes = np.linspace(0.1, 1.0, 10)
        plt.figure(figsize=(10, 6))

        for name, clf in models.items():
            train_sizes, train_scores, test_scores = learning_curve(
                clf, X, y, train_sizes=train_sizes, cv=5,
                scoring='accuracy', n_jobs=-1)

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            plt.plot(train_sizes, train_mean, label=f'{name} (Training)')
            plt.plot(train_sizes, test_mean, label=f'{name} (Testing)')
            plt.fill_between(train_sizes, train_mean - train_std,
                             train_mean + train_std, alpha=0.1)
            plt.fill_between(train_sizes, test_mean - test_std,
                             test_mean + test_std, alpha=0.1)

        plt.xlabel('Training Examples')
        plt.ylabel('Accuracy Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    # Add these imports at the top of your file
    from sklearn.model_selection import learning_curve

    # Plot learning curves
    plot_learning_curve(models, X, y, 'Learning Curves for Different Models')

    # 5. Feature Importance (for Random Forest)
    rf_model = trained_models["Random Forest"]
    feature_names = vectorizer.get_feature_names_out()

    # Get feature importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features

    plt.figure(figsize=(12, 8))
    plt.title('Top 20 Most Important Features')
    plt.barh(range(20), importances[indices])
    plt.yticks(range(20), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


# Call the visualization function
visualize_results()


# Add a function to save the visualizations
def save_visualizations():
    try:
        plt.savefig('model_performance.png')
        plt.savefig('dataset_distribution.png')
        plt.savefig('text_length_distribution.png')
        plt.savefig('learning_curves.png')
        plt.savefig('feature_importance.png')
        print("Visualizations saved successfully!")
    except Exception as e:
        print(f"Error saving visualizations: {str(e)}")


# Save the visualizations
save_visualizations()

