import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer

df1 = pd.read_csv("train (2).csv", sep=';')
df1.head()
df2 = pd.read_csv("evaluation.csv", sep=";")
df2.head()
df3 = pd.read_csv("test (1).csv", sep=";")
df = pd.concat([df1, df2,df3], ignore_index=True)
df.head()
df.shape
df.dtypes
_ = [print(f'{df[column].value_counts()}\n') for column in df.columns]
df.nunique()
df['Unnamed: 0']
df.label
df.label.value_counts()
df.isna().sum()
df = df.drop(columns = ['Unnamed: 0'])
df.shape
df.dtypes
sns.countplot(data=df, x='label')
plt.title('Distribution of labels')
plt.xlabel('label')
plt.ylabel("Count")
plt.show()

def preprocess_text(text):
    """
    Function to cleas=n and tokenize text.

    Params:

        text: str

    returns:
        list

    """
    text = re.sub(r'\W+', ' ', text)

    tokens = text.lower().split()
    stopwords = set(['the', 'a','and','is','to','in'])
    return [word for word in tokens if word not in stopwords]

all_words = df['title'].dropna().apply(preprocess_text).sum()
word_counts = Counter(all_words).most_common(10)


word_freq_df = pd.DataFrame(word_counts, columns=['word', 'frequency'])

sns.barplot(data=word_freq_df, x='word', y='frequency')
plt.title('Top 10 words in titles')
plt.xlabel('Word')
plt.ylabel("Frequency")
plt.xticks(rotation=90)
plt.show()

def plot_top_ngrams(text_data, n=2, top_n=10):

    """
    Function to extract and plot top n-grams.

    Params:
        text_data: str
        n: int, optional
        top_n: int, optional
    """
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    ngrams = vectorizer.fit_transform(text_data.dropna())
    ngram_counts = pd.DataFrame(ngrams.sum(axis=0), columns=vectorizer.get_feature_names_out(), index=['count']).T
    top_ngrams = ngram_counts.nlargest(top_n, 'count')

    # plot top n-grams
    sns.barplot(data=top_ngrams.reset_index(), x='count', y='index')
    plt.title(f'Top {top_n} {n}-grams')
    plt.xlabel('Frequency')
    plt.ylabel(f'{n}-gram')
    plt.show()

# plot top 2-grams for text column
plot_top_ngrams(df['text'], n=2, top_n=10)

df['text_length'] = df['text'].apply(lambda x : len(str(x)))

sns.histplot(data=df, x='text_length', hue = 'label', bins=30, kde=True)
plt.title("Text Lenth Distribution by Label")
plt.xlabel('Text Lenth')
plt.ylabel("Frequency")
plt.show()

tfidf = TfidfVectorizer(stop_words='english', max_features=20)
tfidf_matrix = tfidf.fit_transform(df['text'].fillna(''))
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

tfidf_df['label'] = df['label']

tfidf_avg = tfidf_df.groupby('label').mean().T

tfidf_avg.plot(kind='bar', figsize=(10,6))
plt.title('Average TF_IDF Scores by Label')
plt.ylabel('Average TF-IDF Score')
plt.xlabel('Words')
plt.show()

tfidf = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf.fit_transform(df['text'].fillna(''))
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())
plt.figure(figsize=(10,7))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['label'], palette='viridis', s=50)
plt.title('2D Visualization of Text Embeddings by Label')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Label')
plt.show()

def plot_wordcloud(label):
    """
    Function to generate word cloud for a given label.
    Params:
        label: str
    """

    text = ' '.join(df[df['label']== label]['text'].fillna('').tolist())
    wordcloud = WordCloud(width=800,
                          height=400,
                          background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for label {label}')
    plt.show()


for label in df['label'].unique():
    plot_wordcloud(label)

df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

sns.histplot(data=df,
            x='sentiment',
            hue='label',
            bins=20,
            kde=True)

plt.title("Sentiment Polarity distribution by Label")

plt.xlabel('Sentiment Polarity')

plt.ylabel("Frequency")
plt.show()

# initialize CountVectorizer to get word co-occurrence
vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(df['text'].dropna())
word_cooccurrence = (X.T * X)  # multiply term-document matrix by its transpose to get co-occurrence
word_cooccurrence.setdiag(0)  # remove diagonal (self-co-occurrence)

# convert to DataFrame for heatmap
word_cooccurrence_df = pd.DataFrame(word_cooccurrence.toarray(), index=vectorizer.get_feature_names_out(), columns=vectorizer.get_feature_names_out())

# plot the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(data=word_cooccurrence_df,
            annot=True,
            cbar=False,
            fmt='d')
plt.title('Word Co-Occurrence in Texts')
plt.show()

# calculate text length and word count
df['text_length'] = df['text'].apply(lambda x: len(str(x)))
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# scatter plot of text length vs. word count by label
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df,
                x='text_length',
                y='word_count',
                hue='label')
plt.title('Text Length vs. Word Count by Label')
plt.xlabel('Text Length')
plt.ylabel('Word Count')
plt.show()

# extract bigrams (2-grams) for the text column
vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=20)
X = vectorizer.fit_transform(df['text'].fillna(''))

# create DataFrame of bigram counts
bigrams_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
bigrams_df['label'] = df['label']

# calculate average bigram frequency for each label
bigrams_avg = bigrams_df.groupby('label').mean().T

# plot as heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data=bigrams_avg,
            annot=True,
            cbar=False,
            fmt='.2f')
plt.title('Average Bigram Frequency by Label')
plt.xlabel('Label')
plt.ylabel('Bigram')
plt.show()

# calculate average word length with handling for empty or whitespace-only text entries
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if x and len(str(x).split()) > 0 else 0)

# apply a log transformation to the average word length to manage skewness
df['log_avg_word_length'] = np.log1p(df['avg_word_length'])  # log1p handles zero values

# plot average word length by label with log scale
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='label', y='log_avg_word_length')
plt.title('Log-Scaled Average Word Length by Label')
plt.xlabel('Label')
plt.ylabel('Log-Scaled Average Word Length')
plt.show()

# create TF-IDF vectorizer and apply to the 'text' column
tfidf = TfidfVectorizer(stop_words='english', max_features=50)
tfidf_matrix = tfidf.fit_transform(df['text'].fillna(''))

# get TF-IDF values in a DataFrame with labels as columns
unique_words = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
unique_words['label'] = df['label']

# calculate average TF-IDF for each label
top_words_by_label = unique_words.groupby('label').mean().T
top_words_by_label['max_label'] = top_words_by_label.idxmax(axis=1)  # label with highest tf-idf for each word

# iterate over each label and plot the top 10 words unique to that label
for label in df['label'].unique():
    # filter the top 10 words for the current label
    top_words = top_words_by_label[top_words_by_label['max_label'] == label].nlargest(10, label)
    # create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_words[label],
                y=top_words.index,
                orient='h')
    plt.title(f'Top 10 Unique Words for Label {label}')
    plt.xlabel('Average TF-IDF Score')
    plt.ylabel('Word')
    plt.show()

# extract the top words for analysis
vectorizer = CountVectorizer(stop_words='english', max_features=10)
X = vectorizer.fit_transform(df['text'].fillna(''))
word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
word_counts['label'] = df['label']

# calculate word frequency by label
word_freq_by_label = word_counts.groupby('label').sum().T

# plot stacked bar chart
word_freq_by_label.plot(kind='bar',
                        stacked=True,
                        figsize=(12, 8))
plt.title('Top 10 Words Frequency by Label')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.legend(title='Label')
plt.show()

# plot label distribution as a pie chart
plt.figure(figsize=(8, 8))
df['label'].value_counts().plot.pie(autopct='%1.1f%%',
                                    startangle=90)
plt.title('Label Distribution')
plt.ylabel('')
plt.show()

# calculate word count for each text entry
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# plot word count by label as a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df,
               x='label',
               y='word_count',
               inner='quartile',
               density_norm='width')
plt.title('Word Count Distribution by Label')
plt.xlabel('Label')
plt.ylabel('Word Count')
plt.show()

# calculate sentence count for each text entry
df['sentence_count'] = df['text'].apply(lambda x: len(sent_tokenize(str(x))))

# plot sentence count distribution by label
plt.figure(figsize=(10, 6))
sns.histplot(data=df,
             x='sentence_count',
             hue='label',
             bins=20,
             kde=True)
plt.title('Sentence Count Distribution by Label')
plt.xlabel('Sentence Count')
plt.ylabel('Frequency')
plt.show()

# calculate sentiment polarity for each text entry
df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# plot sentiment polarity by label
plt.figure(figsize=(10, 6))
sns.boxplot(data=df,
            x='label',
            y='sentiment')
plt.title('Sentiment Polarity by Label')
plt.xlabel('Label')
plt.ylabel('Sentiment Polarity')
plt.show()

# create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english', max_features=20)
tfidf_matrix = tfidf.fit_transform(df['text'].fillna(''))
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
tfidf_df['label'] = df['label']

# calculate mean TF-IDF score for each word by label
tfidf_mean = tfidf_df.groupby('label').mean().T

# plot heatmap of TF-IDF scores
plt.figure(figsize=(12, 8))
sns.heatmap(data=tfidf_mean,
            annot=True,
            cbar=False,
            fmt='.2f')
plt.title('TF-IDF Scores by Label')
plt.xlabel('Label')
plt.ylabel('Word')
plt.show()

# calculate text metrics with handling for empty or missing values
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if x and len(str(x).split()) > 0 else 0)
df['sentence_count'] = df['text'].apply(lambda x: len(TextBlob(str(x)).sentences) if x else 0)
df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if x else 0)

# aggregate metrics by label, filling in any NaN values with 0
radar_data = df.groupby('label')[['avg_word_length',
                                  'sentence_count',
                                  'sentiment']].mean().fillna(0)

# create radar chart
categories = list(radar_data.columns)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# iterate through radar_data to plot each label
for i, row in radar_data.iterrows():
    values = row.values.flatten().tolist()
    values += values[:1]  # close the loop
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]

    # plot each label's values
    ax.plot(angles, values, label=f'Label {i}')
    ax.fill(angles, values, alpha=0.1)

# set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# add title and legend
plt.title('Radar Chart of Text Features by Label')
plt.legend(loc='upper right')
plt.show()

# create TF-IDF embeddings
tfidf = TfidfVectorizer(stop_words='english', max_features=100)
tfidf_matrix = tfidf.fit_transform(df['text'].fillna(''))

# apply PCA to reduce to 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# create scatter plot of PCA results
plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['label'], palette="viridis", s=50)
plt.title('PCA of Text Embeddings by Label')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Label')
plt.show()

# calculate word count and sentiment polarity if not already calculated
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# aggregate mean values for word count and sentiment by label
heatmap_data = df.groupby('label')[['word_count', 'sentiment']].mean()

# plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data=heatmap_data,
            annot=True,
            cbar=False,
            fmt='.2f')
plt.title('Average Word Count and Sentiment by Label')
plt.xlabel('Text Metrics')
plt.ylabel('Label')
plt.show()

# scatter plot of word count vs. sentiment by label
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df,
                x='word_count',
                y='sentiment',
                hue='label',
                alpha=0.7)
plt.title('Word Count vs. Sentiment by Label')
plt.xlabel('Word Count')
plt.ylabel('Sentiment Polarity')
plt.legend(title='Label')
plt.show()

# calculate average sentence length with handling for empty or missing text entries
df['avg_sentence_length'] = df['text'].apply(lambda x: np.mean([len(sentence.split()) for sentence in TextBlob(str(x)).sentences]) if x and len(TextBlob(str(x)).sentences) > 0 else 0)

# density plot of average sentence length by label
plt.figure(figsize=(10, 6))
for label in df['label'].unique():
    subset = df[df['label'] == label]
    sns.kdeplot(subset['avg_sentence_length'], label=f'Label {label}', fill=True)
plt.title('Average Sentence Length Distribution by Label')
plt.xlabel('Average Sentence Length')
plt.ylabel('Density')
plt.legend(title='Label')
plt.show()

# calculate label distribution
label_counts = df['label'].value_counts()

# plot donut chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts,  # pass label_counts as the first argument
        labels=label_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.85)

# add center circle to create the donut effect
center_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)

# set title and display plot
plt.title('Label Distribution')
plt.show()

# calculate correlation matrix of text features
correlation_data = df[['word_count',
                       'avg_word_length',
                       'sentence_count',
                       'sentiment']].corr()

# plot correlation matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data=correlation_data,
            annot=True,
            cbar=False,
            vmin=-1,
            vmax=1)
plt.title('Correlation Matrix of Text Features')
plt.show()

# device selection: MPS (for macOS with Apple Silicon), CUDA, or CPU
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using Device: {device}')

class FakeNewsBinaryClassifier(nn.Module):
    """
    A binary classification neural network model for fake news detection.
    This model is designed for binary classification tasks, predicting whether
    an article is true or false based on its title and text content. It consists
    of three fully connected layers with ReLU activations and a final Sigmoid activation
    to output probabilities between 0 and 1.
    """

    def __init__(self, input_dim):
        """
        Initializes the FakeNewsBinaryClassifier model with the specified input dimensions.

        Params:
            input_dim: int
        """
        super(FakeNewsBinaryClassifier, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1),
                                     nn.Sigmoid())

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Params:
            x: torch.Tensor

        Returns:
            torch.Tensor
        """
        return self.network(x)

# combine title and text columns, then vectorize them
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['title'] + " " + df['text']).toarray()
y = df['label'].values  # labels for training

# convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# move tensors to the selected device
X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)

# parameters
num_epochs = 3
kf = KFold(n_splits=3, shuffle=True, random_state=101)
fold_results = []
best_accuracy = 0
best_auc = 0  # initialize best AUC score
best_model_state = None  # initialize to store the best model state

# 5-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
    print(f"Fold {fold+1}")

    # split data for this fold
    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

    # DataLoader for the current fold
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # initialize model, loss function, and optimizer for each fold
    model = FakeNewsBinaryClassifier(input_dim=1000).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop for each fold
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # move inputs and labels to device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

    # evaluate on the validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_preds = (val_outputs >= 0.5).float()

        # calculate metrics for the current fold
        accuracy = accuracy_score(y_val.cpu(), val_preds.cpu())
        auc = roc_auc_score(y_val.cpu(), val_outputs.cpu())
        print(f'Fold {fold + 1} Accuracy: {accuracy:.4f}, AUC: {auc:.4f}')

        # store results for this fold
        fold_results.append((accuracy, auc))
        # save the model if it's the best based on AUC
        if auc > best_auc:
            best_accuracy = accuracy
            best_auc = auc
            best_model_state = model.state_dict()  # save the best model's state_dict

# calculate average metrics across all folds
average_accuracy = np.mean([result[0] for result in fold_results])
average_auc = np.mean([result[1] for result in fold_results])

# provide results
print(f'3-Fold Cross-Validation Results')
print(f'Average Accuracy: {average_accuracy:.4f}')
print(f'Average AUC: {average_auc:.4f}')

# after cross-validation, save the best model
if best_model_state is not None:
    torch.save(best_model_state, 'fake_news_classifier.pth')
    print(f'Best Model w/ Accuracy: {best_accuracy:.4f}, AUC: {best_auc:.4f}')

# load the model for inference
loaded_model = FakeNewsBinaryClassifier(input_dim=1000).to(device)
loaded_model.load_state_dict(torch.load('fake_news_classifier.pth', weights_only=True))
loaded_model.eval()

def predict(model, X):
    """
    Generates a binary prediction ('True' or 'False') for the input data using the specified model.

    Params:
        model: torch.nn.Module
        X: torch.Tensor

    Returns:
        str
    """
    with torch.no_grad():
        output = model(X)
        prediction = (output >= 0.5).float()
    return 'True' if prediction.item() == 1 else 'False'

# new article for inference (no label)
sample_title = 'Palestinians switch off Christmas lights in Bethlehem in anti-Trump protest.'
sample_text = 'RAMALLAH, West Bank (Reuters) - Palestinians switched off Christmas lights at Jesus traditional birthplace.'

# combine and vectorize, then move to device
sample_combined = sample_title + ' ' + sample_text
sample_vector = vectorizer.transform([sample_combined]).toarray()
sample_tensor = torch.tensor(sample_vector, dtype=torch.float32).to(device)

# run inference
print("Prediction:", predict(loaded_model, sample_tensor))