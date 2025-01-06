# Fake News Detection Project

## Overview
Fake news detection is a critical task in today's digital age to combat misinformation. This project aims to build a machine learning pipeline that identifies whether a given news article is fake or genuine. The implementation leverages natural language processing (NLP) techniques, multiple classification algorithms, and data visualization.

## Features
- **Text Preprocessing**: Cleans and prepares the text data for model training.
- **Feature Extraction**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert textual data into numerical features.
- **Model Training**: Trains and evaluates multiple classification models, including Naive Bayes, Logistic Regression, and Random Forest.
- **Model Comparison**: Compares model performance using evaluation metrics and visualizations.
- **Custom Prediction**: Allows users to input custom news text for classification.
- **Reusability**: Saves trained models and vectorizers for future use.
- **Visualization**: Provides insights through dataset distribution, learning curves, and feature importance.

---

## Project Structure
```
project-root
│
├── data
│   └── fake_news_dataset.csv   # Dataset used for training and testing
│
├── notebooks
│   └── exploration.ipynb       # Initial data exploration and visualization
│
├── models
│   └── fake_news_model.pkl     # Saved trained model
│   └── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
│
├── scripts
│   └── preprocess.py           # Text preprocessing functions
│   └── train_model.py          # Model training and evaluation script
│   └── predict.py              # Custom prediction script
│
├── outputs
│   └── confusion_matrix.png    # Confusion matrix visualization
│   └── learning_curves.png     # Learning curves
│   └── feature_importance.png  # Feature importance (Random Forest)
│
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## Setup Instructions

### Prerequisites
- Python 3.7+
- Virtual environment (optional but recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/fake-news-detection.git
cd fake-news-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset
Place your dataset (e.g., `fake_news_dataset.csv`) in the `data` directory. Ensure the dataset has the following columns:
- **title**: Title of the news article
- **text**: Body of the news article
- **label**: Classification label (`1` for genuine, `0` for fake)

### Step 4: Run the Scripts

#### Preprocessing
```bash
python scripts/preprocess.py
```

#### Training Models
```bash
python scripts/train_model.py
```

#### Making Predictions
```bash
python scripts/predict.py "Enter your news text here"
```

---

## Key Components

### 1. Preprocessing
Prepares raw text data for model training:
- Removes special characters and HTML tags
- Tokenizes and lemmatizes text
- Removes stopwords

### 2. Feature Engineering
- **TF-IDF Vectorization**: Converts text into numerical features.

### 3. Models
- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Random Forest Classifier**

### 4. Evaluation Metrics
- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **Learning Curves**

### 5. Visualization
- Dataset distribution
- Confusion matrix
- Learning curves
- Feature importance (for Random Forest)

---

## Results
The trained models achieve the following metrics on the test set:
- **Naive Bayes**: Accuracy - XX%, Precision - XX%, Recall - XX%
- **Logistic Regression**: Accuracy - XX%, Precision - XX%, Recall - XX%
- **Random Forest**: Accuracy - XX%, Precision - XX%, Recall - XX%

(Replace `XX%` with actual results after training.)

---

## Custom Prediction
You can classify custom news articles using the trained model:
1. Run the `predict.py` script.
2. Enter your news text when prompted.

Example:
```bash
python scripts/predict.py "Breaking news: Stock markets hit an all-time high."
```
Output:
```
Prediction: Genuine News
Confidence Score: 92.3%
```

---

## Future Enhancements
- Add hyperparameter tuning for models.
- Use deep learning models (e.g., LSTM, BERT) for improved accuracy.
- Develop a web interface for ease of use.
- Incorporate additional datasets for better generalization.
- Add support for multilingual news classification.

---

## Dependencies
See `requirements.txt` for all dependencies. Major libraries include:
- `nltk` for text preprocessing
- `scikit-learn` for machine learning models
- `matplotlib` and `seaborn` for visualization
- `pandas` and `numpy` for data manipulation

---

## Acknowledgements
- [Kaggle](https://www.kaggle.com/) for providing datasets
- Open-source libraries for enabling this project

---

## Contact
For questions or suggestions, please reach out to [your-email@example.com](mailto:your-email@example.com).

---

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
