import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Configure paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'fake_news_model.pkl'
VECTORIZER_PATH = BASE_DIR / 'tfidf_vectorizer.pkl'
VISUALIZATION_DIR = BASE_DIR / 'visualizations'

# Ensure visualization directory exists
VISUALIZATION_DIR.mkdir(exist_ok=True)


def debug_file_paths(visualization_dir):
    """Check and report the status of all expected files"""
    expected_files = [
        'model_performance.png',
        'feature_importance.png',
        'learning_curve_Random Forest.png',
        'dataset_distribution.png',
        'text_length_distribution.png'
    ]

    debug_info = []
    debug_info.append(f"Current working directory: {os.getcwd()}")
    debug_info.append(f"Visualization directory: {visualization_dir}")
    debug_info.append(f"Visualization directory exists: {visualization_dir.exists()}")

    if visualization_dir.exists():
        debug_info.append("\nFiles in visualization directory:")
        for file in visualization_dir.glob('*'):
            debug_info.append(f"- {file.name}")

        debug_info.append("\nExpected files status:")
        for file in expected_files:
            file_path = visualization_dir / file
            debug_info.append(f"- {file}: {'Found' if file_path.exists() else 'Missing'}")

    return '\n'.join(debug_info)


# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')


download_nltk_data()


def clean_text(text):
    """Clean and preprocess input text."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)  # Fixed regex pattern
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer with proper error handling."""
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        with open(VECTORIZER_PATH, 'rb') as file:
            vectorizer = pickle.load(file)
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Required files not found: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def load_visualization(filename, visualization_dir):
    """Enhanced visualization loading with detailed error reporting"""
    try:
        image_path = visualization_dir / filename
        st.write(f"Attempting to load: {image_path}")  # Debug info

        if not visualization_dir.exists():
            st.error(f"Visualization directory not found: {visualization_dir}")
            return None

        if not image_path.exists():
            st.warning(f"Visualization file not found: {filename}")
            return None

        image = Image.open(image_path)
        st.success(f"Successfully loaded: {filename}")  # Debug info
        return image

    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None


def main():
    st.title("📰 Fake News Detector")

    # Setup paths
    base_dir = Path(__file__).parent
    visualization_dir = base_dir / 'visualizations'

    # Create visualization directory if it doesn't exist
    visualization_dir.mkdir(exist_ok=True)

    # Create sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigate to",
        ["News Analysis", "Model Performance", "Dataset Statistics"]
    )

    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    if model is None or vectorizer is None:
        st.stop()

    if page == "News Analysis":
        show_news_analysis(model, vectorizer)
    elif page == "Model Performance":
        show_model_performance(visualization_dir)
    elif page == "Dataset Statistics":
        show_dataset_statistics(visualization_dir)
    else:
        pass

def show_news_analysis(model, vectorizer):
    st.header("Analyze News Article")

    # Create two columns for input methods
    col1, col2 = st.columns(2)

    with col1:
        user_input = st.text_area(
            "Enter the news article text:",
            height=300,
            placeholder="Paste your article here..."
        )

    with col2:
        uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])
        if uploaded_file:
            user_input = uploaded_file.getvalue().decode()

    if st.button("Analyze Article", key="analyze_button"):
        if user_input:
            with st.spinner("Analyzing..."):
                # Preprocess and predict
                cleaned_text = clean_text(user_input)
                text_vector = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_vector)[0]
                probability = model.predict_proba(text_vector)[0]

                # Display results
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)

                with col1:
                    if prediction == 1:
                        st.success("📰 This article appears to be **REAL**")
                    else:
                        st.error("⚠️ This article appears to be **FAKE**")

                with col2:
                    confidence = max(probability) * 100
                    st.metric(
                        label="Confidence Level",
                        value=f"{confidence:.1f}%"
                    )
        else:
            st.warning("Please enter or upload an article to analyze.")


def show_model_performance(visualization_dir):
    st.header("Model Performance Metrics")

    # Add debug information
    with st.expander("Debug Information"):
        st.code(debug_file_paths(visualization_dir))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance Comparison")
        model_perf = load_visualization('model_performance.png', visualization_dir)
        if model_perf:
            st.image(model_perf)

        st.subheader("Feature Importance")
        feature_imp = load_visualization('feature_importance.png', visualization_dir)
        if feature_imp:
            st.image(feature_imp)

    with col2:
        st.subheader("Random Forest Learning Curve")
        learning_curve = load_visualization('learning_curve_Random Forest.png', visualization_dir)
        if learning_curve:
            st.image(learning_curve)

        st.subheader("Naive Bayes Learning Curve")
        learning_curve = load_visualization('learning_curve_Naive Bayes.png', visualization_dir)
        if learning_curve:
            st.image(learning_curve)

        st.subheader("Logistic Regression Learning Curve")
        learning_curve = load_visualization('learning_curve_Logistic Regression.png', visualization_dir)
        if learning_curve:
            st.image(learning_curve)


def show_dataset_statistics(visualization_dir):
    st.header("Dataset Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Distribution")
        dist_plot = load_visualization('dataset_distribution.png', visualization_dir)
        if dist_plot:
            st.image(dist_plot)

    with col2:
        st.subheader("Text Length Distribution")
        length_dist = load_visualization('text_length_distribution.png', visualization_dir)
        if length_dist:
            st.image(length_dist)


if __name__ == "__main__":
    main()