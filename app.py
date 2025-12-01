import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data (only needed first time)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load the saved models
@st.cache_resource
def load_models():
    # Load Naive Bayes model
    with open('naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    
    # Load Logistic Regression model
    with open('logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    
    # Load TF-IDF vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return nb_model, lr_model, vectorizer

# Text preprocessing function
def preprocess_text(text):
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Clean text: remove non-alphabetic characters and convert to lowercase
    clean_text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    
    # Split into words, remove stopwords, and lemmatize
    words = [lemmatizer.lemmatize(word) for word in clean_text.split() 
             if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# Streamlit app
def main():
    st.title("📊 Sentiment Analysis App")
    st.write("Enter text below to analyze its sentiment!")
    
    # Load models
    try:
        nb_model, lr_model, vectorizer = load_models()
    except FileNotFoundError:
        st.error("Model files not found! Please make sure you've trained and saved the models first.")
        return
    
    # Text input
    user_input = st.text_area("Enter your text here:", height=100)
    
    # Model selection
    model_choice = st.selectbox(
        "Choose a model:",
        ["Naive Bayes", "Logistic Regression"]
    )
    
    # Predict button
    if st.button("Analyze Sentiment"):
        if user_input:
            # Preprocess the text
            processed_text = preprocess_text(user_input)
            
            # Convert to TF-IDF features
            text_features = vectorizer.transform([processed_text]).toarray()
            
            # Make prediction based on selected model
            if model_choice == "Naive Bayes":
                prediction = nb_model.predict(text_features)[0]
                confidence = max(nb_model.predict_proba(text_features)[0])
            else:  # Logistic Regression
                prediction = lr_model.predict(text_features)[0]
                confidence = max(lr_model.predict_proba(text_features)[0])
            
            # Display results
            st.subheader("Results:")
            
            # Sentiment mapping
            sentiment_map = {-1: "Negative 😔", 0: "Neutral 😐", 1: "Positive 😊"}
            color_map = {-1: "red", 0: "gray", 1: "green"}
            
            sentiment = sentiment_map[prediction]
            color = color_map[prediction]
            
            st.markdown(f"**Sentiment:** <span style='color: {color}; font-size: 24px;'>{sentiment}</span>", 
                       unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence:.2%}")
            st.write(f"**Model Used:** {model_choice}")
            
            # Show processed text
            with st.expander("Show processed text"):
                st.write(f"Original: {user_input}")
                st.write(f"Processed: {processed_text}")
                
        else:
            st.warning("Please enter some text to analyze!")
    
    # Add some information about the app
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses machine learning to analyze the sentiment of text.
    
    **Sentiment Categories:**
    - 😊 Positive: Happy, good emotions
    - 😐 Neutral: Neither positive nor negative  
    - 😔 Negative: Sad, bad emotions
    
    **Models Available:**
    - Naive Bayes
    - Logistic Regression
    """)

if __name__ == "__main__":
    main()
