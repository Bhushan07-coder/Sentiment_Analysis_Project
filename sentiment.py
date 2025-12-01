import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from nltk import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
import nltk
import pickle
import pandas as pd
import numpy as np

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load data
data = pd.read_csv(r'/Users/bhushankumar/Downloads/SENTIMENT_NET/Datasets/Twitter_Data.csv')

def count_values_in_columns(data, features):
    total = data.loc[:, features].value_counts(dropna=False)
    percentage = round(data.loc[:, features].value_counts(dropna=False, normalize=True)*100, 2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])

print("Category Distribution:")
print(count_values_in_columns(data, 'category'))

# Separate data by sentiment
positive = data[data['category'] == 1]
negative = data[data['category'] == -1]
neutral = data[data['category'] == 0]

def create_wordcloud(text, path):
    stopwords_set = set(STOPWORDS)
    wc = WordCloud(
        background_color='white',
        max_words=3000,
        stopwords=stopwords_set,
        random_state=42,
        width=900, 
        height=500,
        repeat=True
    )
    wc.generate(str(text))
    wc.to_file(path)
    print(f'Word Cloud saved to {path}')
    
    # Display wordcloud
    plt.figure(figsize=(15, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud - {path}')
    plt.show()

# Create word clouds
print("Creating word clouds...")
create_wordcloud(data['clean_text'].values, "All.png")
create_wordcloud(positive['clean_text'].values, "positive.png")
create_wordcloud(negative['clean_text'].values, "negative.png")

# Text preprocessing
print("Preprocessing text...")
print(f"Initial data shape: {data.shape}")

# Check for missing values
print(f"Missing values in clean_text: {data['clean_text'].isnull().sum()}")
print(f"Missing values in category: {data['category'].isnull().sum()}")

# Remove rows with missing values first
data = data.dropna(subset=['clean_text', 'category'])
print(f"After removing NaN values: {data.shape}")

s = PorterStemmer()
l = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Apply lemmatization (better than stemming for this use case)
corpus2 = []
for text in data['clean_text']:
    # Convert to string and handle any remaining NaN
    text_str = str(text) if pd.notna(text) else ""
    
    # Skip empty strings
    if not text_str or text_str.lower() in ['nan', 'none', '']:
        corpus2.append("")
        continue
    
    # Clean text and lemmatize
    clean_text = re.sub('[^a-zA-Z]', ' ', text_str).lower()
    words = [l.lemmatize(word) for word in clean_text.split() 
             if word not in stop_words and len(word) > 2]
    processed_text = ' '.join(words)
    
    # Ensure we don't add empty strings
    corpus2.append(processed_text if processed_text else "empty")

data['clean_text'] = corpus2

# Remove rows with empty processed text
data = data[data['clean_text'] != ""]
data = data[data['clean_text'] != "empty"]
data = data.reset_index(drop=True)

print(f"After preprocessing: {data.shape}")
print(f"Sample processed texts:")
for i in range(min(3, len(data))):
    print(f"  {i+1}: {data['clean_text'].iloc[i][:100]}...")

# Save cleaned data
data[['clean_text', 'category']].to_csv('cleaned.csv', index=False)
print("Cleaned data saved to 'cleaned.csv'")

# Load cleaned data and verify
data_cl = pd.read_csv('cleaned.csv')

# Additional cleaning for the loaded data
print(f"Loaded data shape: {data_cl.shape}")
print(f"NaN values in clean_text: {data_cl['clean_text'].isnull().sum()}")

# Remove any remaining NaN values
data_cl = data_cl.dropna(subset=['clean_text', 'category'])

# Convert any remaining NaN to empty string and filter out
data_cl['clean_text'] = data_cl['clean_text'].fillna('')
data_cl = data_cl[data_cl['clean_text'].str.len() > 0]

print(f"Final cleaned data shape: {data_cl.shape}")

# Create TF-IDF features
print("Creating TF-IDF features...")
try:
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 3), min_df=2)
    X_tfidf = tfidf.fit_transform(data_cl['clean_text']).toarray()
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
except Exception as e:
    print(f"Error in TF-IDF transformation: {e}")
    print("Checking for problematic texts...")
    for i, text in enumerate(data_cl['clean_text'].head(10)):
        print(f"Text {i}: '{text}' (type: {type(text)})")
    raise

X = data_cl['clean_text']
y = data_cl['category']

# Create feature dataframe
feature_names = tfidf.get_feature_names_out()
data_tfidf = pd.DataFrame(X_tfidf, columns=feature_names)
data_tfidf['output'] = y

print(f"Dataset shape: {data_tfidf.shape}")
print(f"Features: {len(feature_names)}")

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train models
print("Training Naive Bayes model...")
model = MultinomialNB(alpha=0.1)
model.fit(X_train, Y_train)

print("Training Logistic Regression model...")
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, Y_train)

# Make predictions
y_pred_nb = model.predict(X_test)
y_pred_log = log_model.predict(X_test)

# Print classification reports
print("\n" + "="*50)
print("NAIVE BAYES CLASSIFICATION REPORT")
print("="*50)
print(classification_report(Y_test, y_pred_nb, target_names=['Negative', 'Neutral', 'Positive']))

print("\n" + "="*50)
print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("="*50)
print(classification_report(Y_test, y_pred_log, target_names=['Negative', 'Neutral', 'Positive']))

# Save models
print("\nSaving models...")
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(log_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Models saved successfully!")
print("Files created:")
print("- naive_bayes_model.pkl")
print("- logistic_regression_model.pkl") 
print("- tfidf_vectorizer.pkl")

# Display model accuracy
from sklearn.metrics import accuracy_score
nb_accuracy = accuracy_score(Y_test, y_pred_nb)
lr_accuracy = accuracy_score(Y_test, y_pred_log)

print(f"\nModel Accuracies:")
print(f"Naive Bayes: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")
print(f"Logistic Regression: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

print("\nTraining completed! You can now run the Streamlit app.")