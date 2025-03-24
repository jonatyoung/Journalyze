import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

with open('./data/scholar_results.txt', 'r', encoding='utf-8') as file:
    judul_list = [line.strip() for line in file if line.strip()]

df = pd.DataFrame(judul_list, columns=['judul'])

df['cleaned_judul'] = df['judul'].astype(str).apply(preprocess_text)

df.to_csv('./data/preprocessed/preprocessed_data.csv', index=False)