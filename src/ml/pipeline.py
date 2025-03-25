import numpy as np
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
import torch
from tqdm.auto import tqdm

tqdm.pandas()

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

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

def get_bert_embedding(text):
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors="pt")
    
    with torch.no_grad():  # Tidak perlu hitung gradien
        outputs = model(**tokens)
    
    # Mengambil embedding dari token [CLS] (alternatif: bisa juga pakai mean pooling)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

df['bert_embedding'] = df['cleaned_judul'].progress_apply(get_bert_embedding)

embeddings = np.vstack(df['bert_embedding'].values)
np.save("./data/preprocessed/bert_embeddings.npy", embeddings)