import pandas as pd
import numpy as np
import pickle
from bertopic import BERTopic
import umap

# 1. Load Data
df = pd.read_csv("/home/jonathan/workdir/project/mlops/topic_modelling/data/preprocessed/preprocessed_data.csv")  # Ganti dengan nama file yang sesuai
texts = df['cleaned_judul'].tolist()  # Ambil teks judul bersih
embeddings = np.load("/home/jonathan/workdir/project/mlops/topic_modelling/data/bert_embeddings.npy")  # Load embeddings dari BERT

# 2. UMAP untuk Dimensionality Reduction (Opsional, mempercepat pemrosesan)
umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)

# 3. Training BERTopic dengan Precomputed Embeddings
topic_model = BERTopic(umap_model=umap_model, embedding_model=None)  # Tanpa model tambahan
topics, probs = topic_model.fit_transform(texts, embeddings)

# 4. Menampilkan Hasil Topik
print(topic_model.get_topic_info())  # Tampilkan daftar topik

# 5. Visualisasi Topik
topic_model.visualize_barchart(top_n_topics=10)  # Grafik distribusi topik
topic_model.visualize_topics()  # Representasi topik di 2D space


# 7. Simpan Model dalam Format Pickle
with open("/home/jonathan/workdir/project/mlops/topic_modelling/models/bertopic_model.pkl", "wb") as file:
    pickle.dump(topic_model, file)

# 8. (Opsional) Menyimpan Hasil Topik ke CSV
df['topic'] = topics
df.to_csv("/home/jonathan/workdir/project/mlops/topic_modelling/data/hasil_topik.csv", index=False)