import os
import fitz  # PyMuPDF para extraer texto de PDFs
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import nltk
import re
from nltk.corpus import stopwords
import hdbscan  # Importar HDBSCAN

# Descargar stopwords en español si no están descargadas
nltk.download('stopwords')
spanish_stopwords = set(stopwords.words('spanish'))  # Corregir aquí a español

# Función para extraer texto de PDFs
def extract_text_from_pdfs(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            with fitz.open(pdf_path) as doc:
                text = "\n".join(page.get_text() for page in doc)
                documents.append(text)
    return documents

# Función de preprocesamiento de texto
def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'\W+', ' ', text)  # Eliminar caracteres especiales
    words = text.split()  # Tokenizar
    words = [word for word in words if word not in spanish_stopwords]  # Eliminar stopwords
    return " ".join(words)

# Ruta de los PDFs
pdf_folder = "Documents/Repositorio"
documents = extract_text_from_pdfs(pdf_folder)

# Aplicar preprocesamiento a los textos extraídos
documents = [preprocess_text(doc) for doc in documents]

# Vectorización BoW
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents).toarray()

# Normalización para VAE
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Definir el Autoencoder Variacional
input_dim = X.shape[1]
latent_dim = 5

inputs = keras.Input(shape=(input_dim,))
h = keras.layers.Dense(10, activation='relu')(inputs)
z_mean = keras.layers.Dense(latent_dim)(h)
z_log_var = keras.layers.Dense(latent_dim)(h)

# Sampling
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

decoder_h = keras.layers.Dense(10, activation='relu')
decoder_out = keras.layers.Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_out(h_decoded)

vae = keras.Model(inputs, outputs)

# Pérdida del VAE
reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Entrenar VAE
vae.fit(X_scaled, X_scaled, epochs=50, batch_size=5)

# Extraer la representación latente
encoder = keras.Model(inputs, z_mean)
X_latent = encoder.predict(X_scaled)

# Asegurar valores no negativos
X_latent = np.maximum(0, X_latent)

# Aplicar HDBSCAN sobre la representación latente
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
clusterer.fit(X_latent)  # Ajustar el modelo de clustering

# Obtener etiquetas de los clústeres
labels = clusterer.labels_

# Función para encontrar palabras más relevantes de cada clúster
def get_top_words_per_cluster(X, vectorizer, labels, num_words=10):
    # Crear un diccionario para almacenar las palabras clave de cada clúster
    cluster_words = {}
    
    for cluster_id in np.unique(labels):
        if cluster_id != -1:  # Ignorar el ruido (-1)
            cluster_docs = X[labels == cluster_id]
            # Sumar todas las palabras de los documentos en el clúster
            cluster_word_freq = np.sum(cluster_docs, axis=0)
            # Obtener los índices de las palabras más frecuentes
            sorted_idx = np.argsort(cluster_word_freq.flatten())[::-1]
            top_words = [vectorizer.get_feature_names_out()[i] for i in sorted_idx[:num_words]]
            cluster_words[cluster_id] = top_words
    return cluster_words

# Obtener las palabras más relevantes de cada clúster
cluster_words = get_top_words_per_cluster(X, vectorizer, labels)

# Mostrar las palabras clave de cada clúster
for cluster_id, words in cluster_words.items():
    print(f"Cluster {cluster_id}:")
    print(", ".join(words))
    print()

# Mostrar los resultados de los clusters
for i, doc in enumerate(documents):
    print(f"Documento {i+1}: {doc[:200]}...")  # Mostrar solo una parte del documento
    print(f"Cluster asignado: {labels[i]}\n")  # Mostrar el cluster asignado

# Opcional: Mostrar la cantidad de documentos en cada cluster
print("Distribución de clusters:")
unique, counts = np.unique(labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"Cluster {cluster_id}: {count} documentos")
