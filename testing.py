import fitz  # PyMuPDF
import os
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
import re
print("Libraries imported")

# Adicional se descargan recursos de nlkt para tokenizar y lematizar
# 📌 Descargar recursos de NLTK si no están
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
print("NLTK resources downloaded")

## Seteamos diferentes stopwords para aislarlos del texto
# 📌 Stopwords personalizadas
stop_words = set(stopwords.words("english")).union({
    "abstract", "sample", "madrid", "introduction", "conclusion", "method", "study", "approach", 
    "paper", "result", "propose", "data", "information", "model", "analysis",
    "table", "figure", "algorithm", "system", "value", "based", "case", "using", "abrahamgutierrez", "abrahamgutierrezupmes"
})

## Lematizamos el texto (es decir, lo transformamos a su raíz, esto mediante un diccionario que tiene la biblioteca)
lemmatizer = WordNetLemmatizer()

# 📌 Función de limpieza mejorada
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar signos de puntuación
    tokens = word_tokenize(text)  # Tokenización
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatización y stopwords
    return " ".join(tokens)

# 📌 Extraer textos, títulos y keywords de PDFs
def extract_text_titles_keywords(pdf_path):
    doc = fitz.open(pdf_path)
    full_text, titles, keywords = [], [], []
    found_keywords = False

    for page_num, page in enumerate(doc):
        raw_text = page.get_text("text")
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            full_text.append(text)
                            if span["size"] > 12:  # Títulos grandes
                                titles.append(text)

        # 📌 Extraer Keywords
        if page_num == 0:
            keywords_match = re.search(r"(?i)(?:Keywords|Palabras Clave|KEYWORDS)[:\s]*(.*)", raw_text)
            if keywords_match:
                extracted_keywords = keywords_match.group(1).strip()
                if len(extracted_keywords) > 2:
                    keywords.append(extracted_keywords)
                    found_keywords = True

    if not found_keywords:
        keywords.append("")  # Evitar NaN en el DataFrame

    return " ".join(full_text), " | ".join(titles), " | ".join(keywords)

# 📌 Extraer de todos los PDFs
def extract_text_from_pdfs_in_folder(folder_path):
    pdf_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text, titles, keywords = extract_text_titles_keywords(pdf_path)
            pdf_texts.append({
                "Documento": filename,
                "Titulos_Extraidos": titles,
                "Keywords_Extraidas": keywords,
                "Texto_Original": text
            })
    return pd.DataFrame(pdf_texts)

# 📂 📌 Ruta de PDFs
folder_path = "Documents/Repositorio" 

# 📌 Extraer texto, títulos y keywords
df_pdfs_original = extract_text_from_pdfs_in_folder(folder_path)
# GENERACION DE DATAFRAME CON DATOS ORIGINALES
df_pdfs_original.to_csv("textos_originales.csv", index=False)
# 📌 Aplicar limpieza mejorada
df_pdfs_original["Texto_Procesado"] = df_pdfs_original["Texto_Original"].apply(clean_text)
df_pdfs_original["Titulos_Procesados"] = df_pdfs_original["Titulos_Extraidos"].apply(clean_text)
df_pdfs_original["Keywords_Procesadas"] = df_pdfs_original["Keywords_Extraidas"].apply(clean_text)
# 📌 🔥 **DAR MÁS PESO A TÍTULOS Y KEYWORDS**
df_pdfs_original["Texto_Final"] = (
    (df_pdfs_original["Titulos_Procesados"] + " ") * 3 +  # 🔥 Títulos tienen 3X peso
    (df_pdfs_original["Keywords_Procesadas"] + " ") * 2 +  # 🔥 Keywords tienen 2X peso
    df_pdfs_original["Texto_Procesado"]  # Texto normal
)

# 📌 Guardar DataFrames
df_pdfs_original.to_csv("textos_procesados_con_pesos.csv", index=False)
# 📌 TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=50, stop_words="english", ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df_pdfs_original["Texto_Final"]).toarray()

# 📌 Guardar TF-IDF en CSV
pd.DataFrame(X_tfidf, columns=vectorizer.get_feature_names_out()).to_csv("tfidf_vectors_pesados.csv", index=False)

df_tfidf = pd.DataFrame(X_tfidf, columns=vectorizer.get_feature_names_out())

print("\n✅ Vista previa del TF-IDF DataFrame:")
print(df_tfidf.head())  # Imprime las primeras 5 filas
# 📌 1️⃣ Cargar el DataFrame con los textos procesados
df = pd.read_csv("textos_procesados_con_pesos.csv")


# 📌 2️⃣ Dar más peso a títulos y keywords
df["Texto_Final"] = df.apply(
    lambda row: f"{' '.join([str(row['Titulos_Procesados'])]*3)} "
                f"{' '.join([str(row['Keywords_Procesadas'])]*2)} "
                f"{str(row['Texto_Procesado'])}",
    axis=1
)

# 📌 4️⃣ Escalar los embeddings TF-IDF
scaler = MinMaxScaler()
X_tfidf_scaled = scaler.fit_transform(X_tfidf)
# Definimos la dimesion latente (comprimira la entrada en solo 20 dimensiones)
latent_dim = 3  
# Capa de entrada
input_layer = keras.Input(shape=(X_tfidf_scaled.shape[1],))
#Creamos el encoder con 3 capas densas
encoder = layers.Dense(512, activation="relu")(input_layer)
# Agregamos BatchNormalization para normalizar los valores de las capas
encoder = layers.BatchNormalization()(encoder)
# Agregamos Dropout para evitar overfitting
encoder = layers.Dropout(0.3)(encoder)
encoder = layers.Dense(256, activation="relu")(encoder)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.Dropout(0.3)(encoder)
encoder = layers.Dense(128, activation="relu")(encoder)
# Se definen 2 capas, que modelan la distribucion gaussiana del espacio latente
z_mean = layers.Dense(latent_dim, name="z_mean")(encoder)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoder)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])
decoder = layers.Dense(128, activation="relu")(z)
decoder = layers.Dropout(0.2)(decoder)
decoder = layers.Dense(512, activation="relu")(decoder)
decoder = layers.Dense(X_tfidf_scaled.shape[1], activation="sigmoid")(decoder)

vae = keras.Model(input_layer, decoder)
reconstruction_loss = tf.keras.losses.mean_squared_error(input_layer, decoder)
reconstruction_loss *= X_tfidf_scaled.shape[1]
kl_loss = -0.3 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

# 📌 7️⃣ Entrenar el modelo VAE
print("\n🚀 Entrenando el VAE con más capacidad y regularización...")
vae.fit(X_tfidf_scaled, X_tfidf_scaled, epochs=10, batch_size=128, 
        validation_split=0.2)
# 📌 8️⃣ Extraer embeddings latentes
encoder_model = keras.Model(input_layer, z_mean)
embeddings_latentes = encoder_model.predict(X_tfidf_scaled)
# 📌 9️⃣ Aplicar reducción de dimensionalidad con PCA
n_samples = embeddings_latentes.shape[0]
n_features = embeddings_latentes.shape[1]
n_pca_components = min(10, n_samples, n_features)

pca = PCA(n_components=n_pca_components)
embeddings_pca = pca.fit_transform(embeddings_latentes)
# 📌 🔟 Aplicar HDBSCAN primero
# HDBSCAN agrupa datos basándose en densidad
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=6,  
    min_samples=4,       
    cluster_selection_method='eom',
    allow_single_cluster=True  
)

df["Tópico_Descubierto"] = clusterer.fit_predict(embeddings_pca)
# 📌 1️⃣1️⃣ Ajustar el número de tópicos con K-Means si es necesario
num_topics = int(input("Ingrese el número de tópicos deseados: "))
num_detected = len(set(df["Tópico_Descubierto"])) - (1 if -1 in df["Tópico_Descubierto"].values else 0)

if num_detected < num_topics:
    print(f"🔄 Ajustando los tópicos con K-Means para llegar a {num_topics}...")
    kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
    df["Tópico_Descubierto"] = kmeans.fit_predict(embeddings_pca)
    # 📌 1️⃣2️⃣ Obtener frases clave representativas
feature_names = vectorizer.get_feature_names_out()
top_phrases_per_topic = []

for i in range(num_topics):
    cluster_docs = df[df["Tópico_Descubierto"] == i]["Texto_Final"]
    
    if cluster_docs.empty:
        top_phrases_per_topic.append(["Unknown Topic"])
        continue
    
    cluster_tfidf = vectorizer.transform(cluster_docs)
    avg_tfidf = np.mean(cluster_tfidf, axis=0).flatten()
    top_phrase_indices = np.argsort(avg_tfidf.A1)[::-1][:7]
    top_phrases = [feature_names[idx] for idx in top_phrase_indices]
    top_phrases_per_topic.append(top_phrases)
# 📌 1️⃣3️⃣ Filtrar términos irrelevantes
stop_phrases = {"et al", "pp", "conference", "journal", "vol", "dataset", "recommendation", "user", "et", "al", "ieee", "acm", "elsevier", "springer", "copyright"}
def clean_topic_name(name):
    words = name.split()
    return " ".join([word for word in words if word.lower() not in stop_phrases])

# 📌 1️⃣4️⃣ Generar nombres de tópicos más naturales
def generate_topic_name(phrases):
    phrases = [clean_topic_name(p) for p in phrases]
    phrases = list(dict.fromkeys(phrases))
    if len(phrases) >= 3:
        return f"{phrases[0]} and {phrases[1]} in {phrases[2]}"
    elif len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    else:
        return phrases[0] if phrases else "Unknown Topic"

topic_labels = [generate_topic_name(phrases) for phrases in top_phrases_per_topic]
# 📌 1️⃣5️⃣ Asignar nombres interpretables a los tópicos
df["Nombre_Topico"] = df["Tópico_Descubierto"].map(lambda x: topic_labels[x])

# 📌 1️⃣6️⃣ Guardar resultados finales
df.to_csv("topicos_mejorados.csv", index=False)

# 📌 🔥 Mostrar resumen
print("\n📌 Cantidad de documentos en cada tópico:")
print(df["Tópico_Descubierto"].value_counts())

print("\n📌 Tópicos detectados con nombres interpretables:")
for i, name in enumerate(topic_labels):
    print(f"Tópico {i}: {name}")
    
    
    
print("\n📌 Tópicos Descubiertos por HDBSCAN:"
      f"\n{df['Tópico_Descubierto'].value_counts()}")
print("\n📌 Vista previa de los datos:"
      f"\n{df.head()}")


