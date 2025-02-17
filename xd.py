import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords

# Descargar stopwords en español
nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')

# Simulación de datos textuales
documents = [
    "El hotel es grande y cómodo con parqueadero amplio",
    "Los clientes del hotel elogian la comodidad y el desayuno",
    "El auto sufrió una falla eléctrica en la Amazonía",
    "El mantenimiento del motor Volkswagen requiere atención frecuente",
    "El turismo en Cuenca crece debido a su infraestructura hotelera"
]

# Preprocesamiento: Vectorización BoW
vectorizer = CountVectorizer(stop_words=spanish_stopwords)
X = vectorizer.fit_transform(documents).toarray()

# Normalización para VAE usando MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Definir el Autoencoder Variacional
input_dim = X.shape[1]
latent_dim = 5  # Espacio latente reducido

# Encoder
inputs = keras.Input(shape=(input_dim,))
h = keras.layers.Dense(10, activation='relu')(inputs)
z_mean = keras.layers.Dense(latent_dim)(h)
z_log_var = keras.layers.Dense(latent_dim)(h)

# Sampling (Reparametrization Trick)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_h = keras.layers.Dense(10, activation='relu')
decoder_out = keras.layers.Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_out(h_decoded)

# Definir el modelo
vae = keras.Model(inputs, outputs)

# Pérdida VAE
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

# Asegurar valores no negativos en la representación latente
X_latent = np.maximum(0, X_latent)  # Aplicar ReLU para garantizar valores positivos

# Aplicar LDA sobre la representación latente
lda = LatentDirichletAllocation(n_components=6, random_state=42)
topics = lda.fit_transform(X_latent)

# Mostrar tópicos asignados
for i, doc in enumerate(documents):
    print(f"Documento {i+1}: {doc}")
    print(f"Distribución de tópicos: {topics[i]}\n")

# Mostrar las palabras clave de cada tópico
words = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Tópico {topic_idx + 1}:")
    print(" ".join([words[i] for i in topic.argsort()[:-10 - 1:-1]]))  # Top 10 palabras
    print()
