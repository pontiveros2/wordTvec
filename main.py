import pandas as pd
import spacy
from gensim.models import Word2Vec
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np

# 1. PREPARACIÓN DE DATOS
nlp = spacy.load("es_core_news_sm")

with open("Juan_Salvador_Gaviota.txt", "r", encoding="latin-1") as f:
     texto_principito = f.read()


print("Procesando texto con SpaCy...")
doc = nlp(texto_principito)

sentences = []
for sent in doc.sents:
    # Lematizamos y quitamos stop words para reducir ruido
    tokens = [
        token.lemma_.lower() 
        for token in sent 
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    if len(tokens) > 1: # Solo oraciones con algo de contenido
        sentences.append(tokens)

print(f"Total de oraciones procesadas: {len(sentences)}")
print(f"Ejemplo (tokens): {sentences[0]}")

# 2. ENTRENAMIENTO DE WORD2VEC
# Creamos vectores densos (Embeddings)
print("\nEntrenando red neuronal Word2Vec...")

model = Word2Vec(
    sentences,
    vector_size=20,  # 10 dimensiones (neuronas en la capa oculta)
    window=10,        # Contexto
    min_count=1,     # Aceptamos palabras únicas por ser un dataset mini
    workers=multiprocessing.cpu_count(),
    seed=42          # Semilla para reproducibilidad
)

# 3. EXPLORACIÓN SEMÁNTICA
def mostrar_similares(palabra):
    try:
        # topn=3 nos da las 3 palabras más cercanas en el espacio vectorial
        similares = model.wv.most_similar(palabra, topn=3)
        print(f"\nPalabras más cercanas semánticamente a '{palabra}':")
        for sim in similares:
            # sim es una tupla (palabra, puntuación)
            print(f"  - {sim[0]} (Similitud: {sim[1]:.4f})")
    except KeyError:
        print(f"\nLa palabra '{palabra}' no está en el vocabulario.")

# Probamos conceptos (Nota: con texto tan corto, las asociaciones serán raras)
mostrar_similares("sobrevivir")
mostrar_similares("suavidad")

# 4. VISUALIZACIÓN 3D (PCA de Embeddings)
# Aquí ocurre la magia: proyectamos las 10 dimensiones a 3 para verlas.

# Obtenemos todas las palabras y sus vectores
vocabulario = list(model.wv.index_to_key)
vectores = model.wv[vocabulario]

# Reducción de dimensionalidad
pca = PCA(n_components=3)
vectores_3d = pca.fit_transform(vectores)

# Crear DataFrame para facilitar ploteo
df_3d = pd.DataFrame(vectores_3d, columns=['x', 'y', 'z'])
df_3d['palabra'] = vocabulario

# Gráfico
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df_3d['x'], df_3d['y'], df_3d['z'], c='crimson', s=80, edgecolors='white', alpha=0.8)

# Etiquetas
for i, row in df_3d.iterrows():
    ax.text(row['x'], row['y'], row['z'], f" {row['palabra']}", size=10)

ax.set_title('Espacio Semántico (Word Embeddings) - Juan Salvador Gaviota', fontsize=14)
ax.set_xlabel('Dimensión Latente 1')
ax.set_ylabel('Dimensión Latente 2')
ax.set_zlabel('Dimensión Latente 3')

plt.tight_layout()
plt.show()

print(f"\nAsí ve la máquina la palabra 'zorro' (Vector de 10 dimensiones):")
try:
    print(model.wv["zorro"])
except:
    print("La palabra zorro no apareció en el texto dummy, intenta con 'cordero'.")