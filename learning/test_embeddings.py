import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import AgglomerativeClustering

model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = [
    "Rain is expected tomorrow.",
    "Stock prices are fluctuating a lot.",
    "Today is sunny and warm.",
    "A sunny day with clear skies.",
    "The weather is nice and bright today.",
    "Heavy rain is predicted for the weekend.",
    "The stock market is experiencing volatility.",
    "Tomorrow will bring more sunshine.",
    "A beautiful day to go for a walk.",
    "It's getting cloudy, might rain soon.",
    "I love sunny days at the beach.",
    "The forecast indicates possible thunderstorms.",
    "Investors are worried about market trends.",
    "It feels great to be outside on such a lovely day.",
    "There might be a chance of rain later.",
    "Prices are rising due to market instability.",
    "A perfect day for a picnic in the park.",
    "Don't forget your umbrella, just in case.",
    "The sun is shining, and it's quite warm.",
    "There will be sunny spells throughout the day.",
    "It's a good day for outdoor activities."
]

#################### Clustering of sentences based on their embeddings ####################

# embeddings = model.encode(corpus, convert_to_tensor=True)
# embeddings_np = embeddings.cpu().numpy()

# clustering_model = AgglomerativeClustering(n_clusters=3)
# clustering_labels = clustering_model.fit_predict(embeddings_np)

# clusters = {}
# for sentence, label in zip(corpus, clustering_labels):
#     if label not in clusters:
#         clusters[label] = []
#     clusters[label].append(sentence)     
    
# for label, sentences in clusters.items():
#     print(f"Cluster {label}:")
#     for sentence in sentences:
#         print(f" - {sentence}")
#     print()
    
    
# cluster_names = {
#     0: "Sunny Weather",
#     1: "Stock market Volatility",
#     2: "Rainy Weather"
# }

# for cluster_id, sentences in clusters.items():
#     print(f"Cluster {cluster_id} ({cluster_names[cluster_id]}):")
#     for sentence in sentences:
#         print(f" - {sentence}")
#     print()

########################################################################################################################

#################### Semantic Search ####################

# df = pd.read_csv('test/dataset/webtext.csv')
# df.dropna(inplace=True)
# df = df.sample(n=1500).reset_index(drop=True)

# corpus = df['text']

# corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# def semantic_search(query):
#     query_embedding = model.encode(query, convert_to_tensor=True)
    
#     hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
#     return hits[0][0]

# while True:
#     # Get user input for the search query
#     query = input("Enter your search query (or type 'exit' to quit): ")
#     if query.lower() == 'exit':
#         break

#     # Perform semantic search
#     result = semantic_search(query)

#     print("\n\n")
#     print("Input Query :", query)
#     print("\n")
#     # Display the most relevant sentence
#     print(f"Most relevant sentence: '{corpus[result['corpus_id']]}' (Score: {result['score']})")

########################################################################################################################



#################### Paraphrasing ####################