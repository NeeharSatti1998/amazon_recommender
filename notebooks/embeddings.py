from node2vec import Node2Vec
import networkx as nx
import pickle
import pandas as pd

with open(r"C:\Users\neeha\Downloads\rec_sys\models\amazon_user_product_graph_n.pkl", "rb") as f:
    G = pickle.load(f)

# Initializing the Node2Vec model with walk parameters
node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=100, workers=2)

# Training the Word2Vec model on the generated walks
model = node2vec.fit(window=5, min_count=1, batch_words=4)

# Collecting embeddings for all nodes
embeddings = []
for node in G.nodes():
    vec = model.wv[str(node)]
    embeddings.append([node] + list(vec))

embedding_df = pd.DataFrame(embeddings)
embedding_df.columns = ['node'] + [f'dim_{i}' for i in range(1, 65)]


embedding_df.to_csv("data/amazon_node2vec_embeddings_n.csv", index=False)

print("Embeddings saved to amazon_node2vec_embeddings_n.csv")