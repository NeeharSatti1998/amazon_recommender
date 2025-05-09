# Amazon Product Recommendation System (Graph-Based)

This project implements a personalized product recommender system for Amazon products using user reviews. The recommendation engine is built using graph-based modeling, Node2Vec embeddings, and cosine similarity. A Streamlit web app provides a user-friendly interface where users can enter their `user_id` to get relevant product suggestions.

---

## Features

- Graph modeling of user-product interactions
- Node2Vec embedding of graph nodes
- Cosine similarity to compute recommendations
- Streamlit UI with:
  - Search bar for user ID
  - Rated product display (title, image, rating, Amazon link)
  - Top-N product recommendations
- Amazon logo and product image fallback support

---

## Folder Structure

```
rec_sys/
├── app/
│ └── app.py # Streamlit app code
├── models/
│ └── amazon_user_product_graph.pkl
├── amazon_node2vec_embeddings.csv
├── recommendation.py # Embedding and recommendation generation
├── notebooks/
│ └── amazon_data_cleaning.ipynb
  └── embeddings.py
  └── graph_modeling.ipynb
  └── recommendations.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

> Note: The large CSV files are excluded from this repo to meet GitHub file size limits.

---

## Dataset Source

We used the **Amazon Consumer Reviews Dataset** from Kaggle:

**Download it here:**  
[Kaggle Dataset - Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data)

After downloading, place these files into the `/data/` directory:
- `amazon_reviews.csv`
- `amazon_reviews_2.csv`
- `amazon_reviews_3.csv`

---

## How to Run

1. **Clone the repository**

```bash
git clone https://github.com/NeeharSatti1998/amazon_recommender.git
cd amazon_recommender
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download dataset**

Download the CSVs from Kaggle (link above) and place them in the `data/` folder.

4. **Run preprocessing**

```bash
# Clean the data
python notebooks/amazon_data_cleaning.ipynb

# Generate node embeddings
python notebooks/embeddings.py
```

5. **Launch the app**

```bash
streamlit run app/app.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
networkx
node2vec
streamlit
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## How it Works

- Constructs a bipartite graph from user-product interactions
- Applies Node2Vec to learn embeddings for users and products
- Computes cosine similarity between a user and all products
- Filters out already-rated products and displays new ones
- Embeds product names, image previews, and links to Amazon

---

## Notes

- Recommendations are personalized to each user's historical ratings
- Products that users have already rated are not recommended again
- Duplicate reviews are deduplicated using highest rating priority

---

## Credits

- Dataset: [Datafiniti on Kaggle](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data)
- Graph Embedding: `node2vec` Python package
- Interface: Streamlit
