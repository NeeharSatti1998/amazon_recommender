import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Logo and Title
st.image("amazon_logo.png", width=200)
st.title("Amazon Product Recommender")

# User Input
user_id = st.text_input("Enter your user ID to get personalized recommendations")

# Load and combine raw data
df_1 = pd.read_csv(r'C:\Users\neeha\Downloads\rec_sys\data\amazon_reviews.csv')
df_2 = pd.read_csv(r'C:\Users\neeha\Downloads\rec_sys\data\amazon_reviews_2.csv')
df_3 = pd.read_csv(r'C:\Users\neeha\Downloads\rec_sys\data\amazon_reviews_3.csv')
raw_df = pd.concat([df_1, df_2, df_3], ignore_index=True)
raw_df.drop_duplicates(subset=['reviews.username', 'asins', 'reviews.rating'], inplace=True)

# Cache loading embeddings and cleaned interactions
@st.cache_data
def load_data():
    embeddings = pd.read_csv("data/amazon_node2vec_embeddings_n.csv")
    interactions = pd.read_csv("data/amazon_reviews_cleaned_n.csv")
    return embeddings, interactions

embedding_df, interactions = load_data()

# Explode ASINs and clean metadata
raw_df['asins'] = raw_df['asins'].astype(str).str.split(',')
raw_df = raw_df.explode('asins')
raw_df['asins'] = raw_df['asins'].str.strip()

def clean_name(value):
    if pd.isna(value): return None
    cleaned = str(value).splitlines()[0].split(",")[0].strip()
    return cleaned if len(cleaned) > 5 else None

raw_df['product_name'] = raw_df['name'].apply(clean_name)
raw_df = raw_df[raw_df['product_name'].notna()]

raw_df['image_url'] = raw_df['imageURLs'].astype(str).str.split(',').str[0]
raw_df['product_url'] = raw_df['sourceURLs'].astype(str).str.split(',').str[0]

# Mapping ASINs to product metadata
asin_to_name = raw_df[['asins', 'product_name']].dropna().drop_duplicates().set_index('asins')['product_name'].to_dict()
asin_to_image = raw_df.set_index('asins')['image_url'].to_dict()
asin_to_url = raw_df.set_index('asins')['product_url'].to_dict()

# Prepare embeddings
user_embeddings = embedding_df[embedding_df['node'].isin(interactions['user_id'].unique())].set_index('node')
product_embeddings = embedding_df[embedding_df['node'].isin(interactions['product_id'].unique())].set_index('node')

# Recommend top-N products
def recommend_products(user_id, N=5):
    if user_id not in user_embeddings.index:
        return pd.DataFrame()

    user_vector = user_embeddings.loc[user_id].values.reshape(1, -1)
    similarities = cosine_similarity(user_vector, product_embeddings.values)[0]
    product_ids = product_embeddings.index.tolist()

    sim_df = pd.DataFrame({
        'product_id': product_ids,
        'similarity': similarities
    })

    rated = interactions[interactions['user_id'] == user_id]['product_id'].tolist()
    sim_df = sim_df[~sim_df['product_id'].isin(rated)]
    sim_df['product_id'] = sim_df['product_id'].str.strip()
    sim_df = sim_df.drop_duplicates(subset='product_id')
    sim_df['name'] = sim_df['product_id'].map(asin_to_name)
    sim_df['image'] = sim_df['product_id'].map(asin_to_image)
    sim_df['url'] = sim_df['product_id'].map(asin_to_url)
    sim_df = sim_df[sim_df['name'].notna()]
    sim_df = sim_df.drop_duplicates(subset='name') 
    return sim_df.sort_values(by='similarity', ascending=False).head(N)

# Display user ratings and recommendations
if user_id:
    if user_id in interactions['user_id'].values:
        with st.expander(f"Products rated by {user_id}"):
            user_ratings = interactions[interactions['user_id'] == user_id].copy()
            user_ratings['product_id'] = user_ratings['product_id'].astype(str).str.strip()
            user_ratings['name'] = user_ratings['product_id'].map(asin_to_name)
            user_ratings = user_ratings[user_ratings['name'].notna()]
            user_ratings = user_ratings.sort_values('rating', ascending=False).drop_duplicates('name')  
            user_ratings = user_ratings[user_ratings['name'].notna()]
            user_ratings['image'] = user_ratings['product_id'].map(asin_to_image)
            user_ratings['url'] = user_ratings['product_id'].map(asin_to_url)

            for _, row in user_ratings.iterrows():
                st.markdown(f"**{row['name']}** ({row['rating']})")
                img_url = row['image'] if isinstance(row['image'], str) and row['image'].startswith("http") else "default.png"
                st.image(img_url, width=150)
                if pd.notna(row['url']):
                    st.markdown(f"[ View on Amazon]({row['url']})")
                st.write("---")

    recs = recommend_products(user_id)
    if recs.empty:
        st.warning("User ID not found or no recommendations available.")
    else:
        with st.expander(f"Top Recommendations for {user_id}", expanded=True):
            for _, row in recs.iterrows():
                st.markdown(f"**{row['name']}**")
                img_url = row['image'] if isinstance(row['image'], str) and row['image'].startswith("http") else "default.png"
                st.image(img_url, width=150)
                if pd.notna(row['url']):
                    st.markdown(f"[View on Amazon]({row['url']})")
                st.write("---")
