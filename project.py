import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Step 1: Load and Prepare Data
@st.cache_data
def load_data():
    df = pd.read_csv("Fashion Dataset.csv")

    # Basic cleaning
    df = df.dropna(subset=['p_id', 'name'])
    df['brand'] = df['brand'].fillna('Unknown')
    df['colour'] = df['colour'].fillna('Unknown')
    df['ratingCount'] = df['ratingCount'].fillna(0)
    df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].mean())
    df['description'] = df['description'].fillna('')
    df['p_attributes'] = df['p_attributes'].fillna('')

    # Feature Engineering (add price bucket)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['price'] = df['price'].fillna(df['price'].median())

    df['price_bucket'] = pd.cut(df['price'],
                               bins=3,
                               labels=['low', 'medium', 'high'])

    # Combine textual features
    df['text_features'] = (
        df['brand'].astype(str) + ' ' +
        df['colour'].astype(str) + ' ' +
        df['price_bucket'].astype(str) + ' ' +   # ✅ NEW
        df['description'].astype(str) + ' ' +
        df['p_attributes'].astype(str)
    )

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=4000)
    tfidf_matrix = tfidf.fit_transform(df['text_features'])

    # Similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Popularity score (log scaled)
    df['popularity_score'] = df['avg_rating'] * np.log1p(df['ratingCount'])
    df['popularity_norm'] = df['popularity_score'] / df['popularity_score'].max()

    return df, cosine_sim

df, cosine_sim = load_data()

# Step 2: Hybrid Recommender
def hybrid_recommend(product_name, top_n=5, alpha=0.7, same_brand=False):
    # Find product index
    idx_list = df[df['name'].str.lower() == product_name.lower()].index

    if len(idx_list) == 0:
        return None

    idx = idx_list[0]

    # Get similarity scores for selected product
    content_scores = cosine_sim[idx]

    # Hybrid score
    hybrid_scores = alpha * content_scores + (1 - alpha) * df['popularity_norm'].values

    # Create temp dataframe with scores 
    df_temp = df.copy()

    df_temp['content_score'] = content_scores
    df_temp['pop_score'] = df['popularity_norm']
    df_temp['final_score'] = hybrid_scores

    # Remove the selected product
    df_temp = df_temp[df_temp.index != idx]

    # Optional filter: same brand
    if same_brand:
        selected_brand = df.loc[idx, 'brand']
        df_temp = df_temp[df_temp['brand'] == selected_brand]

    # Sort by final score
    df_temp = df_temp.sort_values(by='final_score', ascending=False)

    # Return top N
    return df_temp.head(top_n)[
        ['name', 'brand', 'price', 'avg_rating', 'ratingCount',
         'content_score', 'pop_score']
    ]

# Step 3: Streamlit UI 
st.set_page_config(page_title="Fashion Recommender", layout="centered")

st.title("🛍️ AI-Powered Fashion Recommender")
st.markdown("Smart recommendations using **content similarity + popularity score**")

# Product selection
product_list = df['name'].dropna().unique()
selected_product = st.selectbox("Select a product:", sorted(product_list))

# Show selected product
selected_row = df[df['name'] == selected_product].iloc[0]

st.subheader("Selected Product")
st.write(f"**{selected_row['name']}**")
st.write(f"Brand: {selected_row['brand']}")
st.write(f"Price: ₹{selected_row['price']}")
st.markdown("---")

# Controls
alpha = st.slider("Content vs Popularity Weight (α)", 0.0, 1.0, 0.7, 0.1)
top_n = st.slider("Number of Recommendations", 1, 10, 5)
same_brand = st.checkbox("Recommend from same brand only")

# Generate recommendations
if st.button("🔍 Get Recommendations"):
    with st.spinner("Finding best matches..."):

        recommendations = hybrid_recommend(
            selected_product,
            top_n=top_n,
            alpha=alpha,
            same_brand=same_brand
        )

        if recommendations is None:
            st.error("Product not found.")
        else:
            st.success(f"Top {top_n} Recommendations")

            for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
                with st.container():
                    st.markdown(f"## #{i} {row['name']}")
                    
                    st.write(f"Brand: {row['brand']}")
                    st.write(f"Price: ₹{int(row['price'])}")
                    st.write(f"⭐ Rating: {row['avg_rating']:.2f} ({int(row['ratingCount'])} reviews)")
                    
                    # Scores
                    st.caption(
                        f"Match Score: {row['content_score']:.2f} | "
                        f"Popularity Score: {row['pop_score']:.2f}"
                    )
            
                    st.markdown("---")
st.caption("Hybrid Recommender using TF-IDF + Popularity + Feature Engineering")