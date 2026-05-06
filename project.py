import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

st.write("FILES:", os.listdir())

# STEP 1: LOAD DATA
@st.cache_resource
def load_data():
    df = pd.read_csv("Fashion Dataset.csv")

    # Basic cleaning
    df = df.dropna(subset=['p_id', 'name'])

    df['brand'] = df['brand'].fillna('Unknown')
    df['colour'] = df['colour'].fillna('Unknown')
    df['description'] = df['description'].fillna('')
    df['p_attributes'] = df['p_attributes'].fillna('')

    df['ratingCount'] = (
        df['ratingCount']
        .astype(str)
        .str.replace(',', '')
    )
    df['ratingCount'] = pd.to_numeric(df['ratingCount'], errors='coerce').fillna(0)

    df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].mean())

    # Feature engineering
    df['text_features'] = (
        df['brand'] + " " +
        df['colour'] + " " +
        df['description'] + " " +
        df['p_attributes']
    )

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=1500)
    tfidf_matrix = tfidf.fit_transform(df['text_features'])

    # popularity score
    df['popularity_score'] = df['avg_rating'] * df['ratingCount']
    df['popularity_norm'] = df['popularity_score'] / (df['popularity_score'].max() + 1)

    return df, tfidf_matrix


df, tfidf_matrix = load_data()

# STEP 2: RECOMMENDER
def hybrid_recommend(product_name, top_n=5, alpha=0.7, same_brand=False):

    idx = df[df['name'].str.lower() == product_name.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]

    # similarity
    content_scores = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()

    df_temp = df.copy()

    df_temp['content_score'] = content_scores
    df_temp['pop_score'] = df_temp['popularity_norm']

    df_temp['hybrid_score'] = (
        alpha * df_temp['content_score'] +
        (1 - alpha) * df_temp['pop_score']
    )

    # remove selected product
    df_temp = df_temp.drop(index=idx)

    selected_brand = str(df.loc[idx, 'brand']).lower()
    df_temp['brand'] = df_temp['brand'].astype(str).str.lower()

    if same_brand:
        df_temp = df_temp[df_temp['brand'] == selected_brand]

    return df_temp.sort_values('hybrid_score', ascending=False).head(top_n)

# STEP 3: UI
st.set_page_config(page_title="Fashion Recommender", layout="centered")

st.title("🛍️ AI Fashion Recommender")
st.markdown("TF-IDF + Popularity based hybrid system")

product_list = df['name'].dropna().unique()
selected_product = st.selectbox("Select product:", sorted(product_list))

selected_row = df[df['name'] == selected_product].iloc[0]

st.subheader("Selected Product")
st.write(f"**{selected_row['name']}**")
st.write(f"Brand: {selected_row['brand']}")
st.write(f"Price: ₹{selected_row['price']}")
st.markdown("---")

alpha = st.slider("Content vs Popularity (α)", 0.0, 1.0, 0.7)
top_n = st.slider("Number of recommendations", 1, 10, 5)
same_brand = st.checkbox("Recommend from same brand only")

if st.button("Get Recommendations"):

    recommendations = hybrid_recommend(
        selected_product,
        top_n=top_n,
        alpha=alpha,
        same_brand=same_brand  
    )

    if recommendations is None:
        st.error("Product not found")
    else:
        st.success("Top Recommendations")

        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            st.markdown(f"### {i}. {row['name']}")
            st.write(f"Brand: {row['brand']}")
            st.write(f"Price: ₹{row['price']}")
            st.write(
                f"⭐ Rating: {row['avg_rating']:.2f} "
                f"({int(row['ratingCount'])} reviews)"
            )
            st.caption(
                f"Content: {row['content_score']:.2f} | "
                f"Popularity: {row['pop_score']:.2f}"
            )
            st.markdown("---")

st.caption("Hybrid recommender using TF-IDF + popularity scoring")