# ==========================================================
# 🎮 GAME RECOMMENDATION SYSTEM (CONTENT-BASED)
# Side-by-side recommended games (3 per row) + Cover Image + Download Link + Project Info & EDA
# ==========================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================================
@st.cache_data(show_spinner=True)
def load_data(games_path: str, metadata_path: str) -> pd.DataFrame:
    games = pd.read_csv(games_path)

    meta_rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                meta_rows.append(json.loads(line.strip()))
            except:
                pass

    meta = pd.DataFrame(meta_rows)
    df = pd.merge(games, meta, on="app_id", how="inner")

    df["description"] = df.get("description", "").fillna("")
    df["tags"] = df.get("tags", [[] for _ in range(len(df))])

    def normalize_tags(x):
        if isinstance(x, list):
            return [str(t) for t in x]
        if isinstance(x, str):
            return [t.strip() for t in x.split(",") if t.strip()]
        return []

    df["tags"] = df["tags"].apply(normalize_tags)
    df["tags_text"] = df["tags"].apply(lambda tags: " ".join(tags))
    df["combined_text"] = (df["description"] + " " + df["tags_text"]).str.lower()

    df["title"] = df["title"].astype(str)
    df = df[df["combined_text"].str.strip() != ""].reset_index(drop=True)

    return df


# ==========================================================
# 2. ML MODEL (CONTENT-BASED)
# ==========================================================
class GameRecommender:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        self.matrix = self.vectorizer.fit_transform(df["combined_text"])
        self.title_map = {t.lower(): i for i, t in enumerate(df["title"])}

    def get_index(self, title):
        key = title.lower().strip()
        if key not in self.title_map:
            raise ValueError("Game not found")
        return self.title_map[key]

    def similar_games(self, title, k=10):
        idx = self.get_index(title)
        sims = cosine_similarity(self.matrix[idx], self.matrix).flatten()
        sims[idx] = -1
        top_idx = sims.argsort()[::-1][:k]

        result = self.df.loc[top_idx, ["app_id", "title", "rating", "price_final", "tags",
                                       "positive_ratio", "user_reviews"]].copy()
        result["similarity"] = np.round(sims[top_idx], 3)
        return result.reset_index(drop=True)

    def recommend_from_text(self, text, k=10):
        vec = self.vectorizer.transform([text.lower()])
        sims = cosine_similarity(vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]

        result = self.df.loc[top_idx, ["app_id", "title", "rating", "price_final", "tags",
                                       "positive_ratio", "user_reviews"]].copy()
        result["similarity"] = np.round(sims[top_idx], 3)
        return result.reset_index(drop=True)


@st.cache_resource(show_spinner=True)
def build_recommender(df):
    return GameRecommender(df)


# ==========================================================
# 3. CARD DISPLAY WITH IMAGE + DOWNLOAD LINK
# ==========================================================
def display_cards(df, per_row=3):
    for i in range(0, len(df), per_row):
        cols = st.columns(per_row)
        for j in range(per_row):
            if i + j < len(df):
                row = df.iloc[i + j]

                img_url = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{row['app_id']}/header.jpg"
                store_url = f"https://store.steampowered.com/app/{row['app_id']}/"

                with cols[j]:
                    st.image(img_url, width=300)
                    st.markdown(f"#### 🎮 {row['title']}")
                    st.write(f"⭐ **Similarity:** `{row['similarity']}`")
                    st.write(f"🏆 **Rating:** `{row['rating']}`")
                    st.write(f"🗳 **Reviews:** `{row['user_reviews']}`")
                    st.write(f"💲 **Price:** `${row['price_final']}`")
                    st.write(f"🏷 **Tags:** {', '.join(row['tags'][:6])}")

                    st.markdown(f"[🕹 Click to View / Download]({store_url})", unsafe_allow_html=True)
                    st.markdown("---")


# ==========================================================
# 4. SIMPLE EDA GENERATOR
# ==========================================================
def generate_eda(df):
    num_games = len(df)
    num_unique_titles = df["title"].nunique()

    all_tags = []
    for t in df["tags"]:
        all_tags.extend(t)
    top_tags = pd.Series(all_tags).value_counts().head(20)

    rating_counts = None
    if "rating" in df.columns:
        rating_counts = df["rating"].value_counts()

    return num_games, num_unique_titles, top_tags, rating_counts


# ==========================================================
# 5. STREAMLIT UI
# ==========================================================
def main():
    st.set_page_config(page_title="Game Recommendation System", page_icon="🎮", layout="wide")
    st.title("🎮 Game Recommendation System (Steam Dataset)")

    st.markdown("""
This project is a **Content-Based Game Recommender** built using **Unsupervised Learning**.

### 🧠 ML Technique Used  
- **TF-IDF Vectorization** → Converts game descriptions + tags into numerical vectors  
- **Cosine Similarity** → Finds closest matches based on text meaning  
- No target labels → **Unsupervised Learning**

---
""")

    base = Path(__file__).resolve().parent
    df = load_data(base / "games.csv", base / "games_metadata.json")
    model = build_recommender(df)

    # Sidebar
    mode = st.sidebar.radio("Select Mode", ["ℹ️ Project Info & EDA", "🎮 Similar Games", "🧠 Text Search", "🏷 View Tags"])
    k = st.sidebar.slider("Number of Recommendations", 3, 21, 9)

    # ======================================================
    # MODE A: PROJECT INFO & EDA
    # ======================================================
    if mode == "ℹ️ Project Info & EDA":
        st.subheader("📊 Project Information & Dataset EDA")

        num_games, num_unique, top_tags, rating_dist = generate_eda(df)

        col1, col2 = st.columns(2)
        col1.metric("Total Games", num_games)
        col2.metric("Unique Titles", num_unique)

        with st.expander("🔝 Top 20 Most Common Tags"):
            st.dataframe(pd.DataFrame({"Tag": top_tags.index, "Count": top_tags.values}))

        if rating_dist is not None:
            with st.expander("⭐ Rating Distribution"):
                st.bar_chart(rating_dist)

        with st.expander("📂 Sample Dataset (First 10 Rows)"):
            st.dataframe(df.head(10), use_container_width=True)


    # ======================================================
    # MODE B: SIMILAR GAMES
    # ======================================================
    elif mode == "🎮 Similar Games":
        st.subheader("🎯 Game-Based Similarity Recommendations")
        title = st.selectbox("Select Game:", sorted(df["title"].unique()))
        if st.button("🔍 Find Similar Games"):
            results = model.similar_games(title, k)
            display_cards(results, per_row=3)


    # ======================================================
    # MODE C: TEXT SEARCH
    # ======================================================
    elif mode == "🧠 Text Search":
        st.subheader("🧠 Recommend Based on Text Input")
        text = st.text_input("Type what kind of game you like (e.g., open world zombie rpg)")
        if st.button("🎯 Recommend"):
            results = model.recommend_from_text(text, k)
            display_cards(results, per_row=3)


    # ======================================================
    # MODE D: VIEW TAGS
    # ======================================================
    else:
        st.subheader("🏷 View Game Tags")
        title = st.selectbox("Select Game:", sorted(df["title"].unique()))
        if st.button("📌 Show Tags"):
            tags = model.df.loc[model.get_index(title), "tags"]
            st.success(", ".join(tags))


if __name__ == "__main__":
    main()
