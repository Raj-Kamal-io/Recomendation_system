"""
TechCart - Electronics Recommendation Demo
Loads the trained models from Notebook 2 (TF-IDF, SVD/KNN, Hybrid, LightGCN)
and serves a small e-commerce style site that demonstrates all four.
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for, abort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)
app.secret_key = "techcart-dev-secret"  # fine for a local demo, change if you deploy this for real


# ---------------------------------------------------------------------------
# Load data + models once at startup
# ---------------------------------------------------------------------------

def _load_pickle(name):
    path = os.path.join(MODELS_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


print("Loading product catalog ...")
products_df = pd.read_csv(os.path.join(DATA_DIR, "clean_amazon_products.csv"))
products_df["price"] = products_df["price"].fillna(0)
products_df["stars"] = products_df["stars"].fillna(0)
asin_to_row = products_df.set_index("asin", drop=False)

print("Loading TF-IDF (Content-Based) model ...")
tfidf_vectorizer = _load_pickle("tfidf_vectorizer.pkl")
tfidf_matrix = _load_pickle("tfidf_matrix.pkl")
product_index_lookup = pd.Series(products_df.index, index=products_df["title"]).drop_duplicates()

print("Loading Hybrid model artifacts ...")
hybrid_artifacts = _load_pickle("hybrid_artifacts.pkl")
candidate_items = hybrid_artifacts["candidate_items"]
candidate_item_tfidf = hybrid_artifacts["candidate_item_tfidf"]
asin_to_idx = hybrid_artifacts["asin_to_idx"]
best_cf_model_name = hybrid_artifacts["best_cf_model_name"]

print("Loading best Collaborative Filtering model (%s) ..." % best_cf_model_name)
best_cf_model = _load_pickle("best_cf_model.pkl")

print("Loading LightGCN artifacts ...")
lightgcn = _load_pickle("lightgcn_artifacts.pkl")
final_user_embeddings = lightgcn["final_user_embeddings"]
final_item_embeddings = lightgcn["final_item_embeddings"]
lgcn_user2idx = lightgcn["user2idx"]
lgcn_item2idx = lightgcn["item2idx"]
lgcn_items = lightgcn["lgcn_items"]

print("Loading demo shopper profiles ...")
demo_users = _load_pickle("demo_users.pkl")

print("TechCart is ready. %d products loaded, %d demo shoppers available." % (
    len(products_df), len(demo_users)
))


# ---------------------------------------------------------------------------
# Recommendation helpers (same logic as Notebook 2)
# ---------------------------------------------------------------------------

from sklearn.metrics.pairwise import cosine_similarity


def get_current_shopper():
    return session.get("shopper_id")


@app.context_processor
def inject_globals():
    return {
        "shopper_id": get_current_shopper(),
        "demo_users": list(demo_users.keys()),
    }


def shopper_history(user_id):
    """Rated items for a demo shopper, as a list of dicts (asin, rating, title, category_name)."""
    return demo_users.get(user_id, [])


def content_based_similar(asin, top_n=8):
    """Products similar to a given product, by TF-IDF cosine similarity."""
    if asin not in asin_to_row.index:
        return []
    product_idx = asin_to_row.index.get_loc(asin)
    sims = cosine_similarity(tfidf_matrix[product_idx], tfidf_matrix).flatten()
    order = np.argsort(-sims)
    results = []
    for idx in order:
        if idx == product_idx:
            continue
        results.append((products_df.iloc[idx], sims[idx]))
        if len(results) >= top_n:
            break
    return results


def search_products(query, top_n=24):
    if not query:
        return []
    mask = products_df["title"].str.contains(query, case=False, na=False)
    return products_df[mask].head(top_n)


def hybrid_recommend(user_id, top_n=10, alpha=0.5):
    """Content profile (liked items) blended with the best CF model (LightGCN) affinity score."""
    history = shopper_history(user_id)
    seen = {h["asin"] for h in history} if history else set()

    # Get LightGCN user embedding (fallback to average user embedding if cold start)
    if user_id in lgcn_user2idx:
        u = lgcn_user2idx[user_id]
        u_emb = final_user_embeddings[u]
    else:
        u_emb = final_user_embeddings.mean(axis=0)

    # Compute LightGCN scores for all candidate items
    cf_scores = []
    for asin in candidate_items:
        if asin in lgcn_item2idx:
            idx = lgcn_item2idx[asin]
            score = float(final_item_embeddings[idx] @ u_emb)
        else:
            score = 0.0
        cf_scores.append(score)
    cf_scores = np.array(cf_scores)

    if not history:
        # Cold start: Return products ranked by LightGCN scores alone
        top_positions = np.argsort(-cf_scores)[:top_n]
        results = []
        for pos in top_positions:
            asin = candidate_items[pos]
            if asin in asin_to_row.index:
                # Normalize cold-start score to [0,1] approximately
                norm_score = (cf_scores[pos] - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)
                results.append((asin_to_row.loc[asin], norm_score))
        return results

    liked = [h for h in history if h["rating"] >= 4] or history
    liked_indices = [asin_to_idx[h["asin"]] for h in liked if h["asin"] in asin_to_idx]
    
    if not liked_indices:
        # Fallback to LightGCN scores if no items have TF-IDF mappings
        top_positions = np.argsort(-cf_scores)[:top_n]
        results = []
        for pos in top_positions:
            asin = candidate_items[pos]
            if asin in asin_to_row.index:
                norm_score = (cf_scores[pos] - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)
                results.append((asin_to_row.loc[asin], norm_score))
        return results

    profile_vector = np.asarray(tfidf_matrix[liked_indices].mean(axis=0))
    content_scores = cosine_similarity(profile_vector, candidate_item_tfidf).flatten()

    content_norm = (content_scores - content_scores.min()) / (
        content_scores.max() - content_scores.min() + 1e-9
    )
    cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)
    hybrid_scores = alpha * content_norm + (1 - alpha) * cf_norm

    mask = np.array([asin in seen for asin in candidate_items])
    hybrid_scores[mask] = -np.inf

    top_positions = np.argsort(-hybrid_scores)[:top_n]
    results = []
    for pos in top_positions:
        asin = candidate_items[pos]
        if asin in asin_to_row.index:
            results.append((asin_to_row.loc[asin], hybrid_scores[pos]))
    return results


def lightgcn_recommend(user_id, top_n=10):
    """Graph-propagated embeddings (multi-hop signal), scored by dot product."""
    if user_id in lgcn_user2idx:
        u = lgcn_user2idx[user_id]
        u_emb = final_user_embeddings[u]
    else:
        # Cold start fallback: use the average user embedding
        u_emb = final_user_embeddings.mean(axis=0)

    scores = final_item_embeddings @ u_emb

    history = shopper_history(user_id)
    seen = {h["asin"] for h in history} if history else set()
    seen_mask = np.array([asin in seen for asin in lgcn_items])
    scores = scores.copy()
    scores[seen_mask] = -np.inf

    top_positions = np.argsort(-scores)[:top_n]
    results = []
    for pos in top_positions:
        asin = lgcn_items[pos]
        if asin in asin_to_row.index:
            results.append((asin_to_row.loc[asin], scores[pos]))
    return results


def predicted_rating(user_id, asin):
    if not user_id:
        return None
    try:
        return round(best_cf_model.predict(user_id, asin).est, 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    shopper_id = get_current_shopper()
    
    # If logged in, compute personalized recommendations
    hybrid_recs = []
    lightgcn_recs = []
    history = []
    if shopper_id:
        h_results = hybrid_recommend(shopper_id, top_n=12)
        # Convert rows to dicts and add match score
        hybrid_recs = []
        for row, score in h_results:
            d = row.to_dict()
            d["score"] = f"{score*100:.1f}%"
            d["predicted_rating"] = predicted_rating(shopper_id, d["asin"])
            hybrid_recs.append(d)

        lgcn_results = lightgcn_recommend(shopper_id, top_n=12)
        lightgcn_recs = []
        for row, score in lgcn_results:
            d = row.to_dict()
            d["score"] = f"{score:.3f}"
            d["predicted_rating"] = predicted_rating(shopper_id, d["asin"])
            lightgcn_recs.append(d)

        history = shopper_history(shopper_id)

    trending = (
        products_df[products_df["isBestSeller"] == True]
        .sort_values("popularity_score", ascending=False)
        .head(12)
    )
    if len(trending) < 12:
        trending = products_df.sort_values("popularity_score", ascending=False).head(12)

    categories = products_df["category_name"].value_counts().head(8).index.tolist()

    return render_template(
        "dashboard.html",
        popular_products=trending.to_dict("records"),
        trending=trending.to_dict("records"),
        categories=categories,
        total_products=len(products_df),
        shopper_id=shopper_id,
        hybrid_recs=hybrid_recs,
        lightgcn_recs=lightgcn_recs,
        history=history,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("user_id", "").strip()
        if user_id:
            session["shopper_id"] = user_id
            session["user_id"] = user_id
            if user_id not in demo_users:
                demo_users[user_id] = []
            return redirect(url_for("dashboard"))
    
    return render_template("login.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip()
    page = int(request.args.get("page", 1))
    per_page = 24

    filtered = products_df
    if query:
        filtered = filtered[filtered["title"].str.contains(query, case=False, na=False)]
    if category:
        filtered = filtered[filtered["category_name"].str.contains(category, case=False, na=False)]

    total_items = len(filtered)
    total_pages = max(1, int(np.ceil(total_items / per_page)))
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    results = filtered.iloc[start_idx:end_idx]

    return render_template(
        "search.html",
        query=query,
        category=category,
        products=results.to_dict("records"),
        total_items=total_items,
        total_pages=total_pages,
        current_page=page,
        shopper_id=get_current_shopper(),
    )


@app.route("/category/<path:category_name>")
def category(category_name):
    return redirect(url_for("search", category=category_name))


@app.route("/recommendations/<asin>")
def recommendations(asin):
    if asin not in asin_to_row.index:
        abort(404)
    product = asin_to_row.loc[asin]
    similar = content_based_similar(asin, top_n=12)

    recs = []
    for row, score in similar:
        d = row.to_dict()
        d["score"] = f"{score * 100:.1f}%"
        recs.append(d)

    shopper_id = get_current_shopper()
    my_rating = predicted_rating(shopper_id, asin) if shopper_id else None

    # Get CF predicted ratings for similar items
    for r in recs:
        r["predicted_rating"] = predicted_rating(shopper_id, r["asin"]) if shopper_id else None

    return render_template(
        "recommendation.html",
        product=product,
        recommendations=recs,
        my_rating=my_rating,
        cf_model_name=best_cf_model_name,
        alpha=0.5
    )


@app.route("/comparison")
def comparison():
    return render_template("comparison.html")



@app.route("/switch-shopper", methods=["POST"])
def switch_shopper():
    user_id = request.form.get("shopper_id", "").strip()
    if user_id in demo_users:
        session["shopper_id"] = user_id
        session["user_id"] = user_id
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/sign-out", methods=["POST"])
def sign_out():
    session.pop("shopper_id", None)
    session.pop("user_id", None)
    return redirect(request.referrer or url_for("dashboard"))


@app.errorhandler(404)
def not_found(e):
    return "<h3>404 - Page Not Found</h3><p>The requested URL was not found on the server.</p><a href='/'>Return to Home</a>", 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
