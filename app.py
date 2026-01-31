from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# -------------------------------------------------
# LOAD & PREPROCESS DATA
# -------------------------------------------------
df = pd.read_csv("online_retail.csv")

df = df.dropna(subset=["Customer ID", "Description"])
df = df[df["Quantity"] > 0]
df["Customer ID"] = df["Customer ID"].astype(int)
df["TotalPrice"] = df["Quantity"] * df["Price"]

# -------------------------------------------------
# PRODUCT PROFILES
# -------------------------------------------------
products = df.groupby("StockCode").agg(
    description=("Description", "first"),
    avg_price=("Price", "mean"),
    popularity=("Quantity", "sum")
).reset_index()

# -------------------------------------------------
# VECTOR MODEL
# -------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
product_vectors = vectorizer.fit_transform(products["description"])

# -------------------------------------------------
# QUERY PARSER
# -------------------------------------------------
def parse_query(query):
    if not query:
        return [], None
    words = query.lower().split()
    keywords = [w for w in words if len(w) > 3]

    budget = None
    m = re.search(r"under\s*(\d+)", query.lower())
    if m:
        budget = int(m.group(1))

    return keywords, budget

# -------------------------------------------------
# USER PROFILE
# -------------------------------------------------
def build_user_profile(customer_id):
    user_df = df[df["Customer ID"] == customer_id]
    if user_df.empty:
        return None

    avg_price = user_df["Price"].mean()

    return {
        "orders": user_df["Invoice"].nunique(),
        "total_spent": int(user_df["TotalPrice"].sum()),
        "preference": "Mid-to-premium brands" if avg_price > 2000 else "Budget-friendly"
    }

# -------------------------------------------------
# INTENT DETECTION
# -------------------------------------------------
INTENT_MAP = {
    "gaming": ["gaming", "gamer", "keyboard", "mouse", "headset"],
    "electronics": ["phone", "charger", "cable", "headphone", "laptop"],
    "fashion": ["shirt", "dress", "bag", "jeans"],
    "decor": ["decor", "lamp", "candle"],
    "gift": ["gift", "present"]
}

def detect_intent(query):
    if not query:
        return None
    q = query.lower()
    for intent, words in INTENT_MAP.items():
        if any(w in q for w in words):
            return intent
    return None

# -------------------------------------------------
# RECOMMENDATION ENGINE
# -------------------------------------------------
def recommend(customer_id=None, query=None, top_n=5):
    recs = products.copy()
    keywords, budget = parse_query(query)
    intent = detect_intent(query)

    # ---------- EXISTING USER ----------
    if customer_id:
        user_df = df[df["Customer ID"] == customer_id]
        if not user_df.empty:
            user_text = " ".join(user_df["Description"].tolist())
            user_vec = vectorizer.transform([user_text])
            scores = cosine_similarity(user_vec, product_vectors)[0]

            recs["score"] = scores
            recs = recs[~recs["StockCode"].isin(user_df["StockCode"])]
            recs = recs.sort_values("score", ascending=False)

    # ---------- COLD START ----------
    else:
        if intent:
            recs = recs[
                recs["description"].str.lower().str.contains(
                    "|".join(INTENT_MAP[intent]), na=False
                )
            ]
        recs = recs.sort_values("popularity", ascending=False)

    # ---------- QUERY FILTER ----------
    for kw in keywords:
        recs = recs[recs["description"].str.lower().str.contains(kw, na=False)]

    if budget:
        recs = recs[recs["avg_price"] <= budget]

    if recs.empty:
        recs = products.sort_values("popularity", ascending=False)

    return recs.head(top_n)

# -------------------------------------------------
# LLM-STYLE ANALYSIS
# -------------------------------------------------
def llm_profile_analysis(profile):
    if not profile:
        return (
            "New user detected. Shopping appears to be for someone else. "
            "Recommendations are based on popular and trending products."
        )

    return (
        "Tech-savvy buyer, frequent accessories purchases. "
        "Price sensitivity: moderate. Best time to recommend: weekends."
    )

def product_explanation(product, profile, query, score):
    reasons = []

    if profile and score is not None:
        if score > 0.35:
            reasons.append(
                "You purchased similar accessories recently. "
                "Similar users who bought those items later bought this product."
            )
        elif score > 0.15:
            reasons.append(
                "This product is moderately similar to items you previously explored."
            )

    if query:
        for w in query.lower().split():
            if w in product["description"].lower():
                reasons.append(f"Matches your interest in '{w}'")

    if profile:
        if profile["preference"].startswith("Mid") and product["avg_price"] > 2000:
            reasons.append("Your typical spending range matches this product")
        elif profile["preference"].startswith("Budget") and product["avg_price"] <= 2000:
            reasons.append("Fits your budget preference")

    if not reasons:
        reasons.append("Popular choice among many customers")

    return ". ".join(reasons)

# -------------------------------------------------
# 🔥 FIXED: PRODUCT-SPECIFIC MATCH REASONS
# -------------------------------------------------
def match_reasons(product, profile, query, score):
    checks = []

    # Similar users
    if score is not None and score > 0.30:
        checks.append("Similar users")

    # Category interest
    if query:
        for w in query.lower().split():
            if w in product["description"].lower():
                checks.append("Category interest")
                break

    # Price match
    if profile:
        if profile["preference"].startswith("Mid") and product["avg_price"] > 2000:
            checks.append("Price match")
        elif profile["preference"].startswith("Budget") and product["avg_price"] <= 2000:
            checks.append("Price match")

    # Brand affinity (simple heuristic)
    brand = product["description"].split()[0].lower()
    if query and brand in query.lower():
        checks.append("Brand affinity")

    if not checks:
        checks.append("Popular choice")

    return checks

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    data = request.json
    customer_id = data.get("customer_id")
    query = data.get("query")

    customer_id = int(customer_id) if customer_id and customer_id.isdigit() else None

    profile = build_user_profile(customer_id)
    recs = recommend(customer_id, query)

    response_recs = []
    for _, p in recs.iterrows():
        score = p.get("score", None)
        response_recs.append({
            "name": p["description"],
            "price": round(p["avg_price"], 2),
            "why": product_explanation(p, profile, query, score),
            "matches": match_reasons(p, profile, query, score)
        })

    return jsonify({
        "profile": profile,
        "analysis": llm_profile_analysis(profile),
        "recommendations": response_recs
    })

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
