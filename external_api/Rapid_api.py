import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
from notification.notification import send_mail, send_slack_notification
from external_api import sentiment_rapid_spike


# -----------------------------
# API CONFIG
# -----------------------------
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
}

SEARCH_URL = "https://real-time-amazon-data.p.rapidapi.com/search"
REVIEW_URL = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"

COUNTRY = "US"
SEARCH_PAGE = 1
REVIEW_PAGE = 1


# -----------------------------
# CATEGORY KEYWORDS
# -----------------------------
CATEGORY_KEYWORDS = {
    "Electricals_Power_Backup": ["inverter", "ups", "power backup", "generator"],
    "Home_Appliances": ["air conditioner", "refrigerator", "washing machine"],
    "Kitchen_Appliances": ["mixer", "grinder", "microwave"],
    "Computers_Tablets": ["laptop", "tablet"],
    "Mobile_Accessories": ["charger", "earphones", "power bank"],
    "Wearables": ["smartwatch", "fitness band"],
    "TV_Audio_Entertainment": ["smart tv", "speaker"],
}


# -----------------------------
# SEARCH PRODUCTS
# -----------------------------
def search_products(query):
    params = {
        "query": query,
        "page": SEARCH_PAGE,
        "country": COUNTRY,
        "sort_by": "RELEVANCE"
    }
    response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("products", [])


# -----------------------------
# FETCH REVIEWS
# -----------------------------
def fetch_reviews(asin):
    params = {
        "asin": asin,
        "country": COUNTRY,
        "page": REVIEW_PAGE,
        "sort_by": "TOP_REVIEWS"
    }
    response = requests.get(REVIEW_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("reviews", [])


# -----------------------------
# MAIN RAPID PIPELINE
# -----------------------------
def rapid_api(notification_mode="email"):
    try:
        all_reviews = []

        for category, keywords in tqdm(CATEGORY_KEYWORDS.items(), desc="Categories"):
            for keyword in keywords:
                try:
                    products = search_products(keyword)

                    for product in products[:5]:  # rate-limit safety
                        asin = product.get("asin")
                        if not asin:
                            continue

                        reviews = fetch_reviews(asin)

                        for r in reviews:
                            all_reviews.append({
                                "category": category,
                                "keyword_used": keyword,
                                "asin": asin,
                                "product_title": product.get("title"),
                                "brand": product.get("brand"),
                                "price": product.get("price"),
                                "rating": r.get("rating"),
                                "review_title": r.get("review_title"),
                                "review_text": r.get("review_text"),
                                "review_date": r.get("review_date"),
                                "reviewer": r.get("reviewer_name"),
                                "verified_purchase": r.get("verified_purchase"),
                                "collected_at": datetime.utcnow()
                            })

                except Exception as e:
                    print(f"❌ Error for keyword '{keyword}': {e}")

        # -----------------------------
        # SAVE DATA
        # -----------------------------
        df = pd.DataFrame(all_reviews)

        df.drop_duplicates(
            subset=["asin", "review_text"],
            inplace=True
        )

        df.to_csv("Final Data/amazon_reviews_categorized.csv", index=False)
        print(f"✅ Saved {len(df)} reviews")

        # -----------------------------
        # SENTIMENT SPIKE DETECTION
        # -----------------------------
        result_df = sentiment_rapid_spike.rapid_sentiment_spike(df)

        # -----------------------------
        # NOTIFICATIONS
        # -----------------------------
        if result_df.empty:
            message = (
                "Rapid API data extracted successfully.\n"
                "No major Amazon sentiment spikes detected this week."
            )
        else:
            message = (
                "🚨 Amazon sentiment spikes detected.\n"
                "Please find attached weekly sentiment alert report."
            )

        if notification_mode == "email":
            send_mail(
                subject="Amazon (Rapid API) Sentiment Report",
                text=message,
                df=result_df if not result_df.empty else None
            )

        elif notification_mode == "slack":
            send_slack_notification(message)

    except Exception as e:
        print("❌ Rapid API pipeline failed:", e)

        error_msg = f"Rapid API extraction failed due to: {e}"

        if notification_mode == "email":
            send_mail(
                subject="❌ Rapid API Extraction Failed",
                text=error_msg
            )
        else:
            send_slack_notification(error_msg)

if __name__ == "__main__":
    rapid_api(notification_mode="email")