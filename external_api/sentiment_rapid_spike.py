import pandas as pd


def rapid_sentiment_spike(df):
    # -----------------------------
    # CONFIG
    # -----------------------------
    DATE_COL = "review_date"
    RATING_COL = "rating"
    CATEGORY_COL = "category"

    WEEK_WINDOW = 2
    SPIKE_THRESHOLD = 0.3
    TREND_SHIFT_THRESHOLD = 0.3

    LABEL_MAP = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    # -----------------------------
    # CLEAN & PREP DATA
    # -----------------------------

    # Parse review date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    df = df.dropna(subset=[DATE_COL, RATING_COL, CATEGORY_COL])
    df = df.sort_values(DATE_COL)

    # -----------------------------
    # DERIVE SENTIMENT FROM RATING
    # -----------------------------
    def rating_to_sentiment(r):
        if r >= 4:
            return "positive"
        elif r == 3:
            return "neutral"
        else:
            return "negative"

    df["sentiment_label"] = df[RATING_COL].apply(rating_to_sentiment)
    df["sentiment_score"] = df["sentiment_label"].map(LABEL_MAP)

    # -----------------------------
    # WEEKLY AGGREGATION
    # -----------------------------
    weekly_cat = (
        df.groupby([CATEGORY_COL, df[DATE_COL].dt.to_period("W")])["sentiment_score"]
        .mean()
        .reset_index(name="weekly_sentiment")
    )

    weekly_cat[DATE_COL] = weekly_cat[DATE_COL].dt.start_time

    # -----------------------------
    # ROLLING WINDOW CALCS
    # -----------------------------
    weekly_cat["rolling_avg"] = (
        weekly_cat
        .groupby(CATEGORY_COL)["weekly_sentiment"]
        .rolling(WEEK_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    weekly_cat["prev_rolling_avg"] = (
        weekly_cat.groupby(CATEGORY_COL)["rolling_avg"].shift(1)
    )

    weekly_cat["delta"] = weekly_cat["rolling_avg"] - weekly_cat["prev_rolling_avg"]
    weekly_cat["prev_delta"] = (
        weekly_cat.groupby(CATEGORY_COL)["delta"].shift(1)
    )

    # -----------------------------
    # DETECT ALERTS
    # -----------------------------
    alerts = []

    for _, row in weekly_cat.iterrows():
        if pd.isna(row["delta"]):
            continue

        if row["delta"] <= -SPIKE_THRESHOLD:
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "NEGATIVE SPIKE",
                "change": round(row["delta"], 3)
            })

        if row["delta"] >= SPIKE_THRESHOLD:
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "POSITIVE SPIKE",
                "change": round(row["delta"], 3)
            })

        if (
            pd.notna(row["prev_delta"])
            and row["delta"] * row["prev_delta"] < 0
            and abs(row["delta"]) >= TREND_SHIFT_THRESHOLD
        ):
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "TREND SHIFT",
                "change": round(row["delta"], 3)
            })

    alert_df = pd.DataFrame(alerts)

    # -----------------------------
    # FILTER ONLY LAST WEEK
    # -----------------------------
    if not alert_df.empty:
        latest_date = alert_df["date"].max()
        last_week_start = latest_date - pd.Timedelta(days=7)

        alert_df = alert_df[alert_df["date"] >= last_week_start]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    if alert_df.empty:
        print("✅ No Amazon (Rapid) sentiment spikes in the last week.")
    else:
        print("🚨 LAST WEEK AMAZON SENTIMENT ALERTS")
        print(
            alert_df
            .sort_values(["date", "category"])
            .reset_index(drop=True)
        )

    return alert_df

#if __name__ == "__main__":
#    df = pd.read_csv("Final Data/rapid_api_reviews_final.csv")
#    rapid_sentiment_spike(df)