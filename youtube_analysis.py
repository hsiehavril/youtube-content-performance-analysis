"""
YouTube content performance: Excel pivots -> Python rebuild.

Steps: (1) clean, (2) fix the Excel pivot problems (repeated videos, skewed means),
(3) compare categories with medians, (4) model log10(views) and compare two models.
Change PATH to where your youtube.xlsx lives.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

PATH = "youtube.xlsx"

# ---------- 1. Load and clean ----------
# find the sheet that holds the raw video rows (the one with a video_id column)
sheets = pd.read_excel(PATH, sheet_name=None)
raw_name = next((n for n, d in sheets.items() if "video_id" in d.columns), None)
if raw_name is None:
    raise ValueError("No sheet with a 'video_id' column. Sheets found: " + ", ".join(sheets))
print("using sheet:", raw_name)
df = sheets[raw_name]
print("raw rows:", len(df))

df = df.drop(columns=["Early vs Late performance"])        # every value is "Old", no information
df["title"] = df["title"].astype(str)                      # a few titles are numbers
print("missing video_id:", df["video_id"].isna().sum(),
      "| repeated video_id rows:", df["video_id"].duplicated().sum())

# The same video appears on many rows (one per snapshot). Keep one row per video
# (the snapshot with the most views) so repeated videos don't dominate the averages.
df = (df.dropna(subset=["video_id", "views", "publish_date", "category_name"])
        .query("views > 0")
        .sort_values("views", ascending=False)
        .drop_duplicates(subset="video_id"))
print("clean rows (one per video):", len(df))

# ---------- 2. Feature engineering ----------
df["year"] = df["publish_date"].dt.year
df["month"] = df["publish_date"].dt.month
df["dow"] = df["publish_date"].dt.dayofweek
df["title_len"] = df["title"].str.len()
df["title_caps_ratio"] = df["title"].apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
df["log_views"] = np.log10(df["views"])                    # views are heavily right-skewed

# ---------- 3. Mean vs median by category ----------
cat = (df.groupby("category_name")
         .agg(n=("views", "size"),
              mean_views=("views", "mean"),
              median_views=("views", "median"),
              median_engagement_rate=("Engagement Rate", "median"))
         .sort_values("median_views", ascending=False))
print(cat.round(3))

top = cat[cat["n"] >= 100].sort_values("median_views")     # ignore tiny categories
plt.figure(figsize=(8, 6))
plt.barh(top.index, top["median_views"])
plt.xlabel("Median views")
plt.title("Median views by category (categories with 100+ videos)")
plt.tight_layout()
plt.savefig("median_views_by_category.png", dpi=150)

# ---------- 4. Model log10(views) ----------
X = pd.get_dummies(
    df[["category_name", "month", "dow", "year"]].astype({"month": str, "dow": str, "year": str}),
    drop_first=True)
X["title_len"] = df["title_len"].values
X["title_caps_ratio"] = df["title_caps_ratio"].values
y = df["log_views"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

lin = LinearRegression().fit(X_tr, y_tr)
rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
for name, model in [("Linear regression", lin), ("Random forest", rf)]:
    pred = model.predict(X_te)
    print(f"{name}: R2 = {r2_score(y_te, pred):.3f}, RMSE = {mean_squared_error(y_te, pred) ** 0.5:.3f} (log10 views)")

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print("\nTop random-forest features:\n", imp.round(3))
