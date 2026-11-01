"""
Engagement-rate regression: replicate the Excel model, then fix the repeated-video problem.

Model A: all rows, HC3 robust SE            (replicates the Excel workbook)
Model B: all rows, SE clustered by video_id (repeated videos are not independent)
Model C: one row per video, HC3 robust SE   (cleanest unit of analysis)

Run in Colab after uploading youtube.xlsx. Change PATH if your file is elsewhere.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PATH = "youtube.xlsx"

# find the sheet that holds the raw video rows (the one with a video_id column)
sheets = pd.read_excel(PATH, sheet_name=None)
raw_name = next((n for n, d in sheets.items() if "video_id" in d.columns), None)
if raw_name is None:
    raise ValueError("No sheet with a 'video_id' column. Sheets found: " + ", ".join(sheets)
                     + ". Upload the original raw-data file.")
print("using sheet:", raw_name)
df = sheets[raw_name]
df = df.dropna(subset=["video_id", "views", "publish_date", "category_name"])
df = df[df["views"] > 0].copy()
df["month"] = df["publish_date"].dt.month
df["er_pct"] = df["Engagement Rate"] * 100          # percentage points, like the Excel sheet

# one row per video: keep the snapshot with the most views
df_one = df.sort_values("views", ascending=False).drop_duplicates(subset="video_id").copy()

# drop tiny categories (e.g. Trailers has 1 video, Movies 10): they cannot be estimated reliably
# and a category with one video breaks HC3 standard errors
counts = df_one["category_name"].value_counts()
keep = counts[counts >= 30].index
print("dropped categories:", sorted(set(counts.index) - set(keep)))
df = df[df["category_name"].isin(keep)].copy()
df_one = df_one[df_one["category_name"].isin(keep)].copy()
print("rows with video_id:", len(df), "| unique videos:", len(df_one))

formula = ("er_pct ~ C(category_name, Treatment('Autos & Vehicles'))"
           " + C(month, Treatment(1))")

model_a = smf.ols(formula, data=df).fit(cov_type="HC3")
model_b = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["video_id"])[0]})
model_c = smf.ols(formula, data=df_one).fit(cov_type="HC3")

for name, m in [("A: all rows, HC3", model_a),
                ("B: all rows, clustered by video_id", model_b),
                ("C: one row per video, HC3", model_c)]:
    print(f"Model {name}: n = {int(m.nobs):,}, R2 = {m.rsquared:.3f}")

# ---- compare coefficients and standard errors side by side ----
def tidy(m, label):
    out = pd.DataFrame({f"coef_{label}": m.params, f"se_{label}": m.bse, f"p_{label}": m.pvalues})
    out.index = (out.index.str.replace(r"C\(category_name, Treatment\('Autos & Vehicles'\)\)\[T\.", "cat: ", regex=True)
                          .str.replace(r"C\(month, Treatment\(1\)\)\[T\.", "month ", regex=True)
                          .str.replace("]", "", regex=False))
    return out

comp = pd.concat([tidy(model_a, "A"), tidy(model_b, "B"), tidy(model_c, "C")], axis=1)
pd.set_option("display.width", 200)
print(comp.round(3).to_string())

# ---- sensitivity: drop very low-view videos (their rates are unstable) ----
df_big = df_one[df_one["views"] >= 10_000]
model_d = smf.ols(formula, data=df_big).fit(cov_type="HC3")
print(f"\nModel D: one row per video, views >= 10,000: n = {int(model_d.nobs):,}, R2 = {model_d.rsquared:.3f}")
print(tidy(model_d, "D").round(3).to_string())
