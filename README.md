# YouTube Content Performance Analysis

What content categories are associated with higher reach and higher engagement on YouTube, and how much do the results change once repeated observations are handled correctly?

This project started as an Excel analysis and was rebuilt in Python after I found that repeated snapshots of the same videos were making the first-round results look more certain than they were.

## Question

1. Which categories get the most views, and which get the most engagement relative to views (engagement rate)?
2. How much of the variation in views and engagement rate can category, publish month, and title features explain?
3. Do the conclusions hold after fixing the unit-of-analysis problem?

## Data

- Source: [YouTube Trending Videos Dataset on Kaggle](https://www.kaggle.com/datasets/thedevastator/youtube-trending-videos-dataset/data) (published by thedevastator)
- 161,470 rows, 13 columns: video ID, title, category, publish date, views, likes, comment count, and derived fields.
- Engagement = likes + comments. Engagement rate = (likes + comments) / views.
- The rows are repeated snapshots: 1,799 rows have no video ID, and 105,584 rows repeat a video ID that already appears. After keeping one row per video there are about 55,900 unique videos, mostly published in 2017 and 2018.
- The dataset is a collection of trending videos, so results describe videos that reached a trending list, not YouTube in general.

## Method

**Round 1: Excel.** Pivot tables for average views, average engagement, and engagement rate by category and month; a median-vs-mean comparison; and an OLS regression of engagement rate on category and month with HC3 robust standard errors.

**Round 2: Python** (`youtube_analysis.py`, `youtube_regression.py`).

1. Cleaned the data: dropped an uninformative column (every value was "Old"), removed rows with missing key fields, and kept one row per video (the snapshot with the most views).
2. Compared categories using medians, because views are heavily right-skewed.
3. Predicted log10(views) with linear regression and a random forest (80/20 train/test split).
4. Re-ran the engagement-rate regression four ways:
   - A: all rows, HC3 standard errors (replicates the Excel model)
   - B: all rows, standard errors clustered by video ID
   - C: one row per video, HC3 standard errors
   - D: one row per video, views of at least 10,000 (sensitivity check)
5. Excluded categories with fewer than 30 videos (Movies, Trailers). The reference category is Autos & Vehicles.

## Findings

- **Averages were misleading.** For Music, average views were about 3.6 million but the median was about 318,000 (one row per video), so a few viral videos pulled the average up.
- **Clustering mattered.** Clustering standard errors by video ID made them about 1.9 times larger than in the replicated Excel model. Entertainment, Film & Animation, People & Blogs, Nonprofits & Activism, and Travel & Events were no longer statistically significant at the 5% level.
- **Deduplication changed some estimates.** Comedy fell from +2.0 to +1.3 percentage points relative to Autos & Vehicles, Music from +1.2 to +0.6, and People & Blogs flipped from +0.2 to -0.8.
- **Some results were stable across all four models.** Gaming, How-to & Style, Education, and Science & Technology were associated with higher engagement rates; Shows, News & Politics, and Sports with lower ones.
- **Fit is limited.** Category and month explained about 8% (A, B), 10% (C), and 11% (D) of the variation in engagement rate. For log views, linear regression had R² of 0.08 and a random forest 0.17 on the test set.

## Limitations

- Observational data: associations, not causes.
- No control for channel size, video age, topic, promotion, audience, or recommendation exposure.
- Month is modeled without year, so month effects cannot be read as seasonality.
- Videos with very few views can produce unstable engagement rates (model D checks this).
- Keeping the snapshot with the most views is one reasonable deduplication rule, not the only one.
- Small categories (for example Pets & Animals, about 500 videos) give unstable estimates.

## What I would do next

- Add channel-level data to compare videos within similar channels and time windows.
- Model log engagement rate or use a fractional-response model.
- Add year effects and video age.

## Files

| File | Description |
|---|---|
| `youtube_analysis.py` | Cleaning, median comparison, views models |
| `youtube_regression.py` | Engagement-rate regressions A to D |
| `median_views_by_category.png` | Median views by category |

## How to run

1. Open the notebook in Google Colab, or run the scripts locally with Python 3.
2. Upload the data file and set `PATH` at the top of each script.
3. Required packages: pandas, numpy, openpyxl, scikit-learn, matplotlib, statsmodels.

## About

Wei-Chu Hsieh. B.A. in Communication, UC Davis. Contact: hsiehavril@gmail.com
