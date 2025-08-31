## Data Cleaning and Preprocessing Notes

These notes follow Kaggle’s tutorials on handling missing values, scaling/normalization, and working with dates.
https://www.kaggle.com/code/alexisbcook/handling-missing-values


1. Handling Missing Values

Detect: df.isnull().sum() gives missing count per column.

Quantify: calculate percentage of missing cells relative to dataset size.

Remove:

Rows → df.dropna()

Columns → df.dropna(axis=1)

Impute:

Backfill with next value → df.fillna(method='bfill')

Replace leftovers with zeros → .fillna(0)

2. Scaling and Normalization

Scaling: Adjusts data to a fixed range (e.g. [0,1]).

Example: mlxtend.preprocessing.minmax_scaling

Normalization: Transforms data to follow a normal (Gaussian) distribution.

Example: Box-Cox transformation with scipy.stats.boxcox

Use case: Kickstarter goals scaled and pledges normalized for better distribution.

3. Working with Dates

Detect corrupted date strings by checking length.

Correct invalid entries manually.

Convert strings to datetime with pd.to_datetime.

Extract components (e.g. .dt.day) for analysis.

Example: Earthquake dataset → plot earthquakes by day of month.

# Key Takeaways

Always inspect missing data before deciding removal or imputation.

Scaling puts features in comparable ranges; normalization reshapes distributions.

Clean and standardize dates before extracting features for time series or analysis.