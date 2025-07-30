import pandas as pd
import numpy as np

###Handling Missing Values###

sf_permits = pd.read_csv("C:/Users/danie/py/progs/Building_Permits.csv", low_memory=False)
np.random.seed(0) 

print(sf_permits.head())

#What percentage of the values in the dataset are missing? 

missing_values_count = sf_permits.isnull().sum()

total_missing = missing_values_count.sum()

total_cells = sf_permits.shape[0] * sf_permits.shape[1]

percent_missing = (total_missing/total_cells) * 100
print(percent_missing)

# Now a good practice is to check if the missing value does not exist or if it was not recorded.

# remove all the rows that contain a missing value
print(sf_permits.dropna())

#Create a new DataFrame called `sf_permits_with_na_dropped` that has all of the columns with empty values removed.  

sf_permits_with_na_dropped = sf_permits.dropna(axis=1)

print("Columns in original dataset: %d \n" % sf_permits.shape[1])

print("Columns with na's dropped: %d" % sf_permits_with_na_dropped.shape[1])


# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
sf_permits_with_na_imputed = sf_permits.fillna(method='bfill', axis=0).fillna(0)
print(sf_permits_with_na_imputed)

###Scaling and Normalization###

# Scaling means that you're transforming your data so that it fits within a specific scale, like 0-100 or 0-1

#Normalization: Scaling just changes the range of your data. Normalization is a more radical transformation. The point of normalization is to change your observations so that they can be described as a normal distribution.
#Normal distribution: Also known as the "bell curve", this is a specific statistical distribution where a roughly equal observations fall above and below the mean, the mean and the median are the same, and there are more observations closer to the mean.
#The normal distribution is also known as the Gaussian distribution.

# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pd.read_csv("C:/Users/danie/py/my_datasets/ks-projects-201801.csv ", low_memory=False)
np.random.seed(0)

# select the usd_goal_real column
original_data = pd.DataFrame(kickstarters_2017.usd_goal_real)

# scale the goals from 0 to 1
scaled_data = minmax_scaling(original_data, columns=['usd_goal_real'])

print('Original data\nPreview:\n', original_data.head())
print('Minimum value:', float(original_data.min()),
      '\nMaximum value:', float(original_data.max()))
print('_'*30)

print('\nScaled data\nPreview:\n', scaled_data.head())
print('Minimum value:', float(scaled_data.min()),
      '\nMaximum value:', float(scaled_data.max()))

# select the usd_goal_real column
original_goal_data = pd.DataFrame(kickstarters_2017.goal)
scaled_goal_data = minmax_scaling(original_goal_data, columns=['goal'])

# get the index of all positive pledges (Box-Cox only takes positive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = pd.Series(stats.boxcox(positive_pledges)[0], 
                               name='usd_pledged_real', index=positive_pledges.index)

print('Original data\nPreview:\n', positive_pledges.head())
print('Minimum value:', float(positive_pledges.min()),
      '\nMaximum value:', float(positive_pledges.max()))
print('_'*30)

print('\nNormalized data\nPreview:\n', normalized_pledges.head())
print('Minimum value:', float(normalized_pledges.min()),
      '\nMaximum value:', float(normalized_pledges.max()))

# plot normalized data
ax = sns.histplot(normalized_pledges, kde=True)
ax.set_title("Normalized data")
plt.show()

# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

# read in our data
earthquakes = pd.read_csv("C:/Users/danie/py/my_datasets/database.csv")

# set seed for reproducibility
np.random.seed(0)

print(earthquakes.head())

print(earthquakes[3378:3383])
date_lengths = earthquakes.Date.str.len()
print(date_lengths)
date_lengths.value_counts()
print(date_lengths.value_counts())
indices = np.where([date_lengths == 24])[1]
print('Indices with corrupted data:', indices)
earthquakes.loc[indices]

earthquakes.loc[3378, "Date"] = "02/23/1975"
earthquakes.loc[7512, "Date"] = "04/28/1985"
earthquakes.loc[20650, "Date"] = "03/13/2011"
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y")

# try to get the day of the month from the date column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
print(day_of_month_earthquakes)

# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.displot(day_of_month_earthquakes, kde=False, bins=31)
plt.show()
