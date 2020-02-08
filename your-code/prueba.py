# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Before your start:
# - Read the README.md file
# - Comment as much as you can and use the resources in the README.md file
# - Happy learning!

# %%
#Import your libraries
import pandas, numpy
from datetime import datetime
from sklearn.model_selection import train_test_split

# %% [markdown]
# 
# # Challenge 1 - Import and Describe the Dataset
# %% [markdown]
# #### In this challenge we will use the `austin_weather.csv` file. 
# 
# #### First import it into a data frame called `austin`.

# %%
# Your code here
#%%
data = pandas.read_csv('./austin_weather.csv')
data
# %% [markdown]
# #### Next, describe the dataset you have loaded: 
# - Look at the variables and their types
# - Examine the descriptive statistics of the numeric variables 
# - Look at the first five rows of all variables to evaluate the categorical variables as well

# %%
# Your code here
#%%
data.info();


# %%
# Your code here
#%%
data.describe()


# %%
# Your code here
#%%
data.head()

# %% [markdown]
# #### Given the information you have learned from examining the dataset, write down three insights about the data in a markdown cell below
# %% [markdown]
# #### Your Insights:
# 
# 1. There are 21 variables in the dataset. 3 of them are numeric and the rest contain some text.
# 
# 2. The average temperature in Austin ranged between around 70 degrees F and around 93 degrees F. The highest temperature observed during this period was 107 degrees F and the lowest was 19 degrees F.
# 
# 3. When we look at the head function, we see that a lot of variables contain numeric data even though these columns are of object type. This means we might have to do some data cleansing.
# 
# %% [markdown]
# #### Let's examine the DewPointAvgF variable by using the `unique()` function to list all unique values in this dataframe.
# 
# Describe what you find in a markdown cell below the code. What did you notice? What do you think made Pandas to treat this column as *object* instead of *int64*? 

# %%
# Your code here
#%%
data.DewPointAvgF.unique()


# %%
# Your observation here
## Lo que hizo pandas tratar al array como OBJECT, fue el guion atravezado que tiene en la posición 56.


# %%
#%%
len(data.DewPointAvgF.unique()) # 66
print("This MOFO right here:",data.DewPointAvgF.unique()[56])

# %% [markdown]
# The following is a list of columns misrepresented as `object`. Use this list to convert the columns to numeric using the `pandas.to_numeric` function in the next cell. If you encounter errors in converting strings to numeric values, you need to catch those errors and force the conversion by supplying `errors='coerce'` as an argument for `pandas.to_numeric`. Coercing will replace non-convertable elements with `NaN` which represents an undefined numeric value. This makes it possible for us to conveniently handle missing values in subsequent data processing.
# 
# *Hint: you may use a loop to change one column at a time but it is more efficient to use `apply`.*

# %%
#%%
wrong_type_columns = ['DewPointHighF', 'DewPointAvgF', 'DewPointLowF', 'HumidityHighPercent', 
                      'HumidityAvgPercent', 'HumidityLowPercent', 'SeaLevelPressureHighInches', 
                      'SeaLevelPressureAvgInches' ,'SeaLevelPressureLowInches', 'VisibilityHighMiles',
                      'VisibilityAvgMiles', 'VisibilityLowMiles', 'WindHighMPH', 'WindAvgMPH', 
                      'WindGustMPH', 'PrecipitationSumInches']
len(wrong_type_columns)


# %%
#%%
data_clean = data[wrong_type_columns].apply(pandas.to_numeric, errors='coerce')
data_clean

# %% [markdown]
# #### Check if your code has worked by printing the data types again. You should see only two `object` columns (`Date` and `Events`) now. All other columns should be `int64` or `float64`.

# %%
#%%
data_clean1 = data[['Date','TempHighF','TempAvgF','TempLowF','Events']]
#print(data_clean1.dtypes)
#print(data_clean.dtypes)
data_concat = pandas.concat([data_clean1,data_clean], axis=1)

print(data_concat.dtypes)
data_concat

# %% [markdown]
# # Challenge 2 - Handle the Missing Data
# %% [markdown]
# #### Now that we have fixed the type mismatch, let's address the missing data.
# 
# By coercing the columns to numeric, we have created `NaN` for each cell containing characters. We should choose a strategy to address these missing data.
# 
# The first step is to examine how many rows contain missing data.
# 
# We check how much missing data we have by applying the `.isnull()` function to our dataset. To find the rows with missing data in any of its cells, we apply `.any(axis=1)` to the function. `austin.isnull().any(axis=1)` will return a column containing true if the row contains at least one missing value and false otherwise. Therefore we must subset our dataframe with this column. This will give us all rows with at least one missing value. 
# 
# #### In the next cell, identify all rows containing at least one missing value. Assign the dataframes with missing values to a variable called `missing_values`.

# %%
# Your code here
#%%
missing_values = data_concat[data_concat.isnull().any(axis=1)]
missing_values


# %%
#%%
data_concat.isnull().sum()

# %% [markdown]
# There are multiple strategies to handle missing data. Below lists the most common ones data scientists use:
# 
# * Removing all rows or all columns containing missing data. This is the simplest strategy. It may work in some cases but not others.
# 
# * Filling all missing values with a placeholder value. 
#     * For categorical data, `0`, `-1`, and `9999` are some commonly used placeholder values. 
#     * For continuous data, some may opt to fill all missing data with the mean. This strategy is not optimal since it can increase the fit of the model.
# 
# * Filling the values using some algorithm. 
# 
# #### In our case, we will use a hybrid approach which is to first remove the data that contain most missing values then fill in the rest of the missing values with the *linear interpolation* algorithm.
# %% [markdown]
# #### Next, count the number of rows of `austin` and `missing_values`.

# %%
#%%
print('NUMERO DE REGISTROS POR COLUMNA: ',len(data_concat))
print('NUMERO DE REGISTROS POR COLUMNA: ',len(missing_values))

# %% [markdown]
# #### Calculate the ratio of missing rows to total rows

# %%
# Your code here
#%%
ratio = round((len(missing_values)/len(data_concat))*100,4)
print('Missing Rows Ratio:', ratio,'%')

# %% [markdown]
# As you can see, there is a large proportion of missing data (over 10%). Perhaps we should evaluate which columns have the most missing data and remove those columns. For the remaining columns, we will perform a linear approximation of the missing data.
# 
# We can find the number of missing rows in each column using the `.isna()` function. We then chain the `.sum` function to the `.isna()` function and find the number of missing rows per column

# %%
#%%
# Your code here
data_concat.isna().sum()

# %% [markdown]
# #### As you can see from the output, the majority of missing data is in one column called `PrecipitationSumInches`. What's the number of missing values in this column in ratio to its total number of rows?

# %%
#%%
# Your code here
print('[Ratio MISSING DATA]:[Precipitation Sum Inches]:',
      round((data_concat.isna().sum().PrecipitationSumInches)/len(data_concat.PrecipitationSumInches)*100,4))

# %% [markdown]
# Over 10% data missing! Therefore, we prefer to remove this column instead of filling its missing values. It doesn't make sense to *guess* its missing values because the estimation will be too 
# 
# #### Remove this column from `austin` using the `.drop()` function. Use the `inplace=True` argument.
# 
# *Hints:*
# 
# * By supplying `inplace=True` to `drop()`, the original dataframe object will be changed in place and the function will return `None`. In contrast, if you don't supply `inplace=True`, which is equivalent to supplying `inplace=False` because `False` is the default value, the original dataframe object will be kept and the function returns a copy of the transformed dataframe object. In the latter case, you'll have to assign the returned object back to your variable.
# 
# * Also, since you are dropping a column instead of a row, you'll need to supply `axis=1` to `drop()`.
# 
# [Reference for `pandas.DataFrame.drop`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)

# %%
#%%
# Your code here 
data_concat.drop('PrecipitationSumInches', inplace=True, axis=1)


# Print `austin` to confirm the column is indeed removed

data_concat

# %% [markdown]
# #### Next we will perform linear interpolation of the missing data.
# 
# This means that we will use a linear algorithm to estimate the missing data. Linear interpolation assumes that there is a straight line between the points and the missing point will fall on that line. This is a good enough approximation for weather related data. Weather related data is typically a time series. Therefore, we do not want to drop rows from our data if possible. It is prefereable to estimate the missing values rather than remove the rows. However, if you have data from a single point in time, perhaps a better solution would be to remove the rows. 
# 
# If you would like to read more about linear interpolation, you can do so [here](https://en.wikipedia.org/wiki/Linear_interpolation).
# 
# In the following cell, use the `.interpolate()` function on the entire dataframe. This time pass the `inplace=False` argument to the function and assign the interpolated dataframe to a new variable called `austin_fixed` so that we can compare with `austin`.

# %%
#%%
# Your code here
austin_fixed = data_concat.interpolate(inplace=False)
austin_fixed

# %% [markdown]
# #### Check to make sure `austin_fixed` contains no missing data. Also check `austin` - it still contains missing data.

# %%
#%%
# Your code here
austin_fixed.isnull().sum()

# %% [markdown]
# # Challenge 3 - Processing the `Events` Column
# %% [markdown]
# #### Our dataframe contains one true text column - the Events column. We should evaluate this column to determine how to process it.
# 
# Use the `value_counts()` function to evaluate the contents of this column

# %%
#%%
# Your code here:
austin_fixed.Events.value_counts()

# %% [markdown]
# Reading the values of `Events` and reflecting what those values mean in the context of data, you realize this column indicates what weather events had happened in a particular day.
# 
# #### What is the largest number of events happened in a single day? Enter your answer in the next cell.

# %%
# Your answer:
#La mayor cantidad de eventos en un solo día fueron 3: Fog, Rain, Thunderstorm (Son 3 eventos diferentes el mismo día).
#No se si mencionar "Rain" como el evento que mas numero de sucesos tuvo, porque aparece en todas las opciones eventuales, esto
# a su vez nos indica que el clima obtenido en este datafreim es de un clima cálido y que calentamiento global hizo que nevara 1 día.

# %% [markdown]
# #### We want to transform the string-type `Events` values to the numbers. This will allow us to apply machine learning algorithms easily.
# 
# How? We will create a new column for each type of events (i.e. *Rain*, *Snow*, *Fog*, *Thunderstorm*. In each column, we use `1` to indicate if the corresponding event happened in that day and use `0` otherwise.
# 
# Below we provide you a list of all event types. Loop the list and create a dummy column with `0` values for each event in `austin_fixed`. To create a new dummy column with `0` values, simply use `austin_fixed[event] = 0`.


#%%
event_list = ['Snow', 'Fog', 'Rain', 'Thunderstorm']

# Your code here
for x in event_list:
    austin_fixed[x] = 0
    
# Print your new dataframe to check whether new columns have been created:
austin_fixed

# %% [markdown]
# #### Next, populate the actual values in the dummy columns of  `austin_fixed`.
# 
# You will check the *Events* column. If its string value contains `Rain`, then the *Rain* column should be `1`. The same for `Snow`, `Fog`, and `Thunderstorm`.
# 
# *Hints:*
# 
# * Use [`pandas.Series.str.contains()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.contains.html) to create the value series of each new column.
# 
# * What if the values you populated are booleans instead of numbers? You can cast the boolean values to numbers by using `.astype(int)`. For instance, `pd.Series([True, True, False]).astype(int)` will return a new series with values of `[1, 1, 0]`.

# %%
# Your code here
#%%
austin_fixed.Snow = austin_fixed.Events.str.contains('Snow').astype(int)
austin_fixed.Fog = austin_fixed.Events.str.contains('Fog').astype(int)
austin_fixed.Rain = austin_fixed.Events.str.contains('Rain').astype(int)
austin_fixed.Thunderstorm = austin_fixed.Events.str.contains('Thunderstorm').astype(int)

# %% [markdown]
# #### Print out `austin_fixed` to check if the event columns are populated with the intended values

# %%
# Your code here
austin_fixed[['Snow','Fog','Rain','Thunderstorm']]

# %% [markdown]
# #### If your code worked correctly, now we can drop the `Events` column as we don't need it any more.

# %%
# Your code here
try:
    austin_fixed.drop('Events', inplace=True, axis=1)
except:
    print('Ya eliminaste previamente la columna EVENTS')


# %%
austin_fixed.dtypes

# %% [markdown]
# # Challenge 4 - Processing The `Date` Column
# 
# The `Date` column is another non-numeric field in our dataset. A value in that field looks like `'2014-01-06'` which consists of the year, month, and day connected with hyphens. One way to convert the date string to numerical is using a similar approach as we used for `Events`, namely splitting the column into numerical `Year`, `Month`, and `Day` columns. In this challenge we'll show you another way which is to use the Python `datetime` library's `toordinal()` function. Depending on what actual machine learning analysis you will conduct, each approach has its pros and cons. Our goal today is to practice data preparation so we'll skip the discussion here.
# 
# Here you can find the [reference](https://docs.python.org/3/library/datetime.html) and [example](https://stackoverflow.com/questions/39846918/convert-date-to-ordinal-python) for `toordinal`. The basic process is to first convert the string to a `datetime` object using `datetime.datetime.strptime`, then convert the `datetime` object to numerical using `toordinal`.
# 
# #### In the cell below, convert the `Date` column values from string to numeric values using `toordinal()`.

# %%
# Your code here
austin_fixed.Date = datetime.strptime('2013-01-01', '%Y-%m-%d').toordinal()
austin_fixed

# %% [markdown]
# #### Print `austin_fixed` to check your `Date` column.

# %%
austin_fixed.head(5)

# %% [markdown]
# # Challenge 5 - Sampling and Holdout Sets
# %% [markdown]
# #### Now that we have processed the data for machine learning, we will separate the data to test and training sets.
# 
# We first train the model using only the training set. We check our metrics on the training set. We then apply the model to the test set and check our metrics on the test set as well. If the metrics are significantly more optimal on the training set, then we know we have overfit our model. We will need to revise our model to ensure it will be more applicable to data outside the test set.
# %% [markdown]
# #### In the next cells we will separate the data into a training set and a test set using the `train_test_split()` function in scikit-learn.
# 
# When using `scikit-learn` for machine learning, we first separate the data to predictor and response variables. This is the standard way of passing datasets into a model in `scikit-learn`. The `scikit-learn` will then find out whether the predictors and responses fit the model.
# 
# In the next cell, assign the `TempAvgF` column to `y` and the remaining columns to `X`. Your `X` should be a subset of `austin_fixed` containing the following columns: 
# 
# ```['Date',
#  'TempHighF',
#  'TempLowF',
#  'DewPointHighF',
#  'DewPointAvgF',
#  'DewPointLowF',
#  'HumidityHighPercent',
#  'HumidityAvgPercent',
#  'HumidityLowPercent',
#  'SeaLevelPressureHighInches',
#  'SeaLevelPressureAvgInches',
#  'SeaLevelPressureLowInches',
#  'VisibilityHighMiles',
#  'VisibilityAvgMiles',
#  'VisibilityLowMiles',
#  'WindHighMPH',
#  'WindAvgMPH',
#  'WindGustMPH',
#  'Snow',
#  'Fog',
#  'Rain',
#  'Thunderstorm']```
#  
#  Your `y` should be a subset of `austin_fixed` containing one column `TempAvgF`.

# %%
# Your code here:

# %% [markdown]
# In the next cell, import `train_test_split` from `sklearn.model_selection`

# %%
#Your code here:

# %% [markdown]
# Now that we have split the data to predictor and response variables and imported the `train_test_split()` function, split `X` and `y` into `X_train`, `X_test`, `y_train`, and `y_test`. 80% of the data should be in the training set and 20% in the test set. `train_test_split()` reference can be accessed [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
# 
# 
# Enter your code in the cell below:

# %%
#Your code here:

# %% [markdown]
# #### Congratulations! Now you have finished the preparation of the dataset!
# %% [markdown]
# # Bonus Challenge 1
# 
# #### While the above is the common practice to prepare most datasets, when it comes to time series data, we sometimes do not want to randomly select rows from our dataset.
# 
# This is because many time series algorithms rely on observations having equal time distances between them. In such cases, we typically select the majority of rows as the test data and the last few rows as the training data. We don't use `train_test_split()` to select the train/test data because it returns random selections.
# 
# In the following cell, compute the number of rows that account for 80% of our data and round it to the next integer. Assign this number to `ts_rows`.

# %%
# Your code here:

# %% [markdown]
# Assign the first `ts_rows` rows of `X` to `X_ts_train` and the remaining rows to `X_ts_test`.

# %%
# Your code here:

# %% [markdown]
# Assign the first `ts_rows` rows of `y` to `y_ts_train` and the remaining rows to `y_ts_test`.

# %%
# Your code here:

# %% [markdown]
# # Bonus Challenge 2
# 
# As explained in the README, the main purpose of this lab is to show you the typical process of preparing data for machine learning which sometimes takes up 90% of your time. Data cleaning is a valuable skill to learn and you need to be proficient at various techniques including the ones we showed you above as well as others you'll learn in the future. In the real world this skill will help you perform your job successfully and efficiently.
# 
# Now that we're done with data praparation, if you want to expeirence what you'll do in the rest 10% of your time, let's make the final leap.
# 
# We will use scikit-learn's [*Support Vector Machines*](https://scikit-learn.org/stable/modules/svm.html) to compute the fit of our training data set, the test on our test data set.
# 
# #### In the cell below, import `svm` from `sklearn`:

# %%
# Your code here

# %% [markdown]
# #### Now, call `svm.SVC.fit()` on `X_train` and `y_train`. Assign the returned value to a variable called `clf` which stands for *classifier*. Then obtain the test score for `X_test` and `y_test` by calling `clf.score()`.

# %%
# Your code here

# %% [markdown]
# #### You now see the model fit score of your test data set. If it's extremely low, it means your selected model is not a good fit and you should try other models.
# 
# #### In addition to fitting `X_train`, `y_train`, `X_test`, and `y_test`, you can also fit `X_ts_train`, `y_ts_train`, `X_ts_test`, and `y_ts_test` if you completed Bonus Challenge 1.

# %%
# Your code here

# %% [markdown]
# #### We hope you have learned a lot of useful stuff in this lab!
