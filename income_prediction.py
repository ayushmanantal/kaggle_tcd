#Combining Files
import pandas as pd
import numpy as np
frames = pd.DataFrame()
data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
data_test = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
frames = [data, data_test]
df_concat = pd.concat(frames, ignore_index=False, sort=False)
df_concat

#Select Columns with any NaNs
df_concat.loc[:, df_concat.isnull().any()]

#dropping Income column
df_concat = df_concat.drop(['Income'], axis=1)

#dropping Instance column
df_concat = df_concat.drop(['Instance'], axis=1)

#dropping Hair Color column
df_concat = df_concat.drop(['Hair Color'], axis=1)

#checking column with many unique categories
print(df_concat['Country'].value_counts().sort_values(ascending=False).head(10))

#finding out useful categorical variables
for col_name in df_concat.columns:
    if df_concat[col_name].dtypes == 'object':
        unique_cat = len(df_concat[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(
            col_name=col_name, unique_cat=unique_cat))

#removing null values in categorical vars
df_concat.Gender.fillna(df_concat.Gender.mode()[0],inplace=True)
df_concat.Profession.fillna(df_concat.Profession.mode()[0],inplace=True)
df_concat.Country.fillna(df_concat.Country.mode()[0],inplace=True)
df_concat['University Degree'].fillna(df_concat['University Degree'].mode()[0],inplace=True)
df_concat.head(6)

#checking data
df_concat

#Label Encoding
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
cat = ['Gender','Profession','University Degree','Country']
df_concat[cat] = df_concat[cat].astype('category')
df_concat[cat] = df_concat[cat].apply(lambda col: lb_make.fit_transform(col))
df_concat.sample(50)

#for checking the null values
df_concat.isnull().sum()

#removing null values in Age and Year of Record
df_concat.Age.fillna(df_concat.Age.mean(),inplace=True)
df_concat['Year of Record'].fillna(df_concat['Year of Record'].mean(),inplace=True)

# Checking dataset
df_concat['Year of Record'] = df_concat['Year of Record'].astype(np.int64)
df_concat.info()

#check for null values in main data
df_concat

#normalizing data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
norm = ['Year of Record','Size of City', 'Age']
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
df_concat[norm] = scaler.fit_transform(df_concat[norm])

#seperating data
training_data = df_concat[:111993]
training_data = training_data.drop(["Income in EUR"],axis=1)
test_data = df_concat[111993:]
test_data = test_data.drop(["Income in EUR"],axis=1)
test_data.head()

#income data imported
income = pd.read_csv('income.csv')

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(training_data, income, test_size = 0.30, random_state = 42)

#importing xgboost model
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.7, learning_rate = 0.05, max_depth = 6, alpha = 10, gamma = 5, n_estimators = 3000)

#predicting test data
preds = xg_reg.predict(test_data)

#fetching the result into a csv
df = pd.DataFrame({'Income': preds.flatten()})
df.to_csv('submission_file.csv')
