import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# Loading the data
df = pd.read_csv("/Users/architg/Desktop/Work/Misc./Federated Learning Tutorial (DLW22)/Network Intrusion Dataset/script_train.csv")

# Data processing
df['protocol_type'] = df['protocol_type'].astype('category')
df['protocol_type'] = df['protocol_type'].cat.codes

df['service'] = df['service'].astype('category')
df['service'] = df['service'].cat.codes

df['flag'] = df['flag'].astype('category')
df['flag'] = df['flag'].cat.codes

df['class'] = df['class'].astype('category')
df['class'] = df['class'].cat.codes

# Creating the final dataset
df_final = df[[
    (df.corr()['class']).sort_values()[0:5].index[0],
    (df.corr()['class']).sort_values()[0:5].index[1],
    (df.corr()['class']).sort_values()[0:5].index[2],
    (df.corr()['class']).sort_values()[0:5].index[3],
    (df.corr()['class']).sort_values()[0:5].index[4],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[0],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[1],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[2],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[3],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[4],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[5],
              ]]


# Creating the train & test
train, test = train_test_split(df, test_size=0.2)

x_train = train[[
    (df.corr()['class']).sort_values()[0:5].index[0],
    (df.corr()['class']).sort_values()[0:5].index[1],
    (df.corr()['class']).sort_values()[0:5].index[2],
    (df.corr()['class']).sort_values()[0:5].index[3],
    (df.corr()['class']).sort_values()[0:5].index[4],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[1],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[2],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[3],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[4],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[5]
]]

y_train = train[[
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[0],
]]

x_test = test[[
    (df.corr()['class']).sort_values()[0:5].index[0],
    (df.corr()['class']).sort_values()[0:5].index[1],
    (df.corr()['class']).sort_values()[0:5].index[2],
    (df.corr()['class']).sort_values()[0:5].index[3],
    (df.corr()['class']).sort_values()[0:5].index[4],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[1],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[2],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[3],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[4],
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[5]
]]

y_test = test[[
    (df.corr()['class']).sort_values(ascending=False)[0:6].index[0],
]]

# Building the model
model = LogisticRegression(
    penalty="l2",
    max_iter=1,  # local epoch
    warm_start=True,  # prevent refreshing weights when fitting
)
model.fit(x_train, y_train)

# Measuring accuracy
accuracy = metrics.accuracy_score(y_test, model.predict(x_test))
print(accuracy)
