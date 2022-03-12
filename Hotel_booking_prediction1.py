# -*- coding: utf-8 -*-
import pandas as pd

data_set = 'hotel_bookings.csv'
X = pd.read_csv(data_set)

# to fill company and agency type from null value to an arbitrary float value to
# represent that the cilents that made the bookings are not through any agency or company 
X.agent = X.agent.fillna('0.0')
X.company = X.company.fillna('0.0')

# to remove any other null values from the dataset
X = X.dropna()

y = X.pop('required_car_parking_spaces')

from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# selecting columns that are type object to transform
cols = [0,4,12,13,14,15,19,20,26,29,30]
o_cols = [22]

trans = make_column_transformer((OneHotEncoder(sparse=False,handle_unknown='ignore'),cols),
                                (OrdinalEncoder(),o_cols),
                                remainder= 'passthrough')
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=0) 

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
pipe = make_pipeline(trans,clf)
pipe.fit(X_train,y_train)
pred=pipe.predict(X_test)

# to determine the accuracy of the prediction using KNeighborsClassifier
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test)*100)

# to have an overview of what is the distribution of hotel customers that requires a parking lot
import seaborn as sns
sns.displot(y)


