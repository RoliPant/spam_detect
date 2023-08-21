import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#data collection an preprocessing our data here we lod our data from csv file to pandas dataset
#then we replace null values with a null string or enmpty string
raw_mail_data = pd.read_csv(r'C:\Users\ROLI\Desktop\mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# printing the first 5 rows of the dataframe
#checking the number of rows and columns or checking the size of the dataset
mail_data.head()
mail_data.shape

# label spam mail as 0;  ham mail as 1;
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

# separating the data as texts and label
X = mail_data['Message']
Y = mail_data['Category']

#printing the messages and their categories
print(X)
print(Y)

#spliting the data and giving it size and then random state is used to replenish the code and split the values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

#tells the number of rows and colums in normal model and then training and testing model
print(X.shape)
print(X_train.shape)
print(X_test.shape)

#convert our values into numerical values replaces words repeated multiple times with numerical values
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

#coverting data into the variables (feature vectors(numerical values)) and fit is only used once
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train)
print(X_train_features)

model = LogisticRegression()

# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)

# prediction on training data
prediction_on_training_data = model.predict(X_train_features)
#here we give the true values and the prediction values and compare them
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

input_mail = ["Hi I am roli how are you"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction
prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')