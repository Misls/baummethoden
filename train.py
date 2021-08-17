#Import Pandas library
import pandas as pd

#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image', 'class']

#Read csv-file
data = pd.read_csv('data_banknote_authentication.txt', names=attribute_names)

#Shuffle data
data = data.sample(frac=1)

#Shows the first 5 rows of the data
data.head()

#Get the absolute number of how many instances in our data belong to class zero
count_real = len(data.loc[data['class']==0])
print('Real bills absolute: ' + str(count_real))

#Get the absolute number of how many instances in our data belong to class one
count_fake = len(data.loc[data['class']==1])
print('Fake bills abolute: ' +str(count_fake))

#Get the relative number of how many instances in our data belong to class zero
percentage_real = count_real/(count_fake+count_real)
print('Real bills in percent: ' + str(round(percentage_real,3)))

#Get the relative number of how many instances in our data belong to class one
percentage_fake = count_fake/(count_real+count_fake)
print('Fake bills in percent: ' + str(round(percentage_fake,3)))

#'class'-column
y_variable = data['class']

#all columns that are not the 'class'-column -> all columns that contain the attributes
x_variables = data.loc[:, data.columns != 'class']

#import method from sklearn to split our data into training and test data
from sklearn.model_selection import train_test_split
import numpy as np

#splits into training and test data
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=0.2)

# shapes of our data splits
print(x_train.shape) 
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#import DecisionTreeClassifier from the Sklearn library
from sklearn.tree import DecisionTreeClassifier

#Create a classifier object 
model = DecisionTreeClassifier()

#Classfier builds Decision Tree with training data
model = model.fit(x_train, y_train) 

#Shows importances of the attributes according to our model 
model.feature_importances_

#Get predicted values from test data 
y_pred = model.predict(x_test) 

from sklearn.metrics import classification_report, confusion_matrix  

#Create the matrix that shows how often predicitons were done correctly and how often theey failed.
conf_mat = confusion_matrix(y_test, y_pred)

#The diagonal ones are the correctly predicted instances. The sum of this number devided by the number of all instances gives us the accuracy in percent.
accuracy = (conf_mat[0,0] + conf_mat[1,1]) /(conf_mat[0,0]+conf_mat[0,1]+ conf_mat[1,0]+conf_mat[1,1])

print('Accuracy: ' + str(round(accuracy,4)))
print('Confusion matrix:')
print(conf_mat)  
print('classification report:')
print(classification_report(y_test, y_pred)) 

from sklearn.model_selection import KFold, cross_val_score

#k_fold object
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

#scores reached with different splits of training/test data 
k_fold_scores = cross_val_score(model, x_variables, y_variable, cv=k_fold, n_jobs=1)

#arithmetic mean of accuracy scores 
mean_accuracy = np.mean(k_fold_scores)

print('k-fold mean accuracy:{}' .format(round(mean_accuracy, 4)))

#save model as pkl-file
import pickle

Pkl_Filename = "Pickle_CART_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)