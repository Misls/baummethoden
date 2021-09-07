import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import KFold, cross_val_score
import pickle

#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image', 'class']

#Read csv-file
data = pd.read_csv('data_banknote_authentication.txt', names=attribute_names)

#Shuffle data
data = data.sample(frac=1)
#'class'-column
y_variable = data['class']

#all columns that are not the 'class'-column -> all columns that contain the attributes
x_variables = data.loc[:, data.columns != 'class']

#splits into training and test data
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=0.2)

#Create a classifier object 
model = DecisionTreeClassifier()

#Classfier builds Decision Tree with training data
model = model.fit(x_train, y_train) 

#save model as pkl-file
Pkl_Filename = "Pickle_CART_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)

x_test.to_pickle("./Xdata.pkl")